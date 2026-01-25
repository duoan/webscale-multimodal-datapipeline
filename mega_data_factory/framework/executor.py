"""
Executor: Coordinates workers and manages pipeline execution

Provides Executor for orchestrating the entire data processing pipeline.
"""

from collections.abc import Iterator
from typing import Any

import ray

from .backend import DedupBackend
from .config import PipelineConfig
from .metrics import MetricsAggregator, MetricsCollector, MetricsWriter
from .operator import Deduplicator
from .registry import DataLoaderRegistry, DataWriterRegistry, OperatorRegistry
from .worker import RayWorker


class Executor:
    """Executor coordinates workers and manages pipeline execution."""

    def __init__(self, config: PipelineConfig):
        """Initialize executor from configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Check if rejected samples collection is enabled
        self.collect_rejected = config.executor.rejected_samples.enabled

        # Initialize metrics collection
        self.metrics_enabled = config.executor.metrics.enabled
        self.metrics_collector: MetricsCollector | None = None
        self.metrics_writer: MetricsWriter | None = None
        self.metrics_aggregator: MetricsAggregator | None = None

        if self.metrics_enabled:
            self.metrics_collector = MetricsCollector()
            self.metrics_writer = MetricsWriter(config.executor.metrics.output_path)
            self.metrics_aggregator = MetricsAggregator(self.metrics_collector.run_id)

            # Store config snapshot for metrics
            config_snapshot = {
                "max_samples": config.executor.max_samples,
                "batch_size": config.executor.batch_size,
                "num_stages": len(config.stages),
                "stage_names": [s.name for s in config.stages],
            }
            self.metrics_collector.set_config(config_snapshot)

        # Create data loader
        self.data_loader = DataLoaderRegistry.create(config.data_loader.type, config.data_loader.params)

        # Initialize Ray (before creating operators that might need it)
        if not ray.is_initialized():
            # In distributed clusters, Ray is already initialized.
            # For local development, only limit if explicitly configured in config.
            # In cluster mode or when num_cpus is None, let Ray manage resources automatically.
            if config.executor.num_cpus is not None:
                ray.init(num_cpus=config.executor.num_cpus, ignore_reinit_error=True)
            else:
                # No explicit limit - let Ray use all available CPUs
                # This works correctly in both local and cluster mode
                ray.init(ignore_reinit_error=True)

        # Create shared dedup backend if there are Deduplicator operators
        shared_dedup_backend = None
        if ray.is_initialized():
            # Check if we have any Deduplicator operators across all stages
            has_dedup = False
            for stage_config in config.stages:
                for op_config in stage_config.operators:
                    if op_config.enabled:
                        class_name = op_config.get_class_name()
                        if class_name in OperatorRegistry._operators:
                            op_class = OperatorRegistry._operators[class_name]
                            if issubclass(op_class, Deduplicator):
                                has_dedup = True
                                break
                if has_dedup:
                    break

            # Create shared Ray backend for all Deduplicator operators (with bucketing)
            if has_dedup:
                # Use bucketing to avoid single actor bottleneck
                num_buckets = config.executor.dedup_num_buckets
                # Enable representative tracking when collecting rejected samples
                # This allows us to record which sample was the "first seen" for each dedup key
                track_representative = self.collect_rejected
                shared_dedup_backend = DedupBackend(
                    num_buckets=num_buckets,
                    name_prefix="pipeline_dedup_backend",
                    track_representative=track_representative,
                )
                # Reset backend state at start of each run to clear previous data
                shared_dedup_backend.reset()

        # Create data writer (each worker will get its own instance)
        # Store writer config to create instances per worker
        self.writer_config = config.data_writer

        # Setup rejected samples writer config if enabled
        self.rejected_writer_config = None
        if self.collect_rejected:
            rejected_config = config.executor.rejected_samples
            # Use configured writer type or fall back to main writer type
            rejected_writer_type = rejected_config.writer_type or config.data_writer.type
            # Use configured output path or derive from main output path
            rejected_output_path = rejected_config.output_path
            if not rejected_output_path:
                # Derive from main output path by appending "_rejected"
                main_output_path = config.data_writer.params.get("output_path", "output")
                rejected_output_path = f"{main_output_path}_rejected"

            self.rejected_writer_config = {
                "type": rejected_writer_type,
                "params": {
                    **config.data_writer.params,
                    "output_path": rejected_output_path,
                    "table_name": config.data_writer.params.get("table_name", "default") + "_rejected",
                },
            }
            print(f"Rejected samples collection enabled. Output: {rejected_output_path}")

        # Distributed loading is always enabled for mega-scale processing
        print(
            f"Distributed data loading: {config.data_loader.num_workers} workers, "
            f"batch_size={config.executor.batch_size}"
        )

        # Create DataLoaderWorker actors
        self.loader_workers: list[Any] = []
        self._create_loader_workers(config)

        # Create stages (Ray Actors)
        # Store workers grouped by stage (each stage has multiple replicas)
        self.stages: list[list[Any]] = []  # List of stage groups
        self.stage_indices: list[int] = []  # Round-robin indices per stage for load balancing

        # All stages write to the same Iceberg table
        # Iceberg supports concurrent writes and schema evolution
        writer_params = self.writer_config.params.copy()

        for stage_config in config.stages:
            # Create operators for this stage
            stage_operators = []
            for op_config in stage_config.operators:
                if op_config.enabled:
                    # Derive class name from operator name
                    class_name = op_config.get_class_name()
                    op_instance = OperatorRegistry.create(class_name, op_config.params)

                    # If this is a Deduplicator operator and we have a shared backend, use it
                    if isinstance(op_instance, Deduplicator) and shared_dedup_backend is not None:
                        op_instance.backend = shared_dedup_backend

                    stage_operators.append(op_instance)

            # Dynamic worker allocation with min/max replicas
            min_replicas = stage_config.worker.min_replicas
            max_replicas = stage_config.worker.max_replicas

            print(
                f"  Creating workers for stage '{stage_config.name}': "
                f"min={min_replicas}, max={max_replicas}"
            )

            # Try to create up to max_replicas workers
            stage_workers = []  # Successfully created workers
            failed_count = 0

            for replica_id in range(max_replicas):
                try:
                    # All workers share the same writer configuration
                    # Iceberg supports concurrent writes from multiple workers to the same table
                    worker_writer_params = writer_params.copy()
                    worker_writer = DataWriterRegistry.create(self.writer_config.type, worker_writer_params)

                    # Create rejected writer if enabled
                    rejected_writer = None
                    if self.collect_rejected and self.rejected_writer_config:
                        rejected_writer = DataWriterRegistry.create(
                            self.rejected_writer_config["type"],
                            self.rejected_writer_config["params"].copy(),
                        )

                    worker_name = f"{stage_config.name}_{replica_id}" if max_replicas > 1 else stage_config.name

                    # Extract num_cpus and num_gpus from resources if present, otherwise use defaults
                    num_cpus = stage_config.worker.resources.get("cpu", 1)
                    num_gpus = stage_config.worker.resources.get("gpu", 0)
                    # Create a copy of resources without 'cpu' and 'gpu' (Ray handles these separately)
                    ray_resources = {k: v for k, v in stage_config.worker.resources.items() if k not in ("cpu", "gpu")}

                    # Create Ray Actor worker with name for Ray Dashboard visibility
                    # Ray Actor names must be unique
                    actor_name = f"pipeline_{worker_name}"
                    worker = RayWorker.options(
                        name=actor_name,
                        num_cpus=num_cpus,
                        num_gpus=num_gpus,
                        resources=ray_resources if ray_resources else None,
                    ).remote(
                        worker_name,
                        stage_operators,
                        data_writer=worker_writer,
                        rejected_writer=rejected_writer,
                        collect_rejected=self.collect_rejected,
                    )

                    stage_workers.append(worker)

                except Exception as e:
                    failed_count += 1
                    print(f"    ⚠️  Failed to create worker {replica_id}: {e}")
                    # Stop if we can't reach min_replicas even if we create all remaining workers
                    if (len(stage_workers) + (max_replicas - replica_id - 1)) < min_replicas:
                        raise RuntimeError(
                            f"Cannot reach min_replicas={min_replicas} for stage '{stage_config.name}': "
                            f"only {len(stage_workers)} created, {failed_count} failed"
                        )

            # Check we have at least min_replicas
            actual_workers = len(stage_workers)
            if actual_workers < min_replicas:
                raise RuntimeError(
                    f"Failed to create minimum workers for stage '{stage_config.name}': "
                    f"created {actual_workers}, required {min_replicas}"
                )

            if failed_count > 0:
                print(
                    f"    ⚠️  {failed_count} workers failed to create, "
                    f"using {actual_workers}/{max_replicas} workers"
                )
            print(f"    ✅ {actual_workers}/{max_replicas} workers created for stage '{stage_config.name}'")

            # Add all ready workers for this stage as a group
            self.stages.append(stage_workers)
            self.stage_indices.append(0)

    def _create_loader_workers(self, config: PipelineConfig):
        """Create DataLoaderWorker actors for distributed data loading.

        Two-layer architecture:
        1. Executor (here): Scan files and assign to workers
        2. Workers: Read assigned files

        Args:
            config: Pipeline configuration
        """
        from .loader_worker import DataLoaderWorker

        num_workers = config.data_loader.num_workers
        batch_size = config.executor.batch_size
        checkpoint_interval = config.data_loader.checkpoint_interval

        # Calculate max_records per worker if executor.max_samples is set
        max_records_per_worker = None
        if config.executor.max_samples is not None:
            max_records_per_worker = config.executor.max_samples // num_workers
            print(
                f"Max samples: {config.executor.max_samples} total, "
                f"~{max_records_per_worker} per worker ({num_workers} workers)"
            )

        print(f"Creating {num_workers} DataLoaderWorker actors...")

        # Layer 1: Executor scans files if loader supports it
        assigned_files_per_worker = None
        if hasattr(self.data_loader, "get_file_list"):
            print("  Scanning data files...")
            all_files = self.data_loader.get_file_list()
            total_files = len(all_files)

            # Divide files among workers
            files_per_worker = total_files // num_workers
            remainder = total_files % num_workers

            assigned_files_per_worker = []
            for worker_id in range(num_workers):
                if worker_id < remainder:
                    start_file = worker_id * (files_per_worker + 1)
                    end_file = start_file + files_per_worker + 1
                else:
                    start_file = worker_id * files_per_worker + remainder
                    end_file = start_file + files_per_worker

                assigned_files = all_files[start_file:end_file]
                assigned_files_per_worker.append(assigned_files)
                print(f"  Worker {worker_id}: {len(assigned_files)} files (file {start_file}-{end_file - 1})")

        # Layer 2: Create workers with assigned files
        for shard_id in range(num_workers):
            assigned_files = assigned_files_per_worker[shard_id] if assigned_files_per_worker else None

            worker = DataLoaderWorker.options(
                name=f"pipeline_loader_{shard_id}",
                num_cpus=1,  # Each loader uses 1 CPU
            ).remote(
                data_loader=self.data_loader,
                shard_id=shard_id,
                num_shards=num_workers,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                assigned_files=assigned_files,  # Pass assigned files
                max_records=max_records_per_worker,  # Pass max_records per worker
            )
            self.loader_workers.append(worker)

        print(f"Created {len(self.loader_workers)} DataLoaderWorker actors")

    def _submit_batch_chain(self, records: list[dict[str, Any]]) -> tuple[ray.ObjectRef, list[ray.ObjectRef]]:
        """Submit a batch through all stages using Ray ObjectRef chaining.

        Core magic: Pass ObjectRefs like a chain. Ray automatically waits for
        previous task to complete and unpacks the result for the next task.

        Only the last stage writes data to improve I/O efficiency.

        Args:
            records: List of input records

        Returns:
            Tuple of (final_ref, all_intermediate_refs) for cleanup
        """
        # Start with initial data (can be actual data or ObjectRef if from elsewhere)
        # Here we put raw data into object store to get a Ref for unified handling
        current_ref = ray.put(records)
        all_refs = [current_ref]  # Track all refs for cleanup

        # Chain through all stages
        # Only the last stage writes data (should_write=True)
        num_stages = len(self.stages)
        for stage_idx, worker_group in enumerate(self.stages):
            # Round-robin select a worker from this stage
            worker_idx = self.stage_indices[stage_idx] % len(worker_group)
            worker = worker_group[worker_idx]
            self.stage_indices[stage_idx] += 1

            # Key point: Pass the previous stage's Ref directly to the next
            # Ray automatically handles dependencies: Worker won't execute until current_ref is ready
            # Only last stage writes data (better I/O efficiency)
            is_last_stage = stage_idx == num_stages - 1
            print(f"    Submitting to stage {stage_idx + 1}/{num_stages} (write={is_last_stage})...")
            current_ref = worker.process_batch_with_records.remote(current_ref, should_write=is_last_stage)
            all_refs.append(current_ref)

        print(f"    All {num_stages} stages submitted, returning final ref")
        return current_ref, all_refs

    def _collect_completed(self, batch_pipeline: dict, max_in_flight: int) -> Iterator[tuple]:
        """Check and yield completed tasks, with ObjectRef cleanup.

        Always checks for completed batches (non-blocking), even if pipeline is not full.
        This ensures we yield results as soon as they're ready for true parallelism.

        Args:
            batch_pipeline: Dict of batch_id -> (future, input_count, all_refs)
            max_in_flight: Maximum number of batches to keep in flight (0 = wait for all)

        Yields:
            (input_count, output_count) for completed batches
        """
        # Get all pending futures (filter out None)
        pending_ids = [bid for bid in batch_pipeline.keys() if batch_pipeline[bid][0] is not None]
        pending_futures = [batch_pipeline[bid][0] for bid in pending_ids]

        if not pending_futures:
            return

        # Always check for ready batches (non-blocking check)
        # If pipeline is full, we must wait for at least one to make room
        # Otherwise, just check what's already ready without waiting
        if max_in_flight > 0 and len(batch_pipeline) >= max_in_flight:
            # Pipeline full: wait for at least one to complete (blocking)
            ready_refs, _ = ray.wait(pending_futures, num_returns=1, timeout=None)
        else:
            # Pipeline not full: just check what's already ready (non-blocking)
            ready_refs, _ = ray.wait(pending_futures, num_returns=1, timeout=0.0)

        if not ready_refs:
            # No batches ready yet (checking {len(pending_futures)} pending batches)
            return

        # Process all completed tasks
        for ref in ready_refs:
            # Find batch_id for this ref
            completed_batch_id = None
            for bid, (fut, _, _) in batch_pipeline.items():
                if fut == ref:
                    completed_batch_id = bid
                    break

            if completed_batch_id is not None:
                input_count, all_refs = batch_pipeline[completed_batch_id][1], batch_pipeline[completed_batch_id][2]
                try:
                    result = ray.get(ref, timeout=1.0)
                    output_count = len(result) if result else 0
                    yield (input_count, output_count)
                except Exception as e:
                    print(f"Task failed for batch {completed_batch_id}: {e}")
                    yield (input_count, 0)
                finally:
                    # CRITICAL: Free all ObjectRefs from Ray object store to prevent memory leak
                    for obj_ref in all_refs:
                        try:
                            ray._private.internal_api.free([obj_ref])
                        except Exception:
                            pass  # Ignore errors during cleanup

                # Remove from pipeline
                del batch_pipeline[completed_batch_id]

    def execute(self) -> Iterator[tuple]:
        """Execute the pipeline with Ray ObjectRef chaining for true pipeline parallelism.

        Uses Ray's ObjectRef passing to chain tasks across stages without blocking.
        Driver submits batches instantly (only constructs task graph), then immediately
        submits next. Stage 1 can be processing batch 2 while Stage 2 processes batch 1.

        Yields:
            Tuple of (input_count, output_count) per batch
        """
        # Wrap execution in metrics tracking if enabled
        if self.metrics_enabled and self.metrics_collector:
            yield from self._execute_with_metrics()
        else:
            yield from self._execute_impl()

    def _execute_impl(self) -> Iterator[tuple]:
        """Internal implementation of execute with distributed loading.

        Data loading is always distributed using DataLoaderWorker actors.
        """
        yield from self._execute_distributed()

    def _execute_distributed(self) -> Iterator[tuple]:
        """Distributed loading with streaming pipeline parallelism.

        Each loader worker streams batches as they're ready, enabling true
        pipeline parallelism where data loading and processing happen concurrently.
        """
        print("🚀 Starting distributed data loading with streaming pipeline...")
        print(f"   Loader workers: {len(self.loader_workers)}")
        print(f"   Processing stages: {len(self.stages)}")
        print(f"   Batch size: {self.config.executor.batch_size}")

        # Calculate max records per worker if max_samples is set
        max_records_per_worker = None
        if self.config.executor.max_samples:
            max_records_per_worker = self.config.executor.max_samples // len(self.loader_workers)
            print(f"   Max records per worker: {max_records_per_worker}")

        # Maintain a pipeline of batches being processed across stages
        batch_pipeline = {}  # batch_id -> (future, input_count, all_refs)
        next_batch_id = 0
        total_workers = sum(len(stage) for stage in self.stages)

        # BACKPRESSURE CONTROL: Aggressively limit in-flight batches
        # Slower downstream stages (e.g., embedding) can cause memory buildup
        # Keep pipeline shallow to prevent backpressure
        max_in_flight = min(4, max(2, total_workers // 4))  # Very conservative
        print(f"   Max in-flight batches: {max_in_flight} (backpressure control)")

        # Track loader states
        active_loaders = set(range(len(self.loader_workers)))  # Worker IDs still loading
        loader_futures = {}  # worker_id -> pending future for next batch
        loader_batch_counts = {i: 0 for i in range(len(self.loader_workers))}
        loader_wait_count = 0  # Track how often loaders wait due to backpressure

        # Start requesting first batch from each loader (up to max_in_flight)
        print("Starting streaming from DataLoaderWorker actors...")
        initial_requests = min(max_in_flight, len(self.loader_workers))
        for worker_id in list(active_loaders)[:initial_requests]:
            future = self.loader_workers[worker_id].get_next_batch.remote(
                max_records=max_records_per_worker
            )
            loader_futures[worker_id] = future

        # Main loop: continuously poll loader workers and submit batches
        while active_loaders or batch_pipeline:
            # BACKPRESSURE: Only request new batches if pipeline has capacity
            pipeline_has_capacity = len(batch_pipeline) < max_in_flight

            # Check for ready batches from any active loader
            if loader_futures:
                ready_futures = list(loader_futures.values())
                ready_refs, _ = ray.wait(ready_futures, num_returns=1, timeout=0.01)

                if ready_refs:
                    # Find which worker produced this batch
                    completed_ref = ready_refs[0]
                    worker_id = None
                    for wid, future in loader_futures.items():
                        if future == completed_ref:
                            worker_id = wid
                            break

                    if worker_id is not None:
                        # Get batch result
                        result = ray.get(completed_ref)
                        # CRITICAL: Free the loader's result ref immediately after getting data
                        try:
                            ray._private.internal_api.free([completed_ref])
                        except Exception:
                            pass
                        del loader_futures[worker_id]

                        batch = result["batch"]
                        completed = result["completed"]

                        if batch:
                            # Submit batch immediately to processing pipeline
                            loader_batch_counts[worker_id] += 1
                            final_ref, all_refs = self._submit_batch_chain(batch)
                            batch_pipeline[next_batch_id] = (final_ref, len(batch), all_refs)
                            next_batch_id += 1

                            # Collect completed processing batches (non-blocking)
                            for res in self._collect_completed(batch_pipeline, max_in_flight):
                                yield res

                        # BACKPRESSURE: Only request next batch if pipeline has capacity
                        if not completed and pipeline_has_capacity:
                            future = self.loader_workers[worker_id].get_next_batch.remote(
                                max_records=max_records_per_worker
                            )
                            loader_futures[worker_id] = future
                        elif not completed:
                            # Loader is waiting due to backpressure
                            loader_wait_count += 1
                            if loader_wait_count % 10 == 0:
                                print(
                                    f"⚠️  Backpressure: Pipeline full ({len(batch_pipeline)}/{max_in_flight}), "
                                    f"loader {worker_id} waiting..."
                                )
                        else:
                            # Loader completed
                            active_loaders.remove(worker_id)
                            print(
                                f"[Loader {worker_id}] Completed. "
                                f"Produced {loader_batch_counts[worker_id]} batches. "
                                f"Active loaders: {len(active_loaders)}"
                            )

            # Collect completed processing batches (non-blocking)
            # This creates capacity for new batches
            for res in self._collect_completed(batch_pipeline, max_in_flight):
                yield res

            # BACKPRESSURE: After collecting, check if we can resume paused loaders
            if pipeline_has_capacity and not loader_futures:
                # Find a paused loader (active but not currently loading)
                for worker_id in active_loaders:
                    if worker_id not in loader_futures:
                        future = self.loader_workers[worker_id].get_next_batch.remote(
                            max_records=max_records_per_worker
                        )
                        loader_futures[worker_id] = future
                        break  # Only resume one at a time

        # Wait for all remaining processing tasks to complete
        print("All loaders completed, waiting for remaining processing...")
        while batch_pipeline:
            for res in self._collect_completed(batch_pipeline, 0):  # 0 means wait for all
                yield res

        print(f"✅ Streaming pipeline completed. Total loaders: {len(self.loader_workers)}")

    def _execute_with_metrics(self) -> Iterator[tuple]:
        """Execute pipeline with metrics tracking enabled."""
        if not self.metrics_collector:
            yield from self._execute_impl()
            return

        with self.metrics_collector.track_run():
            # Execute the pipeline
            yield from self._execute_impl()

            # Collect metrics from all workers after execution completes
            # (must be before track_run() exits so stage metrics are available for run metrics calculation)
            print("\nCollecting metrics from workers...")
            self._collect_metrics_from_workers()

        # Write metrics to Parquet
        # (must be after track_run() exits so run metrics are calculated in finally block)
        if self.config.executor.metrics.write_on_completion and self.metrics_writer:
            print("Writing metrics to Parquet...")
            self._write_metrics()
            print(f"Metrics written to: {self.metrics_writer.output_path}")

            # Generate HTML report if enabled
            if hasattr(self.config.executor.metrics, "generate_report") and self.config.executor.metrics.generate_report:
                self._generate_metrics_report()

    def _collect_metrics_from_workers(self):
        """Collect operator metrics from all workers and aggregate to stage metrics."""
        if not self.metrics_aggregator or not self.metrics_collector:
            return

        for worker_group, stage_config in zip(self.stages, self.config.stages, strict=False):
            stage_name = stage_config.name

            # Collect metrics from all workers in this stage
            stage_metrics = self.metrics_aggregator.collect_stage_metrics(worker_group, stage_name)

            # Add stage metrics to collector
            self.metrics_collector.add_stage_metrics(stage_metrics)

            # Add operator metrics to collector
            for op_metrics in stage_metrics.operator_metrics:
                self.metrics_collector.add_operator_metrics(op_metrics)

    def _write_metrics(self):
        """Write all collected metrics to Parquet files."""
        if not self.metrics_writer or not self.metrics_collector:
            return

        # Get all metrics
        run_metrics = self.metrics_collector.get_run_metrics()
        stage_metrics = self.metrics_collector.get_stage_metrics()
        operator_metrics = self.metrics_collector.get_operator_metrics()

        # Write to Parquet
        self.metrics_writer.write_all(
            run_metrics=run_metrics,
            stage_metrics=stage_metrics,
            operator_metrics=operator_metrics,
        )

    def _generate_metrics_report(self):
        """Generate single-run HTML metrics report and optionally publish to HuggingFace."""
        if not self.metrics_writer or not self.metrics_collector:
            return

        from .metrics import MetricsReporter

        print("\nGenerating single-run metrics report...")
        reporter = MetricsReporter(self.metrics_writer.output_path)

        # Get run_id
        run_id = self.metrics_collector.run_id

        # Generate HTML report for this run
        report_path = reporter.generate_single_run_report(run_id=run_id)
        print(f"✓ Report generated: {report_path}")

        # Publish to HuggingFace if configured
        metrics_config = self.config.executor.metrics
        if hasattr(metrics_config, "huggingface_repo") and metrics_config.huggingface_repo:
            try:
                print(f"\nPublishing report to HuggingFace Space: {metrics_config.huggingface_repo}")
                hf_token = getattr(metrics_config, "huggingface_token", None)
                space_url = reporter.publish_to_huggingface(
                    report_path=report_path,
                    repo_id=metrics_config.huggingface_repo,
                    token=hf_token,
                )
                print(f"✓ Report published: {space_url}")
            except Exception as e:
                print(f"Warning: Failed to publish to HuggingFace: {e}")

    def get_operator_stats(self) -> dict[str, dict[str, Any]]:
        """Collect performance statistics from all operators across all workers.

        Returns:
            Dictionary mapping stage_name -> operator_name -> statistics
            Also includes stage-level summary with total throughput
        """
        stats = {}

        # Collect stats from all workers
        for stage_idx, worker_group in enumerate(self.stages):
            stage_name = f"stage_{stage_idx}"
            stats[stage_name] = {}

            # Get stats from ALL workers and aggregate (not just first worker)
            if worker_group:
                try:
                    # Collect stats from all workers
                    all_worker_stats = []
                    for worker in worker_group:
                        try:
                            worker_stats = ray.get(worker.get_operator_stats.remote())
                            all_worker_stats.append(worker_stats)
                        except Exception:
                            # Skip unavailable workers
                            continue

                    if not all_worker_stats:
                        continue

                    # Aggregate statistics across all workers
                    # Group by operator name
                    aggregated_stats = {}
                    for worker_stats in all_worker_stats:
                        for op_name, op_stats in worker_stats.items():
                            if op_name not in aggregated_stats:
                                aggregated_stats[op_name] = {
                                    "input_records": 0,
                                    "output_records": 0,
                                    "total_time": 0.0,
                                    "min_latency": float("inf"),
                                    "max_latency": 0.0,
                                }

                            # Aggregate records and time (sum across all workers)
                            aggregated_stats[op_name]["input_records"] += op_stats.get("input_records", 0)
                            aggregated_stats[op_name]["output_records"] += op_stats.get("output_records", 0)
                            aggregated_stats[op_name]["total_time"] += op_stats.get("total_time", 0.0)

                            # Aggregate min/max latencies
                            aggregated_stats[op_name]["min_latency"] = min(
                                aggregated_stats[op_name]["min_latency"],
                                op_stats.get("min_latency", 0.0),
                            )
                            aggregated_stats[op_name]["max_latency"] = max(
                                aggregated_stats[op_name]["max_latency"],
                                op_stats.get("max_latency", 0.0),
                            )

                    # Calculate final statistics for each operator
                    for op_name, agg_stats in aggregated_stats.items():
                        input_records = agg_stats["input_records"]
                        output_records = agg_stats["output_records"]
                        total_time = agg_stats["total_time"]

                        if input_records > 0 and total_time > 0:
                            pass_rate = 100.0 * output_records / input_records
                            avg_latency = total_time / input_records
                            throughput = input_records / total_time

                            # Use percentile from first worker as approximation
                            # (full percentile calculation would require all latency data)
                            p50 = p95 = p99 = avg_latency
                            if all_worker_stats:
                                first_op_stats = all_worker_stats[0].get(op_name, {})
                                p50 = first_op_stats.get("p50_latency", avg_latency)
                                p95 = first_op_stats.get("p95_latency", avg_latency)
                                p99 = first_op_stats.get("p99_latency", avg_latency)

                            stats[stage_name][op_name] = {
                                "input_records": input_records,
                                "output_records": output_records,
                                "pass_rate": pass_rate,
                                "total_time": total_time,
                                "avg_latency": avg_latency,
                                "min_latency": agg_stats["min_latency"]
                                if agg_stats["min_latency"] != float("inf")
                                else 0.0,
                                "max_latency": agg_stats["max_latency"],
                                "p50_latency": p50,
                                "p95_latency": p95,
                                "p99_latency": p99,
                                "throughput": throughput,
                            }

                    # Calculate stage-level throughput
                    # Use the slowest operator's time (bottleneck) and total records processed
                    total_stage_time = 0.0
                    total_stage_input = 0
                    total_stage_output = 0
                    for op_stats in stats[stage_name].values():
                        total_stage_time = max(
                            total_stage_time, op_stats.get("total_time", 0.0)
                        )  # Bottleneck (max time)
                        total_stage_input = max(total_stage_input, op_stats.get("input_records", 0))
                        total_stage_output = max(total_stage_output, op_stats.get("output_records", 0))

                    if total_stage_time > 0 and total_stage_input > 0:
                        stage_throughput = total_stage_input / total_stage_time
                        stage_pass_rate = 100.0 * total_stage_output / total_stage_input
                        stats[stage_name]["_stage_summary"] = {
                            "input_records": total_stage_input,
                            "output_records": total_stage_output,
                            "pass_rate": stage_pass_rate,
                            "total_time": total_stage_time,
                            "throughput": stage_throughput,
                        }
                except Exception:
                    # Log error but continue (worker might be unavailable or still processing)
                    # Set empty stats to indicate no statistics available
                    stats[stage_name] = {}

        return stats

    def shutdown(self):
        """Shutdown executor and cleanup resources."""
        # Close all worker writers to flush any buffered data
        # Note: Ray Actor workers' writers are cleaned up with the actor lifecycle
        # No explicit close needed as Ray handles cleanup

        if ray.is_initialized():
            ray.shutdown()
