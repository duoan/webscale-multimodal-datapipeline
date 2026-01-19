"""
Executor: Coordinates workers and manages pipeline execution

Provides Executor for orchestrating the entire data processing pipeline.
"""

from collections.abc import Iterator
from typing import Any

import ray

from .backend import DedupBackend
from .config import PipelineConfig
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
                shared_dedup_backend = DedupBackend(num_buckets=num_buckets, name_prefix="pipeline_dedup_backend")
                # Reset backend state at start of each run to clear previous data
                shared_dedup_backend.reset()

        # Create data writer (each worker will get its own instance)
        # Store writer config to create instances per worker
        self.writer_config = config.data_writer

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

            # Create num_replicas instances for this stage
            num_replicas = stage_config.worker.num_replicas or 1
            stage_workers = []  # Collect all workers for this stage

            for replica_id in range(num_replicas):
                # All workers share the same writer configuration
                # Iceberg supports concurrent writes from multiple workers to the same table
                worker_writer_params = writer_params.copy()
                worker_writer = DataWriterRegistry.create(self.writer_config.type, worker_writer_params)

                worker_name = f"{stage_config.name}_{replica_id}" if num_replicas > 1 else stage_config.name

                # Extract num_cpus from resources if present, otherwise use default
                num_cpus = stage_config.worker.resources.get("cpu", 1)
                # Create a copy of resources without 'cpu' (Ray handles num_cpus separately)
                ray_resources = {k: v for k, v in stage_config.worker.resources.items() if k != "cpu"}

                # Create Ray Actor worker with name for Ray Dashboard visibility
                # Ray Actor names must be unique
                actor_name = f"pipeline_{worker_name}"
                worker = RayWorker.options(
                    name=actor_name, num_cpus=num_cpus, resources=ray_resources if ray_resources else None
                ).remote(worker_name, stage_operators, data_writer=worker_writer, num_cpus=num_cpus)

                stage_workers.append(worker)

            # Add all workers for this stage as a group
            self.stages.append(stage_workers)
            self.stage_indices.append(0)

    def _submit_batch_chain(self, records: list[dict[str, Any]]) -> ray.ObjectRef:
        """Submit a batch through all stages using Ray ObjectRef chaining.

        Core magic: Pass ObjectRefs like a chain. Ray automatically waits for
        previous task to complete and unpacks the result for the next task.

        Only the last stage writes data to improve I/O efficiency.

        Args:
            records: List of input records

        Returns:
            Ray ObjectRef to the final result
        """
        # Start with initial data (can be actual data or ObjectRef if from elsewhere)
        # Here we put raw data into object store to get a Ref for unified handling
        current_ref = ray.put(records)

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

        print(f"    All {num_stages} stages submitted, returning final ref")
        return current_ref

    def _collect_completed(self, batch_pipeline: dict, max_in_flight: int) -> Iterator[tuple]:
        """Check and yield completed tasks.

        Always checks for completed batches (non-blocking), even if pipeline is not full.
        This ensures we yield results as soon as they're ready for true parallelism.

        Args:
            batch_pipeline: Dict of batch_id -> (future, input_count)
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
            for bid, (fut, _) in batch_pipeline.items():
                if fut == ref:
                    completed_batch_id = bid
                    break

            if completed_batch_id is not None:
                input_count = batch_pipeline[completed_batch_id][1]
                try:
                    result = ray.get(ref, timeout=1.0)
                    output_count = len(result) if result else 0
                    yield (input_count, output_count)
                except Exception as e:
                    print(f"Task failed for batch {completed_batch_id}: {e}")
                    yield (input_count, 0)

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
        # Load data
        print("Loading data stream...")
        data_stream = self.data_loader.load()
        print("Data stream loaded, starting to process records...")

        # Process in batches with pipeline parallelism
        records = []
        count = 0

        # Maintain a pipeline of batches being processed across stages
        batch_pipeline = {}  # batch_id -> (future, input_count)
        next_batch_id = 0
        # Limit concurrent batches to avoid overwhelming system
        # Calculate based on number of stages and workers
        total_workers = sum(len(stage) for stage in self.stages)
        max_in_flight = min(8, max(2, total_workers // 2))  # Reasonable limit: 2-8 batches

        print(
            f"Starting to iterate data stream (max_in_flight={max_in_flight}, batch_size={self.config.executor.batch_size})..."
        )
        for record in data_stream:
            records.append(record)
            count += 1

            if count % 50 == 0:
                print(f"  Collected {count} records (batch: {len(records)}/{self.config.executor.batch_size})...")

            if len(records) >= self.config.executor.batch_size:
                print(f"  Batch full ({len(records)} records), submitting to {len(self.stages)} stages...")
                # 1. Submit task chain using ObjectRef chaining
                future = self._submit_batch_chain(records)
                print(f"  Batch submitted, future: {future}")

                # 2. Record task
                batch_pipeline[next_batch_id] = (future, len(records))
                next_batch_id += 1
                records = []

                # 3. Collect all completed batches (non-blocking if pipeline not full)
                # This ensures results are yielded as soon as they're ready
                # For first batch, wait a bit to ensure it starts processing
                if next_batch_id == 1:
                    # First batch: wait a bit then check
                    import time

                    time.sleep(0.1)

                while True:
                    completed = list(self._collect_completed(batch_pipeline, max_in_flight))
                    if not completed:
                        break
                    for res in completed:
                        print(f"  Batch completed: {res}")
                        yield res

            if self.config.executor.max_samples and count >= self.config.executor.max_samples:
                break

        # Process remaining records
        if records:
            future = self._submit_batch_chain(records)
            batch_pipeline[next_batch_id] = (future, len(records))

        # Wait for all remaining tasks to complete
        while batch_pipeline:
            for res in self._collect_completed(batch_pipeline, 0):  # 0 means wait for all
                yield res

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
                                    "total_records": 0,
                                    "total_time": 0.0,
                                    "min_latency": float("inf"),
                                    "max_latency": 0.0,
                                }

                            # Aggregate records and time (sum across all workers)
                            aggregated_stats[op_name]["total_records"] += op_stats.get("total_records", 0)
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
                        total_records = agg_stats["total_records"]
                        total_time = agg_stats["total_time"]

                        if total_records > 0 and total_time > 0:
                            avg_latency = total_time / total_records
                            throughput = total_records / total_time

                            # Use percentile from first worker as approximation
                            # (full percentile calculation would require all latency data)
                            p50 = p95 = p99 = avg_latency
                            if all_worker_stats:
                                first_op_stats = all_worker_stats[0].get(op_name, {})
                                p50 = first_op_stats.get("p50_latency", avg_latency)
                                p95 = first_op_stats.get("p95_latency", avg_latency)
                                p99 = first_op_stats.get("p99_latency", avg_latency)

                            stats[stage_name][op_name] = {
                                "total_records": total_records,
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
                    total_stage_records = 0
                    for op_stats in stats[stage_name].values():
                        total_stage_time = max(
                            total_stage_time, op_stats.get("total_time", 0.0)
                        )  # Bottleneck (max time)
                        total_stage_records = max(
                            total_stage_records, op_stats.get("total_records", 0)
                        )  # Records processed by slowest operator

                    if total_stage_time > 0 and total_stage_records > 0:
                        stage_throughput = total_stage_records / total_stage_time
                        stats[stage_name]["_stage_summary"] = {
                            "total_records": total_stage_records,
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
