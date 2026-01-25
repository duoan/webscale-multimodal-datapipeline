"""
Metrics aggregator for distributed collection

Aggregates metrics from Ray workers across distributed environment.
"""

from datetime import UTC, datetime
from typing import Any

import ray

from .models import OperatorMetrics, StageMetrics


class MetricsAggregator:
    """Aggregates metrics from distributed Ray workers.

    Collects operator-level metrics from workers and aggregates to stage-level.
    Can be used as a regular class or Ray actor for distributed scenarios.
    """

    def __init__(self, run_id: str):
        """Initialize metrics aggregator.

        Args:
            run_id: Run identifier for collected metrics
        """
        self.run_id = run_id

    def collect_from_workers(self, workers: list[Any]) -> list[OperatorMetrics]:
        """Collect operator metrics from all workers in a stage.

        Args:
            workers: List of Ray worker actors

        Returns:
            List of operator metrics from all workers
        """
        all_metrics = []

        for worker_idx, worker in enumerate(workers):
            try:
                # Get stats from worker
                worker_stats = ray.get(worker.get_operator_stats.remote())

                # Convert stats dict to OperatorMetrics
                for op_name, op_stats in worker_stats.items():
                    # Extract stage name from worker (if available)
                    # For now, use a placeholder - will be set by caller
                    stage_name = "unknown"
                    worker_id = f"worker_{worker_idx}"

                    metrics = OperatorMetrics(
                        run_id=self.run_id,
                        stage_name=stage_name,
                        operator_name=op_name,
                        worker_id=worker_id,
                        timestamp=datetime.now(UTC),
                        input_records=op_stats.get("input_records", 0),
                        output_records=op_stats.get("output_records", 0),
                        pass_rate=op_stats.get("pass_rate", 0.0),
                        total_time=op_stats.get("total_time", 0.0),
                        avg_latency=op_stats.get("avg_latency", 0.0),
                        min_latency=op_stats.get("min_latency", 0.0),
                        max_latency=op_stats.get("max_latency", 0.0),
                        p50_latency=op_stats.get("p50_latency", 0.0),
                        p95_latency=op_stats.get("p95_latency", 0.0),
                        p99_latency=op_stats.get("p99_latency", 0.0),
                        throughput=op_stats.get("throughput", 0.0),
                        error_count=0,
                        custom_metrics={},
                    )

                    all_metrics.append(metrics)

            except Exception as e:
                print(f"Failed to collect metrics from worker {worker_idx}: {e}")
                continue

        return all_metrics

    def aggregate_to_stage_metrics(self, operator_metrics: list[OperatorMetrics], stage_name: str) -> StageMetrics:
        """Aggregate operator metrics to stage-level metrics.

        Args:
            operator_metrics: List of operator metrics for this stage
            stage_name: Name of the stage

        Returns:
            Aggregated stage metrics
        """
        if not operator_metrics:
            return StageMetrics(
                run_id=self.run_id,
                stage_name=stage_name,
                timestamp=datetime.now(UTC),
                num_workers=0,
                input_records=0,
                output_records=0,
                pass_rate=0.0,
                total_time=0.0,
                avg_throughput=0.0,
                min_throughput=0.0,
                max_throughput=0.0,
                error_count=0,
                operator_metrics=[],
            )

        # Aggregate metrics
        # Key insight:
        # - Within a stage, operators execute serially
        # - Each operator may have multiple workers (data parallelism)
        # - Workers split and process different parts of data (additive within operator)
        #
        # So: Group by operator_name, sum within operator, then take first/last for stage

        from collections import defaultdict
        ops_by_name = defaultdict(list)
        for m in operator_metrics:
            ops_by_name[m.operator_name].append(m)

        # Preserve operator order by timestamp (execution order)
        operator_first_metrics = []
        for op_name, metrics_list in ops_by_name.items():
            first_metric = min(metrics_list, key=lambda m: m.timestamp)
            operator_first_metrics.append((op_name, first_metric.timestamp))

        operator_first_metrics.sort(key=lambda x: x[1])
        operator_names_ordered = [op_name for op_name, _ in operator_first_metrics]

        # Stage input = first operator's total input (sum across workers)
        # Stage output = last operator's total output (sum across workers)
        if operator_names_ordered:
            first_op = operator_names_ordered[0]
            last_op = operator_names_ordered[-1]

            total_input = sum(m.input_records for m in ops_by_name[first_op])
            total_output = sum(m.output_records for m in ops_by_name[last_op])
        else:
            total_input = 0
            total_output = 0

        num_workers = len({m.worker_id for m in operator_metrics})
        pass_rate = (100.0 * total_output / total_input) if total_input > 0 else 0.0

        # Use max time as bottleneck
        total_time = max((m.total_time for m in operator_metrics), default=0.0)

        # Calculate throughput statistics
        throughputs = [m.throughput for m in operator_metrics if m.throughput > 0]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0
        min_throughput = min(throughputs) if throughputs else 0.0
        max_throughput = max(throughputs) if throughputs else 0.0

        total_errors = sum(m.error_count for m in operator_metrics)

        # Use earliest timestamp
        earliest_timestamp = min((m.timestamp for m in operator_metrics), default=datetime.now(UTC))

        return StageMetrics(
            run_id=self.run_id,
            stage_name=stage_name,
            timestamp=earliest_timestamp,
            num_workers=num_workers,
            input_records=total_input,
            output_records=total_output,
            pass_rate=pass_rate,
            total_time=total_time,
            avg_throughput=avg_throughput,
            min_throughput=min_throughput,
            max_throughput=max_throughput,
            error_count=total_errors,
            operator_metrics=operator_metrics,
        )

    def collect_stage_metrics(self, workers: list[Any], stage_name: str) -> StageMetrics:
        """Collect and aggregate metrics for a stage.

        Convenience method that combines collection and aggregation.

        Args:
            workers: List of Ray worker actors for this stage
            stage_name: Name of the stage

        Returns:
            Aggregated stage metrics
        """
        # Collect operator metrics from all workers
        operator_metrics = self.collect_from_workers(workers)

        # Update stage name for all metrics
        operator_metrics = [
            OperatorMetrics(
                run_id=m.run_id,
                stage_name=stage_name,
                operator_name=m.operator_name,
                worker_id=m.worker_id,
                timestamp=m.timestamp,
                input_records=m.input_records,
                output_records=m.output_records,
                pass_rate=m.pass_rate,
                total_time=m.total_time,
                avg_latency=m.avg_latency,
                min_latency=m.min_latency,
                max_latency=m.max_latency,
                p50_latency=m.p50_latency,
                p95_latency=m.p95_latency,
                p99_latency=m.p99_latency,
                throughput=m.throughput,
                error_count=m.error_count,
                custom_metrics=m.custom_metrics,
            )
            for m in operator_metrics
        ]

        # Aggregate to stage level
        return self.aggregate_to_stage_metrics(operator_metrics, stage_name)


@ray.remote
class DistributedMetricsAggregator:
    """Ray actor for distributed metrics aggregation.

    Use this when you need centralized metrics collection across a cluster.
    For single-node scenarios, MetricsAggregator is sufficient.
    """

    def __init__(self, run_id: str):
        """Initialize distributed aggregator.

        Args:
            run_id: Run identifier
        """
        self.aggregator = MetricsAggregator(run_id)
        self._collected_metrics: list[OperatorMetrics] = []

    def collect_from_workers(self, workers: list[Any]) -> list[OperatorMetrics]:
        """Collect metrics from workers.

        Args:
            workers: List of Ray worker actors

        Returns:
            List of operator metrics
        """
        metrics = self.aggregator.collect_from_workers(workers)
        self._collected_metrics.extend(metrics)
        return metrics

    def aggregate_to_stage_metrics(self, operator_metrics: list[OperatorMetrics], stage_name: str) -> StageMetrics:
        """Aggregate operator metrics to stage metrics.

        Args:
            operator_metrics: List of operator metrics
            stage_name: Stage name

        Returns:
            Stage metrics
        """
        return self.aggregator.aggregate_to_stage_metrics(operator_metrics, stage_name)

    def get_all_metrics(self) -> list[OperatorMetrics]:
        """Get all collected metrics.

        Returns:
            All collected operator metrics
        """
        return list(self._collected_metrics)

    def clear(self):
        """Clear collected metrics."""
        self._collected_metrics.clear()
