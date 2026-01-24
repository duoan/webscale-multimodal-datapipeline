"""
Metrics collector with context manager auto-instrumentation

Provides clean, non-intrusive metrics collection via context managers.
"""

import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from .models import OperatorMetrics, RunMetrics, StageMetrics


class MetricsCollector:
    """Collects metrics at run/stage/operator levels using context managers.

    Thread-safe for use in distributed Ray environment.
    """

    def __init__(self, run_id: str | None = None):
        """Initialize metrics collector.

        Args:
            run_id: Optional run identifier (auto-generated if not provided)
        """
        self.run_id = run_id or self._generate_run_id()
        self._run_start_time: float | None = None
        self._run_end_time: float | None = None
        self._stage_start_times: dict[str, float] = {}
        self._operator_start_times: dict[str, float] = {}

        # Collected metrics
        self._operator_metrics: list[OperatorMetrics] = []
        self._stage_metrics: list[StageMetrics] = []
        self._run_metrics: RunMetrics | None = None

        # Configuration snapshot
        self._config: dict[str, Any] = {}

    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{short_uuid}"

    def set_config(self, config: dict[str, Any]):
        """Set pipeline configuration for this run."""
        self._config = config

    @contextmanager
    def track_run(self):
        """Context manager for tracking entire pipeline run.

        Yields:
            Self for method chaining

        Example:
            with collector.track_run():
                # Execute pipeline
                pass
        """
        self._run_start_time = time.perf_counter()
        start_datetime = datetime.now()

        try:
            yield self
        finally:
            self._run_end_time = time.perf_counter()
            end_datetime = datetime.now()
            duration = self._run_end_time - self._run_start_time

            # Aggregate stage metrics
            total_input = sum(s.input_records for s in self._stage_metrics)
            total_output = sum(s.output_records for s in self._stage_metrics)
            overall_pass_rate = (100.0 * total_output / total_input) if total_input > 0 else 0.0
            avg_throughput = total_input / duration if duration > 0 else 0.0
            total_errors = sum(s.error_count for s in self._stage_metrics)

            self._run_metrics = RunMetrics(
                run_id=self.run_id,
                start_time=start_datetime,
                end_time=end_datetime,
                duration=duration,
                num_stages=len(self._stage_metrics),
                total_input_records=total_input,
                total_output_records=total_output,
                overall_pass_rate=overall_pass_rate,
                avg_throughput=avg_throughput,
                total_errors=total_errors,
                stage_metrics=list(self._stage_metrics),
                config=self._config,
            )

    @contextmanager
    def track_stage(self, stage_name: str):
        """Context manager for tracking stage execution.

        Args:
            stage_name: Name of the stage

        Yields:
            Stage context object

        Example:
            with collector.track_stage("stage_0"):
                # Execute stage
                pass
        """
        stage_key = f"{self.run_id}_{stage_name}"
        self._stage_start_times[stage_key] = time.perf_counter()
        start_datetime = datetime.now()

        # Snapshot operator metrics before stage
        operator_metrics_before = len(self._operator_metrics)

        class StageContext:
            """Context object for stage tracking."""

            def __init__(self, collector: "MetricsCollector", stage_name: str):
                self.collector = collector
                self.stage_name = stage_name
                self.custom_metrics: dict[str, Any] = {}

            def add_custom_metric(self, name: str, value: Any):
                """Add custom metric for this stage."""
                self.custom_metrics[name] = value

        stage_ctx = StageContext(self, stage_name)

        try:
            yield stage_ctx
        finally:
            stage_end_time = time.perf_counter()
            stage_duration = stage_end_time - self._stage_start_times[stage_key]

            # Get operator metrics collected during this stage
            operator_metrics_after = len(self._operator_metrics)
            stage_operator_metrics = self._operator_metrics[operator_metrics_before:operator_metrics_after]

            if stage_operator_metrics:
                # Aggregate operator metrics for this stage
                num_workers = len(stage_operator_metrics)
                total_input = sum(m.input_records for m in stage_operator_metrics)
                total_output = sum(m.output_records for m in stage_operator_metrics)
                pass_rate = (100.0 * total_output / total_input) if total_input > 0 else 0.0
                total_time = max((m.total_time for m in stage_operator_metrics), default=0.0)

                throughputs = [m.throughput for m in stage_operator_metrics if m.throughput > 0]
                avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0
                min_throughput = min(throughputs) if throughputs else 0.0
                max_throughput = max(throughputs) if throughputs else 0.0

                total_errors = sum(m.error_count for m in stage_operator_metrics)

                stage_metrics = StageMetrics(
                    run_id=self.run_id,
                    stage_name=stage_name,
                    timestamp=start_datetime,
                    num_workers=num_workers,
                    input_records=total_input,
                    output_records=total_output,
                    pass_rate=pass_rate,
                    total_time=total_time,
                    avg_throughput=avg_throughput,
                    min_throughput=min_throughput,
                    max_throughput=max_throughput,
                    error_count=total_errors,
                    operator_metrics=stage_operator_metrics,
                )

                self._stage_metrics.append(stage_metrics)

            del self._stage_start_times[stage_key]

    @contextmanager
    def track_operator(self, operator_name: str, stage_name: str, worker_id: str):
        """Context manager for tracking operator execution.

        Args:
            operator_name: Name of the operator
            stage_name: Name of the stage this operator belongs to
            worker_id: Unique identifier for the worker

        Yields:
            Operator context object

        Example:
            with collector.track_operator("ImageMetadataRefiner", "stage_0", "worker_0"):
                # Execute operator
                pass
        """
        op_key = f"{self.run_id}_{stage_name}_{operator_name}_{worker_id}"
        self._operator_start_times[op_key] = time.perf_counter()
        start_datetime = datetime.now()

        class OperatorContext:
            """Context object for operator tracking."""

            def __init__(
                self,
                collector: "MetricsCollector",
                operator_name: str,
                stage_name: str,
                worker_id: str,
            ):
                self.collector = collector
                self.operator_name = operator_name
                self.stage_name = stage_name
                self.worker_id = worker_id
                self.input_records = 0
                self.output_records = 0
                self.error_count = 0
                self.custom_metrics: dict[str, Any] = {}

            def update_from_stats(self, stats: dict[str, Any]):
                """Update metrics from operator stats dictionary."""
                self.input_records = stats.get("input_records", 0)
                self.output_records = stats.get("output_records", 0)
                self.error_count = stats.get("error_count", 0)

            def add_custom_metric(self, name: str, value: Any):
                """Add custom metric for this operator."""
                self.custom_metrics[name] = value

        op_ctx = OperatorContext(self, operator_name, stage_name, worker_id)

        try:
            yield op_ctx
        finally:
            op_end_time = time.perf_counter()
            total_time = op_end_time - self._operator_start_times[op_key]

            # Calculate derived metrics
            pass_rate = (
                (100.0 * op_ctx.output_records / op_ctx.input_records) if op_ctx.input_records > 0 else 0.0
            )
            throughput = op_ctx.input_records / total_time if total_time > 0 else 0.0
            avg_latency = total_time / op_ctx.input_records if op_ctx.input_records > 0 else 0.0

            # Create operator metrics (using simplified stats since we don't have detailed latencies)
            operator_metrics = OperatorMetrics(
                run_id=self.run_id,
                stage_name=stage_name,
                operator_name=operator_name,
                worker_id=worker_id,
                timestamp=start_datetime,
                input_records=op_ctx.input_records,
                output_records=op_ctx.output_records,
                pass_rate=pass_rate,
                total_time=total_time,
                avg_latency=avg_latency,
                min_latency=avg_latency,  # Simplified
                max_latency=avg_latency,  # Simplified
                p50_latency=avg_latency,  # Simplified
                p95_latency=avg_latency,  # Simplified
                p99_latency=avg_latency,  # Simplified
                throughput=throughput,
                error_count=op_ctx.error_count,
                custom_metrics=op_ctx.custom_metrics,
            )

            self._operator_metrics.append(operator_metrics)
            del self._operator_start_times[op_key]

    def add_operator_metrics(self, metrics: OperatorMetrics):
        """Add pre-computed operator metrics.

        Used when integrating with existing operator stats collection.

        Args:
            metrics: Operator metrics to add
        """
        self._operator_metrics.append(metrics)

    def get_operator_metrics(self) -> list[OperatorMetrics]:
        """Get all collected operator metrics."""
        return list(self._operator_metrics)

    def get_stage_metrics(self) -> list[StageMetrics]:
        """Get all collected stage metrics."""
        return list(self._stage_metrics)

    def get_run_metrics(self) -> RunMetrics | None:
        """Get run metrics (only available after run completes)."""
        return self._run_metrics

    def clear(self):
        """Clear all collected metrics."""
        self._operator_metrics.clear()
        self._stage_metrics.clear()
        self._run_metrics = None
        self._stage_start_times.clear()
        self._operator_start_times.clear()
