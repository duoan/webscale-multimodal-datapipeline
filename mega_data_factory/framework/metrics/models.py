"""
Metrics data models for three-level collection

Defines immutable dataclasses for Run, Stage, and Operator level metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class OperatorMetrics:
    """Operator-level metrics for individual processing units.

    Attributes:
        run_id: Unique identifier for the pipeline run
        stage_name: Name of the stage this operator belongs to
        operator_name: Name of the operator
        worker_id: Unique identifier for the Ray worker
        timestamp: When the metrics were collected
        input_records: Total number of input records processed
        output_records: Total number of output records (after filtering)
        pass_rate: Percentage of records that passed through (0-100)
        total_time: Total processing time in seconds
        avg_latency: Average latency per record in seconds
        min_latency: Minimum latency in seconds
        max_latency: Maximum latency in seconds
        p50_latency: 50th percentile latency in seconds
        p95_latency: 95th percentile latency in seconds
        p99_latency: 99th percentile latency in seconds
        throughput: Records processed per second
        error_count: Number of errors encountered
        custom_metrics: Additional operator-specific metrics
    """

    run_id: str
    stage_name: str
    operator_name: str
    worker_id: str
    timestamp: datetime
    input_records: int
    output_records: int
    pass_rate: float
    total_time: float
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    error_count: int = 0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "operator_name": self.operator_name,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "input_records": self.input_records,
            "output_records": self.output_records,
            "pass_rate": self.pass_rate,
            "total_time": self.total_time,
            "avg_latency": self.avg_latency,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "throughput": self.throughput,
            "error_count": self.error_count,
            "custom_metrics": self.custom_metrics,
        }


@dataclass(frozen=True)
class StageMetrics:
    """Stage-level aggregated metrics for a group of operators.

    Attributes:
        run_id: Unique identifier for the pipeline run
        stage_name: Name of the stage
        timestamp: When the metrics were collected
        num_workers: Number of workers in this stage
        input_records: Total input records across all workers
        output_records: Total output records across all workers
        pass_rate: Overall pass rate (0-100)
        total_time: Maximum processing time across workers (bottleneck)
        avg_throughput: Average throughput across all workers
        min_throughput: Minimum throughput across workers
        max_throughput: Maximum throughput across workers
        error_count: Total errors across all workers
        operator_metrics: List of individual operator metrics
    """

    run_id: str
    stage_name: str
    timestamp: datetime
    num_workers: int
    input_records: int
    output_records: int
    pass_rate: float
    total_time: float
    avg_throughput: float
    min_throughput: float
    max_throughput: float
    error_count: int = 0
    operator_metrics: list[OperatorMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "num_workers": self.num_workers,
            "input_records": self.input_records,
            "output_records": self.output_records,
            "pass_rate": self.pass_rate,
            "total_time": self.total_time,
            "avg_throughput": self.avg_throughput,
            "min_throughput": self.min_throughput,
            "max_throughput": self.max_throughput,
            "error_count": self.error_count,
        }


@dataclass(frozen=True)
class RunMetrics:
    """Run-level aggregated metrics for entire pipeline execution.

    Attributes:
        run_id: Unique identifier for the pipeline run
        start_time: When the pipeline started
        end_time: When the pipeline completed
        duration: Total execution time in seconds
        num_stages: Number of stages in the pipeline
        total_input_records: Total input records across all stages
        total_output_records: Total output records across all stages
        overall_pass_rate: Overall pass rate (0-100)
        avg_throughput: Average throughput across entire run
        total_errors: Total errors across entire run
        stage_metrics: List of stage-level metrics
        config: Pipeline configuration snapshot
    """

    run_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    num_stages: int
    total_input_records: int
    total_output_records: int
    overall_pass_rate: float
    avg_throughput: float
    total_errors: int = 0
    stage_metrics: list[StageMetrics] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_stages": self.num_stages,
            "total_input_records": self.total_input_records,
            "total_output_records": self.total_output_records,
            "overall_pass_rate": self.overall_pass_rate,
            "avg_throughput": self.avg_throughput,
            "total_errors": self.total_errors,
            "config": self.config,
        }
