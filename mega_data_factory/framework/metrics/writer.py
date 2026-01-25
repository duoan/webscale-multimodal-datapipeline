"""
Metrics writer for Parquet persistence

Writes metrics data to Parquet files with Superset-compatible schema.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .models import OperatorMetrics, RunMetrics, StageMetrics


class MetricsWriter:
    """Writes metrics to Parquet files organized by level (run/stage/operator).

    Output structure:
        metrics_path/
        ├── runs/run_{timestamp}.parquet
        ├── stages/stages_{timestamp}.parquet
        └── operators/operators_{timestamp}.parquet
    """

    # Parquet schemas for each metrics level
    OPERATOR_SCHEMA = pa.schema(
        [
            ("run_id", pa.string()),
            ("stage_name", pa.string()),
            ("operator_name", pa.string()),
            ("worker_id", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("input_records", pa.int64()),
            ("output_records", pa.int64()),
            ("pass_rate", pa.float64()),
            ("total_time", pa.float64()),
            ("avg_latency", pa.float64()),
            ("min_latency", pa.float64()),
            ("max_latency", pa.float64()),
            ("p50_latency", pa.float64()),
            ("p95_latency", pa.float64()),
            ("p99_latency", pa.float64()),
            ("throughput", pa.float64()),
            ("error_count", pa.int64()),
            ("custom_metrics", pa.string()),  # JSON string
        ]
    )

    STAGE_SCHEMA = pa.schema(
        [
            ("run_id", pa.string()),
            ("stage_name", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("num_workers", pa.int64()),
            ("input_records", pa.int64()),
            ("output_records", pa.int64()),
            ("pass_rate", pa.float64()),
            ("total_time", pa.float64()),
            ("avg_throughput", pa.float64()),
            ("min_throughput", pa.float64()),
            ("max_throughput", pa.float64()),
            ("error_count", pa.int64()),
        ]
    )

    RUN_SCHEMA = pa.schema(
        [
            ("run_id", pa.string()),
            ("start_time", pa.timestamp("us")),
            ("end_time", pa.timestamp("us")),
            ("duration", pa.float64()),
            ("num_stages", pa.int64()),
            ("total_input_records", pa.int64()),
            ("total_output_records", pa.int64()),
            ("overall_pass_rate", pa.float64()),
            ("avg_throughput", pa.float64()),
            ("total_errors", pa.int64()),
            ("config", pa.string()),  # JSON string
        ]
    )

    def __init__(self, output_path: str | Path):
        """Initialize metrics writer.

        Args:
            output_path: Base directory for metrics output
        """
        self.output_path = Path(output_path)
        self.runs_path = self.output_path / "runs"
        self.stages_path = self.output_path / "stages"
        self.operators_path = self.output_path / "operators"

        # Create directories
        self.runs_path.mkdir(parents=True, exist_ok=True)
        self.stages_path.mkdir(parents=True, exist_ok=True)
        self.operators_path.mkdir(parents=True, exist_ok=True)

    def write_operator_metrics(self, metrics: list[OperatorMetrics], run_id: str):
        """Write operator metrics to Parquet.

        Args:
            metrics: List of operator metrics
            run_id: Run identifier for filename
        """
        if not metrics:
            return

        # Convert to Arrow format
        data = {
            "run_id": [m.run_id for m in metrics],
            "stage_name": [m.stage_name for m in metrics],
            "operator_name": [m.operator_name for m in metrics],
            "worker_id": [m.worker_id for m in metrics],
            "timestamp": [m.timestamp for m in metrics],
            "input_records": [m.input_records for m in metrics],
            "output_records": [m.output_records for m in metrics],
            "pass_rate": [m.pass_rate for m in metrics],
            "total_time": [m.total_time for m in metrics],
            "avg_latency": [m.avg_latency for m in metrics],
            "min_latency": [m.min_latency for m in metrics],
            "max_latency": [m.max_latency for m in metrics],
            "p50_latency": [m.p50_latency for m in metrics],
            "p95_latency": [m.p95_latency for m in metrics],
            "p99_latency": [m.p99_latency for m in metrics],
            "throughput": [m.throughput for m in metrics],
            "error_count": [m.error_count for m in metrics],
            "custom_metrics": [json.dumps(m.custom_metrics) for m in metrics],
        }

        table = pa.table(data, schema=self.OPERATOR_SCHEMA)

        # Write to Parquet with compression
        output_file = self.operators_path / f"operators_{run_id}.parquet"
        pq.write_table(
            table,
            output_file,
            compression="snappy",
            use_dictionary=True,
        )

    def write_stage_metrics(self, metrics: list[StageMetrics], run_id: str):
        """Write stage metrics to Parquet.

        Args:
            metrics: List of stage metrics
            run_id: Run identifier for filename
        """
        if not metrics:
            return

        # Convert to Arrow format
        data = {
            "run_id": [m.run_id for m in metrics],
            "stage_name": [m.stage_name for m in metrics],
            "timestamp": [m.timestamp for m in metrics],
            "num_workers": [m.num_workers for m in metrics],
            "input_records": [m.input_records for m in metrics],
            "output_records": [m.output_records for m in metrics],
            "pass_rate": [m.pass_rate for m in metrics],
            "total_time": [m.total_time for m in metrics],
            "avg_throughput": [m.avg_throughput for m in metrics],
            "min_throughput": [m.min_throughput for m in metrics],
            "max_throughput": [m.max_throughput for m in metrics],
            "error_count": [m.error_count for m in metrics],
        }

        table = pa.table(data, schema=self.STAGE_SCHEMA)

        # Write to Parquet with compression
        output_file = self.stages_path / f"stages_{run_id}.parquet"
        pq.write_table(
            table,
            output_file,
            compression="snappy",
            use_dictionary=True,
        )

    def write_run_metrics(self, metrics: RunMetrics):
        """Write run metrics to Parquet.

        Args:
            metrics: Run metrics
        """
        # Convert to Arrow format
        data = {
            "run_id": [metrics.run_id],
            "start_time": [metrics.start_time],
            "end_time": [metrics.end_time],
            "duration": [metrics.duration],
            "num_stages": [metrics.num_stages],
            "total_input_records": [metrics.total_input_records],
            "total_output_records": [metrics.total_output_records],
            "overall_pass_rate": [metrics.overall_pass_rate],
            "avg_throughput": [metrics.avg_throughput],
            "total_errors": [metrics.total_errors],
            "config": [json.dumps(metrics.config)],
        }

        table = pa.table(data, schema=self.RUN_SCHEMA)

        # Write to Parquet with compression
        # Note: run_id already contains "run_" prefix, so don't add it again
        output_file = self.runs_path / f"{metrics.run_id}.parquet"
        pq.write_table(
            table,
            output_file,
            compression="snappy",
            use_dictionary=True,
        )

    def write_all(
        self,
        run_metrics: RunMetrics | None = None,
        stage_metrics: list[StageMetrics] | None = None,
        operator_metrics: list[OperatorMetrics] | None = None,
    ):
        """Write all metrics at once.

        Args:
            run_metrics: Run metrics to write
            stage_metrics: Stage metrics to write
            operator_metrics: Operator metrics to write
        """
        if run_metrics:
            run_id = run_metrics.run_id
            self.write_run_metrics(run_metrics)
        elif stage_metrics:
            run_id = stage_metrics[0].run_id if stage_metrics else "unknown"
        elif operator_metrics:
            run_id = operator_metrics[0].run_id if operator_metrics else "unknown"
        else:
            return

        if stage_metrics:
            self.write_stage_metrics(stage_metrics, run_id)

        if operator_metrics:
            self.write_operator_metrics(operator_metrics, run_id)

    def read_operator_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Read operator metrics from Parquet.

        Args:
            run_id: Run identifier

        Returns:
            List of operator metrics as dictionaries
        """
        parquet_file = self.operators_path / f"operators_{run_id}.parquet"
        if not parquet_file.exists():
            return []

        table = pq.read_table(parquet_file)
        return table.to_pylist()

    def read_stage_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Read stage metrics from Parquet.

        Args:
            run_id: Run identifier

        Returns:
            List of stage metrics as dictionaries
        """
        parquet_file = self.stages_path / f"stages_{run_id}.parquet"
        if not parquet_file.exists():
            return []

        table = pq.read_table(parquet_file)
        return table.to_pylist()

    def read_run_metrics(self, run_id: str) -> dict[str, Any] | None:
        """Read run metrics from Parquet.

        Args:
            run_id: Run identifier

        Returns:
            Run metrics as dictionary, or None if not found
        """
        parquet_file = self.runs_path / f"run_{run_id}.parquet"
        if not parquet_file.exists():
            return None

        table = pq.read_table(parquet_file)
        rows = table.to_pylist()
        return rows[0] if rows else None

    def list_runs(self) -> list[str]:
        """List all available run IDs.

        Returns:
            List of run IDs
        """
        run_files = self.runs_path.glob("run_*.parquet")
        return [f.stem.replace("run_", "") for f in run_files]
