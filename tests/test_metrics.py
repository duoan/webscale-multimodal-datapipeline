"""
Unit tests for metrics module

Tests MetricsCollector, MetricsWriter, and MetricsAggregator.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mega_data_factory.framework.metrics import (
    MetricsAggregator,
    MetricsCollector,
    MetricsWriter,
    OperatorMetrics,
    StageMetrics,
)


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = MetricsCollector()
        assert collector.run_id is not None
        assert collector.run_id.startswith("run_")

    def test_initialization_with_run_id(self):
        """Test collector initialization with custom run_id."""
        custom_id = "test_run_123"
        collector = MetricsCollector(run_id=custom_id)
        assert collector.run_id == custom_id

    def test_track_run_context_manager(self):
        """Test track_run context manager."""
        collector = MetricsCollector()

        with collector.track_run():
            pass

        run_metrics = collector.get_run_metrics()
        assert run_metrics is not None
        assert run_metrics.run_id == collector.run_id
        assert run_metrics.duration >= 0
        assert run_metrics.start_time <= run_metrics.end_time

    def test_track_stage_context_manager(self):
        """Test track_stage context manager."""
        collector = MetricsCollector()

        # Create dummy operator metrics
        op_metrics = OperatorMetrics(
            run_id=collector.run_id,
            stage_name="test_stage",
            operator_name="TestOperator",
            worker_id="worker_0",
            timestamp=datetime.now(),
            input_records=100,
            output_records=90,
            pass_rate=90.0,
            total_time=1.0,
            avg_latency=0.01,
            min_latency=0.005,
            max_latency=0.02,
            p50_latency=0.01,
            p95_latency=0.015,
            p99_latency=0.018,
            throughput=100.0,
        )
        collector.add_operator_metrics(op_metrics)

        with collector.track_stage("test_stage"):
            pass

        stage_metrics = collector.get_stage_metrics()
        assert len(stage_metrics) == 1
        assert stage_metrics[0].stage_name == "test_stage"

    def test_add_operator_metrics(self):
        """Test adding operator metrics directly."""
        collector = MetricsCollector()

        metrics = OperatorMetrics(
            run_id=collector.run_id,
            stage_name="test_stage",
            operator_name="TestOperator",
            worker_id="worker_0",
            timestamp=datetime.now(),
            input_records=100,
            output_records=90,
            pass_rate=90.0,
            total_time=1.0,
            avg_latency=0.01,
            min_latency=0.005,
            max_latency=0.02,
            p50_latency=0.01,
            p95_latency=0.015,
            p99_latency=0.018,
            throughput=100.0,
        )

        collector.add_operator_metrics(metrics)
        operator_metrics = collector.get_operator_metrics()

        assert len(operator_metrics) == 1
        assert operator_metrics[0].operator_name == "TestOperator"


class TestMetricsWriter:
    """Test MetricsWriter functionality."""

    def test_initialization(self):
        """Test writer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)
            assert writer.output_path.exists()
            assert writer.runs_path.exists()
            assert writer.stages_path.exists()
            assert writer.operators_path.exists()

    def test_write_operator_metrics(self):
        """Test writing operator metrics to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)

            metrics = [
                OperatorMetrics(
                    run_id="test_run",
                    stage_name="test_stage",
                    operator_name="TestOperator",
                    worker_id="worker_0",
                    timestamp=datetime.now(),
                    input_records=100,
                    output_records=90,
                    pass_rate=90.0,
                    total_time=1.0,
                    avg_latency=0.01,
                    min_latency=0.005,
                    max_latency=0.02,
                    p50_latency=0.01,
                    p95_latency=0.015,
                    p99_latency=0.018,
                    throughput=100.0,
                )
            ]

            writer.write_operator_metrics(metrics, "test_run")

            # Verify file exists
            output_file = writer.operators_path / "operators_test_run.parquet"
            assert output_file.exists()

            # Read back and verify
            read_metrics = writer.read_operator_metrics("test_run")
            assert len(read_metrics) == 1
            assert read_metrics[0]["operator_name"] == "TestOperator"

    def test_write_empty_metrics(self):
        """Test writing empty metrics list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)
            writer.write_operator_metrics([], "test_run")

            # Should not create file
            output_file = writer.operators_path / "operators_test_run.parquet"
            assert not output_file.exists()

    def test_list_runs(self):
        """Test listing available runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)

            # Write metrics for multiple runs
            for i in range(3):
                metrics = [
                    OperatorMetrics(
                        run_id=f"run_{i}",
                        stage_name="test_stage",
                        operator_name="TestOperator",
                        worker_id="worker_0",
                        timestamp=datetime.now(),
                        input_records=100,
                        output_records=90,
                        pass_rate=90.0,
                        total_time=1.0,
                        avg_latency=0.01,
                        min_latency=0.005,
                        max_latency=0.02,
                        p50_latency=0.01,
                        p95_latency=0.015,
                        p99_latency=0.018,
                        throughput=100.0,
                    )
                ]
                writer.write_operator_metrics(metrics, f"run_{i}")

            runs = writer.list_runs()
            # Note: list_runs looks for run files in runs/ directory,
            # but we wrote to operators/ directory
            assert len(runs) == 0  # Expected since we didn't write run metrics


class TestMetricsAggregator:
    """Test MetricsAggregator functionality."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = MetricsAggregator("test_run")
        assert aggregator.run_id == "test_run"

    def test_aggregate_to_stage_metrics(self):
        """Test aggregating operator metrics to stage metrics."""
        aggregator = MetricsAggregator("test_run")

        operator_metrics = [
            OperatorMetrics(
                run_id="test_run",
                stage_name="test_stage",
                operator_name="Op1",
                worker_id="worker_0",
                timestamp=datetime.now(),
                input_records=100,
                output_records=90,
                pass_rate=90.0,
                total_time=1.0,
                avg_latency=0.01,
                min_latency=0.005,
                max_latency=0.02,
                p50_latency=0.01,
                p95_latency=0.015,
                p99_latency=0.018,
                throughput=100.0,
            ),
            OperatorMetrics(
                run_id="test_run",
                stage_name="test_stage",
                operator_name="Op2",
                worker_id="worker_1",
                timestamp=datetime.now(),
                input_records=100,
                output_records=80,
                pass_rate=80.0,
                total_time=1.5,
                avg_latency=0.015,
                min_latency=0.008,
                max_latency=0.025,
                p50_latency=0.015,
                p95_latency=0.02,
                p99_latency=0.023,
                throughput=66.67,
            ),
        ]

        stage_metrics = aggregator.aggregate_to_stage_metrics(operator_metrics, "test_stage")

        assert stage_metrics.stage_name == "test_stage"
        assert stage_metrics.num_workers == 2
        assert stage_metrics.input_records == 200  # Sum of both operators
        assert stage_metrics.output_records == 170  # Sum of both operators
        assert stage_metrics.pass_rate == pytest.approx(85.0, rel=0.01)  # 170/200

    def test_aggregate_empty_metrics(self):
        """Test aggregating empty metrics list."""
        aggregator = MetricsAggregator("test_run")
        stage_metrics = aggregator.aggregate_to_stage_metrics([], "empty_stage")

        assert stage_metrics.stage_name == "empty_stage"
        assert stage_metrics.num_workers == 0
        assert stage_metrics.input_records == 0
        assert stage_metrics.output_records == 0


class TestMetricsModels:
    """Test metrics data models."""

    def test_operator_metrics_to_dict(self):
        """Test OperatorMetrics to_dict method."""
        metrics = OperatorMetrics(
            run_id="test_run",
            stage_name="test_stage",
            operator_name="TestOperator",
            worker_id="worker_0",
            timestamp=datetime.now(),
            input_records=100,
            output_records=90,
            pass_rate=90.0,
            total_time=1.0,
            avg_latency=0.01,
            min_latency=0.005,
            max_latency=0.02,
            p50_latency=0.01,
            p95_latency=0.015,
            p99_latency=0.018,
            throughput=100.0,
            custom_metrics={"test": "value"},
        )

        d = metrics.to_dict()
        assert d["run_id"] == "test_run"
        assert d["operator_name"] == "TestOperator"
        assert d["custom_metrics"] == {"test": "value"}

    def test_stage_metrics_to_dict(self):
        """Test StageMetrics to_dict method."""
        metrics = StageMetrics(
            run_id="test_run",
            stage_name="test_stage",
            timestamp=datetime.now(),
            num_workers=2,
            input_records=200,
            output_records=180,
            pass_rate=90.0,
            total_time=2.0,
            avg_throughput=100.0,
            min_throughput=90.0,
            max_throughput=110.0,
        )

        d = metrics.to_dict()
        assert d["run_id"] == "test_run"
        assert d["stage_name"] == "test_stage"
        assert d["num_workers"] == 2
