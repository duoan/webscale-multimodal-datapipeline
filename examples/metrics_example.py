"""
Example: Using metrics collection in pipeline

Demonstrates how to use the metrics module for run/stage/operator level metrics.
"""

from datetime import datetime

from mega_data_factory.framework.metrics import (
    MetricsAggregator,
    MetricsCollector,
    MetricsWriter,
    OperatorMetrics,
)


def example_basic_usage():
    """Example 1: Basic metrics collection with context managers."""
    print("=" * 60)
    print("Example 1: Basic Metrics Collection")
    print("=" * 60)

    collector = MetricsCollector()
    print(f"Run ID: {collector.run_id}")

    # Track entire run
    with collector.track_run():
        print("Starting pipeline run...")

        # Simulate stage execution
        with collector.track_stage("stage_0"):
            print("  Stage 0: Processing...")

            # Simulate operator execution (manual metrics)
            op_metrics = OperatorMetrics(
                run_id=collector.run_id,
                stage_name="stage_0",
                operator_name="ImageMetadataRefiner",
                worker_id="worker_0",
                timestamp=datetime.now(),
                input_records=1000,
                output_records=1000,
                pass_rate=100.0,
                total_time=0.5,
                avg_latency=0.0005,
                min_latency=0.0003,
                max_latency=0.001,
                p50_latency=0.0005,
                p95_latency=0.0008,
                p99_latency=0.0009,
                throughput=2000.0,
            )
            collector.add_operator_metrics(op_metrics)

        with collector.track_stage("stage_1"):
            print("  Stage 1: Processing...")

            op_metrics = OperatorMetrics(
                run_id=collector.run_id,
                stage_name="stage_1",
                operator_name="ImageClipEmbeddingRefiner",
                worker_id="worker_0",
                timestamp=datetime.now(),
                input_records=1000,
                output_records=1000,
                pass_rate=100.0,
                total_time=5.0,
                avg_latency=0.005,
                min_latency=0.004,
                max_latency=0.008,
                p50_latency=0.005,
                p95_latency=0.006,
                p99_latency=0.007,
                throughput=200.0,
            )
            collector.add_operator_metrics(op_metrics)

    # Get collected metrics
    run_metrics = collector.get_run_metrics()
    stage_metrics = collector.get_stage_metrics()
    operator_metrics = collector.get_operator_metrics()

    print("\nRun Metrics:")
    print(f"  Duration: {run_metrics.duration:.2f}s")
    print(f"  Stages: {run_metrics.num_stages}")
    print(f"  Total Input: {run_metrics.total_input_records}")
    print(f"  Total Output: {run_metrics.total_output_records}")
    print(f"  Pass Rate: {run_metrics.overall_pass_rate:.1f}%")
    print(f"  Throughput: {run_metrics.avg_throughput:.1f} records/s")

    print(f"\nStage Metrics ({len(stage_metrics)} stages):")
    for stage in stage_metrics:
        print(f"  {stage.stage_name}:")
        print(f"    Workers: {stage.num_workers}")
        print(f"    Input: {stage.input_records}")
        print(f"    Output: {stage.output_records}")
        print(f"    Pass Rate: {stage.pass_rate:.1f}%")
        print(f"    Time: {stage.total_time:.2f}s")

    print(f"\nOperator Metrics ({len(operator_metrics)} operators):")
    for op in operator_metrics:
        print(f"  {op.operator_name} ({op.stage_name}):")
        print(f"    Throughput: {op.throughput:.1f} records/s")
        print(f"    Latency: p50={op.p50_latency * 1000:.2f}ms, p95={op.p95_latency * 1000:.2f}ms")


def example_write_to_parquet():
    """Example 2: Writing metrics to Parquet files."""
    print("\n" + "=" * 60)
    print("Example 2: Writing Metrics to Parquet")
    print("=" * 60)

    collector = MetricsCollector(run_id="example_run_001")
    writer = MetricsWriter("./metrics_example_output")

    # Simulate metrics collection
    with collector.track_run():
        with collector.track_stage("basic_stage"):
            op_metrics = OperatorMetrics(
                run_id=collector.run_id,
                stage_name="basic_stage",
                operator_name="ImageMetadataRefiner",
                worker_id="worker_0",
                timestamp=datetime.now(),
                input_records=1000,
                output_records=998,
                pass_rate=99.8,
                total_time=0.5,
                avg_latency=0.0005,
                min_latency=0.0003,
                max_latency=0.001,
                p50_latency=0.0005,
                p95_latency=0.0008,
                p99_latency=0.0009,
                throughput=2000.0,
            )
            collector.add_operator_metrics(op_metrics)

    # Write to Parquet
    run_metrics = collector.get_run_metrics()
    stage_metrics = collector.get_stage_metrics()
    operator_metrics = collector.get_operator_metrics()

    writer.write_all(
        run_metrics=run_metrics,
        stage_metrics=stage_metrics,
        operator_metrics=operator_metrics,
    )

    print(f"Metrics written to: {writer.output_path}")
    print(f"  - Runs: {writer.runs_path}")
    print(f"  - Stages: {writer.stages_path}")
    print(f"  - Operators: {writer.operators_path}")

    # Read back and verify
    print("\nReading back from Parquet:")
    read_run = writer.read_run_metrics(collector.run_id)
    if read_run:
        print(f"  Run: {read_run['run_id']}")
        print(f"  Duration: {read_run['duration']:.2f}s")
        print(f"  Total Input: {read_run['total_input_records']}")

    read_ops = writer.read_operator_metrics(collector.run_id)
    print(f"  Operators: {len(read_ops)} records read")


def example_aggregation():
    """Example 3: Aggregating distributed metrics."""
    print("\n" + "=" * 60)
    print("Example 3: Metrics Aggregation")
    print("=" * 60)

    aggregator = MetricsAggregator("test_run")

    # Simulate operator metrics from multiple workers
    worker_metrics = [
        OperatorMetrics(
            run_id="test_run",
            stage_name="embedding_stage",
            operator_name="ImageClipEmbeddingRefiner",
            worker_id="worker_0",
            timestamp=datetime.now(),
            input_records=500,
            output_records=500,
            pass_rate=100.0,
            total_time=2.5,
            avg_latency=0.005,
            min_latency=0.004,
            max_latency=0.008,
            p50_latency=0.005,
            p95_latency=0.006,
            p99_latency=0.007,
            throughput=200.0,
        ),
        OperatorMetrics(
            run_id="test_run",
            stage_name="embedding_stage",
            operator_name="ImageClipEmbeddingRefiner",
            worker_id="worker_1",
            timestamp=datetime.now(),
            input_records=500,
            output_records=500,
            pass_rate=100.0,
            total_time=2.8,
            avg_latency=0.0056,
            min_latency=0.004,
            max_latency=0.009,
            p50_latency=0.0055,
            p95_latency=0.007,
            p99_latency=0.008,
            throughput=178.6,
        ),
    ]

    # Aggregate to stage level
    stage_metrics = aggregator.aggregate_to_stage_metrics(worker_metrics, "embedding_stage")

    print(f"Stage: {stage_metrics.stage_name}")
    print(f"  Workers: {stage_metrics.num_workers}")
    print(f"  Total Input: {stage_metrics.input_records}")
    print(f"  Total Output: {stage_metrics.output_records}")
    print(f"  Pass Rate: {stage_metrics.pass_rate:.1f}%")
    print(f"  Bottleneck Time: {stage_metrics.total_time:.2f}s")
    print(f"  Avg Throughput: {stage_metrics.avg_throughput:.1f} records/s")
    print(f"  Throughput Range: {stage_metrics.min_throughput:.1f} - {stage_metrics.max_throughput:.1f}")


if __name__ == "__main__":
    example_basic_usage()
    example_write_to_parquet()
    example_aggregation()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
