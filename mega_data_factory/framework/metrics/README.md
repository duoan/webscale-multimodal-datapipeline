# Metrics Module

Three-level metrics collection system (run/stage/operator) with automatic instrumentation and Parquet output for Superset visualization.

## Features

- **Three-Level Metrics Collection**: Run (entire pipeline), Stage (operator groups), Operator (individual processing units)
- **Context Manager Auto-Instrumentation**: Use `with` statements to automatically collect metrics, non-intrusive
- **Parquet Output**: Export to Parquet format for efficient querying and visualization
- **Distributed-Friendly**: Supports collecting and aggregating metrics from Ray workers
- **Superset Compatible**: Schema optimized for Superset visualization and analysis

## Quick Start

### 1. Enable Metrics (Configuration File)

Add metrics configuration to your pipeline config file:

```yaml
executor:
  max_samples: 1000
  batch_size: 200

  # Metrics configuration
  metrics:
    enabled: true                 # Enable metrics collection
    output_path: "./metrics"      # Output directory
    collect_custom_metrics: false # Whether to collect custom metrics
    write_on_completion: true     # Write to Parquet after run completes
```

### 2. Run Pipeline

```bash
mdf run --config configs/example_with_metrics.yaml
```

### 3. View Output

Metrics are automatically written to:

```
metrics/
├── runs/run_20260124_123456_abc123.parquet       # Run-level metrics
├── stages/stages_20260124_123456_abc123.parquet  # Stage-level metrics
└── operators/operators_20260124_123456_abc123.parquet  # Operator-level metrics
```

## Metrics Reference

### Run Metrics (Pipeline Level)

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique run identifier |
| `start_time` | timestamp | Start time |
| `end_time` | timestamp | End time |
| `duration` | float | Total duration (seconds) |
| `num_stages` | int | Number of stages |
| `total_input_records` | int | Total input records |
| `total_output_records` | int | Total output records |
| `overall_pass_rate` | float | Overall pass rate (0-100) |
| `avg_throughput` | float | Average throughput (records/s) |
| `total_errors` | int | Total error count |
| `config` | string | Configuration snapshot (JSON) |

### Stage Metrics (Stage Level)

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique run identifier |
| `stage_name` | string | Stage name |
| `timestamp` | timestamp | Timestamp |
| `num_workers` | int | Number of workers |
| `input_records` | int | Input record count |
| `output_records` | int | Output record count |
| `pass_rate` | float | Pass rate (0-100) |
| `total_time` | float | Total time (bottleneck) |
| `avg_throughput` | float | Average throughput |
| `min_throughput` | float | Minimum throughput |
| `max_throughput` | float | Maximum throughput |
| `error_count` | int | Error count |

### Operator Metrics (Operator Level)

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique run identifier |
| `stage_name` | string | Stage name |
| `operator_name` | string | Operator name |
| `worker_id` | string | Worker identifier |
| `timestamp` | timestamp | Timestamp |
| `input_records` | int | Input record count |
| `output_records` | int | Output record count |
| `pass_rate` | float | Pass rate (0-100) |
| `total_time` | float | Total time (seconds) |
| `avg_latency` | float | Average latency (seconds) |
| `min_latency` | float | Minimum latency (seconds) |
| `max_latency` | float | Maximum latency (seconds) |
| `p50_latency` | float | P50 latency (seconds) |
| `p95_latency` | float | P95 latency (seconds) |
| `p99_latency` | float | P99 latency (seconds) |
| `throughput` | float | Throughput (records/s) |
| `error_count` | int | Error count |
| `custom_metrics` | string | Custom metrics (JSON) |

## Programming Interface

### Basic Usage

```python
from mega_data_factory.framework.metrics import MetricsCollector, MetricsWriter

# Create collector
collector = MetricsCollector()

# Track entire run
with collector.track_run():
    # Track stage
    with collector.track_stage("stage_0"):
        # Manually add operator metrics
        collector.add_operator_metrics(operator_metrics)

# Write to Parquet
writer = MetricsWriter("./metrics")
writer.write_all(
    run_metrics=collector.get_run_metrics(),
    stage_metrics=collector.get_stage_metrics(),
    operator_metrics=collector.get_operator_metrics(),
)
```

### Distributed Aggregation

```python
from mega_data_factory.framework.metrics import MetricsAggregator

aggregator = MetricsAggregator(run_id="test_run")

# Collect metrics from Ray workers
stage_metrics = aggregator.collect_stage_metrics(workers, "stage_name")
```

### Reading Metrics

```python
from mega_data_factory.framework.metrics import MetricsWriter

writer = MetricsWriter("./metrics")

# List all runs
runs = writer.list_runs()

# Read metrics for specific run
run_metrics = writer.read_run_metrics("run_20260124_123456")
stage_metrics = writer.read_stage_metrics("run_20260124_123456")
operator_metrics = writer.read_operator_metrics("run_20260124_123456")
```

## Superset Visualization

### 1. Connect Parquet Data Source

Add data source in Superset:

- **Database Type**: Parquet
- **Database Path**: `./metrics/operators/*.parquet`

### 2. Create Dashboard

Recommended visualizations:

#### Run Level
- **Time Series**: `total_input_records`, `total_output_records` over time
- **Gauge**: `overall_pass_rate`, `avg_throughput`
- **Table**: Recent runs list

#### Stage Level
- **Bar Chart**: Compare `input_records` across stages
- **Line Chart**: `pass_rate` trends by stage
- **Heatmap**: Stage throughput heatmap

#### Operator Level
- **Box Plot**: Latency distribution (p50/p95/p99)
- **Scatter Plot**: `throughput` vs `pass_rate`
- **Time Series**: Operator throughput over time

### 3. Example Queries

Find slow operators:
```sql
SELECT
    operator_name,
    stage_name,
    AVG(throughput) as avg_throughput,
    AVG(p95_latency) as avg_p95_latency
FROM operators
WHERE run_id = 'run_20260124_123456'
GROUP BY operator_name, stage_name
ORDER BY avg_throughput ASC
LIMIT 10
```

Find stages with lowest pass rate:
```sql
SELECT
    stage_name,
    AVG(pass_rate) as avg_pass_rate,
    SUM(input_records) as total_input
FROM stages
GROUP BY stage_name
ORDER BY avg_pass_rate ASC
```

## Performance Overhead

- **CPU Overhead**: < 1%
- **Memory Overhead**: ~1KB per operator metric
- **Disk Usage**: Typically < 1MB per run with Parquet compression

## Best Practices

1. **Regular Cleanup**: Use cron jobs to regularly clean old metrics files
2. **Partitioned Storage**: Store metrics partitioned by date (e.g., `metrics/2026-01-24/`)
3. **Aggregated Queries**: For long-running pipelines, periodically aggregate metrics to summary tables
4. **Monitoring Alerts**: Set up Superset alerts based on metrics (e.g., pass rate < 90%)

## Troubleshooting

### Metrics Not Generated

Check configuration:
```yaml
executor:
  metrics:
    enabled: true  # Ensure enabled
    write_on_completion: true  # Ensure writing
```

### Parquet Files Empty

Possible causes:
- Pipeline didn't complete normally (crash or terminated)
- No metrics collected (check if operators have statistics)

Solution:
- Check pipeline logs
- Verify metrics are collected with `collector.get_operator_metrics()`

### Superset Cannot Read

Ensure:
- Parquet file path is correct
- Superset has read permissions
- Schema is compatible (verify with `pyarrow`)

## Example Code

See complete examples in `examples/metrics_example.py`.

## API Reference

For detailed API documentation, see source code comments:
- `collector.py`: MetricsCollector
- `writer.py`: MetricsWriter
- `aggregator.py`: MetricsAggregator
- `models.py`: Data model definitions
