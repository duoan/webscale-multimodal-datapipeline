# Metrics Module - Quick Start Guide

## 1. Enable Metrics (30 seconds)

Add to your pipeline config (e.g., `configs/my_pipeline.yaml`):

```yaml
executor:
  metrics:
    enabled: true
    output_path: "./metrics"
```

## 2. Run Your Pipeline

```bash
mdf run --config configs/my_pipeline.yaml
```

That's it! Metrics are automatically collected and saved.

## 3. Check Output

```bash
ls -lh metrics/
```

You'll see:
```
metrics/
├── runs/run_20260124_123456_abc123.parquet
├── stages/stages_20260124_123456_abc123.parquet
└── operators/operators_20260124_123456_abc123.parquet
```

## 4. View Metrics (Python)

```python
from mega_data_factory.framework.metrics import MetricsWriter

writer = MetricsWriter("./metrics")

# List all runs
runs = writer.list_runs()
print(f"Available runs: {runs}")

# Read operator metrics
metrics = writer.read_operator_metrics(runs[0])
for m in metrics:
    print(f"{m['operator_name']}: {m['throughput']:.1f} records/s")
```

## 5. Superset Visualization

### Connect Data Source
1. Open Superset
2. Add Database → Parquet
3. Path: `./metrics/operators/*.parquet`

### Create Charts
- **Time Series**: `throughput` over `timestamp`
- **Bar Chart**: Compare operators by `throughput`
- **Box Plot**: `p50_latency`, `p95_latency`, `p99_latency`

## Metrics Reference

### Run Level (Pipeline)
- `duration`, `total_input_records`, `total_output_records`, `overall_pass_rate`, `avg_throughput`

### Stage Level (Operator Groups)
- `num_workers`, `input_records`, `output_records`, `pass_rate`, `total_time`, `avg_throughput`

### Operator Level (Individual Operators)
- `input_records`, `output_records`, `pass_rate`, `throughput`, `avg_latency`, `p50/p95/p99_latency`

## Advanced Usage

### Disable Metrics

```yaml
executor:
  metrics:
    enabled: false
```

### Custom Output Path

```yaml
executor:
  metrics:
    enabled: true
    output_path: "s3://my-bucket/metrics"  # S3 path
```

### Programmatic Access

```python
from mega_data_factory.framework.metrics import MetricsCollector, MetricsWriter

collector = MetricsCollector()

with collector.track_run():
    # Your pipeline code
    pass

# Save metrics
writer = MetricsWriter("./metrics")
writer.write_all(
    run_metrics=collector.get_run_metrics(),
    stage_metrics=collector.get_stage_metrics(),
    operator_metrics=collector.get_operator_metrics()
)
```

## Example Queries

### Find Slow Operators
```sql
SELECT operator_name, AVG(throughput) as avg_throughput
FROM operators
GROUP BY operator_name
ORDER BY avg_throughput ASC
LIMIT 5
```

### Analyze Pass Rates
```sql
SELECT stage_name, AVG(pass_rate) as avg_pass_rate
FROM stages
GROUP BY stage_name
ORDER BY avg_pass_rate ASC
```

### Track Performance Over Time
```sql
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(throughput) as avg_throughput
FROM operators
WHERE operator_name = 'ImageClipEmbeddingRefiner'
GROUP BY hour
ORDER BY hour DESC
```

## Troubleshooting

**Metrics not generated?**
- Check `enabled: true` in config
- Verify pipeline completed successfully

**Can't read Parquet files?**
```python
import pyarrow.parquet as pq
table = pq.read_table("metrics/operators/operators_*.parquet")
print(table.schema)
```

**Need more details?**
- See `README.md` for full documentation
- Run `python examples/metrics_example.py` for examples

## Performance

- **CPU Overhead**: < 1%
- **Memory**: ~1KB per metric
- **Disk**: < 1MB per run (compressed)
- **Impact**: Negligible on pipeline performance
