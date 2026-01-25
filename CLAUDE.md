# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mega Data Factory is a high-throughput, distributed multimodal data processing pipeline for web-scale (hundreds of billions) datasets. It's built on Ray for distributed computing, with Rust-accelerated and GPU-optimized operators for image processing tasks like quality assessment, embedding extraction, and deduplication.

## Build & Development Commands

```bash
# Install with Rust acceleration (recommended)
uv pip install -e .

# Install without Rust (pure Python fallback)
uv sync

# Build Rust extension module
maturin develop --release

# Run pipeline
mdf run --config configs/z_image.yaml
mdf run -c configs/z_image.yaml --max-samples 1000 --batch-size 500

# Linting
ruff check .
ruff format .

# Run tests
pytest tests/
pytest tests/test_metrics.py -v  # Single test file
pytest tests/test_metrics.py::TestMetricsCollector::test_initialization -v  # Single test
```

## Architecture

### Pipeline Execution Flow

1. **DataLoader** (distributed via `DataLoaderWorker` Ray actors) loads data in parallel shards
2. **Executor** orchestrates batch submission through stages using Ray ObjectRef chaining
3. **Stages** contain operators that process batches (CPU or GPU workers)
4. **DataWriter** writes processed records to Parquet/Iceberg

### Key Design Patterns

**Registry Pattern**: All components (operators, loaders, writers) self-register at import time:
- `OperatorRegistry` - operators registered by class name (PascalCase)
- `DataLoaderRegistry` - data loaders by type name
- `DataWriterRegistry` - data writers by type name

**Operator Hierarchy**:
- `Operator` - base class with automatic stats collection
- `Refiner` - enriches records with new fields (inplace via `refine_batch`)
- `Filter` - removes records based on conditions (`should_keep_batch` returns bool list)
- `Deduplicator` - removes duplicates using shared `DedupBackend` (`get_dedup_keys_batch`)
- `CombinedOperator` - chains multiple operators

**Config-driven**: YAML config defines the entire pipeline. Operator names in config use snake_case (e.g., `image_metadata_refiner`) which converts to PascalCase class names (`ImageMetadataRefiner`).

### Ray Execution Model

- **ObjectRef Chaining**: Batches flow through stages without blocking the driver
- **Backpressure Control**: `max_in_flight` limits concurrent batches to prevent memory buildup
- **Dynamic Workers**: Stages can specify `min_replicas`/`max_replicas` for elastic scaling
- **Shared DedupBackend**: Ray actors provide distributed deduplication with bucketing

### Rust Acceleration

Located in `src/lib.rs`, exposed as `mega_data_factory.rust_operators`:
- `image_assess_quality_batch` - compression artifacts + entropy calculation
- `image_compute_phash_batch` - perceptual hash computation

Uses `rayon` for parallel batch processing. Python fallback exists when Rust module unavailable.

## Directory Structure

```
mega_data_factory/
├── cli.py                 # CLI entry point (mdf command)
├── framework/
│   ├── base.py            # DataLoader, DataWriter ABCs
│   ├── operator.py        # Operator, Refiner, Filter, Deduplicator
│   ├── executor.py        # Pipeline orchestration
│   ├── config.py          # YAML config parsing (PipelineConfig)
│   ├── registry.py        # Component registries
│   ├── worker.py          # RayWorker actor
│   ├── loader_worker.py   # DataLoaderWorker actor
│   ├── backend.py         # DedupBackend (distributed dedup state)
│   └── metrics/           # Metrics collection, aggregation, reporting
├── operators/
│   ├── refiners/          # ImageMetadataRefiner, ImageClipEmbeddingRefiner, etc.
│   ├── filters/           # ImageQualityFilter
│   └── dedup/             # ImagePhashDeduplicator
├── loaders/               # HuggingFaceDataLoader, CommonCrawlLoader
├── writers/               # ParquetDataWriter, IcebergWriter
└── models/                # ML model trainers (aesthetic, AIGC detector, k-means)
```

## Creating New Operators

```python
from mega_data_factory.framework import Refiner, OperatorRegistry
import pyarrow as pa

class MyRefiner(Refiner):
    def refine_batch(self, records: list[dict]) -> None:
        for record in records:
            record["my_field"] = compute(record)

    def get_output_schema(self) -> dict[str, pa.DataType]:
        return {"my_field": pa.float32()}

# Register (typically at module import)
OperatorRegistry.register("MyRefiner", MyRefiner)
```

For `Filter`: implement `should_keep_batch(records) -> list[bool]`
For `Deduplicator`: implement `get_dedup_keys_batch(records) -> list[str]`

## Configuration Structure

```yaml
data_loader:
  type: HuggingFaceDataLoader
  params: {...}
  num_workers: 8  # Parallel loader workers

stages:
  - name: stage_name
    operators:
      - name: operator_snake_case  # Converts to OperatorSnakeCase class
        params: {...}
        enabled: true
    worker:
      min_replicas: 1
      max_replicas: 4
      resources:
        cpu: 1
        gpu: 0.25

data_writer:
  type: ParquetDataWriter
  params:
    output_path: "./output"

executor:
  max_samples: 10000
  batch_size: 1024
  dedup_num_buckets: 16
  metrics:
    enabled: true
    generate_report: true
```
