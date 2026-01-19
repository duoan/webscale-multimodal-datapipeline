# KMeans Clustering Model

KMeans clustering for semantic deduplication by grouping similar image embeddings.

## Overview

This module provides both local and distributed KMeans training for clustering image feature vectors. The cluster IDs can be used as bucket IDs for semantic deduplication in the data pipeline.

## Features

- **Distributed Training**: Scale to large datasets using Ray
- **Parquet Input**: Direct loading from parquet files with embeddings
- **Incremental Updates**: Support for iterative centroid refinement

## Usage

### Distributed Training (Recommended)

For large-scale datasets, use Ray distributed training:

```bash
python models/kmeans/distributed_train.py \
    --data_urls ./data/features_*.parquet \
    --n_clusters 100 \
    --n_workers 4 \
    --output_dir ./kmeans_training
```

**Input Format:**

Parquet files must have two columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique identifier for each record |
| `feature` | array | Feature vector (embedding) for clustering |

**Output:**

- Intermediate files: `iter_{iteration}_shard_{worker_id}_assignments.npy`
- Centroids: `iter_{iteration}/centroids.npy`
- Final model: Saved to specified `--model_path`

### Local Training

For small datasets:

```bash
python models/train_kmeans.py \
    --features_path features.npy \
    --n_clusters 100 \
    --output_path ./models/kmeans/kmeans_model.pkl
```

### Inference

```python
from models.kmeans.inference import KMeansInference

# Load model
inference = KMeansInference(model_path="./models/kmeans/kmeans_model.pkl")

# Get cluster ID for an embedding
cluster_id = inference.predict_cluster(image_embedding)

# Batch prediction
cluster_ids = inference.predict_batch(embeddings_array)
```

## File Structure

```text
models/kmeans/
├── __init__.py             # Module exports
├── trainer.py              # Local KMeans trainer
├── distributed_trainer.py  # Ray distributed trainer
├── distributed_train.py    # Distributed training script
└── inference.py            # Inference wrapper
```

## Integration

Used in `SemanticDeduplicator` operator to assign cluster IDs for deduplication:

```python
# In pipeline config
stages:
  - name: semantic_dedup
    operator: SemanticDeduplicator
    params:
      model_path: ./models/kmeans/kmeans_model.pkl
```
