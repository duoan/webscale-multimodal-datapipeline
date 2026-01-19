# Models Package

This package contains model training and inference code for the Z-Image data pipeline.

## Available Models

| Model | Description | Documentation |
|-------|-------------|---------------|
| [Quality Assessment](./quality_assessment/) | Multi-head CNN for visual degradation detection (color cast, blur, watermark, noise) | [README](./quality_assessment/README.md) |
| [KMeans Clustering](./kmeans/) | Distributed KMeans for semantic deduplication | [README](./kmeans/README.md) |
| [Classifier](./classifier/) | Base classification model trainer | - |

## Directory Structure

```text
models/
├── quality_assessment/     # Visual degradation assessment
│   ├── trainer.py          # Multi-head model & trainer
│   ├── inference.py        # Inference wrapper
│   ├── synthetic_data.py   # Synthetic training data generation
│   └── README.md           # Detailed documentation
├── kmeans/                 # KMeans clustering
│   ├── trainer.py          # Local trainer
│   ├── distributed_trainer.py  # Ray distributed trainer
│   ├── distributed_train.py    # Training script
│   └── inference.py        # Inference wrapper
├── classifier/             # Classification models
│   └── trainer.py          # Base classifier trainer
├── train_kmeans.py         # KMeans training entry point
└── train_quality_assessment.py  # Quality model training entry point
```

## Quick Start

### Quality Assessment

Detect visual degradations (color cast, blurriness, watermark, noise):

```bash
# Generate synthetic training data from HuggingFace
python -m models.train_quality_assessment generate_hf \
    --dataset tiny-imagenet \
    --output_dir ./training_data \
    --num_images 1000

# Train the model
python -m models.train_quality_assessment train \
    --train_images ./training_data/images.npy \
    --train_labels ./training_data/labels.npy \
    --epochs 50
```

See [quality_assessment/README.md](./quality_assessment/README.md) for architecture details.

### KMeans Clustering

Cluster image embeddings for semantic deduplication:

```bash
# Distributed training with Ray
python models/kmeans/distributed_train.py \
    --data_urls ./data/features_*.parquet \
    --n_clusters 100 \
    --n_workers 4

# Local training
python models/train_kmeans.py \
    --features_path features.npy \
    --n_clusters 100
```

## Integration with Pipeline

These models integrate with pipeline operators:

| Model | Operator | Purpose |
|-------|----------|---------|
| Quality Assessment | `VisualDegradationsRefiner` | Score images on degradation factors |
| KMeans | `SemanticDeduplicator` | Assign cluster IDs for deduplication |
