# ImageAestheticQualityRefiner

Predicts aesthetic quality scores using CLIP+MLP model trained on AVA dataset.

Based on [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor).

## Requirements

**Pre-computed CLIP ViT-L-14 embeddings required.**

Run `ImageClipEmbeddingRefiner` with `model_name="ViT-L-14"` first.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_field` | str | **required** | Field containing CLIP embeddings (768-dim) |
| `model_repo` | str | `"ttj/sac-logos-ava1-l14-linearMSE"` | HuggingFace model repo |
| `model_filename` | str | `"model.safetensors"` | Model weights filename |
| `device` | str | `"auto"` | Device selection |
| `inference_batch_size` | int | `32` | Batch size |
| `use_fp16` | bool | `True` | FP16 on CUDA |

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_aesthetic_score` | float | Aesthetic score (typically 1-10, higher = better) |

## Usage

```python
from operators.refiners import ImageClipEmbeddingRefiner, ImageAestheticQualityRefiner

# Step 1: Extract CLIP embeddings
clip_refiner = ImageClipEmbeddingRefiner(model_name="ViT-L-14")
clip_refiner.refine_batch(records)

# Step 2: Score aesthetics
aesthetic_refiner = ImageAestheticQualityRefiner(
    embedding_field="image_clip_emb_vit_l_14",
)
aesthetic_refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  # First: extract embeddings
  - name: image_clip_embedding_refiner
    params:
      model_name: "ViT-L-14"

  # Then: score aesthetics
  - name: image_aesthetic_quality_refiner
    params:
      embedding_field: "image_clip_emb_vit_l_14"
```

## Score Interpretation

| Score | Quality |
|-------|---------|
| < 4.0 | Poor |
| 4.0 - 5.5 | Average |
| 5.5 - 7.0 | Good |
| > 7.0 | Excellent |
