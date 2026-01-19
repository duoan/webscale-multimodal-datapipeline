# ImageAIGCDetectorRefiner

Detects AI-generated images using SigLIP2+MLP classifier.

Based on Imagen 3 findings: AIGC content filtering is crucial for preventing degradation in model output quality and physical realism.

## Requirements

**Pre-computed SigLIP2 embeddings required.**

Run `ImageSigLIPEmbeddingRefiner` first.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_field` | str | **required** | Field containing SigLIP2 embeddings |
| `model_path` | str | None | Local path to classifier weights |
| `model_repo` | str | None | HuggingFace repo (alternative to model_path) |
| `model_filename` | str | `"image_aigc_classifier.pth"` | Model filename |
| `embedding_dim` | int | `1152` | Expected embedding dimension |
| `hidden_dims` | tuple | `(512, 128)` | MLP hidden layer dimensions |
| `threshold` | float | `0.5` | Classification threshold |
| `device` | str | `"auto"` | Device selection |
| `inference_batch_size` | int | `32` | Batch size |
| `use_fp16` | bool | `True` | FP16 on CUDA |

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_aigc_score` | float | Probability of being AI-generated (0-1) |
| `image_is_aigc` | bool | True if score > threshold |

## Usage

```python
from operators.refiners import ImageSigLIPEmbeddingRefiner, ImageAIGCDetectorRefiner

# Step 1: Extract SigLIP embeddings
siglip_refiner = ImageSigLIPEmbeddingRefiner()
siglip_refiner.refine_batch(records)

# Step 2: Detect AIGC
aigc_refiner = ImageAIGCDetectorRefiner(
    embedding_field="image_siglip_emb_so400m_patch14_384",
    model_path="./models/image_aigc_detector/classifier.pth",
    threshold=0.5,
)
aigc_refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  # First: extract embeddings
  - name: image_siglip_embedding_refiner
    params:
      model_name: "google/siglip2-so400m-patch14-384"

  # Then: detect AIGC
  - name: image_aigc_detector_refiner
    params:
      embedding_field: "image_siglip_emb_so400m_patch14_384"
      model_path: "./models/image_aigc_detector/classifier.pth"
      threshold: 0.5
```

## Training Your Own Classifier

See [models/image_aigc_detector/README.md](../../models/image_aigc_detector/README.md) for training instructions.

```bash
# Extract embeddings
python -m models.train_image_aigc_detector extract_embeddings \
    --real_images_dir ./data/real \
    --ai_images_dir ./data/ai_generated \
    --output_dir ./data/embeddings

# Train classifier
python -m models.train_image_aigc_detector train \
    --embeddings_dir ./data/embeddings \
    --output_path ./models/image_aigc_detector/classifier.pth
```

## Reference

- [Imagen 3](https://arxiv.org/abs/2408.07009) - AIGC content detection for data filtering
