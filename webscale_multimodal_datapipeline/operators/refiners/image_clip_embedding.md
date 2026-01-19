# ImageClipEmbeddingRefiner

Extracts CLIP embeddings using OpenCLIP models. GPU-optimized with batch inference.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"ViT-B-32"` | OpenCLIP model name |
| `pretrained` | str | `"openai"` | Pretrained weights |
| `device` | str | `"auto"` | Device: `"auto"`, `"cuda"`, `"mps"`, `"cpu"` |
| `normalize` | bool | `True` | L2 normalize embeddings |
| `inference_batch_size` | int | `32` | GPU batch size |
| `use_fp16` | bool | `True` | FP16 on CUDA |
| `preprocess_workers` | int | `4` | Parallel preprocessing threads |

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_clip_emb_{model}` | list[float] | CLIP embedding vector |

Example: `image_clip_emb_vit_b_32` (512-dim), `image_clip_emb_vit_l_14` (768-dim)

## Usage

```python
from operators.refiners import ImageClipEmbeddingRefiner

# For aesthetic scoring, use ViT-L-14
refiner = ImageClipEmbeddingRefiner(
    model_name="ViT-L-14",
    pretrained="openai",
    device="auto",
)
refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_clip_embedding_refiner
    params:
      model_name: "ViT-L-14"
      pretrained: "openai"
      device: "auto"
      inference_batch_size: 32
      use_fp16: true
```

## Available Models

| Model | Dim | Speed | Quality |
|-------|-----|-------|---------|
| ViT-B-32 | 512 | Fast | Good |
| ViT-B-16 | 512 | Medium | Better |
| ViT-L-14 | 768 | Slow | Best (required for aesthetic scoring) |
