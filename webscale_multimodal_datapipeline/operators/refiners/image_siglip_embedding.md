# ImageSigLIPEmbeddingRefiner

Extracts SigLIP2 embeddings using Google's SigLIP2 models from HuggingFace. GPU-optimized.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"google/siglip2-so400m-patch14-384"` | HuggingFace model name |
| `device` | str | `"auto"` | Device: `"auto"`, `"cuda"`, `"mps"`, `"cpu"` |
| `normalize` | bool | `True` | L2 normalize embeddings |
| `inference_batch_size` | int | `32` | GPU batch size |
| `use_fp16` | bool | `True` | FP16 on CUDA |
| `preprocess_workers` | int | `4` | Parallel preprocessing threads |

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_siglip_emb_{model}` | list[float] | SigLIP2 embedding vector |

Example: `image_siglip_emb_so400m_patch14_384` (1152-dim)

## Usage

```python
from operators.refiners import ImageSigLIPEmbeddingRefiner

refiner = ImageSigLIPEmbeddingRefiner(
    model_name="google/siglip2-so400m-patch14-384",
    device="auto",
)
refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_siglip_embedding_refiner
    params:
      model_name: "google/siglip2-so400m-patch14-384"
      device: "auto"
      inference_batch_size: 32
      use_fp16: true
```

## Available Models

| Model | Dim | Input Size | Description |
|-------|-----|------------|-------------|
| siglip2-so400m-patch14-384 | 1152 | 384x384 | Recommended, best quality |
| siglip2-base-patch16-224 | 768 | 224x224 | Faster, smaller |
| siglip2-large-patch16-256 | 1024 | 256x256 | Medium |

## Note

SigLIP2 embeddings are required for `ImageAIGCDetectorRefiner`.
