# ImageMetadataRefiner

Extracts basic image metadata from image bytes.

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `file_size_bytes` | int | File size in bytes |
| `format` | str | Image format (JPEG, PNG, WEBP, etc.) |

## Usage

```python
from operators.refiners import ImageMetadataRefiner

refiner = ImageMetadataRefiner()
refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_metadata_refiner
```
