# ImageQualityFilter

Filters images based on quality criteria.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_width` | int | `256` | Minimum width in pixels |
| `min_height` | int | `256` | Minimum height in pixels |
| `max_compression_artifacts` | float | `0.8` | Maximum compression artifacts score |
| `min_information_entropy` | float | `3.0` | Minimum information entropy |

## Usage

```python
from operators.filters import ImageQualityFilter

filter = ImageQualityFilter(
    min_width=512,
    min_height=512,
    max_compression_artifacts=0.5,
    min_information_entropy=4.0,
)
keep_flags = filter.should_keep_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_quality_filter
    params:
      min_width: 512
      min_height: 512
      max_compression_artifacts: 0.5
      min_information_entropy: 4.0
```

## Note

Requires `ImageMetadataRefiner` and `ImageTechnicalQualityRefiner` to be run first to populate the required fields.
