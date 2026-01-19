# ImageVisualDegradationsRefiner

Assesses visual degradation factors based on Z-Image paper quality assessment.

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_color_cast` | float | Color cast score (0-1, 0=normal, 1=severe) |
| `image_blurriness` | float | Blurriness score (0-1, 0=sharp, 1=blurry) |
| `image_watermark` | float | Watermark presence (0-1, 0=none, 1=visible) |
| `image_noise` | float | Noise level (0-1, 0=clean, 1=noisy) |
| `image_quality_overall` | float | Overall quality (0-1, higher is better) |

## Usage

```python
from operators.refiners import ImageVisualDegradationsRefiner

refiner = ImageVisualDegradationsRefiner()
refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_visual_degradations_refiner
```

## Reference

Based on Z-Image Technical Report (Section 2.1 - Data Profiling Engine).
