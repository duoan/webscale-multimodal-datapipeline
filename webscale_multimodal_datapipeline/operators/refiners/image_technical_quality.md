# ImageTechnicalQualityRefiner

Assesses technical quality metrics. Auto-uses Rust backend (3x faster) if available.

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_compression_artifacts` | float | Compression artifact score (0-1, lower is better) |
| `image_information_entropy` | float | Shannon entropy (higher = more detail) |

## Usage

```python
from operators.refiners import ImageTechnicalQualityRefiner

refiner = ImageTechnicalQualityRefiner()
refiner.refine_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_technical_quality_refiner
```

## Rust Acceleration

Build Rust extension for ~3x speedup:

```bash
cd rust && ./build.sh
uv pip install dist/*.whl
```

The refiner auto-detects and uses Rust backend when available.
