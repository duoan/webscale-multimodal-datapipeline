# ImagePhashDeduplicator

Removes duplicate images using perceptual hashing. Auto-uses Rust backend (2.5x faster) if available.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hash_size` | int | `16` | Hash size (produces hash_size^2 bit hash) |

## Usage

```python
from operators.dedup import ImagePhashDeduplicator

dedup = ImagePhashDeduplicator(hash_size=16)
keys = dedup.get_dedup_keys_batch(records)
```

## Pipeline Config

```yaml
operators:
  - name: image_phash_deduplicator
    params:
      hash_size: 16
```

## Rust Acceleration

Build Rust extension for ~2.5x speedup:

```bash
cd rust && ./build.sh
uv pip install dist/*.whl
```

The deduplicator auto-detects and uses Rust backend when available.

## Hash Size Guide

| hash_size | Bits | Collision Rate | Use Case |
|-----------|------|----------------|----------|
| 8 | 64 | Higher | Fast, approximate dedup |
| 16 | 256 | Low | Recommended default |
| 32 | 1024 | Very Low | Strict dedup |
