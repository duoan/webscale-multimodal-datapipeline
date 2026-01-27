# TextExactDeduplicator

Remove exact duplicate text documents using content hashing.

## Overview

Fast and memory-efficient deduplication using content fingerprints. Uses xxhash (if available) or MD5 to compute hashes, then leverages the distributed `DedupBackend` to remove duplicates.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_field` | `str` | `"text"` | Field containing text to hash |
| `hash_algorithm` | `str` | `"auto"` | Hash algorithm: "auto", "xxhash", "md5", "sha256" |
| `normalize_whitespace` | `bool` | `True` | Collapse whitespace before hashing |
| `lowercase` | `bool` | `True` | Lowercase text before hashing |
| `include_url` | `bool` | `False` | Include URL in hash (URL+content dedup) |
| `url_field` | `str` | `"url"` | URL field name if include_url=True |

## Usage

### Basic Usage

```yaml
stages:
  - name: deduplication
    operators:
      - name: text_exact_deduplicator
```

### With URL+Content Dedup (FineWeb Style)

```yaml
- name: text_exact_deduplicator
  params:
    include_url: true
    url_field: "url"
```

### Strict Normalization

```yaml
- name: text_exact_deduplicator
  params:
    normalize_whitespace: true
    lowercase: true  # Case-insensitive dedup
```

### With Backend Configuration

```yaml
executor:
  dedup_num_buckets: 32  # More buckets for larger datasets
```

## Hash Algorithms

| Algorithm | Speed | Collision Resistance | Notes |
|-----------|-------|---------------------|-------|
| `xxhash` | Fastest | Good | Recommended if installed |
| `md5` | Fast | Good | Default fallback |
| `sha256` | Slower | Best | For maximum safety |

Install xxhash for best performance:
```bash
pip install xxhash
```

## Normalization Options

### `normalize_whitespace=True` (default)
```
"Hello   world\n\n" → "Hello world"
```

### `lowercase=True`
```
"Hello World" → "hello world"
```

## URL+Content Dedup

When `include_url=True`, hash is computed on:
```
"{url}|{normalized_text}"
```

This allows same content from different URLs to be kept (useful for syndicated content).

## Distributed Deduplication

Uses Ray actors for distributed state:

```
┌─────────────┐
│   Worker    │──hash──▶┌─────────────────┐
├─────────────┤         │ DedupBackend[0] │
│   Worker    │──hash──▶├─────────────────┤
├─────────────┤         │ DedupBackend[1] │
│   Worker    │──hash──▶├─────────────────┤
└─────────────┘         │      ...        │
                        └─────────────────┘
```

## Performance

- **Throughput**: ~100,000+ records/sec (hash computation)
- **Memory**: O(unique_hashes) per bucket
- **Scaling**: Linear with `dedup_num_buckets`

## Reference

- [RefinedWeb (arXiv:2306.01116)](https://arxiv.org/pdf/2306.01116) - Section G.3: Exact deduplication
- [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) - URL+content dedup
- [DCLM (arXiv:2406.11794)](https://arxiv.org/pdf/2406.11794) - Content-based dedup
