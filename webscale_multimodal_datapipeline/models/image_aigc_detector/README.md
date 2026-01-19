# AIGC Content Detector

Binary classifier for detecting AI-generated images using **SigLIP2 + MLP Head**.

Based on Imagen 3 findings: AIGC content filtering is crucial for preventing degradation in model output quality and physical realism.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     SigLIP2 Vision Encoder                  │
│                      (Frozen, 1152-dim)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   [CLS] Token   │
                    │   Embedding     │
                    └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MLP Classifier Head                       │
│  Linear(1152→512) → BN → ReLU → Dropout(0.3)                │
│  Linear(512→128)  → BN → ReLU → Dropout(0.3)                │
│  Linear(128→1)    → Sigmoid                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  AIGC Score     │
                    │  (0.0 - 1.0)    │
                    └─────────────────┘
```

## Why SigLIP2?

1. **State-of-the-art Vision Encoder**: SigLIP2 provides rich semantic features that capture both visual content and style
2. **Frozen Backbone**: No need to fine-tune the massive encoder - just train a lightweight head
3. **Fast Inference**: MLP head is extremely efficient (< 1ms per image on GPU)
4. **Transfer Learning**: SigLIP2's pre-training helps detect "uncanny valley" artifacts

## Quick Start

### 1. Prepare Training Data

You need two sets of images:

- **Real Images**: High-quality photographs (Unsplash, LAION-filtered, ImageNet)
- **AI Images**: Generated from diverse models (FLUX, MJv6, SDXL, DALL-E 3, SD3)

```bash
# Create data directories
mkdir -p data/real_photos data/ai_generated

# Download/collect real photos to data/real_photos/
# Generate/download AI images to data/ai_generated/
```

### 2. Extract Embeddings (Recommended)

Pre-extracting SigLIP2 embeddings speeds up training iteration significantly:

```bash
python -m models.train_aigc_detector extract_embeddings \
    --real_images_dir ./data/real_photos \
    --ai_images_dir ./data/ai_generated \
    --output_dir ./data/aigc_embeddings \
    --max_real 50000 \
    --max_ai 50000 \
    --batch_size 64
```

Or use HuggingFace datasets:

```bash
python -m models.train_aigc_detector extract_embeddings \
    --real_dataset zh-plus/tiny-imagenet \
    --ai_dataset poloclub/diffusiondb \
    --output_dir ./data/aigc_embeddings \
    --max_real 10000 \
    --max_ai 10000
```

### 3. Train Classifier

```bash
python -m models.train_aigc_detector train \
    --embeddings_dir ./data/aigc_embeddings \
    --output_path ./models/aigc_detector/aigc_classifier.pth \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --focal_loss
```

### 4. Evaluate

```bash
python -m models.train_aigc_detector evaluate \
    --model_path ./models/aigc_detector/aigc_classifier.pth \
    --embeddings_dir ./data/aigc_embeddings_test
```

### 5. Use in Pipeline

```python
from operators.refiners.image_aigc_detector import ImageAIGCDetectorRefiner

# Create refiner
refiner = ImageAIGCDetectorRefiner(
    model_path="./models/aigc_detector/aigc_classifier.pth",
    threshold=0.5,
)

# Process records
records = [{"image": {"bytes": image_bytes}}]
refiner.refine_batch(records)

# Check results
for record in records:
    print(f"AIGC Score: {record['image_aigc_score']:.3f}")
    print(f"Is AIGC: {record['image_is_aigc']}")
```

## Training Data Guidelines

### Real Images (Label 0)

**Good Sources:**

- Unsplash (high-quality photography)
- Pexels (curated stock photos)
- LAION-Aesthetics (filtered subset)
- ImageNet (natural images)

**Filtering:**

- Exclude heavily photoshopped images
- Exclude 3D renders (can be confused with AI)
- Prefer uncompressed or high-quality JPEG

### AI-Generated Images (Label 1)

**Must Include:**

- Midjourney v6 (hardest to detect - critical!)
- FLUX.1 (latest diffusion model)
- SDXL (common baseline)
- DALL-E 3 (different architecture)
- Stable Diffusion 3 (latest SD)

**Prompt Strategy:**

- **Photorealistic**: "Professional photograph of...", "DSLR photo..."
- **Portraits**: Human faces, groups, lifestyle
- **Landscapes**: Nature, urban, architecture
- **Abstract**: Art styles, patterns
- **Objects**: Products, food, vehicles

**Important:** Use diverse prompts! Don't let the classifier learn to detect "anime style = AI".

## Data Augmentation

Critical augmentations to prevent overfitting to compression artifacts:

```python
transforms = [
    # JPEG compression (CRITICAL!)
    JpegCompressionTransform(quality_range=(30, 95), p=0.5),

    # Gaussian blur (prevents high-freq overfitting)
    GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

    # Random crop (prevents edge overfitting)
    RandomResizedCrop(384, scale=(0.8, 1.0)),

    # Color jitter
    ColorJitter(brightness=0.2, contrast=0.2),
]
```

## Hard Negative Mining

After initial training, find and upsample hard samples:

```bash
python -m models.train_aigc_detector hard_mining \
    --model_path ./models/aigc_detector/aigc_classifier.pth \
    --embeddings_dir ./data/aigc_embeddings \
    --output_dir ./data/hard_samples \
    --threshold_low 0.3 \
    --threshold_high 0.7
```

Hard samples are:

- AI images misclassified as real (false negatives)
- Real images misclassified as AI (false positives)
- Samples in the uncertain region (0.3-0.7 probability)

## Expected Performance

| Dataset | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| SDXL only | 95%+ | 94% | 96% | 95% |
| Mixed (SDXL+MJ) | 90%+ | 88% | 92% | 90% |
| With MJv6 | 85%+ | 83% | 87% | 85% |

**Note:** Midjourney v6 is significantly harder to detect than other models.

## Pipeline Integration

Add to your `pipeline_config.yaml`:

```yaml
refiners:
  - name: aigc_detector
    class: operators.refiners.image_aigc_detector.ImageAIGCDetectorRefiner
    params:
      model_path: ./models/aigc_detector/aigc_classifier.pth
      threshold: 0.5
      device: auto

filters:
  - name: filter_aigc
    class: operators.filters.image_quality_filter.ImageQualityFilter
    params:
      field: image_is_aigc
      value: false  # Keep only real images
```

## Model Files

After training, you'll have:

- `aigc_classifier.pth` - MLP head weights only (~2MB)
- `aigc_classifier_full.pth` - Full checkpoint with optimizer state

The SigLIP2 backbone is loaded from HuggingFace at runtime.

## References

- [Imagen 3 Paper](https://arxiv.org/abs/2408.07009) - AIGC detection for data filtering
- [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384) - Vision encoder
- [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) - AI image dataset
- [Universal Fake Detect](https://github.com/Yuheng-Li/UniversalFakeDetect) - Academic baseline
