# Quality Assessment Model

Based on the Z-Image paper Section 2.1 (Data Profiling Engine) - **Technical Quality Assessment**, this model evaluates image technical quality by detecting four types of visual degradations.

## Degradation Types

| Degradation | Description | Score Meaning |
|-------------|-------------|---------------|
| Color Cast | Abnormal color tint/shift | 0=normal, 1=severe cast |
| Blurriness | Lack of sharpness/focus | 0=sharp, 1=very blurry |
| Watermark | Visible watermarks | 0=none, 1=obvious watermark |
| Noise | Visual noise artifacts | 0=clean, 1=very noisy |

## Model Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiHeadQualityAssessmentModel                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: (B, 3, 224, 224)                                                │
│         ↓                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Shared Backbone (CNN)                         │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐ │   │
│  │  │ ConvBlock  │→  │ ConvBlock  │→  │ ConvBlock  │→  │ConvBlock │ │   │
│  │  │ 3→32       │   │ 32→64      │   │ 64→128     │   │128→256   │ │   │
│  │  │ +Residual  │   │ +Residual  │   │ +Residual  │   │+Residual │ │   │
│  │  │ +MaxPool   │   │ +MaxPool   │   │ +MaxPool   │   │+MaxPool  │ │   │
│  │  └────────────┘   └────────────┘   └────────────┘   └──────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓                                                               │
│  ┌──────────────────┐                                                   │
│  │ Global Avg Pool  │  (B, 256, H', W') → (B, 256)                      │
│  └──────────────────┘                                                   │
│         ↓                                                               │
│  ┌──────────────────────────────────────┐                               │
│  │      Channel Attention (SE-like)     │                               │
│  │  Linear(256→64) → ReLU → Linear(64→256) → Sigmoid                    │
│  │  features = features * attention                                     │
│  └──────────────────────────────────────┘                               │
│         ↓                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    4 Prediction Heads                            │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐         │   │
│  │  │Color Cast │ │Blurriness │ │ Watermark │ │   Noise   │         │   │
│  │  │  Head     │ │   Head    │ │   Head    │ │   Head    │         │   │
│  │  │256→256→1  │ │256→256→1  │ │256→256→1  │ │256→256→1  │         │   │
│  │  │ +Dropout  │ │ +Dropout  │ │ +Dropout  │ │ +Dropout  │         │   │
│  │  │ +Sigmoid  │ │ +Sigmoid  │ │ +Sigmoid  │ │ +Sigmoid  │         │   │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘         │   │
│  │        ↓             ↓             ↓             ↓               │   │
│  │      [0-1]         [0-1]         [0-1]         [0-1]             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓                                                               │
│  ┌──────────────────────────────────────┐                               │
│  │         Overall Quality Score        │                               │
│  │  overall = 1 - Σ(score_i × weight_i) │                               │
│  │  weights: [0.2, 0.3, 0.25, 0.25]     │                               │
│  └──────────────────────────────────────┘                               │
│                                                                         │
│  Output: {color_cast, blurriness, watermark, noise, overall}            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### ConvBlock

```python
class ConvBlock(nn.Module):
    # Conv2d → BatchNorm → ReLU
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

- **BatchNorm**: Accelerates training convergence and provides regularization
- **ReLU**: Non-linear activation

### ResidualBlock

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return self.relu(out)
```

- **Skip Connection**: Mitigates vanishing gradients, enables deeper networks
- Each stage has a residual block to enhance feature extraction

### Channel Attention

```python
self.channel_attention = nn.Sequential(
    nn.Linear(256, 64),     # Squeeze
    nn.ReLU(inplace=True),
    nn.Linear(64, 256),     # Excitation
    nn.Sigmoid(),
)
# Usage: features = features * attention
```

Similar to **SE-Net (Squeeze-and-Excitation)**:

- Learns importance weights for each channel
- Enhances relevant features, suppresses irrelevant ones
- Particularly effective for detecting different degradation types

### Prediction Head

```python
def _make_head(self, in_features, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),      # 256 → 256
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),                      # Prevent overfitting
        nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, 1),           # 128 → 1
        nn.Sigmoid(),                             # Output [0, 1]
    )
```

- **3-layer MLP**: Sufficient expressive capacity
- **Dropout**: Regularization to prevent overfitting
- **Sigmoid**: Normalizes output to [0, 1]

## Multi-Task Learning

### Why Multi-Head Instead of Separate Models?

| Approach | Pros | Cons |
|----------|------|------|
| **4 separate models** | Task independence | 4x parameters, 4x inference time |
| **Multi-head shared backbone** | Parameter sharing, feature reuse, fast inference | Potential task interference |

Reasons for choosing multi-head:

1. **Feature sharing**: Low-level features (edges, textures) are useful for all tasks
2. **Efficiency**: Only one forward pass needed
3. **Regularization effect**: Multi-task learning provides implicit regularization

### Loss Weight Design

```python
# Loss weight for each task
self.loss_weights = {
    "color_cast": 1.0,
    "blurriness": 1.0,
    "watermark": 1.5,   # Watermark detection is more important
    "noise": 1.0,
}

# Total loss = Σ(weight_i × MSE(pred_i, label_i))
```

Watermark has higher weight because:

- Watermarks severely impact image usability
- Watermark detection is harder (diverse shapes and styles)

### Overall Score Calculation

```python
# Higher degradation score → Lower quality
degradation_weights = [0.2, 0.3, 0.25, 0.25]  # color, blur, wm, noise
overall = 1.0 - (all_scores * weights).sum()
```

**Blurriness has the highest weight (0.3)** because blur has the most significant impact on image quality.

## Training Strategy

```python
class MultiHeadQualityTrainer:
    def train(self, ...):
        # Optimizer: AdamW (with weight decay)
        self.optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=1e-4)

        # Learning rate schedule: Cosine Annealing
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

        # Gradient clipping: Prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # Early Stopping: Prevent overfitting
        if val_loss < best_val_loss:
            best_state = model.state_dict()
        elif patience_counter >= patience:
            break
```

## Synthetic Data Generation

Since real labeled data is difficult to obtain, **synthetic degradation** is used to generate training data:

```python
class SyntheticDegradationGenerator:
    def apply_color_cast(img, intensity):
        # Add color cast (red/green/blue/warm/cool tint)
        tint = random.choice([(255,200,200), ...])
        return img * (1-intensity) + tint * intensity

    def apply_blur(img, intensity):
        # Gaussian / Motion / Box blur
        return img.filter(GaussianBlur(radius=intensity*10))

    def apply_watermark(img, intensity):
        # Text overlay (tiled/center/diagonal/corner)
        draw.text(pos, "WATERMARK", alpha=intensity)

    def apply_noise(img, intensity):
        # Gaussian / Salt-pepper / Poisson noise
        return img + np.random.normal(0, intensity*50)
```

**Automatic label generation**: The applied intensity IS the ground truth label!

### Generate Training Data from HuggingFace

```bash
# List recommended datasets
python -m models.train_quality_assessment list_datasets

# Generate from HuggingFace dataset
python -m models.train_quality_assessment generate_hf \
    --dataset cifar10 \
    --output ./data/quality_training \
    --num-images 5000 \
    --samples-per-image 5
```

## Model Parameters

```
Total Parameters: ~2.5M

Backbone:    ~1.8M (shared)
Attention:   ~66K
4 Heads:     ~650K (~160K each)
```

Compared to ResNet-50 (25M), this model is very lightweight and suitable for batch processing large-scale data.

## Usage Examples

### Inference

```python
from models.quality_assessment import MultiHeadQualityInference

# Load model
model = MultiHeadQualityInference(
    model_path="./models/quality_assessment/model.pth",
    device="mps"  # or "cuda"
)

# Predict
scores = model.predict(image)
print(f"Color Cast: {scores.color_cast:.2f}")
print(f"Blurriness: {scores.blurriness:.2f}")
print(f"Watermark:  {scores.watermark:.2f}")
print(f"Noise:      {scores.noise:.2f}")
print(f"Overall:    {scores.overall:.2f}")
```

Example output:

```
Color Cast: 0.12
Blurriness: 0.05
Watermark:  0.00
Noise:      0.08
Overall:    0.93  (high quality image)
```

### Training

```bash
# Generate synthetic training data
python -m models.train_quality_assessment generate_hf \
    --dataset tiny-imagenet \
    --output ./data/quality_training \
    --num-images 10000

# Train model
python -m models.train_quality_assessment train \
    --data-dir ./data/quality_training \
    --output ./models/quality_assessment/model.pth \
    --epochs 50 \
    --batch-size 32

# Evaluate model
python -m models.train_quality_assessment evaluate \
    --model ./models/quality_assessment/model.pth \
    --data-dir ./data/quality_training
```

## File Structure

```text
models/quality_assessment/
├── __init__.py           # Module exports
├── trainer.py            # Model definition & trainer
├── inference.py          # Inference wrapper
├── synthetic_data.py     # Synthetic data generation
└── README.md             # This document
```

## Training Log

```text
(datapipeline_z_image) duoan ~/PycharmProjects/datapipeline_z_image (master) % python -m models.train_quality_assessment generate_hf \
    --dataset tiny-imagenet \
    --output_dir ./training_data \
    --num_images 500 \
    --samples_per_image 5
============================================================
Generating Synthetic Training Data (from HuggingFace)
============================================================
Using preset: tiny-imagenet
  Description: Tiny ImageNet - 100K images, 200 classes, good variety
Loading HuggingFace dataset: zh-plus/tiny-imagenet (split: train)...
README.md: 3.90kB [00:00, 2.83MB/s]
dataset_infos.json: 3.52kB [00:00, 14.0MB/s]
Loading 500 images...
  Loaded 100/500 images...
  Loaded 200/500 images...
  Loaded 300/500 images...
  Loaded 400/500 images...
  Loaded 500/500 images...
Successfully loaded 500 images from zh-plus/tiny-imagenet
Generating synthetic degraded samples (x5 per image)...

============================================================
Training data generated successfully!
  Source: zh-plus/tiny-imagenet
  Clean images: 500
  Total samples: 3000 (including 5x degraded versions)
  Image size: (224, 224)
  Images saved to: training_data/images.npy
  Labels saved to: training_data/labels.npy
============================================================

To train the model, run:
  python -m models.train_quality_assessment train \
      --train_images training_data/images.npy \
      --train_labels training_data/labels.npy
(datapipeline_z_image) duoan ~/PycharmProjects/datapipeline_z_image (feature/image-data-pipeline) % python -m models.train_quality_assessment train \
    --train_images ./training_data/images.npy \
    --train_labels ./training_data/labels.npy \
    --epochs 50
============================================================
Training Multi-Head Quality Assessment Model
============================================================
Using device: mps

Loading training data...
  Images shape: (3000, 224, 224, 3)
  Labels shape: (3000, 4)
  Train samples: 2700
  Validation samples: 300

Initializing model...

Training for 50 epochs...
  Batch size: 32
  Learning rate: 0.001
  Early stopping patience: 10
------------------------------------------------------------
Epoch 1/50, Train Loss: 0.0485, Val Loss: 0.0534 | CC: 0.0205, Blur: 0.0654, WM: 0.0302, Noise: 0.0628
Epoch 2/50, Train Loss: 0.0351, Val Loss: 0.0361 | CC: 0.0117, Blur: 0.0455, WM: 0.0234, Noise: 0.0480
Epoch 3/50, Train Loss: 0.0334, Val Loss: 0.0413 | CC: 0.0107, Blur: 0.0417, WM: 0.0229, Noise: 0.0468
Epoch 4/50, Train Loss: 0.0311, Val Loss: 0.0358 | CC: 0.0086, Blur: 0.0367, WM: 0.0234, Noise: 0.0439
Epoch 5/50, Train Loss: 0.0296, Val Loss: 0.0340 | CC: 0.0083, Blur: 0.0343, WM: 0.0232, Noise: 0.0409
Epoch 6/50, Train Loss: 0.0286, Val Loss: 0.0330 | CC: 0.0084, Blur: 0.0329, WM: 0.0222, Noise: 0.0396
Epoch 7/50, Train Loss: 0.0266, Val Loss: 0.0271 | CC: 0.0077, Blur: 0.0268, WM: 0.0225, Noise: 0.0382
Epoch 8/50, Train Loss: 0.0252, Val Loss: 0.0277 | CC: 0.0073, Blur: 0.0232, WM: 0.0222, Noise: 0.0370
Epoch 9/50, Train Loss: 0.0250, Val Loss: 0.0268 | CC: 0.0072, Blur: 0.0225, WM: 0.0224, Noise: 0.0366
Epoch 10/50, Train Loss: 0.0238, Val Loss: 0.0285 | CC: 0.0059, Blur: 0.0210, WM: 0.0220, Noise: 0.0350
Epoch 11/50, Train Loss: 0.0232, Val Loss: 0.0255 | CC: 0.0062, Blur: 0.0194, WM: 0.0216, Noise: 0.0346
Epoch 12/50, Train Loss: 0.0227, Val Loss: 0.0233 | CC: 0.0058, Blur: 0.0186, WM: 0.0214, Noise: 0.0345
Epoch 13/50, Train Loss: 0.0222, Val Loss: 0.0238 | CC: 0.0057, Blur: 0.0173, WM: 0.0214, Noise: 0.0336
Epoch 14/50, Train Loss: 0.0220, Val Loss: 0.0252 | CC: 0.0056, Blur: 0.0173, WM: 0.0215, Noise: 0.0329
Epoch 15/50, Train Loss: 0.0206, Val Loss: 0.0284 | CC: 0.0049, Blur: 0.0145, WM: 0.0212, Noise: 0.0313
Epoch 16/50, Train Loss: 0.0205, Val Loss: 0.0238 | CC: 0.0049, Blur: 0.0139, WM: 0.0211, Noise: 0.0317

```

## References

- Z-Image Technical Report (Section 2.1 - Data Profiling Engine)
- SE-Net: Squeeze-and-Excitation Networks
- Multi-Task Learning in Deep Neural Networks
