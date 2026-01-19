"""
AIGC Detector Data Preparation

Tools for collecting and preparing training data for AIGC detection:
1. Real Images: High-quality photographs from photography libraries
2. AI-Generated: Synthetic images from diffusion models (FLUX, MJv6, SDXL, etc.)

Data Augmentation Strategy:
- JPEG Compression: Forces model to learn semantic features instead of compression artifacts
- Gaussian Blur: Prevents reliance on high-frequency noise patterns
- Random Crop: Reduces edge-based overfitting
- Color Jitter: Improves robustness to color variations

Key Insight from Imagen 3:
The classifier must detect "Uncanny Valley" / physical violations,
not just statistical artifacts. Augmentation helps achieve this.
"""

import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class AIGCDatasetConfig:
    """Configuration for AIGC dataset creation."""

    # Target image size for training
    target_size: tuple[int, int] = (384, 384)

    # Augmentation settings
    jpeg_quality_range: tuple[int, int] = (30, 95)
    blur_radius_range: tuple[float, float] = (0.0, 2.0)
    color_jitter_strength: float = 0.2

    # Data collection settings
    samples_per_image: int = 1
    real_to_ai_ratio: float = 1.0  # 1.0 = balanced dataset

    # AI generator sources to include (for documentation)
    ai_generators: list[str] = field(
        default_factory=lambda: [
            "midjourney_v6",
            "flux_1",
            "sdxl",
            "dall_e_3",
            "stable_diffusion_3",
        ]
    )


def get_augmentation_transforms(config: AIGCDatasetConfig | None = None):
    """Get augmentation transforms for training.

    These augmentations are CRITICAL for preventing the classifier from
    relying on high-frequency artifacts (checkerboard patterns, etc.)
    that are easily removed by compression.

    Returns:
        torchvision transform composition
    """
    try:
        from torchvision import transforms
    except ImportError as err:
        raise ImportError("torchvision is required. Install: pip install torchvision") from err

    config = config or AIGCDatasetConfig()

    transform_list = [
        # Random resized crop - prevents edge/corner overfitting
        transforms.RandomResizedCrop(
            config.target_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        # JPEG compression simulation (critical!)
        JpegCompressionTransform(
            quality_range=config.jpeg_quality_range,
            p=0.5,
        ),
        # Gaussian blur (prevents high-freq artifact reliance)
        transforms.GaussianBlur(
            kernel_size=5,
            sigma=config.blur_radius_range,
        ),
        # Color jitter for robustness
        transforms.ColorJitter(
            brightness=config.color_jitter_strength,
            contrast=config.color_jitter_strength,
            saturation=config.color_jitter_strength,
            hue=config.color_jitter_strength / 4,
        ),
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(transform_list)


def get_validation_transforms(config: AIGCDatasetConfig | None = None):
    """Get transforms for validation (no augmentation).

    Returns:
        torchvision transform composition
    """
    try:
        from torchvision import transforms
    except ImportError as err:
        raise ImportError("torchvision is required. Install: pip install torchvision") from err

    config = config or AIGCDatasetConfig()

    return transforms.Compose(
        [
            transforms.Resize(config.target_size),
            transforms.CenterCrop(config.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class JpegCompressionTransform:
    """Apply random JPEG compression to simulate real-world image degradation.

    This is CRITICAL for AIGC detection:
    - AI images often have specific high-frequency "fingerprints"
    - JPEG compression destroys these fingerprints
    - Forces model to learn semantic/structural features instead
    """

    def __init__(self, quality_range: tuple[int, int] = (30, 95), p: float = 0.5):
        """Initialize JPEG compression transform.

        Args:
            quality_range: (min_quality, max_quality) for JPEG compression
            p: Probability of applying compression
        """
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random JPEG compression.

        Args:
            img: PIL Image

        Returns:
            Compressed PIL Image
        """
        if random.random() > self.p:
            return img

        quality = random.randint(*self.quality_range)

        # Compress to JPEG bytes and reload
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        return Image.open(buffer).convert("RGB")


def collect_real_images_from_directory(
    image_dir: str | Path,
    max_images: int | None = None,
    target_size: tuple[int, int] = (384, 384),
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> tuple[list[np.ndarray], list[str]]:
    """Collect real images from a local directory.

    Args:
        image_dir: Directory containing real images
        max_images: Maximum number of images to collect
        target_size: Target image size (H, W)
        extensions: File extensions to include

    Returns:
        Tuple of (images_list, paths_list)
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Directory not found: {image_dir}")

    # Find all image files
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.rglob(f"*{ext}"))
        image_paths.extend(image_dir.rglob(f"*{ext.upper()}"))

    # Shuffle and limit
    random.shuffle(image_paths)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"Loading {len(image_paths)} real images from {image_dir}...")

    images = []
    paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img))
            paths.append(str(path))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    print(f"Successfully loaded {len(images)} images")
    return images, paths


def collect_real_images_from_huggingface(
    dataset_name: str,
    num_images: int = 10000,
    target_size: tuple[int, int] = (384, 384),
    split: str = "train",
    image_column: str = "image",
    streaming: bool = True,
    seed: int = 42,
) -> list[np.ndarray]:
    """Collect real images from a HuggingFace dataset.

    Recommended datasets for real images:
    - "uoft-cs/cifar10" - Small natural images
    - "zh-plus/tiny-imagenet" - ImageNet subset
    - "timm/imagenet-1k-wds" - Full ImageNet (requires authentication)

    Args:
        dataset_name: HuggingFace dataset name
        num_images: Number of images to collect
        target_size: Target image size (H, W)
        split: Dataset split to use
        image_column: Name of the image column
        streaming: Use streaming mode (memory efficient)
        seed: Random seed for shuffling

    Returns:
        List of image arrays
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("datasets is required. Install: pip install datasets") from err

    print(f"Loading {num_images} real images from HuggingFace: {dataset_name}...")

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    images = []
    for i, sample in enumerate(dataset):
        if i >= num_images:
            break

        try:
            img = sample[image_column]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img))
        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{num_images} images...")

    print(f"Successfully loaded {len(images)} real images")
    return images


def create_training_dataset(
    real_images: list[np.ndarray],
    ai_images: list[np.ndarray],
    output_dir: str | Path,
    config: AIGCDatasetConfig | None = None,
) -> tuple[Path, Path]:
    """Create training dataset from real and AI images.

    Combines real and AI-generated images into a single dataset with labels.

    Args:
        real_images: List of real image arrays (label=0)
        ai_images: List of AI-generated image arrays (label=1)
        output_dir: Directory to save the dataset
        config: Dataset configuration

    Returns:
        Tuple of (images_path, labels_path)
    """
    config = config or AIGCDatasetConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating training dataset...")
    print(f"  Real images: {len(real_images)}")
    print(f"  AI images: {len(ai_images)}")

    # Balance dataset if needed
    if config.real_to_ai_ratio != 1.0:
        target_real = int(len(ai_images) * config.real_to_ai_ratio)
        if len(real_images) > target_real:
            real_images = random.sample(real_images, target_real)
            print(f"  Balanced to {len(real_images)} real images")

    # Combine and create labels
    all_images = real_images + ai_images
    all_labels = np.array([0] * len(real_images) + [1] * len(ai_images), dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = all_labels[indices]

    # Convert to numpy array
    images_array = np.stack(all_images, axis=0)

    # Save
    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"

    np.save(images_path, images_array)
    np.save(labels_path, all_labels)

    print("Dataset saved:")
    print(f"  Images: {images_path} (shape: {images_array.shape})")
    print(f"  Labels: {labels_path} (shape: {all_labels.shape})")
    print(f"  Class distribution: {np.bincount(all_labels)}")

    return images_path, labels_path


def extract_embeddings_batch(
    images: list[np.ndarray] | list[Image.Image],
    backbone,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract SigLIP2 embeddings for a batch of images.

    Args:
        images: List of image arrays or PIL Images
        backbone: SigLIP2Backbone instance
        batch_size: Batch size for inference
        show_progress: Whether to show progress

    Returns:
        Embeddings array (N, embedding_dim)
    """
    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]

        # Convert numpy to PIL if needed
        pil_batch = []
        for img in batch:
            if isinstance(img, np.ndarray):
                pil_batch.append(Image.fromarray(img))
            else:
                pil_batch.append(img)

        # Extract embeddings
        embeddings = backbone.extract_embeddings(pil_batch)
        all_embeddings.append(embeddings.cpu().numpy())

        if show_progress and (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Extracted embeddings: {min(i + batch_size, len(images))}/{len(images)}")

    return np.vstack(all_embeddings)


def load_ai_images_placeholder(
    output_dir: str | Path,
    num_images: int = 10000,
    target_size: tuple[int, int] = (384, 384),
    generators: list[str] | None = None,
) -> list[np.ndarray]:
    """Placeholder for loading AI-generated images.

    In practice, you should:
    1. Use the FLUX/SDXL/MJ API to generate images with diverse prompts
    2. Download from existing AI-art datasets
    3. Use a local Stable Diffusion setup

    Example datasets on HuggingFace:
    - "poloclub/diffusiondb" - SDXL generations
    - "imagenet-ai/imagenet-ai" - AI versions of ImageNet

    Args:
        output_dir: Directory containing AI images or to save them
        num_images: Number of images to load/generate
        target_size: Target image size
        generators: List of generator names to include

    Returns:
        List of AI-generated image arrays
    """
    generators = generators or ["flux_1", "sdxl", "midjourney_v6"]
    output_dir = Path(output_dir)

    print("=" * 60)
    print("AI Image Collection Guide")
    print("=" * 60)
    print()
    print("To collect AI-generated images, you need to:")
    print()
    print("1. Generate using APIs:")
    print("   - Midjourney v6: Discord API or unofficial API")
    print("   - FLUX.1: Replicate API or local (requires A100)")
    print("   - SDXL: Local or Replicate API")
    print("   - DALL-E 3: OpenAI API")
    print()
    print("2. Download from HuggingFace datasets:")
    print("   - poloclub/diffusiondb (Stable Diffusion)")
    print("   - imagenet-ai/imagenet-ai (AI ImageNet)")
    print()
    print("3. Use diverse prompts including:")
    print("   - Photorealistic scenes")
    print("   - Portraits and people")
    print("   - Abstract art")
    print("   - Nature and landscapes")
    print("   - Objects and products")
    print()
    print("Save generated images to:")
    print(f"  {output_dir / 'ai_generated/'}")
    print()
    print("Then load with:")
    print("  from webscale_multimodal_datapipeline.models.image_aigc_detector import collect_real_images_from_directory")
    print(f"  ai_images, _ = collect_real_images_from_directory('{output_dir}/ai_generated')")
    print("=" * 60)

    # Check if images already exist
    ai_dir = output_dir / "ai_generated"
    if ai_dir.exists():
        images, _ = collect_real_images_from_directory(ai_dir, max_images=num_images, target_size=target_size)
        if images:
            return images

    raise FileNotFoundError(
        f"No AI images found. Please generate or download AI images to: {ai_dir}\nSee the guide above for instructions."
    )


# Recommended real image datasets for training
RECOMMENDED_REAL_DATASETS: dict[str, dict[str, Any]] = {
    "unsplash-lite": {
        "name": "unsplash-research/unsplash-lite",
        "description": "Curated high-quality photographs from Unsplash",
        "image_column": "image",
    },
    "coco": {
        "name": "detection-datasets/coco",
        "description": "COCO dataset - diverse real photos",
        "image_column": "image",
    },
    "imagenet-subset": {
        "name": "zh-plus/tiny-imagenet",
        "description": "Tiny ImageNet - natural images subset",
        "image_column": "image",
    },
}

# Known AI-generated datasets for training
RECOMMENDED_AI_DATASETS: dict[str, dict[str, Any]] = {
    "diffusiondb": {
        "name": "poloclub/diffusiondb",
        "description": "2M Stable Diffusion generations with prompts",
        "image_column": "image",
        "generator": "stable_diffusion",
    },
    "journeydb": {
        "name": "JourneyDB/JourneyDB",
        "description": "4M+ Midjourney generations",
        "image_column": "image",
        "generator": "midjourney",
    },
}
