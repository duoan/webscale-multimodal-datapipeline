"""
Synthetic Data Generation for Quality Assessment Training

Generates training data by applying synthetic degradations to clean images:
- Color cast (color tints)
- Blurriness (various blur types)
- Watermarks (text overlays)
- Noise (Gaussian, salt-and-pepper, etc.)

Reference: Z-Image Technical Report (Section 2.1 - Technical Quality Assessment)
"""

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .trainer import MultiHeadLabels


class DegradationLevel(Enum):
    """Degradation severity levels."""

    NONE = 0
    LIGHT = 1
    MODERATE = 2
    HEAVY = 3
    SEVERE = 4


@dataclass
class DegradationConfig:
    """Configuration for synthetic degradation generation."""

    # Color cast settings
    color_cast_intensity_range: tuple[float, float] = (0.1, 0.5)
    color_cast_colors: list[tuple[int, int, int]] | None = None

    # Blur settings
    blur_radius_range: tuple[int, int] = (1, 10)
    motion_blur_kernel_range: tuple[int, int] = (5, 20)

    # Watermark settings
    watermark_opacity_range: tuple[float, float] = (0.1, 0.5)
    watermark_texts: list[str] | None = None
    watermark_font_size_range: tuple[int, int] = (20, 100)

    # Noise settings
    gaussian_noise_std_range: tuple[float, float] = (10, 50)
    salt_pepper_amount_range: tuple[float, float] = (0.01, 0.1)

    def __post_init__(self):
        if self.color_cast_colors is None:
            self.color_cast_colors = [
                (255, 200, 200),  # Red tint
                (200, 255, 200),  # Green tint
                (200, 200, 255),  # Blue tint
                (255, 255, 200),  # Yellow tint
                (255, 200, 255),  # Magenta tint
                (200, 255, 255),  # Cyan tint
                (255, 220, 180),  # Warm/sepia tint
                (180, 200, 255),  # Cool tint
            ]

        if self.watermark_texts is None:
            self.watermark_texts = [
                "SAMPLE",
                "WATERMARK",
                "STOCK PHOTO",
                "© COPYRIGHT",
                "PREVIEW",
                "DRAFT",
                "DEMO",
                "www.example.com",
                "shutterstock",
                "getty images",
            ]


class SyntheticDegradationGenerator:
    """Generator for synthetic image degradations.

    Creates training data by applying controlled degradations to clean images.
    """

    def __init__(self, config: DegradationConfig | None = None, seed: int | None = None):
        """Initialize generator.

        Args:
            config: Degradation configuration
            seed: Random seed for reproducibility
        """
        self.config = config or DegradationConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def apply_color_cast(self, img: Image.Image, intensity: float | None = None) -> tuple[Image.Image, float]:
        """Apply color cast to image.

        Args:
            img: Input PIL Image
            intensity: Optional specific intensity (0-1)

        Returns:
            Tuple of (degraded_image, degradation_level)
        """
        if intensity is None:
            intensity = random.uniform(*self.config.color_cast_intensity_range)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img, dtype=np.float32)
        tint_color = random.choice(self.config.color_cast_colors)

        # Apply color tint
        tint_array = np.array(tint_color, dtype=np.float32)
        blended = img_array * (1 - intensity) + tint_array * intensity
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended), intensity

    def apply_blur(
        self, img: Image.Image, intensity: float | None = None, blur_type: str | None = None
    ) -> tuple[Image.Image, float]:
        """Apply blur to image.

        Args:
            img: Input PIL Image
            intensity: Optional specific intensity (0-1)
            blur_type: Type of blur ('gaussian', 'motion', 'box')

        Returns:
            Tuple of (degraded_image, degradation_level)
        """
        if intensity is None:
            intensity = random.uniform(0.1, 1.0)

        if blur_type is None:
            blur_type = random.choice(["gaussian", "motion", "box"])

        # Scale intensity to actual blur parameters
        if blur_type == "gaussian":
            min_r, max_r = self.config.blur_radius_range
            radius = int(min_r + (max_r - min_r) * intensity)
            result = img.filter(ImageFilter.GaussianBlur(radius=max(1, radius)))

        elif blur_type == "motion":
            # Simulate motion blur with direction
            min_k, max_k = self.config.motion_blur_kernel_range
            kernel_size = int(min_k + (max_k - min_k) * intensity)
            kernel_size = max(3, kernel_size)

            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            direction = random.choice(["horizontal", "vertical", "diagonal"])

            if direction == "horizontal":
                kernel[kernel_size // 2, :] = 1.0 / kernel_size
            elif direction == "vertical":
                kernel[:, kernel_size // 2] = 1.0 / kernel_size
            else:
                np.fill_diagonal(kernel, 1.0 / kernel_size)

            # Apply convolution
            img_array = np.array(img, dtype=np.float32)
            if img.mode == "RGB":
                result_array = np.zeros_like(img_array)
                for c in range(3):
                    result_array[:, :, c] = self._convolve2d(img_array[:, :, c], kernel)
            else:
                result_array = self._convolve2d(img_array, kernel)

            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
            result = Image.fromarray(result_array)

        else:  # box blur
            min_r, max_r = self.config.blur_radius_range
            radius = int(min_r + (max_r - min_r) * intensity)
            result = img.filter(ImageFilter.BoxBlur(radius=max(1, radius)))

        return result, intensity

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution."""
        from scipy import ndimage

        return ndimage.convolve(image, kernel, mode="reflect")

    def apply_watermark(self, img: Image.Image, intensity: float | None = None) -> tuple[Image.Image, float]:
        """Apply watermark overlay to image.

        Args:
            img: Input PIL Image
            intensity: Optional specific intensity (0-1, controls opacity)

        Returns:
            Tuple of (degraded_image, degradation_level)
        """
        if intensity is None:
            intensity = random.uniform(*self.config.watermark_opacity_range)

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Create watermark overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Select watermark text and parameters
        text = random.choice(self.config.watermark_texts)
        min_size, max_size = self.config.watermark_font_size_range
        font_size = random.randint(min_size, max_size)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        # Calculate text position (random or tiled)
        style = random.choice(["center", "diagonal", "tiled", "corner"])
        alpha = int(255 * intensity)
        color = (128, 128, 128, alpha)  # Gray watermark

        if style == "center":
            # Center watermark
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img.size[0] - text_width) // 2
            y = (img.size[1] - text_height) // 2
            draw.text((x, y), text, fill=color, font=font)

        elif style == "diagonal":
            # Diagonal watermark
            temp_img = Image.new("RGBA", (img.size[0] * 2, img.size[1] * 2), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (temp_img.size[0] - text_width) // 2
            y = (temp_img.size[1] - text_height) // 2
            temp_draw.text((x, y), text, fill=color, font=font)
            temp_img = temp_img.rotate(45, expand=False)
            # Crop to original size
            left = (temp_img.size[0] - img.size[0]) // 2
            top = (temp_img.size[1] - img.size[1]) // 2
            overlay = temp_img.crop((left, top, left + img.size[0], top + img.size[1]))

        elif style == "tiled":
            # Tiled watermarks
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            spacing_x = text_width + 50
            spacing_y = text_height + 50
            for y in range(-text_height, img.size[1] + text_height, spacing_y):
                for x in range(-text_width, img.size[0] + text_width, spacing_x):
                    draw.text((x, y), text, fill=color, font=font)

        else:  # corner
            corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            margin = 10

            if corner == "top_left":
                x, y = margin, margin
            elif corner == "top_right":
                x, y = img.size[0] - text_width - margin, margin
            elif corner == "bottom_left":
                x, y = margin, img.size[1] - text_height - margin
            else:
                x, y = img.size[0] - text_width - margin, img.size[1] - text_height - margin

            draw.text((x, y), text, fill=color, font=font)

        # Composite
        result = Image.alpha_composite(img, overlay)
        return result.convert("RGB"), intensity

    def apply_noise(
        self, img: Image.Image, intensity: float | None = None, noise_type: str | None = None
    ) -> tuple[Image.Image, float]:
        """Apply noise to image.

        Args:
            img: Input PIL Image
            intensity: Optional specific intensity (0-1)
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')

        Returns:
            Tuple of (degraded_image, degradation_level)
        """
        if intensity is None:
            intensity = random.uniform(0.1, 1.0)

        if noise_type is None:
            noise_type = random.choice(["gaussian", "salt_pepper", "poisson"])

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img, dtype=np.float32)

        if noise_type == "gaussian":
            min_std, max_std = self.config.gaussian_noise_std_range
            std = min_std + (max_std - min_std) * intensity
            noise = np.random.normal(0, std, img_array.shape)
            noisy = img_array + noise

        elif noise_type == "salt_pepper":
            min_amount, max_amount = self.config.salt_pepper_amount_range
            amount = min_amount + (max_amount - min_amount) * intensity
            noisy = img_array.copy()
            # Salt
            salt_mask = np.random.random(img_array.shape[:2]) < (amount / 2)
            noisy[salt_mask] = 255
            # Pepper
            pepper_mask = np.random.random(img_array.shape[:2]) < (amount / 2)
            noisy[pepper_mask] = 0

        else:  # poisson
            # Scale by intensity
            vals = len(np.unique(img_array))
            vals = max(1, int(vals * (1 - intensity * 0.5)))
            noisy = np.random.poisson(img_array / 255.0 * vals) / float(vals) * 255

        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy), intensity

    def generate_degraded_sample(
        self,
        img: Image.Image,
        degradations: list[str] | None = None,
        intensities: dict[str, float] | None = None,
    ) -> tuple[Image.Image, MultiHeadLabels]:
        """Generate a degraded image sample with labels.

        Args:
            img: Clean input image
            degradations: List of degradations to apply ['color_cast', 'blurriness', 'watermark', 'noise']
            intensities: Optional dict of specific intensities for each degradation

        Returns:
            Tuple of (degraded_image, labels)
        """
        intensities = intensities or {}

        # Initialize labels with zeros
        labels = {
            "color_cast": 0.0,
            "blurriness": 0.0,
            "watermark": 0.0,
            "noise": 0.0,
        }

        result = img

        # Randomly select degradations if not specified
        if degradations is None:
            available = ["color_cast", "blurriness", "watermark", "noise"]
            num_degradations = random.randint(0, len(available))
            degradations = random.sample(available, num_degradations)

        # Apply each degradation
        for deg_type in degradations:
            intensity = intensities.get(deg_type)

            if deg_type == "color_cast":
                result, level = self.apply_color_cast(result, intensity)
                labels["color_cast"] = level

            elif deg_type == "blurriness":
                result, level = self.apply_blur(result, intensity)
                labels["blurriness"] = level

            elif deg_type == "watermark":
                result, level = self.apply_watermark(result, intensity)
                labels["watermark"] = level

            elif deg_type == "noise":
                result, level = self.apply_noise(result, intensity)
                labels["noise"] = level

        return result, MultiHeadLabels(**labels)

    def generate_dataset(
        self,
        images: list[Image.Image],
        samples_per_image: int = 5,
        include_clean: bool = True,
    ) -> tuple[list[np.ndarray], list[MultiHeadLabels]]:
        """Generate a synthetic dataset from clean images.

        Args:
            images: List of clean PIL Images
            samples_per_image: Number of degraded versions per image
            include_clean: Whether to include original clean images

        Returns:
            Tuple of (image_arrays, labels)
        """
        all_images = []
        all_labels = []

        for img in images:
            # Optionally include clean version
            if include_clean:
                clean_array = np.array(img.convert("RGB"))
                all_images.append(clean_array)
                all_labels.append(MultiHeadLabels(color_cast=0.0, blurriness=0.0, watermark=0.0, noise=0.0))

            # Generate degraded versions
            for _ in range(samples_per_image):
                degraded, labels = self.generate_degraded_sample(img)
                degraded_array = np.array(degraded.convert("RGB"))
                all_images.append(degraded_array)
                all_labels.append(labels)

        return all_images, all_labels


def create_training_data_from_directory(
    image_dir: str,
    output_dir: str,
    samples_per_image: int = 5,
    target_size: tuple[int, int] = (224, 224),
    config: DegradationConfig | None = None,
) -> tuple[str, str]:
    """Create training data from a directory of clean images.

    Args:
        image_dir: Directory containing clean images
        output_dir: Directory to save generated data
        samples_per_image: Number of degraded samples per clean image
        target_size: Target image size (H, W)
        config: Optional degradation configuration

    Returns:
        Tuple of paths to (images.npy, labels.npy)
    """
    from pathlib import Path

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in valid_extensions:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    print(f"Loaded {len(images)} images from {image_dir}")

    # Generate synthetic data
    generator = SyntheticDegradationGenerator(config)
    image_arrays, labels = generator.generate_dataset(images, samples_per_image=samples_per_image)

    # Convert to numpy arrays
    images_np = np.stack(image_arrays, axis=0)
    labels_np = np.array(
        [[lbl.color_cast, lbl.blurriness, lbl.watermark, lbl.noise] for lbl in labels],
        dtype=np.float32,
    )

    # Save
    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"

    np.save(images_path, images_np)
    np.save(labels_path, labels_np)

    print(f"Generated {len(image_arrays)} samples")
    print(f"Images saved to {images_path}")
    print(f"Labels saved to {labels_path}")

    return str(images_path), str(labels_path)


def create_training_data_from_huggingface(
    dataset_name: str,
    output_dir: str,
    num_images: int = 1000,
    samples_per_image: int = 5,
    target_size: tuple[int, int] = (224, 224),
    config: DegradationConfig | None = None,
    split: str = "train",
    image_column: str = "image",
    streaming: bool = True,
    seed: int = 42,
) -> tuple[str, str]:
    """Create training data from a HuggingFace dataset.

    This function loads clean images from a HuggingFace dataset and generates
    synthetic degraded versions for training the quality assessment model.

    Recommended clean image datasets:
    - "zh-plus/tiny-imagenet" - Small ImageNet subset (100K images)
    - "ILSVRC/imagenet-1k" - Full ImageNet (requires auth)
    - "cifar10" - CIFAR-10 (60K small images)
    - "food101" - Food images (101K images)
    - "Maysee/tiny-imagenet" - Another tiny ImageNet
    - "uoft-cs/cifar100" - CIFAR-100
    - "ethz/food101" - Food images

    Args:
        dataset_name: HuggingFace dataset name (e.g., "zh-plus/tiny-imagenet")
        output_dir: Directory to save generated data
        num_images: Number of images to load from the dataset
        samples_per_image: Number of degraded samples per clean image
        target_size: Target image size (H, W)
        config: Optional degradation configuration
        split: Dataset split to use (default: "train")
        image_column: Name of the image column in the dataset
        streaming: Whether to use streaming mode (recommended for large datasets)
        seed: Random seed for reproducibility

    Returns:
        Tuple of paths to (images.npy, labels.npy)

    Example:
        >>> images_path, labels_path = create_training_data_from_huggingface(
        ...     dataset_name="zh-plus/tiny-imagenet",
        ...     output_dir="./training_data",
        ...     num_images=500,
        ...     samples_per_image=5,
        ... )
    """
    from pathlib import Path

    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("Please install the 'datasets' package: pip install datasets") from err

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HuggingFace dataset: {dataset_name} (split: {split})...")

    # Load dataset
    if streaming:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        dataset = dataset.shuffle(seed=seed)
    else:
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.shuffle(seed=seed)

    # Load images
    images = []
    loaded = 0

    print(f"Loading {num_images} images...")

    for item in dataset:
        if loaded >= num_images:
            break

        try:
            # Get image from dataset
            img = item.get(image_column)

            if img is None:
                # Try common alternative column names
                for alt_col in ["img", "photo", "picture", "file"]:
                    img = item.get(alt_col)
                    if img is not None:
                        break

            if img is None:
                continue

            # Convert to PIL Image if needed
            if not isinstance(img, Image.Image):
                if isinstance(img, dict) and "bytes" in img:
                    from io import BytesIO

                    img = Image.open(BytesIO(img["bytes"]))
                elif isinstance(img, (bytes, bytearray)):
                    from io import BytesIO

                    img = Image.open(BytesIO(img))
                else:
                    continue

            # Convert to RGB and resize
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
            loaded += 1

            if loaded % 100 == 0:
                print(f"  Loaded {loaded}/{num_images} images...")

        except Exception as e:
            print(f"  Warning: Failed to load image: {e}")
            continue

    print(f"Successfully loaded {len(images)} images from {dataset_name}")

    if len(images) == 0:
        raise ValueError(
            f"No images could be loaded from dataset '{dataset_name}'. "
            f"Please check the dataset name and image column (tried: {image_column})"
        )

    # Generate synthetic data
    print(f"Generating synthetic degraded samples (x{samples_per_image} per image)...")
    generator = SyntheticDegradationGenerator(config, seed=seed)
    image_arrays, labels = generator.generate_dataset(images, samples_per_image=samples_per_image)

    # Convert to numpy arrays
    images_np = np.stack(image_arrays, axis=0)
    labels_np = np.array(
        [[lbl.color_cast, lbl.blurriness, lbl.watermark, lbl.noise] for lbl in labels],
        dtype=np.float32,
    )

    # Save
    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"

    np.save(images_path, images_np)
    np.save(labels_path, labels_np)

    print(f"\n{'=' * 60}")
    print("Training data generated successfully!")
    print(f"  Source: {dataset_name}")
    print(f"  Clean images: {len(images)}")
    print(f"  Total samples: {len(image_arrays)} (including {samples_per_image}x degraded versions)")
    print(f"  Image size: {target_size}")
    print(f"  Images saved to: {images_path}")
    print(f"  Labels saved to: {labels_path}")
    print(f"{'=' * 60}")

    return str(images_path), str(labels_path)


# Recommended datasets for quality assessment training
RECOMMENDED_DATASETS = {
    "tiny-imagenet": {
        "name": "zh-plus/tiny-imagenet",
        "description": "Tiny ImageNet - 100K images, 200 classes, good variety",
        "image_column": "image",
    },
    "cifar10": {
        "name": "cifar10",
        "description": "CIFAR-10 - 60K small images (32x32), 10 classes",
        "image_column": "img",
    },
    "cifar100": {
        "name": "uoft-cs/cifar100",
        "description": "CIFAR-100 - 60K small images, 100 classes",
        "image_column": "img",
    },
    "food101": {
        "name": "ethz/food101",
        "description": "Food-101 - 101K food images, good for real-world variety",
        "image_column": "image",
    },
    "flowers": {
        "name": "nelorth/oxford-flowers",
        "description": "Oxford Flowers - 8K flower images",
        "image_column": "image",
    },
    "cats-dogs": {
        "name": "microsoft/cats_vs_dogs",
        "description": "Cats vs Dogs - 25K images",
        "image_column": "image",
    },
}
