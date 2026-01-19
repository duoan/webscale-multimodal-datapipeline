"""
Quality Assessment Model Inference

Inference code for quality assessment models based on Z-Image paper:
- Multi-head model for scoring degradation factors (color cast, blurriness, watermarks, noise)
- Single-head model for overall quality (legacy)

Reference: Z-Image Technical Report (Section 2.1 - Data Profiling Engine)
"""

from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .trainer import (
    DegradationScores,
    MultiHeadQualityAssessmentModel,
    QualityAssessmentModel,
)


class QualityAssessmentInference:
    """Inference wrapper for single-head quality assessment models (legacy)."""

    def __init__(
        self,
        model_path: str | None = None,
        model: QualityAssessmentModel | None = None,
        device: str = "cpu",
    ):
        """Initialize inference.

        Args:
            model_path: Path to saved model
            model: Pre-loaded model (alternative to model_path)
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = QualityAssessmentModel()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
        else:
            raise ValueError("Either model_path or model must be provided")

        self.model.eval()

    def predict(self, image: np.ndarray) -> float:
        """Predict quality score for a single image.

        Args:
            image: Image array of shape (H, W, C) or (C, H, W)

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Preprocess image
        if image.ndim == 3 and image.shape[2] == 3:
            # (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            score = output.item()

        return float(score)

    def predict_from_bytes(self, image_bytes: bytes) -> float:
        """Predict quality score from image bytes.

        Args:
            image_bytes: Image bytes data

        Returns:
            Quality score between 0.0 and 1.0
        """
        img = Image.open(BytesIO(image_bytes))
        img_array = np.array(img)
        return self.predict(img_array)


class MultiHeadQualityInference:
    """Inference wrapper for multi-head quality assessment models.

    Predicts multiple degradation scores:
    - color_cast: Color cast/tint level (0-1)
    - blurriness: Blur level (0-1)
    - watermark: Watermark visibility (0-1)
    - noise: Noise level (0-1)
    - overall: Overall quality score (0-1, higher = better quality)
    """

    def __init__(
        self,
        model_path: str | None = None,
        model: MultiHeadQualityAssessmentModel | None = None,
        device: str = "cpu",
        input_size: tuple[int, int] = (224, 224),
    ):
        """Initialize inference.

        Args:
            model_path: Path to saved model
            model: Pre-loaded model (alternative to model_path)
            device: Device to use ('cpu', 'cuda', or 'mps')
            input_size: Expected input size (H, W) for resizing
        """
        self.device = torch.device(device)
        self.input_size = input_size

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = MultiHeadQualityAssessmentModel()
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle both full checkpoint and state_dict only
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
        else:
            raise ValueError("Either model_path or model must be provided")

        self.model.eval()

    def _preprocess(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Preprocessed tensor of shape (1, C, H, W)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Resize if needed
        if image.shape[:2] != self.input_size:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((self.input_size[1], self.input_size[0]), Image.Resampling.LANCZOS)
            image = np.array(pil_img)

        # Ensure (H, W, C) -> (C, H, W)
        if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))

        # Take only RGB channels
        if image.shape[0] == 4:
            image = image[:3]

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Add batch dimension
        tensor = torch.from_numpy(image).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: np.ndarray | Image.Image) -> DegradationScores:
        """Predict degradation scores for a single image.

        Args:
            image: Image array of shape (H, W, C) or PIL Image

        Returns:
            DegradationScores with all degradation metrics
        """
        tensor = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(tensor)

        return DegradationScores(
            color_cast=float(outputs["color_cast"].item()),
            blurriness=float(outputs["blurriness"].item()),
            watermark=float(outputs["watermark"].item()),
            noise=float(outputs["noise"].item()),
            overall=float(outputs["overall"].item()),
        )

    def predict_dict(self, image: np.ndarray | Image.Image) -> dict[str, float]:
        """Predict degradation scores and return as dictionary.

        Args:
            image: Image array or PIL Image

        Returns:
            Dictionary with all degradation scores
        """
        return self.predict(image).to_dict()

    def predict_from_bytes(self, image_bytes: bytes) -> DegradationScores:
        """Predict degradation scores from image bytes.

        Args:
            image_bytes: Image bytes data

        Returns:
            DegradationScores with all degradation metrics
        """
        img = Image.open(BytesIO(image_bytes))
        return self.predict(img)

    def predict_batch(self, images: list[np.ndarray | Image.Image]) -> list[DegradationScores]:
        """Predict degradation scores for a batch of images.

        Args:
            images: List of image arrays or PIL Images

        Returns:
            List of DegradationScores
        """
        # Preprocess all images
        tensors = [self._preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            outputs = self.model(batch)

        results = []
        batch_size = len(images)
        for i in range(batch_size):
            results.append(
                DegradationScores(
                    color_cast=float(outputs["color_cast"][i].item()),
                    blurriness=float(outputs["blurriness"][i].item()),
                    watermark=float(outputs["watermark"][i].item()),
                    noise=float(outputs["noise"][i].item()),
                    overall=float(outputs["overall"][i].item()),
                )
            )

        return results

    def predict_batch_from_bytes(self, image_bytes_list: list[bytes]) -> list[DegradationScores]:
        """Predict degradation scores for a batch of images from bytes.

        Args:
            image_bytes_list: List of image bytes

        Returns:
            List of DegradationScores
        """
        images = [Image.open(BytesIO(b)) for b in image_bytes_list]
        return self.predict_batch(images)


def get_auto_device() -> str:
    """Automatically select the best available device.

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_quality_model(
    model_path: str,
    device: str | None = None,
    multi_head: bool = True,
) -> QualityAssessmentInference | MultiHeadQualityInference:
    """Convenience function to load a quality assessment model.

    Args:
        model_path: Path to saved model
        device: Device to use (auto-detected if None)
        multi_head: Whether to use multi-head model

    Returns:
        Inference wrapper for the model
    """
    if device is None:
        device = get_auto_device()

    if multi_head:
        return MultiHeadQualityInference(model_path=model_path, device=device)
    else:
        return QualityAssessmentInference(model_path=model_path, device=device)
