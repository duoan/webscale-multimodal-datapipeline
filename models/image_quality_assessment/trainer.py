"""
Quality Assessment Model Trainer

Trains models for visual degradations assessment based on Z-Image paper:
- Color cast detection
- Blurriness detection
- Watermark detection
- Noise detection

Reference: Z-Image Technical Report (Section 2.1 - Data Profiling Engine)
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset


class DegradationType(Enum):
    """Types of visual degradations assessed by the model."""

    COLOR_CAST = "color_cast"
    BLURRINESS = "blurriness"
    WATERMARK = "watermark"
    NOISE = "noise"


@dataclass
class DegradationScores:
    """Container for all degradation scores."""

    color_cast: float
    blurriness: float
    watermark: float
    noise: float
    overall: float  # Weighted combination of all scores

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "color_cast": self.color_cast,
            "blurriness": self.blurriness,
            "watermark": self.watermark,
            "noise": self.noise,
            "overall": self.overall,
        }


class MultiHeadLabels(NamedTuple):
    """Labels for multi-head training."""

    color_cast: float
    blurriness: float
    watermark: float
    noise: float


class ImageQualityDataset(Dataset):
    """Dataset for image quality assessment training."""

    def __init__(self, images: list[np.ndarray], labels: list[float]):
        """Initialize dataset.

        Args:
            images: List of image arrays
            labels: List of quality scores (0.0 to 1.0)
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class MultiHeadQualityDataset(Dataset):
    """Dataset for multi-head quality assessment training.

    Each sample has labels for all degradation types:
    - color_cast: 0-1 (0=no color cast, 1=severe color cast)
    - blurriness: 0-1 (0=sharp, 1=very blurry)
    - watermark: 0-1 (0=no watermark, 1=visible watermark)
    - noise: 0-1 (0=clean, 1=very noisy)
    """

    def __init__(
        self,
        images: list[np.ndarray],
        labels: list[MultiHeadLabels] | list[dict[str, float]],
        transform=None,
    ):
        """Initialize dataset.

        Args:
            images: List of image arrays (H, W, C) or (C, H, W)
            labels: List of MultiHeadLabels or dicts with degradation scores
            transform: Optional transform to apply to images
        """
        self.images = images
        self.labels = [MultiHeadLabels(**lbl) if isinstance(lbl, dict) else lbl for lbl in labels]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            # Ensure (C, H, W) format
            if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                image = np.transpose(image, (2, 0, 1))
            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
            image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label_tensor = torch.tensor(
            [label.color_cast, label.blurriness, label.watermark, label.noise],
            dtype=torch.float32,
        )

        return image, label_tensor


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class QualityAssessmentModel(nn.Module):
    """Neural network model for single-head quality assessment (legacy)."""

    def __init__(self, input_channels: int = 3, num_classes: int = 1):
        """Initialize model.

        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (1 for regression)
        """
        super().__init__()
        # Simple CNN architecture - can be replaced with more sophisticated models
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)  # Output between 0 and 1
        return x


class MultiHeadQualityAssessmentModel(nn.Module):
    """Multi-head neural network for visual degradation assessment.

    Based on Z-Image paper's quality assessment model that scores images on:
    - Color cast: Detects abnormal color tints
    - Blurriness: Measures lack of sharpness/focus
    - Watermark: Detects visible watermarks
    - Noise: Measures visual noise levels

    Architecture:
    - Shared backbone for feature extraction
    - Separate prediction heads for each degradation type
    - Uses attention mechanism for better localization
    """

    def __init__(
        self,
        input_channels: int = 3,
        backbone_channels: list[int] | None = None,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """Initialize multi-head model.

        Args:
            input_channels: Number of input channels (3 for RGB)
            backbone_channels: Channel progression for backbone [32, 64, 128, 256]
            head_hidden_dim: Hidden dimension for prediction heads
            dropout: Dropout rate for regularization
        """
        super().__init__()

        if backbone_channels is None:
            backbone_channels = [32, 64, 128, 256]

        self.backbone_channels = backbone_channels

        # Shared backbone for feature extraction
        backbone_layers = []
        in_ch = input_channels
        for out_ch in backbone_channels:
            backbone_layers.extend(
                [
                    ConvBlock(in_ch, out_ch),
                    ResidualBlock(out_ch),
                    nn.MaxPool2d(2, 2),
                ]
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*backbone_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Channel attention for feature refinement
        self.channel_attention = nn.Sequential(
            nn.Linear(backbone_channels[-1], backbone_channels[-1] // 4),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_channels[-1] // 4, backbone_channels[-1]),
            nn.Sigmoid(),
        )

        # Prediction heads for each degradation type
        feature_dim = backbone_channels[-1]

        self.head_color_cast = self._make_head(feature_dim, head_hidden_dim, dropout)
        self.head_blurriness = self._make_head(feature_dim, head_hidden_dim, dropout)
        self.head_watermark = self._make_head(feature_dim, head_hidden_dim, dropout)
        self.head_noise = self._make_head(feature_dim, head_hidden_dim, dropout)

        # Weights for computing overall quality score
        # Higher weights for more impactful degradations
        self.register_buffer(
            "degradation_weights",
            torch.tensor([0.2, 0.3, 0.25, 0.25]),  # color, blur, watermark, noise
        )

    def _make_head(self, in_features: int, hidden_dim: int, dropout: float) -> nn.Module:
        """Create a prediction head."""
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the backbone.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)

        # Apply channel attention
        attention = self.channel_attention(pooled)
        pooled = pooled * attention

        return pooled

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with scores for each degradation type and overall score
        """
        features = self.extract_features(x)

        # Get predictions from each head
        color_cast = self.head_color_cast(features)
        blurriness = self.head_blurriness(features)
        watermark = self.head_watermark(features)
        noise = self.head_noise(features)

        # Stack all predictions
        all_scores = torch.cat([color_cast, blurriness, watermark, noise], dim=1)

        # Compute overall quality (lower degradation = higher quality)
        # overall = 1 - weighted_sum(degradations)
        overall = 1.0 - (all_scores * self.degradation_weights).sum(dim=1, keepdim=True)

        result = {
            "color_cast": color_cast,
            "blurriness": blurriness,
            "watermark": watermark,
            "noise": noise,
            "overall": overall,
            "all_scores": all_scores,
        }

        if return_features:
            result["features"] = features

        return result

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only overall quality score (for compatibility).

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Overall quality score tensor of shape (B, 1)
        """
        return self.forward(x)["overall"]


class QualityAssessmentTrainer:
    """Trainer for quality assessment models (single-head, legacy)."""

    def __init__(self, model: nn.Module | None = None, device: str = "cpu"):
        """Initialize trainer.

        Args:
            model: Optional pre-initialized model
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        if model is None:
            self.model = QualityAssessmentModel().to(self.device)
        else:
            self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def train(self, train_loader: DataLoader, epochs: int = 10, lr: float = 0.001):
        """Train the model.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            lr: Learning rate
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def save(self, model_path: str):
        """Save trained model.

        Args:
            model_path: Path to save the model
        """
        import os

        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path: str):
        """Load trained model.

        Args:
            model_path: Path to the saved model
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")


class MultiHeadQualityTrainer:
    """Trainer for multi-head quality assessment models.

    Trains the model to predict multiple degradation factors simultaneously:
    - Color cast
    - Blurriness
    - Watermark presence
    - Noise level
    """

    def __init__(
        self,
        model: MultiHeadQualityAssessmentModel | None = None,
        device: str = "cpu",
        loss_weights: dict[str, float] | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Optional pre-initialized model
            device: Device to use ('cpu', 'cuda', or 'mps')
            loss_weights: Optional weights for each degradation loss
        """
        self.device = torch.device(device)

        if model is None:
            self.model = MultiHeadQualityAssessmentModel().to(self.device)
        else:
            self.model = model.to(self.device)

        # Loss weights for each head (can be tuned)
        self.loss_weights = loss_weights or {
            "color_cast": 1.0,
            "blurriness": 1.0,
            "watermark": 1.5,  # Higher weight for watermark detection
            "noise": 1.0,
        }

        self.criterion = nn.MSELoss(reduction="none")
        self.optimizer = None
        self.scheduler = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "color_cast_loss": [],
            "blurriness_loss": [],
            "watermark_loss": [],
            "noise_loss": [],
        }

    def _compute_loss(
        self, outputs: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-task loss.

        Args:
            outputs: Model outputs dict with all degradation scores
            labels: Ground truth labels tensor (B, 4)

        Returns:
            Tuple of (total_loss, per_head_losses_dict)
        """
        all_scores = outputs["all_scores"]  # (B, 4)

        # Compute MSE loss for each head
        per_sample_losses = self.criterion(all_scores, labels)  # (B, 4)

        # Weight the losses
        weights = torch.tensor(
            [
                self.loss_weights["color_cast"],
                self.loss_weights["blurriness"],
                self.loss_weights["watermark"],
                self.loss_weights["noise"],
            ],
            device=self.device,
        )

        weighted_losses = per_sample_losses * weights
        total_loss = weighted_losses.mean()

        # Per-head losses for logging
        per_head_losses = {
            "color_cast": per_sample_losses[:, 0].mean().item(),
            "blurriness": per_sample_losses[:, 1].mean().item(),
            "watermark": per_sample_losses[:, 2].mean().item(),
            "noise": per_sample_losses[:, 3].mean().item(),
        }

        return total_loss, per_head_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the multi-head model.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization weight
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_head_losses = dict.fromkeys(self.loss_weights, 0.0)

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss, head_losses = self._compute_loss(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                for k, v in head_losses.items():
                    train_head_losses[k] += v

            self.scheduler.step()

            # Average losses
            num_batches = len(train_loader)
            train_loss /= num_batches
            for k in train_head_losses:
                train_head_losses[k] /= num_batches

            self.history["train_loss"].append(train_loss)
            for k, v in train_head_losses.items():
                self.history[f"{k}_loss"].append(v)

            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Logging
            if verbose:
                val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}{val_str} | "
                    f"CC: {train_head_losses['color_cast']:.4f}, "
                    f"Blur: {train_head_losses['blurriness']:.4f}, "
                    f"WM: {train_head_losses['watermark']:.4f}, "
                    f"Noise: {train_head_losses['noise']:.4f}"
                )

        # Restore best model if early stopping was used
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss, _ = self._compute_loss(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with metrics for each degradation type
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                all_preds.append(outputs["all_scores"].cpu())
                all_labels.append(labels)

        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Compute metrics
        mse = functional.mse_loss(preds, labels, reduction="none").mean(dim=0)
        mae = (preds - labels).abs().mean(dim=0)

        metrics = {
            "color_cast_mse": mse[0].item(),
            "blurriness_mse": mse[1].item(),
            "watermark_mse": mse[2].item(),
            "noise_mse": mse[3].item(),
            "color_cast_mae": mae[0].item(),
            "blurriness_mae": mae[1].item(),
            "watermark_mae": mae[2].item(),
            "noise_mae": mae[3].item(),
            "overall_mse": mse.mean().item(),
            "overall_mae": mae.mean().item(),
        }

        return metrics

    def save(self, model_path: str, save_full: bool = False):
        """Save trained model.

        Args:
            model_path: Path to save the model
            save_full: Whether to save full checkpoint (including optimizer state)
        """
        import os

        os.makedirs(
            os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
            exist_ok=True,
        )

        if save_full:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "loss_weights": self.loss_weights,
                "history": self.history,
            }
            torch.save(checkpoint, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)

        print(f"Model saved to {model_path}")

    def load(self, model_path: str, load_full: bool = False):
        """Load trained model.

        Args:
            model_path: Path to the saved model
            load_full: Whether to load full checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        if load_full and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint.get("loss_weights"):
                self.loss_weights = checkpoint["loss_weights"]
            if checkpoint.get("history"):
                self.history = checkpoint["history"]
        else:
            # Assume it's just state_dict
            state_dict = (
                checkpoint
                if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint
                else checkpoint.get("model_state_dict", checkpoint)
            )
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"Model loaded from {model_path}")
