"""
AIGC Content Detector Trainer

Trains a binary classifier to detect AI-generated images using:
- Frozen SigLIP2 Vision Encoder (feature extraction)
- Lightweight MLP Head (2-3 layers classification)

The model learns to distinguish between:
- Real images (Label 0): Natural photographs
- AI-generated images (Label 1): Synthetic content from diffusion models

Key Design Decisions:
1. Freeze SigLIP2 backbone to preserve pre-trained semantic understanding
2. Use MLP head for fast training and low computational cost
3. Support hard negative mining for difficult samples
4. Apply augmentations to prevent overfitting to high-frequency artifacts

Reference: Imagen 3 Paper - AIGC Content Detection for Data Filtering
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def get_auto_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class AIGCDetectorConfig:
    """Configuration for AIGC detector training."""

    # SigLIP2 model settings
    siglip_model_name: str = "google/siglip2-so400m-patch14-384"
    embedding_dim: int = 1152  # SigLIP2 so400m output dimension

    # MLP Head architecture
    hidden_dims: tuple[int, ...] = (512, 128)
    dropout_rate: float = 0.3
    use_batch_norm: bool = True

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 30
    warmup_epochs: int = 2

    # Loss settings
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_smoothing: float = 0.1

    # Early stopping
    early_stopping_patience: int = 5

    def __post_init__(self):
        # Adjust embedding_dim based on model
        if "so400m" in self.siglip_model_name:
            self.embedding_dim = 1152
        elif "base" in self.siglip_model_name:
            self.embedding_dim = 768
        elif "large" in self.siglip_model_name:
            self.embedding_dim = 1024


class AIGCDataset(Dataset):
    """Dataset for AIGC detection training.

    Supports:
    - Loading from pre-extracted embeddings (recommended for speed)
    - Loading from raw images (requires SigLIP2 model)

    Labels:
    - 0: Real image (natural photograph)
    - 1: AI-generated image (synthetic)
    """

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        images: list[np.ndarray] | None = None,
        transform=None,
    ):
        """Initialize dataset.

        Args:
            embeddings: Pre-computed SigLIP2 embeddings (N, embedding_dim)
            labels: Binary labels (N,) - 0 for real, 1 for AI-generated
            images: Raw images if embeddings not provided (N, H, W, C)
            transform: Optional transform to apply to images
        """
        if embeddings is not None:
            self.embeddings = embeddings
            self.labels = labels
            self.mode = "embedding"
        elif images is not None:
            self.images = images
            self.labels = labels
            self.mode = "image"
            self.transform = transform
        else:
            raise ValueError("Either embeddings or images must be provided")

    def __len__(self):
        if self.mode == "embedding":
            return len(self.embeddings)
        return len(self.images)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.mode == "embedding":
            embedding = torch.from_numpy(self.embeddings[idx]).float()
            return embedding, label

        # Image mode
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Helps the model focus on hard-to-classify examples (e.g., photorealistic
    AI images that are difficult to distinguish from real photos).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits (N,) or (N, 1)
            targets: Ground truth labels (N,)

        Returns:
            Focal loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # p_t = p if y=1, else 1-p
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AIGCClassifierHead(nn.Module):
    """Lightweight MLP head for AIGC binary classification.

    Takes SigLIP2 [CLS] token embedding as input and outputs
    a single logit for binary classification (real vs AI-generated).

    Architecture:
    - 2-3 fully connected layers with ReLU activation
    - Dropout for regularization
    - Optional BatchNorm for training stability
    """

    def __init__(
        self,
        input_dim: int = 1152,
        hidden_dims: tuple[int, ...] = (512, 128),
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """Initialize classifier head.

        Args:
            input_dim: Input embedding dimension (SigLIP2 output)
            hidden_dims: Tuple of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use BatchNorm layers
        """
        super().__init__()

        self.input_dim = input_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Final classification layer (single output for binary classification)
        layers.append(nn.Linear(prev_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings (B, input_dim)

        Returns:
            Logits (B, 1) for binary classification
        """
        return self.classifier(x)


class SigLIP2Backbone(nn.Module):
    """Frozen SigLIP2 Vision Encoder for feature extraction.

    Loads a pre-trained SigLIP2 model and extracts [CLS] token embeddings.
    The backbone is frozen during training - only the MLP head is trained.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        device: str = "auto",
        use_fp16: bool = True,
    ):
        """Initialize SigLIP2 backbone.

        Args:
            model_name: HuggingFace model name for SigLIP2
            device: Device to run on ("cpu", "cuda", "mps", or "auto")
            use_fp16: Use FP16 for faster inference (CUDA only)
        """
        super().__init__()

        # Import transformers here to avoid import errors if not installed
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as err:
            raise ImportError(
                "transformers is required for SigLIP2. Install: pip install transformers"
            ) from err

        # Handle device selection
        if device == "auto":
            device = get_auto_device()

        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f"Loading SigLIP2 model: {model_name} on {device}...")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()
            print("SigLIP2 using FP16 half precision")

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.vision_config.hidden_size
        print(f"SigLIP2 embedding dimension: {self.embedding_dim}")

    @torch.inference_mode()
    def extract_embeddings(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract [CLS] token embeddings from images.

        Args:
            images: List of PIL images

        Returns:
            Embeddings tensor (B, embedding_dim)
        """
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        # Get vision embeddings
        outputs = self.model.vision_model(**inputs)

        # Use pooler output (CLS token) if available, else use mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Mean pooling over sequence dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.float()  # Always return float32

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        """Forward pass - alias for extract_embeddings."""
        return self.extract_embeddings(images)


class AIGCDetectorTrainer:
    """Trainer for AIGC detection classifier.

    Trains a lightweight MLP head on top of frozen SigLIP2 embeddings
    to distinguish between real and AI-generated images.

    Supports:
    - Training from pre-extracted embeddings (fast)
    - Training from raw images (with on-the-fly embedding extraction)
    - Hard negative mining for difficult samples
    - Focal loss for class imbalance handling
    """

    def __init__(
        self,
        config: AIGCDetectorConfig | None = None,
        device: str = "auto",
        backbone: SigLIP2Backbone | None = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            device: Device to run on
            backbone: Optional pre-initialized SigLIP2 backbone (for image mode)
        """
        self.config = config or AIGCDetectorConfig()

        if device == "auto":
            device = get_auto_device()
        self.device = torch.device(device)

        # Initialize classifier head
        self.model = AIGCClassifierHead(
            input_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm,
        ).to(self.device)

        # Backbone for image mode (lazy loading)
        self.backbone = backbone

        # Loss function
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(
                label_smoothing=self.config.label_smoothing if hasattr(nn.BCEWithLogitsLoss, "label_smoothing") else 0.0
            )

        self.optimizer = None
        self.scheduler = None

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        print(f"AIGC Detector Trainer initialized on {device}")
        print(f"  Embedding dim: {self.config.embedding_dim}")
        print(f"  MLP hidden dims: {self.config.hidden_dims}")
        print(f"  Loss: {'Focal Loss' if self.config.use_focal_loss else 'BCE'}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the AIGC detector.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Cosine annealing with warmup
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation phase
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                self.history["val_precision"].append(val_metrics["precision"])
                self.history["val_recall"].append(val_metrics["recall"])
                self.history["val_f1"].append(val_metrics["f1"])

                # Early stopping check
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Logging
            if verbose:
                val_str = ""
                if val_metrics:
                    val_str = (
                        f" | Val Loss: {val_metrics['loss']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.4f}, "
                        f"F1: {val_metrics['f1']:.4f}"
                    )
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"LR: {lr:.6f}{val_str}"
                )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(embeddings).squeeze(-1)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(train_loader), correct / total

    @torch.inference_mode()
    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for embeddings, labels in val_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(embeddings).squeeze(-1)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Precision, Recall, F1 for AI-generated class (label=1)
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @torch.inference_mode()
    def predict(self, embeddings: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Predict AIGC probability for embeddings.

        Args:
            embeddings: Input embeddings (N, embedding_dim)
            threshold: Classification threshold

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)

        logits = self.model(embeddings_tensor).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(np.int32)

        return preds, probs

    def save(self, model_path: str, save_full: bool = False) -> None:
        """Save trained model.

        Args:
            model_path: Path to save the model
            save_full: Whether to save full checkpoint (including optimizer)
        """
        import os

        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)

        if save_full:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "history": self.history,
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            }
            torch.save(checkpoint, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)

        print(f"Model saved to {model_path}")

    def load(self, model_path: str, load_full: bool = False) -> None:
        """Load trained model.

        Args:
            model_path: Path to the saved model
            load_full: Whether to load full checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=not load_full)

        if load_full and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint.get("history"):
                self.history = checkpoint["history"]
        else:
            state_dict = (
                checkpoint
                if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint
                else checkpoint["model_state_dict"]
            )
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"Model loaded from {model_path}")

    def get_hard_negatives(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        threshold_low: float = 0.3,
        threshold_high: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identify hard negative samples for retraining.

        Hard negatives are samples where the model is uncertain:
        - AI images misclassified as real (prob < threshold for label=1)
        - Real images misclassified as AI (prob > threshold for label=0)

        Args:
            embeddings: All embeddings
            labels: True labels
            threshold_low: Lower probability threshold
            threshold_high: Upper probability threshold

        Returns:
            Tuple of (hard_embeddings, hard_labels)
        """
        _, probs = self.predict(embeddings)

        # Find uncertain predictions
        is_hard = np.zeros(len(labels), dtype=bool)

        # Real images with high AI probability (false positives)
        is_hard |= (labels == 0) & (probs > threshold_high)

        # AI images with low AI probability (false negatives)
        is_hard |= (labels == 1) & (probs < threshold_low)

        # Uncertain region
        is_hard |= (probs > threshold_low) & (probs < threshold_high)

        hard_indices = np.where(is_hard)[0]
        print(f"Found {len(hard_indices)} hard samples ({100*len(hard_indices)/len(labels):.1f}%)")

        return embeddings[hard_indices], labels[hard_indices]
