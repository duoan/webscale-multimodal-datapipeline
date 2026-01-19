"""
Classifier Model Trainer

Base trainer for classification models.
"""

import numpy as np


class ClassifierTrainer:
    """Base class for classifier trainers."""

    def __init__(self):
        """Initialize trainer."""
        self.model = None

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Train classifier model.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
        """
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels for features.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted labels array
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def save(self, model_path: str):
        """Save trained model."""
        raise NotImplementedError("Subclasses must implement save()")

    def load(self, model_path: str):
        """Load trained model."""
        raise NotImplementedError("Subclasses must implement load()")
