"""
KMeans Model Inference

Inference code for KMeans clustering models.
Used to assign cluster IDs to records for semantic deduplication.
"""

from typing import Any

import joblib
import numpy as np


class KMeansInference:
    """Inference wrapper for KMeans models."""

    def __init__(self, model_path: str | None = None, model: Any | None = None):
        """Initialize inference.

        Args:
            model_path: Path to saved KMeans model
            model: Pre-loaded KMeans model (alternative to model_path)
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = joblib.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")

    def predict_cluster(self, feature: np.ndarray) -> int:
        """Predict cluster ID for a single feature vector.

        Args:
            feature: Feature vector of shape (n_features,)

        Returns:
            Cluster ID (bucket ID for deduplication)
        """
        # Reshape for single sample
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        return int(self.model.predict(feature)[0])

    def predict_clusters(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster IDs for multiple feature vectors.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Cluster labels array of shape (n_samples,)
        """
        return self.model.predict(features)
