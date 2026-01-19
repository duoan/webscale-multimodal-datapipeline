"""
KMeans Model Trainer

Trains KMeans clustering models for semantic deduplication.
"""

import os

import joblib
import numpy as np
from sklearn.cluster import KMeans


class KMeansTrainer:
    """Trainer for KMeans clustering models."""

    def __init__(self, n_clusters: int = 100, random_state: int = 42, n_init: int = 10):
        """Initialize KMeans trainer.

        Args:
            n_clusters: Number of clusters (buckets) to create
            random_state: Random seed for reproducibility
            n_init: Number of times to run k-means with different centroid seeds
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.model: KMeans | None = None

    def train(self, features: np.ndarray) -> KMeans:
        """Train KMeans model on feature vectors.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
                     Can be image embeddings, phash vectors, etc.

        Returns:
            Trained KMeans model
        """
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init, verbose=1)
        self.model.fit(features)
        return self.model

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for features.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Cluster labels array of shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(features)

    def save(self, model_path: str):
        """Save trained model to disk.

        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path: str) -> KMeans:
        """Load trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded KMeans model
        """
        self.model = joblib.load(model_path)
        return self.model
