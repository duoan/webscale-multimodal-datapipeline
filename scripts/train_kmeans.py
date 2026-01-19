"""
Train KMeans Model for Semantic Deduplication

Example script to train a KMeans model for clustering images.
The cluster IDs can then be used as bucket IDs for semantic deduplication.
"""

import numpy as np
import argparse
from webscale_multimodal_datapipeline.models.kmeans.trainer import KMeansTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train KMeans model for semantic deduplication")
    parser.add_argument("--features_path", type=str, required=True,
                       help="Path to feature file (numpy array or pickle)")
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of clusters (default: 100)")
    parser.add_argument("--output_path", type=str, default="./models/kmeans/kmeans_model.pkl",
                       help="Path to save trained model (default: ./models/kmeans/kmeans_model.pkl)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Load features
    print(f"Loading features from {args.features_path}...")
    if args.features_path.endswith('.npy'):
        features = np.load(args.features_path)
    else:
        import pickle
        with open(args.features_path, 'rb') as f:
            features = pickle.load(f)
    
    print(f"Features shape: {features.shape}")
    
    # Train model
    print(f"Training KMeans with {args.n_clusters} clusters...")
    trainer = KMeansTrainer(
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )
    model = trainer.train(features)
    
    # Save model
    trainer.save(args.output_path)
    print(f"Training completed. Model saved to {args.output_path}")


if __name__ == "__main__":
    main()
