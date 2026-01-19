"""
Distributed KMeans Training Script

Example script to train distributed KMeans using Ray.
"""

import argparse

import ray

from webscale_multimodal_datapipeline.models.kmeans.distributed_trainer import DistributedKMeansTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train distributed KMeans model using Ray",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with parquet files
  python models/kmeans/distributed_train.py \\
      --data_urls ./data/features_*.parquet \\
      --n_clusters 100 \\
      --n_workers 4

  # Train with URLs from file
  python models/kmeans/distributed_train.py \\
      --data_urls_file parquet_urls.txt \\
      --n_clusters 100 \\
      --n_workers 8 \\
      --output_dir ./kmeans_output

Note: Parquet files must have two columns:
  - 'id' (string): Unique identifier for each record
  - 'feature' (array/embedding): Feature vector for clustering
        """,
    )
    parser.add_argument(
        "--data_urls",
        type=str,
        nargs="+",
        default=None,
        help="List of parquet file URLs (must have 'id' and 'feature' columns)",
    )
    parser.add_argument(
        "--data_urls_file", type=str, default=None, help="Path to file containing parquet URLs (one per line)"
    )
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters (default: 100)")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of Ray workers (default: 4)")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum iterations (default: 100)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kmeans_training",
        help="Output directory for intermediate results (default: ./kmeans_training)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/kmeans/kmeans_model.pkl",
        help="Path to save final model (default: ./models/kmeans/kmeans_model.pkl)",
    )

    args = parser.parse_args()

    # Load data URLs
    if args.data_urls_file:
        with open(args.data_urls_file) as f:
            data_urls = [line.strip() for line in f if line.strip()]
    elif args.data_urls:
        data_urls = args.data_urls
    else:
        parser.error("Either --data_urls or --data_urls_file must be provided")

    print(f"Found {len(data_urls)} parquet file URLs")
    print("Parquet files must have 'id' (string) and 'feature' (embedding) columns")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.n_workers)

    try:
        # Create trainer
        trainer = DistributedKMeansTrainer(
            n_clusters=args.n_clusters,
            data_urls=data_urls,
            n_workers=args.n_workers,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            random_state=args.random_state,
            output_dir=args.output_dir,
        )

        # Train model
        trainer.train()

        # Save model
        trainer.save(args.model_path)
        print("\nTraining completed successfully!")
        print(f"Model saved to: {args.model_path}")
        print(f"Training artifacts in: {args.output_dir}")

    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
