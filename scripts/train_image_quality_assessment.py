"""
Train Quality Assessment Model

Script to train multi-head quality assessment models for visual degradations
based on the Z-Image paper.

Supports:
1. Training from pre-generated numpy data
2. Generating synthetic training data from clean images
3. Multi-head model training for degradation factors:
   - Color cast
   - Blurriness
   - Watermarks
   - Noise

Reference: Z-Image Technical Report (Section 2.1 - Data Profiling Engine)

Usage Examples:
    # Generate synthetic data and train:
    python -m models.train_quality_assessment generate_data \\
        --image_dir ./clean_images \\
        --output_dir ./training_data

    # Train multi-head model:
    python -m models.train_quality_assessment train \\
        --train_images ./training_data/images.npy \\
        --train_labels ./training_data/labels.npy \\
        --output_path ./models/quality_assessment/multihead_model.pth

    # Train with validation split:
    python -m models.train_quality_assessment train \\
        --train_images ./training_data/images.npy \\
        --train_labels ./training_data/labels.npy \\
        --val_split 0.1 \\
        --epochs 50 \\
        --early_stopping 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from webscale_multimodal_datapipeline.models.image_quality_assessment import (
    RECOMMENDED_DATASETS,
    DegradationConfig,
    MultiHeadQualityDataset,
    MultiHeadQualityTrainer,
    create_training_data_from_directory,
    create_training_data_from_huggingface,
    get_auto_device,
)


def cmd_generate_data(args: argparse.Namespace) -> None:
    """Generate synthetic training data from local images."""
    print("=" * 60)
    print("Generating Synthetic Training Data (from local directory)")
    print("=" * 60)

    # Configure degradation parameters if custom settings provided
    config = DegradationConfig()
    if args.blur_max:
        config.blur_radius_range = (1, args.blur_max)
    if args.noise_max:
        config.gaussian_noise_std_range = (10, args.noise_max)

    images_path, labels_path = create_training_data_from_directory(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        samples_per_image=args.samples_per_image,
        target_size=(args.image_size, args.image_size),
        config=config,
    )

    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print(f"  Images: {images_path}")
    print(f"  Labels: {labels_path}")
    print("=" * 60)


def cmd_generate_from_hf(args: argparse.Namespace) -> None:
    """Generate synthetic training data from HuggingFace dataset."""
    print("=" * 60)
    print("Generating Synthetic Training Data (from HuggingFace)")
    print("=" * 60)

    # Check if using a preset
    dataset_name = args.dataset
    image_column = args.image_column

    if dataset_name in RECOMMENDED_DATASETS:
        preset = RECOMMENDED_DATASETS[dataset_name]
        print(f"Using preset: {dataset_name}")
        print(f"  Description: {preset['description']}")
        dataset_name = preset["name"]
        if image_column == "image":  # Use preset's column if not explicitly set
            image_column = preset["image_column"]

    # Configure degradation parameters
    config = DegradationConfig()
    if args.blur_max:
        config.blur_radius_range = (1, args.blur_max)
    if args.noise_max:
        config.gaussian_noise_std_range = (10, args.noise_max)

    images_path, labels_path = create_training_data_from_huggingface(
        dataset_name=dataset_name,
        output_dir=args.output_dir,
        num_images=args.num_images,
        samples_per_image=args.samples_per_image,
        target_size=(args.image_size, args.image_size),
        config=config,
        split=args.split,
        image_column=image_column,
        streaming=not args.no_streaming,
        seed=args.seed,
    )

    print("\nTo train the model, run:")
    print("  python -m models.train_quality_assessment train \\")
    print(f"      --train_images {images_path} \\")
    print(f"      --train_labels {labels_path}")


def cmd_list_datasets(args: argparse.Namespace) -> None:
    """List recommended HuggingFace datasets for training."""
    print("=" * 60)
    print("Recommended HuggingFace Datasets for Quality Assessment")
    print("=" * 60)
    print()

    for key, info in RECOMMENDED_DATASETS.items():
        print(f"  {key}")
        print(f"    Dataset: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Image column: {info['image_column']}")
        print()

    print("Usage example:")
    print("  python -m models.train_quality_assessment generate_hf \\")
    print("      --dataset tiny-imagenet \\")
    print("      --output_dir ./training_data \\")
    print("      --num_images 500")
    print()
    print("Or use any HuggingFace dataset directly:")
    print("  python -m models.train_quality_assessment generate_hf \\")
    print("      --dataset 'username/dataset-name' \\")
    print("      --image_column 'image' \\")
    print("      --output_dir ./training_data")


def cmd_train(args: argparse.Namespace) -> None:
    """Train the multi-head quality assessment model."""
    print("=" * 60)
    print("Training Multi-Head Quality Assessment Model")
    print("=" * 60)

    # Determine device
    device = args.device if args.device != "auto" else get_auto_device()
    print(f"Using device: {device}")

    # Load data
    print("\nLoading training data...")
    train_images = np.load(args.train_images)
    train_labels = np.load(args.train_labels)

    print(f"  Images shape: {train_images.shape}")
    print(f"  Labels shape: {train_labels.shape}")

    # Validate data
    if train_labels.shape[1] != 4:
        raise ValueError(
            f"Expected labels with 4 columns (color_cast, blurriness, watermark, noise), got {train_labels.shape[1]}"
        )

    # Convert labels to list of dicts for MultiHeadQualityDataset
    labels_list = [
        {
            "color_cast": float(row[0]),
            "blurriness": float(row[1]),
            "watermark": float(row[2]),
            "noise": float(row[3]),
        }
        for row in train_labels
    ]

    # Create full dataset
    full_dataset = MultiHeadQualityDataset(
        images=list(train_images),
        labels=labels_list,
    )

    # Split into train/val if requested
    val_loader = None
    if args.val_split > 0:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
    else:
        train_dataset = full_dataset
        print(f"  Total samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Initialize trainer
    print("\nInitializing model...")
    trainer = MultiHeadQualityTrainer(device=device)

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.early_stopping}")
    print("-" * 60)

    _history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        verbose=True,
    )

    # Evaluate on full training set
    print("\n" + "-" * 60)
    print("Final Evaluation on Training Set:")
    metrics = trainer.evaluate(train_loader)
    print(f"  Overall MSE: {metrics['overall_mse']:.4f}")
    print(f"  Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"  Color Cast MSE: {metrics['color_cast_mse']:.4f}")
    print(f"  Blurriness MSE: {metrics['blurriness_mse']:.4f}")
    print(f"  Watermark MSE: {metrics['watermark_mse']:.4f}")
    print(f"  Noise MSE: {metrics['noise_mse']:.4f}")

    # Save model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path), save_full=args.save_full)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Model saved to: {output_path}")
    print("=" * 60)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model on test data."""
    print("=" * 60)
    print("Evaluating Quality Assessment Model")
    print("=" * 60)

    device = args.device if args.device != "auto" else get_auto_device()
    print(f"Using device: {device}")

    # Load data
    print("\nLoading test data...")
    test_images = np.load(args.test_images)
    test_labels = np.load(args.test_labels)

    labels_list = [
        {
            "color_cast": float(row[0]),
            "blurriness": float(row[1]),
            "watermark": float(row[2]),
            "noise": float(row[3]),
        }
        for row in test_labels
    ]

    test_dataset = MultiHeadQualityDataset(
        images=list(test_images),
        labels=labels_list,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Load model
    print(f"Loading model from {args.model_path}...")
    trainer = MultiHeadQualityTrainer(device=device)
    trainer.load(args.model_path)

    # Evaluate
    print("\nEvaluating...")
    metrics = trainer.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("-" * 60)
    print(f"  Overall MSE: {metrics['overall_mse']:.4f}")
    print(f"  Overall MAE: {metrics['overall_mae']:.4f}")
    print("-" * 60)
    print("  Per-Head Metrics:")
    print(f"    Color Cast - MSE: {metrics['color_cast_mse']:.4f}, MAE: {metrics['color_cast_mae']:.4f}")
    print(f"    Blurriness - MSE: {metrics['blurriness_mse']:.4f}, MAE: {metrics['blurriness_mae']:.4f}")
    print(f"    Watermark  - MSE: {metrics['watermark_mse']:.4f}, MAE: {metrics['watermark_mae']:.4f}")
    print(f"    Noise      - MSE: {metrics['noise_mse']:.4f}, MAE: {metrics['noise_mae']:.4f}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train multi-head quality assessment model for visual degradations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate data subcommand
    gen_parser = subparsers.add_parser(
        "generate_data",
        help="Generate synthetic training data from clean images",
    )
    gen_parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing clean images",
    )
    gen_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated data",
    )
    gen_parser.add_argument(
        "--samples_per_image",
        type=int,
        default=5,
        help="Number of degraded samples per clean image (default: 5)",
    )
    gen_parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target image size (default: 224)",
    )
    gen_parser.add_argument(
        "--blur_max",
        type=int,
        default=None,
        help="Maximum blur radius (default: 10)",
    )
    gen_parser.add_argument(
        "--noise_max",
        type=float,
        default=None,
        help="Maximum noise std (default: 50)",
    )

    # Generate data from HuggingFace subcommand
    hf_parser = subparsers.add_parser(
        "generate_hf",
        help="Generate synthetic training data from a HuggingFace dataset",
    )
    hf_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name or preset (e.g., 'tiny-imagenet', 'zh-plus/tiny-imagenet')",
    )
    hf_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated data",
    )
    hf_parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of images to load from dataset (default: 500)",
    )
    hf_parser.add_argument(
        "--samples_per_image",
        type=int,
        default=5,
        help="Number of degraded samples per clean image (default: 5)",
    )
    hf_parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target image size (default: 224)",
    )
    hf_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    hf_parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Name of the image column in the dataset (default: image)",
    )
    hf_parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode (downloads full dataset)",
    )
    hf_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    hf_parser.add_argument(
        "--blur_max",
        type=int,
        default=None,
        help="Maximum blur radius (default: 10)",
    )
    hf_parser.add_argument(
        "--noise_max",
        type=float,
        default=None,
        help="Maximum noise std (default: 50)",
    )

    # List recommended datasets subcommand
    subparsers.add_parser(
        "list_datasets",
        help="List recommended HuggingFace datasets for training",
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train the multi-head quality assessment model",
    )
    train_parser.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="Path to training images (numpy .npy file)",
    )
    train_parser.add_argument(
        "--train_labels",
        type=str,
        required=True,
        help="Path to training labels (numpy .npy file with 4 columns)",
    )
    train_parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoints/multihead_quality_model.pth",
        help="Path to save trained model",
    )
    train_parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    train_parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'mps', or 'auto' (default: auto)",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--save_full",
        action="store_true",
        help="Save full checkpoint (including optimizer state)",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model on test data",
    )
    eval_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    eval_parser.add_argument(
        "--test_images",
        type=str,
        required=True,
        help="Path to test images (numpy .npy file)",
    )
    eval_parser.add_argument(
        "--test_labels",
        type=str,
        required=True,
        help="Path to test labels (numpy .npy file)",
    )
    eval_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'mps', or 'auto' (default: auto)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate_data":
        cmd_generate_data(args)
    elif args.command == "generate_hf":
        cmd_generate_from_hf(args)
    elif args.command == "list_datasets":
        cmd_list_datasets(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
