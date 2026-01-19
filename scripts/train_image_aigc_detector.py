"""
Train AIGC Content Detector

Script to train a binary classifier for detecting AI-generated images.
Uses frozen SigLIP2 backbone with a lightweight MLP head.

Based on Imagen 3 findings: AIGC content filtering is crucial for preventing
degradation in model output quality and physical realism.

Usage Examples:
    # Step 1: Extract embeddings from images (one-time preprocessing)
    python -m models.train_image_aigc_detector extract_embeddings \\
        --real_images_dir ./data/real_photos \\
        --ai_images_dir ./data/ai_generated \\
        --output_dir ./data/aigc_embeddings

    # Step 2: Train classifier on embeddings (fast!)
    python -m models.train_image_aigc_detector train \\
        --embeddings_dir ./data/aigc_embeddings \\
        --output_path ./models/aigc_detector/aigc_classifier.pth

    # Alternative: Train directly from images (slower, more memory)
    python -m models.train_image_aigc_detector train_from_images \\
        --real_images_dir ./data/real_photos \\
        --ai_images_dir ./data/ai_generated \\
        --output_path ./models/aigc_detector/aigc_classifier.pth

    # Evaluate trained model
    python -m models.train_image_aigc_detector evaluate \\
        --model_path ./models/aigc_detector/aigc_classifier.pth \\
        --embeddings_dir ./data/aigc_embeddings_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from webscale_multimodal_datapipeline.models.image_aigc_detector import (
    AIGCDataset,
    AIGCDetectorTrainer,
    SigLIP2Backbone,
    collect_real_images_from_directory,
    collect_real_images_from_huggingface,
    get_auto_device,
)
from webscale_multimodal_datapipeline.models.image_aigc_detector.synthetic_data import (
    RECOMMENDED_AI_DATASETS,
    RECOMMENDED_REAL_DATASETS,
    extract_embeddings_batch,
)
from webscale_multimodal_datapipeline.models.image_aigc_detector.trainer import AIGCDetectorConfig


def cmd_extract_embeddings(args: argparse.Namespace) -> None:
    """Extract SigLIP2 embeddings from images."""
    print("=" * 60)
    print("Extracting SigLIP2 Embeddings for AIGC Detection")
    print("=" * 60)

    device = args.device if args.device != "auto" else get_auto_device()
    print(f"Using device: {device}")

    # Initialize SigLIP2 backbone
    print(f"\nLoading SigLIP2 model: {args.siglip_model}...")
    backbone = SigLIP2Backbone(
        model_name=args.siglip_model,
        device=device,
        use_fp16=args.fp16,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect real images
    print("\n" + "-" * 40)
    print("Collecting Real Images")
    print("-" * 40)
    if args.real_images_dir:
        real_images, _ = collect_real_images_from_directory(
            args.real_images_dir,
            max_images=args.max_real,
            target_size=(args.image_size, args.image_size),
        )
    elif args.real_dataset:
        real_images = collect_real_images_from_huggingface(
            args.real_dataset,
            num_images=args.max_real,
            target_size=(args.image_size, args.image_size),
            image_column=args.real_image_column,
        )
    else:
        raise ValueError("Either --real_images_dir or --real_dataset must be provided")

    # Collect AI images
    print("\n" + "-" * 40)
    print("Collecting AI-Generated Images")
    print("-" * 40)
    if args.ai_images_dir:
        ai_images, _ = collect_real_images_from_directory(
            args.ai_images_dir,
            max_images=args.max_ai,
            target_size=(args.image_size, args.image_size),
        )
    elif args.ai_dataset:
        ai_images = collect_real_images_from_huggingface(
            args.ai_dataset,
            num_images=args.max_ai,
            target_size=(args.image_size, args.image_size),
            image_column=args.ai_image_column,
        )
    else:
        raise ValueError("Either --ai_images_dir or --ai_dataset must be provided")

    # Extract embeddings for real images
    print("\n" + "-" * 40)
    print("Extracting Real Image Embeddings")
    print("-" * 40)
    real_embeddings = extract_embeddings_batch(
        real_images,
        backbone,
        batch_size=args.batch_size,
        show_progress=True,
    )
    print(f"Real embeddings shape: {real_embeddings.shape}")

    # Extract embeddings for AI images
    print("\n" + "-" * 40)
    print("Extracting AI Image Embeddings")
    print("-" * 40)
    ai_embeddings = extract_embeddings_batch(
        ai_images,
        backbone,
        batch_size=args.batch_size,
        show_progress=True,
    )
    print(f"AI embeddings shape: {ai_embeddings.shape}")

    # Combine and save
    all_embeddings = np.vstack([real_embeddings, ai_embeddings])
    all_labels = np.array([0] * len(real_embeddings) + [1] * len(ai_embeddings), dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(all_labels))
    all_embeddings = all_embeddings[indices]
    all_labels = all_labels[indices]

    # Save
    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.npy"

    np.save(embeddings_path, all_embeddings)
    np.save(labels_path, all_labels)

    print("\n" + "=" * 60)
    print("Embedding Extraction Complete!")
    print(f"  Embeddings: {embeddings_path} (shape: {all_embeddings.shape})")
    print(f"  Labels: {labels_path} (shape: {all_labels.shape})")
    print(f"  Class distribution: Real={np.sum(all_labels == 0)}, AI={np.sum(all_labels == 1)}")
    print("=" * 60)

    print("\nTo train the classifier, run:")
    print("  python -m models.train_image_aigc_detector train \\")
    print(f"      --embeddings_dir {output_dir}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train the AIGC detector from pre-extracted embeddings."""
    print("=" * 60)
    print("Training AIGC Content Detector")
    print("=" * 60)

    device = args.device if args.device != "auto" else get_auto_device()
    print(f"Using device: {device}")

    # Load embeddings
    print("\nLoading embeddings...")
    embeddings_dir = Path(args.embeddings_dir)
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    labels = np.load(embeddings_dir / "labels.npy")

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Class distribution: Real={np.sum(labels == 0)}, AI={np.sum(labels == 1)}")

    # Create config
    config = AIGCDetectorConfig(
        embedding_dim=embeddings.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        dropout_rate=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_focal_loss=args.focal_loss,
        early_stopping_patience=args.early_stopping,
    )

    # Create dataset
    full_dataset = AIGCDataset(embeddings=embeddings, labels=labels)

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\nDataset split:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = AIGCDetectorTrainer(config=config, device=device)

    # Train
    print(f"\nTraining for {config.num_epochs} epochs...")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  MLP hidden dims: {config.hidden_dims}")
    print(f"  Loss: {'Focal Loss' if config.use_focal_loss else 'BCE'}")
    print("-" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        verbose=True,
    )

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Validation Results:")
    val_metrics = trainer._validate(val_loader)
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")

    # Save model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path), save_full=args.save_full)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Model saved to: {output_path}")
    print("=" * 60)


def cmd_train_from_images(args: argparse.Namespace) -> None:
    """Train AIGC detector directly from images (extracts embeddings on-the-fly)."""
    print("=" * 60)
    print("Training AIGC Detector from Images")
    print("=" * 60)
    print("Note: This extracts embeddings on-the-fly. For faster iteration,")
    print("use 'extract_embeddings' first, then 'train' on embeddings.")
    print("=" * 60)

    device = args.device if args.device != "auto" else get_auto_device()

    # Initialize backbone
    print(f"\nLoading SigLIP2 model: {args.siglip_model}...")
    backbone = SigLIP2Backbone(
        model_name=args.siglip_model,
        device=device,
        use_fp16=args.fp16,
    )

    # Collect images
    print("\nCollecting images...")
    real_images, _ = collect_real_images_from_directory(
        args.real_images_dir,
        max_images=args.max_real,
        target_size=(args.image_size, args.image_size),
    )
    ai_images, _ = collect_real_images_from_directory(
        args.ai_images_dir,
        max_images=args.max_ai,
        target_size=(args.image_size, args.image_size),
    )

    # Extract embeddings
    print("\nExtracting embeddings...")
    real_embeddings = extract_embeddings_batch(real_images, backbone, batch_size=args.batch_size)
    ai_embeddings = extract_embeddings_batch(ai_images, backbone, batch_size=args.batch_size)

    # Combine
    embeddings = np.vstack([real_embeddings, ai_embeddings])
    labels = np.array([0] * len(real_embeddings) + [1] * len(ai_embeddings), dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(labels))
    embeddings = embeddings[indices]
    labels = labels[indices]

    # Continue with training (reuse train logic)
    args.embeddings_dir = None
    config = AIGCDetectorConfig(
        embedding_dim=embeddings.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        dropout_rate=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
    )

    full_dataset = AIGCDataset(embeddings=embeddings, labels=labels)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    trainer = AIGCDetectorTrainer(config=config, device=device)
    trainer.train(train_loader=train_loader, val_loader=val_loader, verbose=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path))

    print(f"\nModel saved to: {output_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained AIGC detector."""
    print("=" * 60)
    print("Evaluating AIGC Content Detector")
    print("=" * 60)

    device = args.device if args.device != "auto" else get_auto_device()

    # Load embeddings
    embeddings_dir = Path(args.embeddings_dir)
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    labels = np.load(embeddings_dir / "labels.npy")

    print(f"Test set: {len(labels)} samples")
    print(f"  Real: {np.sum(labels == 0)}, AI: {np.sum(labels == 1)}")

    # Load model
    config = AIGCDetectorConfig(embedding_dim=embeddings.shape[1])
    trainer = AIGCDetectorTrainer(config=config, device=device)
    trainer.load(args.model_path)

    # Predict
    preds, probs = trainer.predict(embeddings, threshold=args.threshold)

    # Compute metrics
    accuracy = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f"  True Negatives (Real→Real): {tn}")
    print(f"  False Positives (Real→AI): {fp}")
    print(f"  False Negatives (AI→Real): {fn}")
    print(f"  True Positives (AI→AI): {tp}")
    print("=" * 60)


def cmd_hard_mining(args: argparse.Namespace) -> None:
    """Identify hard samples for retraining."""
    print("=" * 60)
    print("Hard Negative Mining for AIGC Detector")
    print("=" * 60)

    device = args.device if args.device != "auto" else get_auto_device()

    # Load embeddings
    embeddings_dir = Path(args.embeddings_dir)
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    labels = np.load(embeddings_dir / "labels.npy")

    # Load model
    config = AIGCDetectorConfig(embedding_dim=embeddings.shape[1])
    trainer = AIGCDetectorTrainer(config=config, device=device)
    trainer.load(args.model_path)

    # Find hard samples
    hard_embeddings, hard_labels = trainer.get_hard_negatives(
        embeddings,
        labels,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
    )

    # Save hard samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "hard_embeddings.npy", hard_embeddings)
    np.save(output_dir / "hard_labels.npy", hard_labels)

    print(f"\nHard samples saved to: {output_dir}")
    print(f"  Embeddings: {hard_embeddings.shape}")
    print(f"  Real: {np.sum(hard_labels == 0)}, AI: {np.sum(hard_labels == 1)}")


def cmd_list_datasets(args: argparse.Namespace) -> None:
    """List recommended datasets for training."""
    print("=" * 60)
    print("Recommended Datasets for AIGC Detection Training")
    print("=" * 60)

    print("\n--- Real Image Datasets ---")
    for key, info in RECOMMENDED_REAL_DATASETS.items():
        print(f"\n  {key}")
        print(f"    Dataset: {info['name']}")
        print(f"    Description: {info['description']}")

    print("\n--- AI-Generated Datasets ---")
    for key, info in RECOMMENDED_AI_DATASETS.items():
        print(f"\n  {key}")
        print(f"    Dataset: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Generator: {info.get('generator', 'unknown')}")

    print("\n" + "=" * 60)
    print("Usage Example:")
    print("  python -m models.train_image_aigc_detector extract_embeddings \\")
    print("      --real_dataset zh-plus/tiny-imagenet \\")
    print("      --ai_dataset poloclub/diffusiondb \\")
    print("      --output_dir ./data/aigc_embeddings")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train AIGC content detector with SigLIP2 + MLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract embeddings subcommand
    extract_parser = subparsers.add_parser(
        "extract_embeddings",
        help="Extract SigLIP2 embeddings from images",
    )
    extract_parser.add_argument("--real_images_dir", type=str, help="Directory with real images")
    extract_parser.add_argument("--ai_images_dir", type=str, help="Directory with AI-generated images")
    extract_parser.add_argument("--real_dataset", type=str, help="HuggingFace dataset for real images")
    extract_parser.add_argument("--ai_dataset", type=str, help="HuggingFace dataset for AI images")
    extract_parser.add_argument("--real_image_column", type=str, default="image")
    extract_parser.add_argument("--ai_image_column", type=str, default="image")
    extract_parser.add_argument("--output_dir", type=str, required=True)
    extract_parser.add_argument("--max_real", type=int, default=10000)
    extract_parser.add_argument("--max_ai", type=int, default=10000)
    extract_parser.add_argument("--image_size", type=int, default=384)
    extract_parser.add_argument("--batch_size", type=int, default=32)
    extract_parser.add_argument("--siglip_model", type=str, default="google/siglip2-so400m-patch14-384")
    extract_parser.add_argument("--device", type=str, default="auto")
    extract_parser.add_argument("--fp16", action="store_true", default=True)

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train classifier from pre-extracted embeddings",
    )
    train_parser.add_argument("--embeddings_dir", type=str, required=True)
    train_parser.add_argument("--output_path", type=str, default="./models/aigc_detector/aigc_classifier.pth")
    train_parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 128])
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--val_split", type=float, default=0.1)
    train_parser.add_argument("--focal_loss", action="store_true", default=True)
    train_parser.add_argument("--early_stopping", type=int, default=5)
    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--save_full", action="store_true")
    train_parser.add_argument("--device", type=str, default="auto")

    # Train from images subcommand
    train_img_parser = subparsers.add_parser(
        "train_from_images",
        help="Train directly from images (extracts embeddings on-the-fly)",
    )
    train_img_parser.add_argument("--real_images_dir", type=str, required=True)
    train_img_parser.add_argument("--ai_images_dir", type=str, required=True)
    train_img_parser.add_argument("--output_path", type=str, default="./models/aigc_detector/aigc_classifier.pth")
    train_img_parser.add_argument("--max_real", type=int, default=10000)
    train_img_parser.add_argument("--max_ai", type=int, default=10000)
    train_img_parser.add_argument("--image_size", type=int, default=384)
    train_img_parser.add_argument("--siglip_model", type=str, default="google/siglip2-so400m-patch14-384")
    train_img_parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 128])
    train_img_parser.add_argument("--dropout", type=float, default=0.3)
    train_img_parser.add_argument("--lr", type=float, default=1e-4)
    train_img_parser.add_argument("--batch_size", type=int, default=32)
    train_img_parser.add_argument("--epochs", type=int, default=30)
    train_img_parser.add_argument("--val_split", type=float, default=0.1)
    train_img_parser.add_argument("--early_stopping", type=int, default=5)
    train_img_parser.add_argument("--seed", type=int, default=42)
    train_img_parser.add_argument("--device", type=str, default="auto")
    train_img_parser.add_argument("--fp16", action="store_true", default=True)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.add_argument("--embeddings_dir", type=str, required=True)
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument("--device", type=str, default="auto")

    # Hard mining subcommand
    hard_parser = subparsers.add_parser("hard_mining", help="Find hard samples for retraining")
    hard_parser.add_argument("--model_path", type=str, required=True)
    hard_parser.add_argument("--embeddings_dir", type=str, required=True)
    hard_parser.add_argument("--output_dir", type=str, required=True)
    hard_parser.add_argument("--threshold_low", type=float, default=0.3)
    hard_parser.add_argument("--threshold_high", type=float, default=0.7)
    hard_parser.add_argument("--device", type=str, default="auto")

    # List datasets subcommand
    subparsers.add_parser("list_datasets", help="List recommended datasets")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract_embeddings":
        cmd_extract_embeddings(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "train_from_images":
        cmd_train_from_images(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "hard_mining":
        cmd_hard_mining(args)
    elif args.command == "list_datasets":
        cmd_list_datasets(args)


if __name__ == "__main__":
    main()
