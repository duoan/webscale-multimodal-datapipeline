"""
AIGC Content Detector Module

Implements a binary classifier to detect AI-generated images vs real photos.
Uses frozen SigLIP2 Vision Encoder with a lightweight MLP head.

Based on Imagen 3 findings: AIGC content filtering is crucial for preventing
degradation in model output quality and physical realism.

Reference: Imagen 3 Paper (Section on Data Filtering)
"""

from .synthetic_data import (
    AIGCDatasetConfig,
    collect_real_images_from_directory,
    collect_real_images_from_huggingface,
    create_training_dataset,
    get_augmentation_transforms,
)
from .trainer import (
    AIGCClassifierHead,
    AIGCDataset,
    AIGCDetectorTrainer,
    SigLIP2Backbone,
    get_auto_device,
)

__all__ = [
    # Model components
    "SigLIP2Backbone",
    "AIGCClassifierHead",
    # Trainer
    "AIGCDetectorTrainer",
    # Dataset
    "AIGCDataset",
    "AIGCDatasetConfig",
    # Data utilities
    "create_training_dataset",
    "collect_real_images_from_directory",
    "collect_real_images_from_huggingface",
    "get_augmentation_transforms",
    # Utils
    "get_auto_device",
]
