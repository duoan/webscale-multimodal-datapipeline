"""
Quality Assessment Models

Models for assessing image quality and visual degradations based on the Z-Image paper.

This module provides:
- Multi-head neural network for scoring degradation factors
- Training infrastructure with multi-task learning
- Synthetic data generation for training
- Inference pipeline for batch processing

Degradation factors assessed:
- Color cast: Abnormal color tints
- Blurriness: Lack of sharpness/focus
- Watermark: Visible watermarks
- Noise: Visual noise levels

Reference: Z-Image Technical Report (Section 2.1 - Data Profiling Engine)
"""

from .inference import (
    MultiHeadQualityInference,
    QualityAssessmentInference,
    get_auto_device,
    load_quality_model,
)
from .synthetic_data import (
    RECOMMENDED_DATASETS,
    DegradationConfig,
    DegradationLevel,
    SyntheticDegradationGenerator,
    create_training_data_from_directory,
    create_training_data_from_huggingface,
)
from .trainer import (
    DegradationScores,
    DegradationType,
    ImageQualityDataset,
    MultiHeadLabels,
    MultiHeadQualityAssessmentModel,
    MultiHeadQualityDataset,
    MultiHeadQualityTrainer,
    QualityAssessmentModel,
    QualityAssessmentTrainer,
)

__all__ = [
    # Models
    "QualityAssessmentModel",
    "MultiHeadQualityAssessmentModel",
    # Trainers
    "QualityAssessmentTrainer",
    "MultiHeadQualityTrainer",
    # Inference
    "QualityAssessmentInference",
    "MultiHeadQualityInference",
    "load_quality_model",
    "get_auto_device",
    # Data
    "ImageQualityDataset",
    "MultiHeadQualityDataset",
    "MultiHeadLabels",
    "DegradationScores",
    "DegradationType",
    # Synthetic data
    "DegradationConfig",
    "DegradationLevel",
    "SyntheticDegradationGenerator",
    "create_training_data_from_directory",
    "create_training_data_from_huggingface",
    "RECOMMENDED_DATASETS",
]
