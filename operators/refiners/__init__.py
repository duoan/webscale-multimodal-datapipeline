"""
Refiner operators package.

This package contains refiner implementations.
Refiners enrich records by adding new information.
Refiners are automatically registered when this package is imported.
"""

from framework import OperatorRegistry

from .image_clip_embedding import ImageClipEmbeddingRefiner
from .image_metadata import ImageMetadataRefiner
from .image_technical_quality import ImageTechnicalQualityRefiner
from .image_visual_degradations import ImageVisualDegradationsRefiner

# Register all refiners with the framework
OperatorRegistry.register("ImageMetadataRefiner", ImageMetadataRefiner)
OperatorRegistry.register("ImageTechnicalQualityRefiner", ImageTechnicalQualityRefiner)  # Auto-uses Rust if available
OperatorRegistry.register("ImageVisualDegradationsRefiner", ImageVisualDegradationsRefiner)
OperatorRegistry.register("ImageClipEmbeddingRefiner", ImageClipEmbeddingRefiner)

__all__ = [
    "ImageMetadataRefiner",
    "ImageTechnicalQualityRefiner",
    "ImageVisualDegradationsRefiner",
    "ImageClipEmbeddingRefiner",
]
