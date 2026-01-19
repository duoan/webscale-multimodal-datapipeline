"""
Refiner operators package.

This package contains refiner implementations.
Refiners enrich records by adding new information.
Refiners are automatically registered when this package is imported.
"""

from webscale_multimodal_datapipeline.framework import OperatorRegistry

from .image_aesthetic_quality import ImageAestheticQualityRefiner
from .image_aigc_detector import ImageAIGCDetectorRefiner
from .image_clip_embedding import ImageClipEmbeddingRefiner
from .image_metadata import ImageMetadataRefiner
from .image_siglip_embedding import ImageSigLIPEmbeddingRefiner
from .image_technical_quality import ImageTechnicalQualityRefiner
from .image_visual_degradations import ImageVisualDegradationsRefiner

# Register all refiners with the framework
OperatorRegistry.register("ImageMetadataRefiner", ImageMetadataRefiner)
OperatorRegistry.register("ImageTechnicalQualityRefiner", ImageTechnicalQualityRefiner)  # Auto-uses Rust if available
OperatorRegistry.register("ImageVisualDegradationsRefiner", ImageVisualDegradationsRefiner)
OperatorRegistry.register("ImageClipEmbeddingRefiner", ImageClipEmbeddingRefiner)
OperatorRegistry.register("ImageSigLIPEmbeddingRefiner", ImageSigLIPEmbeddingRefiner)
OperatorRegistry.register("ImageAestheticQualityRefiner", ImageAestheticQualityRefiner)
OperatorRegistry.register("ImageAIGCDetectorRefiner", ImageAIGCDetectorRefiner)

__all__ = [
    "ImageMetadataRefiner",
    "ImageTechnicalQualityRefiner",
    "ImageVisualDegradationsRefiner",
    "ImageClipEmbeddingRefiner",
    "ImageSigLIPEmbeddingRefiner",
    "ImageAestheticQualityRefiner",
    "ImageAIGCDetectorRefiner",
]
