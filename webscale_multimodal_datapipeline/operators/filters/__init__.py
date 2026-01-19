"""
Filter operators package.

This package contains filter implementations.
Filters are automatically registered when this package is imported.
"""

from webscale_multimodal_datapipeline.framework import OperatorRegistry

from .image_quality_filter import ImageQualityFilter

# Register all filters with the framework
OperatorRegistry.register("ImageQualityFilter", ImageQualityFilter)

__all__ = [
    "ImageQualityFilter",
]
