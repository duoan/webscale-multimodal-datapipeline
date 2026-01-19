"""
Deduplication operators package.

This package contains deduplication implementations.
Dedups are automatically registered when this package is imported.
"""

from webscale_multimodal_datapipeline.framework import OperatorRegistry

from .image_phash_dedup import ImagePhashDeduplicator

# Register all dedup operators with the framework
OperatorRegistry.register("ImagePhashDeduplicator", ImagePhashDeduplicator)

__all__ = [
    "ImagePhashDeduplicator",
]
