"""
Data Loaders package.

This package contains all data loader implementations.
Loaders are automatically registered when this package is imported.
"""

from webscale_multimodal_datapipeline.framework import DataLoaderRegistry

from .huggingface_loader import HuggingFaceDataLoader

# Register all loaders with the framework
DataLoaderRegistry.register("HuggingFaceDataLoader", HuggingFaceDataLoader)

__all__ = [
    "HuggingFaceDataLoader",
]
