"""
Data Loaders package.

This package contains all data loader implementations.
Loaders are automatically registered when this package is imported.
"""

from mega_data_factory.framework import DataLoaderRegistry

from .commoncrawl_loader import CommonCrawlWarcStreamLoader
from .huggingface_loader import HuggingFaceLoader

# Register all loaders with the framework
DataLoaderRegistry.register("HuggingFaceLoader", HuggingFaceLoader)
DataLoaderRegistry.register("CommonCrawlWarcStreamLoader", CommonCrawlWarcStreamLoader)

# Backward compatibility
DataLoaderRegistry.register("HuggingFaceDataLoader", HuggingFaceLoader)
DataLoaderRegistry.register("HuggingFaceFileLoader", HuggingFaceLoader)

__all__ = [
    "HuggingFaceLoader",
    "CommonCrawlWarcStreamLoader",
]
