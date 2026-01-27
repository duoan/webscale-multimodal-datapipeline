"""Deduplication operators package - lazy loaded to avoid heavy dependencies."""

from mega_data_factory.framework import OperatorRegistry

# Text dedup (no heavy dependencies)
from .text_exact_dedup import TextExactDeduplicator

OperatorRegistry.register("TextExactDeduplicator", TextExactDeduplicator)


def _register_image_dedup():
    """Lazy register image dedup that depends on PIL."""
    from .image_phash_dedup import ImagePhashDeduplicator

    OperatorRegistry.register("ImagePhashDeduplicator", ImagePhashDeduplicator)


try:
    _register_image_dedup()
except ImportError:
    pass


__all__ = ["TextExactDeduplicator"]
