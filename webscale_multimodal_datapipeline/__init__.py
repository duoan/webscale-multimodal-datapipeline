"""Webscale Multimodal Data Pipeline.

High-performance, distributed web-scale multimodal data processing pipeline
with Ray, featuring Rust-accelerated and GPU-optimized operators.
"""

__version__ = "0.1.0"

from webscale_multimodal_datapipeline.framework import (
    Deduplicator,
    Executor,
    Filter,
    Operator,
    PipelineConfig,
    Refiner,
)

__all__ = [
    "__version__",
    "Deduplicator",
    "Executor",
    "Filter",
    "Operator",
    "PipelineConfig",
    "Refiner",
]
