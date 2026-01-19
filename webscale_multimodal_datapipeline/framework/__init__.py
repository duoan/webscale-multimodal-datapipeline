"""
Pipeline Framework: Configuration-Driven Distributed Processing Framework

This package provides a flexible framework for building data processing pipelines.
All public APIs are exported from this module.
"""

# Config classes
# Backend classes
from .backend import (
    DedupBackend,
)

# Base classes
from .base import (
    DataLoader,
    DataWriter,
)
from .config import (
    DataLoaderConfig,
    DataWriterConfig,
    ExecutorConfig,
    OperatorConfig,
    PipelineConfig,
    StageConfig,
    StageWorkerConfig,
)

# Executor
from .executor import (
    Executor,
)

# Operator classes
from .operator import (
    CombinedOperator,
    Deduplicator,
    Filter,
    Operator,
    Refiner,
)

# Registry classes
from .registry import (
    DataLoaderRegistry,
    DataWriterRegistry,
    OperatorRegistry,
)

# Worker classes
from .worker import (
    RayWorker,
    Worker,
)

# Export all public APIs
__all__ = [
    # Config
    "OperatorConfig",
    "StageWorkerConfig",
    "StageConfig",
    "DataLoaderConfig",
    "DataWriterConfig",
    "ExecutorConfig",
    "PipelineConfig",
    # Operator
    "Operator",
    "Refiner",
    "Filter",
    "Deduplicator",
    "CombinedOperator",
    # Backend
    "DedupBackend",
    # Registry
    "OperatorRegistry",
    "DataLoaderRegistry",
    "DataWriterRegistry",
    # Base
    "DataLoader",
    "DataWriter",
    # Worker
    "Worker",
    "RayWorker",
    # Executor
    "Executor",
]
