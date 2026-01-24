"""
Metrics: Performance monitoring and analytics for pipeline execution

Provides three-level metrics collection (run/stage/operator) with automatic
instrumentation via context managers and Parquet output for Superset visualization.
"""

from .aggregator import MetricsAggregator
from .collector import MetricsCollector
from .models import OperatorMetrics, RunMetrics, StageMetrics
from .writer import MetricsWriter

__all__ = [
    "MetricsCollector",
    "MetricsWriter",
    "MetricsAggregator",
    "RunMetrics",
    "StageMetrics",
    "OperatorMetrics",
]
