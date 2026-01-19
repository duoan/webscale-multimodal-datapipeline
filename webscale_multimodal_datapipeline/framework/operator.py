"""
Operator: Abstract interface for record processing

Defines the Operator, Refiner, Filter, and Deduplicator base classes.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa

from .backend import DedupBackend


class Operator(ABC):
    """Base class for all operators (batch processing with built-in stats)."""

    def __init__(self):
        self._stats = {
            "total_records": 0,
            "total_time": 0.0,
            "min_latency": float("inf"),
            "max_latency": 0.0,
            "_latencies": [],
        }

    def process_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        """Process a batch of records with automatic stats collection."""
        start_time = time.perf_counter()
        results = self._process_batch_impl(records)
        batch_latency = time.perf_counter() - start_time

        num_records = len(records)
        if num_records > 0:
            self._stats["total_records"] += num_records
            self._stats["total_time"] += batch_latency
            per_record_latency = batch_latency / num_records
            self._stats["min_latency"] = min(self._stats["min_latency"], per_record_latency)
            self._stats["max_latency"] = max(self._stats["max_latency"], per_record_latency)
            self._stats["_latencies"].extend([per_record_latency] * num_records)
            if len(self._stats["_latencies"]) > 10000:
                self._stats["_latencies"] = self._stats["_latencies"][-10000:]

        return results

    @abstractmethod
    def _process_batch_impl(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        """Internal batch processing implementation (subclasses implement this)."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics for this operator.

        Returns:
            Dictionary with performance metrics:
            - total_records: Total number of records processed
            - total_time: Total processing time (seconds)
            - avg_latency: Average latency per record (seconds)
            - min_latency: Minimum latency (seconds)
            - max_latency: Maximum latency (seconds)
            - p50_latency: 50th percentile latency (seconds)
            - p95_latency: 95th percentile latency (seconds)
            - p99_latency: 99th percentile latency (seconds)
            - throughput: Records per second
        """
        stats = self._stats.copy()
        total_records = stats["total_records"]

        if total_records == 0:
            return {
                "total_records": 0,
                "total_time": 0.0,
                "avg_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "throughput": 0.0,
            }

        avg_latency = stats["total_time"] / total_records
        min_latency = stats["min_latency"] if stats["min_latency"] != float("inf") else 0.0
        max_latency = stats["max_latency"]

        # Calculate percentiles
        latencies = stats["_latencies"]
        if latencies:
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        else:
            p50 = p95 = p99 = 0.0

        throughput = total_records / stats["total_time"] if stats["total_time"] > 0 else 0.0

        return {
            "total_records": total_records,
            "total_time": stats["total_time"],
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "throughput": throughput,
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self._stats = {
            "total_records": 0,
            "total_time": 0.0,
            "min_latency": float("inf"),
            "max_latency": 0.0,
            "_latencies": [],
            "start_time": None,
        }

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return the output schema for this operator.

        Returns:
            Dictionary mapping field names to Arrow data types
        """
        return {}


class Refiner(Operator):
    """Refiner operators enrich records by adding new information (inplace)."""

    @abstractmethod
    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Refine a batch of records inplace."""
        pass

    def _process_batch_impl(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if not records:
            return []
        self.refine_batch(records)
        return records

    @abstractmethod
    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner."""
        pass


class Filter(Operator):
    """Filter operators determine whether records should be kept or filtered out."""

    @abstractmethod
    def should_keep_batch(self, records: list[dict[str, Any]]) -> list[bool]:
        """Determine which records should be kept (True = keep, False = filter)."""
        pass

    def _process_batch_impl(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if not records:
            return []
        keep_flags = self.should_keep_batch(records)
        return [record if keep else None for record, keep in zip(records, keep_flags, strict=False)]

    def get_output_schema(self) -> dict[str, pa.DataType]:
        return {}


class Deduplicator(Operator):
    """Deduplicator operators remove duplicate records.

    Deduplicators use a configurable backend to track seen records and filter duplicates.
    Examples:
    - Perceptual hash deduplication (PhashDeduplicator)
    - Exact match deduplication
    - Semantic deduplication (cluster-based, bucket_id = cluster_id)

    For semantic deduplication, the dedup_key can encode cluster_id information,
    and a custom bucket_id_getter can extract it for routing.
    """

    def __init__(self, backend: DedupBackend | None = None):
        """Initialize deduplication operator.

        Args:
            backend: Deduplication backend (should be provided by Executor, can be None initially)
        """
        super().__init__()
        self.backend = backend

    @abstractmethod
    def get_dedup_keys_batch(self, records: list[dict[str, Any]]) -> list[str]:
        """Extract deduplication keys from a batch of records."""
        pass

    def _process_batch_impl(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if self.backend is None:
            raise RuntimeError("Deduplicator backend not set.")
        if not records:
            return []
        keys = self.get_dedup_keys_batch(records)
        is_new = self.backend.batch_mark_seen(keys)
        return [record if new else None for record, new in zip(records, is_new, strict=False)]

    def get_output_schema(self) -> dict[str, pa.DataType]:
        return {}

    def reset(self):
        """Reset deduplication state."""
        self.backend.reset()


class CombinedOperator(Operator):
    """Combines multiple operators into one (batch processing only)."""

    def __init__(self, operators: list[Operator]):
        super().__init__()
        self.operators = operators

    def _process_batch_impl(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if not records:
            return []

        current_batch = records
        for operator in self.operators:
            current_batch = operator.process_batch(current_batch)
            current_batch = [r for r in current_batch if r is not None]
            if not current_batch:
                break

        return current_batch

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics from all operators."""
        return {op.__class__.__name__: op.get_stats() for op in self.operators}

    def get_output_schema(self) -> dict[str, pa.DataType]:
        combined = {}
        for op in self.operators:
            if isinstance(op, Refiner):
                combined.update(op.get_output_schema())
        return combined
