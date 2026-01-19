"""
Worker: Executes operators on records

Provides Worker and RayWorker for executing operators in batch processing.
"""

import logging
from typing import Any

import ray

from .base import DataWriter
from .operator import CombinedOperator, Operator


class Worker:
    """Worker executes operators on records and writes results locally.

    This is the base Worker class for non-Ray execution.
    For Ray execution, use RayWorker (a Ray Actor).
    """

    def __init__(
        self, name: str, operators: list[Operator], data_writer: DataWriter | None = None, num_cpus: int | None = None
    ):
        """Initialize worker.

        Args:
            name: Worker name
            operators: List of operators to run
            data_writer: Data writer for writing results locally (optional)
            num_cpus: Number of CPUs available to this worker
        """
        self.name = name
        self.operators = operators
        self.data_writer = data_writer

        # Set up logging
        self.logger = logging.getLogger(f"Worker.{name}")
        self.logger.setLevel(logging.INFO)

        # Initialize batch counters for progress tracking
        self.batch_count = 0
        self.record_count = 0
        self.processed_count = 0

        # Combine operators if multiple
        if len(operators) == 1:
            self.operator = operators[0]
        else:
            self.operator = CombinedOperator(operators)

    def process(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single record with all operators.

        Args:
            record: Input record

        Returns:
            Processed record, or None if filtered out
        """
        # Use _process_with_stats if available to collect performance metrics
        if hasattr(self.operator, "_process_with_stats"):
            return self.operator._process_with_stats(record)
        return self.operator.process(record)

    def process_batch(self, records: list[dict[str, Any]]) -> int:
        """Process a batch of records and write results locally.

        Args:
            records: List of input records

        Returns:
            Number of processed records that were written
        """
        if not records:
            return 0

        # Process records sequentially
        results = [self.process(record) for record in records]
        # Filter out None values (from filters/dedups)
        processed = [r for r in results if r is not None]

        # Update counters
        self.batch_count += 1
        self.record_count += len(records)
        self.processed_count += len(processed)

        # Write results locally if writer is available
        if self.data_writer and processed:
            self.data_writer.write(processed)

        # Log progress every 10 batches or on first batch
        if self.batch_count % 10 == 1 or self.batch_count == 1:
            self.logger.info(
                f"Progress: {self.batch_count} batches, "
                f"{self.record_count} records processed, "
                f"{self.processed_count} records written "
                f"({self.processed_count}/{self.record_count}={100 * self.processed_count / max(1, self.record_count):.1f}% pass rate)"
            )

        return len(processed)

    def process_batch_no_write(self, records: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Process a batch of records, write to Iceberg, and return processed records.

        Used for multi-worker-type pipeline where each worker writes data
        and passes records to the next worker type.

        Args:
            records: List of input records

        Returns:
            List of processed records (None values filtered out), or None if all filtered
        """
        if not records:
            return []

        # Process records sequentially
        results = [self.process(record) for record in records]
        # Filter out None values (from filters/dedups)
        processed = [r for r in results if r is not None]

        # Update counters
        self.batch_count += 1
        self.record_count += len(records)
        self.processed_count += len(processed)

        # Write results to Iceberg/Parquet (supports concurrent writes)
        if self.data_writer and processed:
            self.data_writer.write(processed)

        # Log progress every 10 batches or on first batch
        if self.batch_count % 10 == 1 or self.batch_count == 1:
            self.logger.info(
                f"Progress: {self.batch_count} batches, "
                f"{self.record_count} records processed, "
                f"{self.processed_count} records passed "
                f"({self.processed_count}/{self.record_count}={100 * self.processed_count / max(1, self.record_count):.1f}% pass rate)"
            )

        return processed if processed else None

    def get_operator_stats(self) -> dict[str, Any]:
        """Get performance statistics from all operators in this worker.

        Returns:
            Dictionary mapping operator class names to their statistics
        """
        stats = {}
        # All operators have get_stats from Operator base class
        if hasattr(self.operator, "operators"):
            # CombinedOperator - get stats from individual operators
            for op in self.operator.operators:
                op_name = op.__class__.__name__
                stats[op_name] = op.get_stats()
        else:
            # Single operator
            op_name = self.operator.__class__.__name__
            stats[op_name] = self.operator.get_stats()
        return stats


@ray.remote
class RayWorker:
    """Ray Actor for distributed batch processing.

    Each RayWorker processes batches of records on a Ray node.
    Workers are created as Ray Actors and process batches remotely.
    """

    def __init__(
        self, name: str, operators: list[Operator], data_writer: DataWriter | None = None, num_cpus: int | None = None
    ):
        """Initialize Ray worker.

        Args:
            name: Worker name
            operators: List of operators to run
            data_writer: Data writer for writing results locally (optional)
            num_cpus: Number of CPUs available to this worker
        """
        self.name = name
        self.operators = operators
        self.data_writer = data_writer

        # Set up logging (Ray logs appear in Ray Dashboard)
        self.logger = logging.getLogger(f"RayWorker.{name}")
        self.logger.setLevel(logging.INFO)

        # Initialize batch counters for progress tracking
        self.batch_count = 0
        self.record_count = 0
        self.processed_count = 0

        # Combine operators if multiple
        if len(operators) == 1:
            self.operator = operators[0]
        else:
            self.operator = CombinedOperator(operators)

    def process_batch(self, records: list[dict[str, Any]]) -> int:
        """Process a batch of records and write results locally on Ray node.

        Args:
            records: List of input records

        Returns:
            Number of processed records that were written
        """
        if not records:
            return 0

        # Process batch using process_batch if available (better performance for batch operations)
        # Use _process_batch_with_stats to collect performance statistics
        if hasattr(self.operator, "_process_batch_with_stats"):
            results = self.operator._process_batch_with_stats(records)
        elif hasattr(self.operator, "process_batch"):
            # Operator has process_batch but no stats wrapper, use it directly
            results = self.operator.process_batch(records)
        else:
            # Fallback to per-record processing with statistics
            if hasattr(self.operator, "_process_with_stats"):
                results = [self.operator._process_with_stats(record) for record in records]
            else:
                results = [self.operator.process(record) for record in records]

        # Filter out None values (from filters/dedups)
        processed = [r for r in results if r is not None]

        # Update counters
        self.batch_count += 1
        self.record_count += len(records)
        self.processed_count += len(processed)

        # Write results locally if writer is available (on Ray node)
        if self.data_writer and processed:
            self.data_writer.write(processed)

        # Log progress every 10 batches or on first batch (visible in Ray Dashboard)
        if self.batch_count % 10 == 1 or self.batch_count == 1:
            self.logger.info(
                f"Progress: {self.batch_count} batches, "
                f"{self.record_count} records processed, "
                f"{self.processed_count} records written "
                f"({self.processed_count}/{self.record_count}={100 * self.processed_count / max(1, self.record_count):.1f}% pass rate)"
            )

        return len(processed)

    def process_batch_with_records(self, records_or_ref, should_write: bool = False) -> list[dict[str, Any]] | None:
        """Process a batch of records and optionally write to Iceberg.

        Used for multi-stage pipeline where only the last stage writes data.
        Supports both direct records and Ray ObjectRef for chaining.

        Args:
            records_or_ref: List of input records or Ray ObjectRef to records
            should_write: If True, write results to Iceberg. Only last stage should set this to True.

        Returns:
            List of processed records (None values filtered out), or None if all filtered
        """
        self.logger.info(
            f"process_batch_with_records called (should_write={should_write}, type={type(records_or_ref)})"
        )

        # Handle Ray ObjectRef (for chaining) - Ray will automatically resolve it
        if isinstance(records_or_ref, ray.ObjectRef):
            self.logger.info("Resolving ObjectRef...")
            records = ray.get(records_or_ref)
            self.logger.info(f"ObjectRef resolved, got {len(records) if records else 0} records")
        else:
            records = records_or_ref

        # Handle None/empty records (upstream may have filtered everything)
        if not records:
            self.logger.info("Empty records, returning []")
            return []

        # Process batch using process_batch if available (better performance for batch operations)
        # This is especially beneficial for Deduplicator which can batch remote calls
        # Use _process_batch_with_stats to collect performance statistics
        if hasattr(self.operator, "_process_batch_with_stats"):
            results = self.operator._process_batch_with_stats(records)
        elif hasattr(self.operator, "process_batch"):
            # Operator has process_batch but no stats wrapper, use it directly
            results = self.operator.process_batch(records)
        else:
            # Fallback to per-record processing with statistics
            if hasattr(self.operator, "_process_with_stats"):
                results = [self.operator._process_with_stats(record) for record in records]
            else:
                results = [self.operator.process(record) for record in records]

        # Filter out None values (from filters/dedups)
        processed = [r for r in results if r is not None]

        # Update counters
        self.batch_count += 1
        self.record_count += len(records)
        self.processed_count += len(processed)

        # Only write if this is the last stage (better I/O efficiency)
        if should_write and self.data_writer and processed:
            self.data_writer.write(processed)

        # Log progress every 10 batches or on first batch (visible in Ray Dashboard)
        if self.batch_count % 10 == 1 or self.batch_count == 1:
            self.logger.info(
                f"Progress: {self.batch_count} batches, "
                f"{self.record_count} records processed, "
                f"{self.processed_count} records passed "
                f"({self.processed_count}/{self.record_count}={100 * self.processed_count / max(1, self.record_count):.1f}% pass rate)"
            )

        return processed if processed else None

    def get_operator_stats(self) -> dict[str, Any]:
        """Get performance statistics from all operators in this worker.

        Returns:
            Dictionary mapping operator class names to their statistics
        """
        stats = {}
        # All operators have get_stats from Operator base class
        if hasattr(self.operator, "operators"):
            # CombinedOperator - get stats from individual operators
            for op in self.operator.operators:
                op_name = op.__class__.__name__
                stats[op_name] = op.get_stats()
        else:
            # Single operator
            op_name = self.operator.__class__.__name__
            stats[op_name] = self.operator.get_stats()
        return stats
