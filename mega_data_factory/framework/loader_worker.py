"""
DataLoaderWorker: Ray Actor for distributed data loading

Enables parallel data loading across multiple workers, with each worker
loading a shard of the dataset and producing batches.
"""

from typing import Any

import ray

from .base import DataLoader


@ray.remote
class DataLoaderWorker:
    """Ray Actor for distributed data loading.

    Each worker loads a disjoint shard of the dataset and produces batches
    for downstream processing stages.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        shard_id: int,
        num_shards: int,
        batch_size: int,
        checkpoint_interval: int = 1000,
        iterator_refresh_interval: int = 10,
        assigned_files: list[str] | None = None,
        max_records: int | None = None,
    ):
        """Initialize data loader worker.

        Args:
            data_loader: DataLoader instance
            shard_id: This worker's shard ID (0 to num_shards-1)
            num_shards: Total number of shards
            batch_size: Number of records per batch
            checkpoint_interval: Save checkpoint every N records
            iterator_refresh_interval: Refresh iterator every N batches (0 = never)
            assigned_files: List of files assigned to this worker
            max_records: Maximum records to load (None = unlimited)
        """
        self.data_loader = data_loader
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.iterator_refresh_interval = iterator_refresh_interval
        self.assigned_files = assigned_files
        self.max_records = max_records

        if assigned_files is None:
            raise ValueError("assigned_files is required for file-based loading")

        self.records_processed = 0
        self.checkpoint = None
        self.batches_since_refresh = 0

        # Initialize the data stream once
        self._data_stream = None
        self._initialize_stream()

    def _initialize_stream(self):
        """Initialize or refresh the data stream iterator."""
        print(
            f"[DataLoaderWorker {self.shard_id}] Initializing data stream "
            f"(processed: {self.records_processed})"
        )

        if not hasattr(self.data_loader, "load_files"):
            raise ValueError(
                f"Loader {type(self.data_loader).__name__} does not support load_files()"
            )

        self._data_stream = self.data_loader.load_files(
            file_list=self.assigned_files,
            worker_id=self.shard_id,
            checkpoint=self.checkpoint,
        )

        self.batches_since_refresh = 0

    def get_next_batch(
        self,
        max_records: int | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get the next batch from this shard (streaming mode).

        Uses a persistent iterator that is refreshed periodically (based on
        iterator_refresh_interval) to prevent memory accumulation.

        Args:
            max_records: Optional override for maximum records (None = use self.max_records)
            **kwargs: Additional parameters (not used in streaming mode)

        Returns:
            Dictionary with:
                - 'batch': List of records (or None if completed)
                - 'records_processed': Total records processed so far
                - 'completed': Boolean indicating if loading is complete
        """
        # Use instance max_records if not overridden
        effective_max_records = max_records if max_records is not None else self.max_records

        # Check if we've reached max_records
        if effective_max_records and self.records_processed >= effective_max_records:
            return {
                "batch": None,
                "records_processed": self.records_processed,
                "completed": True,
            }

        # Refresh iterator if configured and threshold reached
        if (
            self.iterator_refresh_interval > 0
            and self.batches_since_refresh >= self.iterator_refresh_interval
        ):
            print(
                f"[DataLoaderWorker {self.shard_id}] Refreshing iterator "
                f"(batches: {self.batches_since_refresh})"
            )
            self._initialize_stream()

        batch = []
        records_in_this_batch = 0

        try:
            for record in self._data_stream:
                batch.append(record)
                self.records_processed += 1
                records_in_this_batch += 1

                # Return batch when full
                if len(batch) >= self.batch_size:
                    # Update checkpoint to current position
                    self.checkpoint = self.data_loader.create_checkpoint(
                        shard_id=self.shard_id,
                        records_processed=self.records_processed,
                    )

                    # Save checkpoint periodically
                    if self.records_processed % self.checkpoint_interval == 0:
                        self._save_checkpoint()

                    # Track batches for iterator refresh
                    self.batches_since_refresh += 1

                    return {
                        "batch": batch,
                        "records_processed": self.records_processed,
                        "completed": False,
                    }

                # Check max_records limit
                if effective_max_records and self.records_processed >= effective_max_records:
                    # Update checkpoint
                    self.checkpoint = self.data_loader.create_checkpoint(
                        shard_id=self.shard_id,
                        records_processed=self.records_processed,
                    )
                    self._save_checkpoint()

                    # Return partial batch if any
                    return {
                        "batch": batch if batch else None,
                        "records_processed": self.records_processed,
                        "completed": True,
                    }

            # Iterator exhausted - return final partial batch if any
            self.checkpoint = self.data_loader.create_checkpoint(
                shard_id=self.shard_id,
                records_processed=self.records_processed,
            )
            self._save_checkpoint()

            return {
                "batch": batch if batch else None,
                "records_processed": self.records_processed,
                "completed": True,
            }

        except StopIteration:
            # Iterator exhausted
            self.checkpoint = self.data_loader.create_checkpoint(
                shard_id=self.shard_id,
                records_processed=self.records_processed,
            )
            self._save_checkpoint()

            return {
                "batch": batch if batch else None,
                "records_processed": self.records_processed,
                "completed": True,
            }

    def _save_checkpoint(self):
        """Save checkpoint for resume support.

        Currently stores checkpoint in memory. In production, this should
        be persisted to external storage (Redis, S3, or file system).
        """
        self.checkpoint = self.data_loader.create_checkpoint(
            shard_id=self.shard_id,
            records_processed=self.records_processed,
        )
        print(
            f"[DataLoaderWorker {self.shard_id}] Checkpoint: "
            f"{self.records_processed} records processed"
        )

    def get_checkpoint(self) -> dict[str, Any]:
        """Get current checkpoint data.

        Returns:
            Checkpoint dictionary
        """
        return self.checkpoint or {}

    def restore_checkpoint(self, checkpoint: dict[str, Any]):
        """Restore from checkpoint.

        Args:
            checkpoint: Checkpoint data from previous run
        """
        self.checkpoint = checkpoint
        self.records_processed = checkpoint.get("records_processed", 0)
        print(
            f"[DataLoaderWorker {self.shard_id}] Restored checkpoint: "
            f"{self.records_processed} records"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics.

        Returns:
            Dictionary with shard_id, records_processed, and checkpoint info
        """
        return {
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "records_processed": self.records_processed,
            "has_checkpoint": self.checkpoint is not None,
        }
