"""
Base: Abstract base classes for data loaders and writers

Provides abstract interfaces for DataLoader and DataWriter.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class DataLoader(ABC):
    """Abstract base class for data loaders.

    File-based loaders implement:
    - get_file_list() -> list[str]: Scan and return all data files
    - load_files(file_list, ...) -> Iterator: Load assigned files
    """

    def create_checkpoint(self, shard_id: int, records_processed: int) -> dict[str, Any]:
        """Create checkpoint data for resume support.

        Args:
            shard_id: Shard ID being processed
            records_processed: Number of records processed so far in this shard

        Returns:
            Checkpoint data dictionary
        """
        return {
            "shard_id": shard_id,
            "records_processed": records_processed,
        }


class DataWriter(ABC):
    """Abstract base class for data writers.

    Writers that need cleanup should implement a close() method.
    """

    @abstractmethod
    def write(self, data: list[dict[str, Any]]):
        """Write a batch of processed data.

        Args:
            data: List of processed records to write
        """
