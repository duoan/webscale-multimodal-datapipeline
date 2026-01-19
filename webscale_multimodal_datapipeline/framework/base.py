"""
Base: Abstract base classes for data loaders and writers

Provides abstract interfaces for DataLoader and DataWriter.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, **kwargs) -> Iterator[dict[str, Any]]:
        """Load data and return an iterator over records.

        Args:
            **kwargs: Additional parameters for loading

        Yields:
            Records as dictionaries
        """
        pass


class DataWriter(ABC):
    """Abstract base class for data writers."""

    @abstractmethod
    def write(self, data: list[dict[str, Any]]):
        """Write a batch of processed data.

        Args:
            data: List of processed records to write
        """
        pass

    def close(self):
        """Close writer and flush any buffered data."""
        pass
