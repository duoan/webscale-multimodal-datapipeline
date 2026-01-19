"""
HuggingFace Data Loader

Loads datasets from HuggingFace Hub.
"""

from collections.abc import Iterator
from typing import Any

from datasets import Image as HFImage
from datasets import load_dataset

from webscale_multimodal_datapipeline.framework import DataLoader


class HuggingFaceDataLoader(DataLoader):
    """DataLoader for HuggingFace datasets."""

    def __init__(self, dataset_name: str, split: str = "train", streaming: bool = True):
        """Initialize HuggingFace data loader.

        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to load
            streaming: Whether to use streaming mode
        """
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming

    def load(self, **kwargs) -> Iterator[dict[str, Any]]:
        """Load dataset and return iterator.

        Args:
            **kwargs: Additional parameters for dataset loading

        Yields:
            Records as dictionaries
        """
        print(f"Loading HuggingFace dataset: {self.dataset_name} (split: {self.split}, streaming: {self.streaming})...")
        ds = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming, **kwargs)
        print("Dataset loaded, starting to iterate records...")

        if self.streaming:
            ds = ds.cast_column("image", HFImage(decode=False))

        record_count = 0
        for record in ds:
            record_count += 1
            if record_count == 1:
                print("First record received, continuing...")
            elif record_count % 50 == 0:
                print(f"  Yielded {record_count} records from data stream...")
            yield record
