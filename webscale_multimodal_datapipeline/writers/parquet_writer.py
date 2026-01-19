"""
Parquet Data Writer

Writes data to Parquet files with incremental writes using PyArrow (fast, no pandas overhead).
"""

import os
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from webscale_multimodal_datapipeline.framework import DataWriter


class ParquetDataWriter(DataWriter):
    """DataWriter that writes to Parquet files using PyArrow."""

    def __init__(self, output_path: str, table_name: str = "profiles"):
        """Initialize Parquet writer.

        Args:
            output_path: Directory path for output files
            table_name: Name of the table/file
        """
        self.output_path = output_path
        self.table_name = table_name

    def write(self, data: list[dict[str, Any]]):
        """Write data to Parquet files using PyArrow (fast, no pandas conversion).

        Args:
            data: List of processed records to write
        """
        if not data:
            return

        os.makedirs(self.output_path, exist_ok=True)

        # Convert directly to PyArrow Table (no pandas overhead)
        arrow_table = pa.Table.from_pylist(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = os.path.join(self.output_path, f"{self.table_name}_{timestamp}.parquet")

        # Write with compression and optimized settings
        pq.write_table(
            arrow_table,
            parquet_path,
            compression="snappy",  # Fast compression
            row_group_size=50000,  # Larger row groups for better performance
            use_dictionary=True,  # Dictionary encoding for better compression
        )

    def close(self):
        """Close writer (no-op for Parquet)."""
        pass
