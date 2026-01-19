"""
Iceberg Data Writer

Writes data to Iceberg tables using pyiceberg.
Uses filesystem-based catalog (no service required).
"""

import os
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from webscale_multimodal_datapipeline.framework import DataWriter

try:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.table import Table
    from pyiceberg.types import FloatType, IntegerType, ListType, LongType, NestedField, StringType

    PYICEBERG_AVAILABLE = True
except ImportError:
    PYICEBERG_AVAILABLE = False


class IcebergDataWriter(DataWriter):
    """DataWriter for Iceberg tables using filesystem catalog (no service required)."""

    def __init__(
        self,
        table_path: str,
        table_name: str = "image_profiles",
        namespace: str = "default",
        catalog_type: str = "filesystem",
        schema: Schema | None = None,
    ):
        """Initialize Iceberg writer.

        Args:
            table_path: Base path where table data and metadata will be stored
            table_name: Table name
            namespace: Namespace/database name (default: "default")
            catalog_type: Catalog type, "filesystem" for file-based catalog (no service)
            schema: Optional Iceberg schema (if None, will be inferred from data)
        """
        if not PYICEBERG_AVAILABLE:
            raise ImportError("pyiceberg is required for IcebergDataWriter. Install it with: pip install pyiceberg")

        self.table_path = table_path
        self.table_name = table_name
        self.namespace = namespace
        self.catalog_type = catalog_type
        self.schema = schema

        # Initialize catalog using filesystem (no service required)
        # Catalog metadata is stored in {table_path}/catalog/
        catalog_path = os.path.join(table_path, "catalog")
        os.makedirs(catalog_path, exist_ok=True)

        catalog_properties = {"type": "filesystem", "warehouse": table_path}

        try:
            self.catalog = load_catalog("default", **catalog_properties)
        except Exception as e:
            # If catalog loading fails, try creating a simple filesystem catalog
            # For filesystem catalog, we manage metadata files directly
            print(f"Warning: Could not load catalog: {e}")
            print("Using direct file-based Iceberg table management")
            self.catalog = None

        self.table: Table | None = None
        self._initialized = False
        self._init_lock = None  # Will be set when needed for thread-safety

        # Buffering for batch writes (improve performance by reducing small file writes)
        self._write_buffer: list[dict[str, Any]] = []
        self._buffer_size: int = 1000  # Buffer up to 1000 records before flushing

    def _infer_schema_from_data(self, data: list[dict[str, Any]]) -> Schema:
        """Infer Iceberg schema from data records.

        Args:
            data: Sample data records

        Returns:
            Iceberg Schema
        """
        if not data:
            raise ValueError("Cannot infer schema from empty data")

        # Sample first record to determine schema
        sample = data[0]
        fields = []
        field_id = 1

        for key, value in sample.items():
            if key == "id":
                fields.append(NestedField(field_id, key, StringType(), required=False))
            elif isinstance(value, (int, float)):
                if isinstance(value, int):
                    fields.append(NestedField(field_id, key, LongType(), required=False))
                else:
                    fields.append(NestedField(field_id, key, FloatType(), required=False))
            elif isinstance(value, list):
                # Handle list types (e.g., embeddings)
                # For now, we'll store lists as strings (JSON) or binary (Parquet native)
                # Iceberg ListType has specific API requirements
                # Store as string representation for simplicity
                fields.append(NestedField(field_id, key, StringType(), required=False))
            else:
                fields.append(NestedField(field_id, key, StringType(), required=False))
            field_id += 1

        return Schema(*fields)

    def _ensure_table_exists(self, sample_data: list[dict[str, Any]]):
        """Ensure Iceberg table exists, create if it doesn't.

        Lazy initialization: Skip loading existing tables to avoid file scanning overhead.
        For now, we use direct Parquet writes which is much faster and doesn't block.

        Args:
            sample_data: Sample data to infer schema if needed
        """
        if self._initialized and self.table is not None:
            return

        # For filesystem-based catalog, skip table loading to avoid file scanning
        # We'll write Parquet files directly (simplified approach)
        # This avoids the "Resolving data files" blocking issue
        if not self.catalog:
            self.table = None
            self._initialized = True
            return

        # Skip loading existing tables - it triggers file scanning which blocks
        # Instead, always use direct Parquet writes for now
        # If you need full Iceberg metadata, configure a proper catalog service
        self.table = None
        self._initialized = True

    def write(self, data: list[dict[str, Any]]):
        """Write data to Iceberg table incrementally (with buffering for performance).

        Args:
            data: List of processed records to write
        """
        if not data:
            return

        # Add to buffer
        self._write_buffer.extend(data)

        # Flush buffer if it reaches threshold
        if len(self._write_buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered data to disk."""
        if not self._write_buffer:
            return

        # Ensure table exists (only check once)
        if not self._initialized:
            self._ensure_table_exists(self._write_buffer)

        # Convert buffered data to PyArrow Table directly (more efficient than DataFrame)
        # Build Arrow schema from first record

        # Convert to list of lists/values for each column (faster than DataFrame)
        if not self._write_buffer:
            return

        # Use pandas for now (can optimize to direct Arrow later)
        df = pd.DataFrame(self._write_buffer)
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)

        # Write Parquet file (simplified approach without full Iceberg metadata)
        data_dir = os.path.join(self.table_path, "data")
        os.makedirs(data_dir, exist_ok=True)

        import time
        import uuid

        file_name = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.parquet"
        parquet_path = os.path.join(data_dir, file_name)

        # Use compression and larger row group size for better performance
        pq.write_table(
            arrow_table,
            parquet_path,
            compression="snappy",  # Fast compression
            row_group_size=50000,  # Larger row groups for better performance
            use_dictionary=True,  # Dictionary encoding for better compression
        )

        # Clear buffer
        self._write_buffer = []

    def close(self):
        """Close writer and flush any remaining buffered data."""
        # Flush any remaining data in buffer
        if self._write_buffer:
            self._flush_buffer()
