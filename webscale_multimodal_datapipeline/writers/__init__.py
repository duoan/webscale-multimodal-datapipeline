"""
Data Writers package.

This package contains all data writer implementations.
Writers are automatically registered when this package is imported.
"""

from webscale_multimodal_datapipeline.framework import DataWriterRegistry

from .iceberg_writer import IcebergDataWriter
from .parquet_writer import ParquetDataWriter

# Register all writers with the framework
DataWriterRegistry.register("ParquetDataWriter", ParquetDataWriter)
DataWriterRegistry.register("IcebergDataWriter", IcebergDataWriter)

__all__ = [
    "ParquetDataWriter",
    "IcebergDataWriter",
]
