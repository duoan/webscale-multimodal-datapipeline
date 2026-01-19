"""
Image Metadata Refiner

Extracts basic image metadata: width, height, file size, format.
This is a Refiner that enriches records with metadata information.
"""

from io import BytesIO
from typing import Any

import pyarrow as pa
from PIL import Image

from webscale_multimodal_datapipeline.framework import Refiner

# Field name constants
FIELD_WIDTH = "image_width"
FIELD_HEIGHT = "image_height"
FIELD_FILE_SIZE = "image_file_size_bytes"
FIELD_FORMAT = "image_format"

OUTPUT_FIELDS = [FIELD_WIDTH, FIELD_HEIGHT, FIELD_FILE_SIZE, FIELD_FORMAT]


class ImageMetadataRefiner(Refiner):
    """Refiner for extracting basic image metadata.

    Output fields:
    - image_width: Image width in pixels
    - image_height: Image height in pixels
    - image_file_size_bytes: File size in bytes
    - image_format: Image format (JPEG, PNG, etc.)
    """

    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Extract basic image metadata for a batch of records (inplace)."""
        for record in records:
            img_obj = record.get("image", {})

            if isinstance(img_obj, dict) and "bytes" in img_obj:
                image_bytes = img_obj["bytes"]
                try:
                    img = Image.open(BytesIO(image_bytes))
                    w, h = img.size
                    record[FIELD_WIDTH] = w
                    record[FIELD_HEIGHT] = h
                    record[FIELD_FILE_SIZE] = len(image_bytes)
                    record[FIELD_FORMAT] = img.format or "UNKNOWN"
                except Exception:
                    record[FIELD_WIDTH] = 0
                    record[FIELD_HEIGHT] = 0
                    record[FIELD_FILE_SIZE] = len(image_bytes)
                    record[FIELD_FORMAT] = "ERROR"
            else:
                record[FIELD_WIDTH] = 0
                record[FIELD_HEIGHT] = 0
                record[FIELD_FILE_SIZE] = 0
                record[FIELD_FORMAT] = "ERROR"

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner."""
        return {
            FIELD_WIDTH: pa.int32(),
            FIELD_HEIGHT: pa.int32(),
            FIELD_FILE_SIZE: pa.int64(),
            FIELD_FORMAT: pa.string(),
        }
