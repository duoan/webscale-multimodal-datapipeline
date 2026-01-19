"""
Quality Filter

Filters records based on quality criteria.
"""

from typing import Any

from webscale_multimodal_datapipeline.framework import Filter

# Import field name constants from refiners
from webscale_multimodal_datapipeline.operators.refiners.image_metadata import FIELD_HEIGHT, FIELD_WIDTH
from webscale_multimodal_datapipeline.operators.refiners.image_technical_quality import (
    FIELD_COMPRESSION_ARTIFACTS,
    FIELD_INFORMATION_ENTROPY,
)


class ImageQualityFilter(Filter):
    """Filter records based on image quality metrics.

    Uses fields from ImageMetadataRefiner and ImageTechnicalQualityRefiner:
    - image_width, image_height
    - image_compression_artifacts, image_information_entropy
    """

    def __init__(
        self,
        min_width: int = 256,
        min_height: int = 256,
        max_compression_artifacts: float = 0.8,
        min_information_entropy: float = 3.0,
    ):
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height
        self.max_compression_artifacts = max_compression_artifacts
        self.min_information_entropy = min_information_entropy

    def should_keep_batch(self, records: list[dict[str, Any]]) -> list[bool]:
        """Determine which records meet quality criteria."""
        results = []
        for record in records:
            width = record.get(FIELD_WIDTH, 0)
            height = record.get(FIELD_HEIGHT, 0)
            compression_artifacts = record.get(FIELD_COMPRESSION_ARTIFACTS, 0.0)
            information_entropy = record.get(FIELD_INFORMATION_ENTROPY, 0.0)

            keep = (
                width >= self.min_width
                and height >= self.min_height
                and compression_artifacts <= self.max_compression_artifacts
                and information_entropy >= self.min_information_entropy
            )
            results.append(keep)
        return results
