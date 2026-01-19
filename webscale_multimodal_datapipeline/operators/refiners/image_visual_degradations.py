"""
Visual Degradations Refiner

Assesses visual degradations using the multi-head quality assessment model.
This is a Refiner that enriches records with visual quality scores based on
the Z-Image paper's approach to technical quality assessment.

Degradation factors assessed:
- Color cast: Abnormal color tints
- Blurriness: Lack of sharpness/focus
- Watermark: Visible watermarks
- Noise: Visual noise levels

Reference: Z-Image Technical Report (Section 2.1 - Data Profiling Engine)
"""

from typing import Any

import pyarrow as pa

from webscale_multimodal_datapipeline.framework import Refiner

# Field name constants (vd = visual degradations)
FIELD_COLOR_CAST = "img_vd_color_cast"
FIELD_BLURRINESS = "img_vd_blurriness"
FIELD_WATERMARK = "img_vd_watermark"
FIELD_NOISE = "img_vd_noise"
FIELD_OVERALL_QUALITY = "img_vd_overall_quality"

# All output fields
OUTPUT_FIELDS = [
    FIELD_COLOR_CAST,
    FIELD_BLURRINESS,
    FIELD_WATERMARK,
    FIELD_NOISE,
    FIELD_OVERALL_QUALITY,
]


class ImageVisualDegradationsRefiner(Refiner):
    """Refiner for model-based image visual degradations assessment.

    Uses a multi-head neural network to score images on multiple degradation factors.

    Output fields (all float32, range 0-1):
    - vd_color_cast: Color cast/tint level (higher = more color cast)
    - vd_blurriness: Blur level (higher = more blurry)
    - vd_watermark: Watermark visibility (higher = more visible)
    - vd_noise: Noise level (higher = more noise)
    - vd_overall_quality: Overall quality score (higher = better quality)
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        input_size: tuple[int, int] = (224, 224),
    ):
        """Initialize with model path.

        Args:
            model_path: Path to trained multi-head model file
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            input_size: Expected input size (H, W) for the model
        """
        super().__init__()
        self.model_path = model_path
        self.input_size = input_size
        self._device = device
        self._inference = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded (lazy loading).

        Returns:
            True if model is available, False otherwise
        """
        if self._model_loaded:
            return self._inference is not None

        self._model_loaded = True

        if self.model_path is None:
            return False

        try:
            from webscale_multimodal_datapipeline.models.image_quality_assessment.inference import (
                MultiHeadQualityInference,
                get_auto_device,
            )

            device = self._device or get_auto_device()
            self._inference = MultiHeadQualityInference(
                model_path=self.model_path,
                device=device,
                input_size=self.input_size,
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to load visual degradations model: {e}")
            return False

    def _get_default_result(self) -> dict[str, None]:
        """Get default result with all fields set to None."""
        return dict.fromkeys(OUTPUT_FIELDS)

    def _scores_to_dict(self, scores) -> dict[str, float]:
        """Convert model scores to output dictionary."""
        return {
            FIELD_COLOR_CAST: scores.color_cast,
            FIELD_BLURRINESS: scores.blurriness,
            FIELD_WATERMARK: scores.watermark,
            FIELD_NOISE: scores.noise,
            FIELD_OVERALL_QUALITY: scores.overall,
        }

    def _set_default_values(self, record: dict[str, Any]) -> None:
        """Set all output fields to None in the record."""
        for field in OUTPUT_FIELDS:
            record[field] = None

    def refine(self, record: dict[str, Any]) -> dict[str, Any]:
        """Assess visual degradations using model and return new fields.

        Args:
            record: Input record with image data

        Returns:
            Dictionary with degradation scores
        """
        if not self._ensure_model_loaded():
            return self._get_default_result()

        # Extract image bytes
        img_obj = record.get("image", {})
        if isinstance(img_obj, dict) and "bytes" in img_obj:
            image_bytes = img_obj["bytes"]
        else:
            return self._get_default_result()

        try:
            scores = self._inference.predict_from_bytes(image_bytes)
            return self._scores_to_dict(scores)
        except Exception:
            return self._get_default_result()

    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Refine a batch of records inplace (optimized batch processing).

        Args:
            records: List of records to refine
        """
        if not records:
            return

        if not self._ensure_model_loaded():
            for record in records:
                self._set_default_values(record)
            return

        # Collect valid image bytes
        valid_indices = []
        image_bytes_list = []

        for i, record in enumerate(records):
            img_obj = record.get("image", {})
            if isinstance(img_obj, dict) and "bytes" in img_obj:
                valid_indices.append(i)
                image_bytes_list.append(img_obj["bytes"])

        # Set defaults for all records first
        for record in records:
            self._set_default_values(record)

        if not image_bytes_list:
            return

        try:
            # Batch prediction
            scores_list = self._inference.predict_batch_from_bytes(image_bytes_list)

            # Update records with predictions
            for idx, scores in zip(valid_indices, scores_list, strict=False):
                records[idx].update(self._scores_to_dict(scores))

        except Exception as e:
            print(f"Warning: Batch prediction failed: {e}")

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner."""
        return {field: pa.float32() for field in OUTPUT_FIELDS}
