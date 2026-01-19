"""
Image CLIP Embedding Refiner

Extracts image embeddings using OpenCLIP models.
This refiner enriches records with CLIP embedding features for semantic analysis.
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import open_clip
import pyarrow as pa
import torch
from PIL import Image

from webscale_multimodal_datapipeline.framework import Refiner


class ImageClipEmbeddingRefiner(Refiner):
    """Refiner for extracting image embeddings using OpenCLIP."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto",
        normalize: bool = True,
        feature_field_name: str | None = None,
        inference_batch_size: int = 32,
        use_fp16: bool = True,
        preprocess_workers: int = 4,
    ):
        """Initialize CLIP embedding refiner.

        Args:
            model_name: OpenCLIP model name (e.g., "ViT-B-32", "ViT-L-14")
            pretrained: Pretrained weights identifier (e.g., "openai", "laion400m_e32")
            device: Device to run model on ("cpu", "cuda", "mps", or "auto")
            normalize: Whether to normalize embeddings to unit length
            feature_field_name: Name of the output field for embedding
            inference_batch_size: Batch size for GPU inference (default: 32)
            use_fp16: Use FP16 half precision for faster inference (default: True)
            preprocess_workers: Number of threads for parallel image preprocessing
        """
        super().__init__()
        self.inference_batch_size = inference_batch_size
        self.preprocess_workers = preprocess_workers

        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize

        # Handle device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                print("Auto-detected MPS (Mac GPU)")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Auto-detected CUDA")
            else:
                device = "cpu"
                print("Using CPU")

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        # FP16 support (not for MPS which has limited fp16 support)
        self.use_fp16 = use_fp16 and device == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # Set feature field name
        if feature_field_name is None:
            normalized_model_name = model_name.lower().replace("-", "_").replace(" ", "_")
            self.feature_field_name = f"image_clip_emb_{normalized_model_name}"
        else:
            self.feature_field_name = feature_field_name

        # Load model
        print(f"Loading OpenCLIP model: {model_name}/{pretrained} on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

        # Convert to FP16 if enabled
        if self.use_fp16:
            self.model = self.model.half()
            print("Using FP16 half precision")

        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Thread pool initialized lazily (can't pickle ThreadPoolExecutor for Ray)
        self._executor = None

        print(f"Model loaded. Embedding dim: {self.model.visual.output_dim}")
        print(f"Output field: {self.feature_field_name}")

    def _preprocess_image(self, record: dict[str, Any]) -> tuple[int, Any] | None:
        """Preprocess a single image (for parallel execution)."""
        img_obj = record.get("image", {})
        if isinstance(img_obj, dict) and "bytes" in img_obj:
            try:
                img = Image.open(BytesIO(img_obj["bytes"]))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return self.preprocess(img)
            except Exception:
                pass
        return None

    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Extract CLIP embeddings from a batch of images inplace (GPU batch inference)."""
        if not records:
            return

        embedding_dim = self.model.visual.output_dim
        zero_embedding = [0.0] * embedding_dim

        # Initialize all records with zero embeddings first
        for record in records:
            record[self.feature_field_name] = zero_embedding

        # Process in mini-batches
        for batch_start in range(0, len(records), self.inference_batch_size):
            batch_end = min(batch_start + self.inference_batch_size, len(records))
            batch_records = records[batch_start:batch_end]

            # Lazy init thread pool (can't pickle for Ray serialization)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.preprocess_workers)

            # Parallel preprocess using thread pool
            results = list(self._executor.map(self._preprocess_image, batch_records))

            # Collect valid tensors and indices
            tensors = []
            valid_indices = []
            for i, tensor in enumerate(results):
                if tensor is not None:
                    tensors.append(tensor)
                    valid_indices.append(i)

            if not tensors:
                continue

            # Batch inference
            try:
                batch_tensor = torch.stack(tensors).to(self.device, dtype=self.dtype)
                with torch.inference_mode():
                    features = self.model.encode_image(batch_tensor)
                    if self.normalize:
                        features = features / features.norm(dim=-1, keepdim=True)
                    # Convert all at once (faster than per-row tolist)
                    embeddings = features.float().cpu().numpy().tolist()

                for j, idx in enumerate(valid_indices):
                    batch_records[idx][self.feature_field_name] = embeddings[j]
            except Exception:
                pass

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner.

        Note: Arrow doesn't have native list types with fixed length,
        so we use list<item: float> for the embedding feature.
        """
        return {
            self.feature_field_name: pa.list_(pa.float32()),  # Variable-length list of floats
        }
