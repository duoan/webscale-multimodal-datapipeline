"""
Image SigLIP Embedding Refiner

Extracts image embeddings using Google's SigLIP2 models from HuggingFace Transformers.
This refiner enriches records with SigLIP embedding features for semantic analysis.

SigLIP2 models provide state-of-the-art image representations that can be used for:
- AIGC content detection
- Image similarity search
- Semantic clustering
- Zero-shot classification

Available models:
- google/siglip2-so400m-patch14-384 (1152-dim, recommended)
- google/siglip2-base-patch16-224 (768-dim, faster)
- google/siglip2-large-patch16-256 (1024-dim)
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import pyarrow as pa
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from webscale_multimodal_datapipeline.framework import Refiner


class ImageSigLIPEmbeddingRefiner(Refiner):
    """Refiner for extracting image embeddings using SigLIP2 models.

    Uses Google's SigLIP2 vision encoder to extract semantic image embeddings.
    The embeddings can be reused by other refiners (e.g., AIGC detector) to
    avoid redundant computation.

    Output fields:
    - image_siglip_emb_{model_suffix}: Embedding vector (list of floats)
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        device: str = "auto",
        normalize: bool = True,
        feature_field_name: str | None = None,
        inference_batch_size: int = 32,
        use_fp16: bool = True,
        preprocess_workers: int = 4,
    ):
        """Initialize SigLIP embedding refiner.

        Args:
            model_name: HuggingFace model name for SigLIP2
                        (e.g., "google/siglip2-so400m-patch14-384")
            device: Device to run model on ("cpu", "cuda", "mps", or "auto")
            normalize: Whether to normalize embeddings to unit length
            feature_field_name: Name of the output field for embedding.
                               If None, auto-generated from model name.
            inference_batch_size: Batch size for GPU inference (default: 32)
            use_fp16: Use FP16 half precision for faster inference (default: True)
            preprocess_workers: Number of threads for parallel image preprocessing
        """
        super().__init__()

        self.model_name = model_name
        self.normalize = normalize
        self.inference_batch_size = inference_batch_size
        self.preprocess_workers = preprocess_workers

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
            # Extract model suffix for field name (e.g., "so400m_patch14_384")
            model_suffix = model_name.split("/")[-1].lower().replace("-", "_").replace("siglip2_", "")
            self.feature_field_name = f"image_siglip_emb_{model_suffix}"
        else:
            self.feature_field_name = feature_field_name

        # Load model and processor
        print(f"Loading SigLIP2 model: {model_name} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Freeze model and set to eval mode
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Convert to FP16 if enabled
        if self.use_fp16:
            self.model = self.model.half()
            print("Using FP16 half precision")

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.vision_config.hidden_size

        # Thread pool initialized lazily (can't pickle ThreadPoolExecutor for Ray)
        self._executor = None

        print(f"Model loaded. Embedding dim: {self.embedding_dim}")
        print(f"Output field: {self.feature_field_name}")

    def _preprocess_image(self, record: dict[str, Any]) -> Image.Image | None:
        """Preprocess a single image (for parallel execution)."""
        img_obj = record.get("image", {})
        if isinstance(img_obj, dict) and "bytes" in img_obj:
            try:
                img = Image.open(BytesIO(img_obj["bytes"]))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception:
                pass
        return None

    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Extract SigLIP embeddings from a batch of images inplace (GPU batch inference)."""
        if not records:
            return

        zero_embedding = [0.0] * self.embedding_dim

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
            images = list(self._executor.map(self._preprocess_image, batch_records))

            # Collect valid images and indices
            valid_images = []
            valid_indices = []
            for i, img in enumerate(images):
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(i)

            if not valid_images:
                continue

            # Batch inference
            try:
                # Process images through SigLIP2 processor
                inputs = self.processor(images=valid_images, return_tensors="pt")
                inputs = {
                    k: v.to(self.device, dtype=self.dtype if k != "input_ids" else v.dtype) for k, v in inputs.items()
                }

                with torch.inference_mode():
                    # Get vision model outputs
                    outputs = self.model.vision_model(**{k: v for k, v in inputs.items() if k != "input_ids"})

                    # Use pooler output (CLS token) if available, else mean pooling
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    else:
                        # Mean pooling over sequence dimension
                        features = outputs.last_hidden_state.mean(dim=1)

                    # Normalize if requested
                    if self.normalize:
                        features = features / features.norm(dim=-1, keepdim=True)

                    # Convert to float32 for output
                    embeddings = features.float().cpu().numpy().tolist()

                for j, idx in enumerate(valid_indices):
                    batch_records[idx][self.feature_field_name] = embeddings[j]

            except Exception as e:
                print(f"Warning: Batch inference failed: {e}")

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner.

        Note: Arrow doesn't have native list types with fixed length,
        so we use list<item: float> for the embedding feature.
        """
        return {
            self.feature_field_name: pa.list_(pa.float32()),  # Variable-length list of floats
        }
