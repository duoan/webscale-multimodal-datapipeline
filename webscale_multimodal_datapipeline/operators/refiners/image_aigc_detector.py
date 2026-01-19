"""
Image AIGC Content Detector Refiner

Detects AI-generated images using a trained SigLIP2 + MLP classifier.
This refiner enriches records with AIGC probability scores for filtering
synthetic content from training datasets.

Based on Imagen 3 findings: AIGC content filtering is crucial for preventing
degradation in model output quality and physical realism.

Requirements:
- Pre-computed SigLIP2 embeddings (use ImageSigLIPEmbeddingRefiner first)
- Trained AIGC classifier weights

Usage:
    # Step 1: Extract SigLIP embeddings
    siglip_refiner = ImageSigLIPEmbeddingRefiner()

    # Step 2: Run AIGC detection on embeddings
    aigc_refiner = ImageAIGCDetectorRefiner(
        embedding_field="image_siglip_emb_so400m_patch14_384",
        model_path="./models/image_aigc_detector/classifier.pth",
    )

Output fields:
- image_aigc_score: Probability that the image is AI-generated (0-1)
- image_is_aigc: Boolean flag (True if score > threshold)
"""

from typing import Any

import numpy as np
import pyarrow as pa
import torch
from huggingface_hub import hf_hub_download

from webscale_multimodal_datapipeline.framework import Refiner
from webscale_multimodal_datapipeline.models.image_aigc_detector import AIGCClassifierHead

# Field name constants
FIELD_AIGC_SCORE = "image_aigc_score"
FIELD_IS_AIGC = "image_is_aigc"

# Default SigLIP2 embedding dimension (so400m model)
DEFAULT_EMBEDDING_DIM = 1152


class ImageAIGCDetectorRefiner(Refiner):
    """Refiner for detecting AI-generated images using SigLIP2 + MLP.

    Uses a trained classifier to predict the probability that an image
    is AI-generated (synthetic) vs real (natural photograph).

    Requires pre-computed SigLIP2 embeddings from ImageSigLIPEmbeddingRefiner.

    Output fields:
    - image_aigc_score: Probability that the image is AI-generated (0.0-1.0)
    - image_is_aigc: Boolean flag (True if score > threshold)
    """

    def __init__(
        self,
        embedding_field: str,
        model_path: str | None = None,
        model_repo: str | None = None,
        model_filename: str = "image_aigc_classifier.pth",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dims: tuple[int, ...] = (512, 128),
        threshold: float = 0.5,
        device: str = "auto",
        inference_batch_size: int = 32,
        use_fp16: bool = True,
    ):
        """Initialize AIGC detector refiner.

        Args:
            embedding_field: Field name containing pre-computed SigLIP2 embeddings.
                             Use "image_siglip_emb_so400m_patch14_384" if using
                             ImageSigLIPEmbeddingRefiner with default so400m model.
            model_path: Local path to trained classifier weights (.pth file)
            model_repo: HuggingFace repo containing the classifier (alternative to model_path)
            model_filename: Filename in the HuggingFace repo
            embedding_dim: Expected embedding dimension (must match classifier input)
            hidden_dims: MLP hidden layer dimensions (must match trained model)
            threshold: Classification threshold (default 0.5)
            device: Device to run on ("cpu", "cuda", "mps", or "auto")
            inference_batch_size: Batch size for inference
            use_fp16: Use FP16 half precision (CUDA only)
        """
        super().__init__()

        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.inference_batch_size = inference_batch_size

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

        # FP16 support (not for MPS)
        self.use_fp16 = use_fp16 and device == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f"Using pre-computed embeddings from field: '{embedding_field}'")
        print(f"Expected embedding dimension: {embedding_dim}")

        # Load AIGC classifier
        self._load_classifier(model_path, model_repo, model_filename, embedding_dim, hidden_dims)

        print("Image AIGC Detector Refiner initialized.")
        print(f"  Output fields: {FIELD_AIGC_SCORE}, {FIELD_IS_AIGC}")
        print(f"  Threshold: {threshold}")

    def _load_classifier(
        self,
        model_path: str | None,
        model_repo: str | None,
        model_filename: str,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        """Load the AIGC classifier head."""
        # Determine model path
        if model_path is not None:
            classifier_path = model_path
        elif model_repo is not None:
            print(f"Downloading classifier from: {model_repo}/{model_filename}")
            classifier_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        else:
            raise ValueError("Either model_path or model_repo must be provided")

        # Initialize classifier with same architecture as training
        self.classifier = AIGCClassifierHead(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout_rate=0.0,  # No dropout during inference
            use_batch_norm=True,
        )

        # Load weights
        print(f"Loading classifier from: {classifier_path}")
        state_dict = torch.load(classifier_path, map_location="cpu", weights_only=True)

        # Handle full checkpoint vs state_dict only
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        self.classifier.load_state_dict(state_dict)
        self.classifier.to(self.device)
        self.classifier.eval()

        if self.use_fp16:
            self.classifier = self.classifier.half()
            print("Classifier using FP16 half precision")

        print("AIGC classifier loaded successfully")

    def _get_embeddings(self, records: list[dict[str, Any]]) -> tuple[torch.Tensor, list[int]]:
        """Extract embeddings from the specified field in records."""
        embeddings = []
        valid_indices = []

        for i, record in enumerate(records):
            emb = record.get(self.embedding_field)
            if emb is not None:
                emb_array = np.array(emb, dtype=np.float32)
                if len(emb_array) == self.embedding_dim:
                    embeddings.append(emb_array)
                    valid_indices.append(i)

        if not embeddings:
            return torch.tensor([]), []

        embeddings_tensor = torch.from_numpy(np.stack(embeddings)).to(self.device, dtype=self.dtype)
        return embeddings_tensor, valid_indices

    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Predict AIGC scores for a batch of images inplace."""
        if not records:
            return

        # Initialize all records with default values
        for record in records:
            record[FIELD_AIGC_SCORE] = 0.0
            record[FIELD_IS_AIGC] = False

        # Process in mini-batches
        for batch_start in range(0, len(records), self.inference_batch_size):
            batch_end = min(batch_start + self.inference_batch_size, len(records))
            batch_records = records[batch_start:batch_end]

            try:
                embeddings, valid_indices = self._get_embeddings(batch_records)

                if len(valid_indices) == 0:
                    continue

                # Predict AIGC scores
                with torch.inference_mode():
                    logits = self.classifier(embeddings.to(self.classifier.classifier[0].weight.dtype))
                    scores = torch.sigmoid(logits).float().cpu().numpy().flatten()

                # Update records
                for j, idx in enumerate(valid_indices):
                    score = float(scores[j])
                    batch_records[idx][FIELD_AIGC_SCORE] = score
                    batch_records[idx][FIELD_IS_AIGC] = score > self.threshold

            except Exception as e:
                print(f"Warning: Batch inference failed: {e}")

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner."""
        return {
            FIELD_AIGC_SCORE: pa.float32(),
            FIELD_IS_AIGC: pa.bool_(),
        }
