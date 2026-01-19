"""
Image Aesthetic Quality Refiner

Predicts aesthetic quality scores using CLIP+MLP model from:
- https://github.com/christophschuhmann/improved-aesthetic-predictor
- https://huggingface.co/ttj/sac-logos-ava1-l14-linearMSE

The model outputs a score (typically 1-10) predicting how visually appealing
an image is, based on training from professional annotators (AVA dataset + LAION logos).

Requirements:
- Pre-computed CLIP ViT-L-14 embeddings (768-dim)
- Use ImageClipEmbeddingRefiner with model_name="ViT-L-14" first

Usage:
    # Step 1: Extract CLIP embeddings with ViT-L-14
    clip_refiner = ImageClipEmbeddingRefiner(model_name="ViT-L-14")

    # Step 2: Run aesthetic scoring on embeddings
    aesthetic_refiner = ImageAestheticQualityRefiner(
        embedding_field="image_clip_emb_vit_l_14",
    )

Output fields:
- image_aesthetic_score: Aesthetic quality score (higher = more visually appealing, typically 1-10)
"""

from typing import Any

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from webscale_multimodal_datapipeline.framework import Refiner

# Field name constant
FIELD_AESTHETIC_SCORE = "image_aesthetic_score"

# Required embedding dimension for the aesthetic predictor (ViT-L/14)
REQUIRED_EMBEDDING_DIM = 768


class AestheticMLP(nn.Module):
    """MLP model for aesthetic score prediction.

    Architecture from: https://github.com/christophschuhmann/improved-aesthetic-predictor
    Input: CLIP ViT-L/14 embeddings (768 dimensions)
    Output: Aesthetic score (typically 1-10)
    """

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ImageAestheticQualityRefiner(Refiner):
    """Refiner for predicting aesthetic quality scores using CLIP+MLP.

    Uses the improved-aesthetic-predictor model trained on AVA dataset + LAION logos.
    The model predicts how visually appealing an image is on a scale of ~1-10.

    Requires pre-computed CLIP ViT-L-14 embeddings (768-dim) from ImageClipEmbeddingRefiner.

    Output fields:
    - image_aesthetic_score: Aesthetic quality score (higher = more visually appealing)
    """

    def __init__(
        self,
        embedding_field: str,
        model_repo: str = "ttj/sac-logos-ava1-l14-linearMSE",
        model_filename: str = "model.safetensors",
        device: str = "auto",
        inference_batch_size: int = 32,
        use_fp16: bool = True,
    ):
        """Initialize aesthetic quality refiner.

        Args:
            embedding_field: Field name containing pre-computed CLIP ViT-L-14 embeddings (768-dim).
                             Use "image_clip_emb_vit_l_14" from ImageClipEmbeddingRefiner.
            model_repo: Hugging Face model repository for the aesthetic predictor MLP
            model_filename: Filename of the model weights in the repo
            device: Device to run model on ("cpu", "cuda", "mps", or "auto")
            inference_batch_size: Batch size for inference (default: 32)
            use_fp16: Use FP16 half precision for faster inference (default: True)
        """
        super().__init__()

        self.embedding_field = embedding_field
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

        # FP16 support (not for MPS which has limited fp16 support)
        self.use_fp16 = use_fp16 and device == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f"Using pre-computed embeddings from field: '{embedding_field}'")
        print(f"Required embedding dimension: {REQUIRED_EMBEDDING_DIM} (CLIP ViT-L/14)")

        # Load aesthetic predictor MLP from Hugging Face
        print(f"Loading aesthetic predictor MLP from: {model_repo}...")
        self._load_aesthetic_mlp(model_repo, model_filename)

        print(f"Aesthetic Quality Refiner initialized. Output field: {FIELD_AESTHETIC_SCORE}")

    def _load_aesthetic_mlp(self, model_repo: str, model_filename: str) -> None:
        """Load the aesthetic predictor MLP from Hugging Face."""
        # Download model from Hugging Face
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)

        # Initialize MLP with CLIP ViT-L/14 embedding size (768)
        self.aesthetic_mlp = AestheticMLP(input_size=REQUIRED_EMBEDDING_DIM)

        # Load weights - handle both safetensors and pth formats
        if model_filename.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        self.aesthetic_mlp.load_state_dict(state_dict)
        self.aesthetic_mlp.to(self.device)
        self.aesthetic_mlp.eval()

        # Convert MLP to FP16 if enabled (but predictions will be cast to float32)
        if self.use_fp16:
            self.aesthetic_mlp = self.aesthetic_mlp.half()
            print("Aesthetic MLP using FP16 half precision")

        print("Aesthetic predictor MLP loaded successfully")

    def _get_embeddings(self, records: list[dict[str, Any]]) -> tuple[torch.Tensor, list[int]]:
        """Extract embeddings from the specified field in records.

        Returns:
            Tuple of (embeddings tensor, valid_indices list)
        """
        embeddings = []
        valid_indices = []

        for i, record in enumerate(records):
            emb = record.get(self.embedding_field)
            if emb is not None:
                emb_array = np.array(emb, dtype=np.float32)
                if len(emb_array) == REQUIRED_EMBEDDING_DIM:
                    embeddings.append(emb_array)
                    valid_indices.append(i)
                else:
                    print(
                        f"Warning: Embedding dim mismatch at index {i}: "
                        f"got {len(emb_array)}, expected {REQUIRED_EMBEDDING_DIM}"
                    )

        if not embeddings:
            return torch.tensor([]), []

        # Stack and convert to tensor
        embeddings_tensor = torch.from_numpy(np.stack(embeddings)).to(self.device, dtype=self.dtype)
        return embeddings_tensor, valid_indices

    def refine_batch(self, records: list[dict[str, Any]]) -> None:
        """Predict aesthetic scores for a batch of images inplace (GPU batch inference)."""
        if not records:
            return

        # Initialize all records with default score (0.0)
        for record in records:
            record[FIELD_AESTHETIC_SCORE] = 0.0

        # Process in mini-batches for efficient GPU usage
        for batch_start in range(0, len(records), self.inference_batch_size):
            batch_end = min(batch_start + self.inference_batch_size, len(records))
            batch_records = records[batch_start:batch_end]

            try:
                embeddings, valid_indices = self._get_embeddings(batch_records)

                if len(valid_indices) == 0:
                    continue

                # Predict aesthetic scores
                with torch.inference_mode():
                    # Cast to the MLP's dtype (may be fp16 or fp32)
                    scores = self.aesthetic_mlp(embeddings.to(self.aesthetic_mlp.layers[0].weight.dtype))
                    # Convert to float32 for output
                    scores = scores.float().cpu().numpy().flatten()

                for j, idx in enumerate(valid_indices):
                    batch_records[idx][FIELD_AESTHETIC_SCORE] = float(scores[j])

            except Exception as e:
                print(f"Warning: Batch inference failed: {e}")

    def get_output_schema(self) -> dict[str, pa.DataType]:
        """Return output schema for new fields added by this refiner."""
        return {
            FIELD_AESTHETIC_SCORE: pa.float32(),
        }
