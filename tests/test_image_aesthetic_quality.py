#!/usr/bin/env python3
"""
Test script for ImageAestheticQualityRefiner.

Tests the CLIP+MLP aesthetic score prediction using the model from:
https://huggingface.co/ttj/sac-logos-ava1-l14-linearMSE

1. Reuse pre-computed embeddings
"""

import io
import time

from PIL import Image

from webscale_multimodal_datapipeline.operators.refiners.image_aesthetic_quality import ImageAestheticQualityRefiner
from webscale_multimodal_datapipeline.operators.refiners.image_clip_embedding import ImageClipEmbeddingRefiner


def create_test_image(
    width: int = 512,
    height: int = 512,
    color: tuple[int, int, int] = (100, 150, 200),
    pattern: str = "solid",
) -> bytes:
    """Create a test image and return as bytes."""
    img = Image.new("RGB", (width, height), color)

    if pattern == "gradient":
        # Create a gradient pattern
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                pixels[x, y] = (r, g, b)
    elif pattern == "noise":
        # Create a noisy pattern
        import numpy as np

        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(noise)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_with_precomputed_embeddings():
    """Test mode 1: Reuse pre-computed CLIP embeddings (recommended)."""
    print("=" * 70)
    print("Mode 1: Reuse Pre-computed CLIP Embeddings (Recommended)")
    print("=" * 70)

    # Step 1: Compute CLIP embeddings with ImageClipEmbeddingRefiner
    print("\n[1] Computing CLIP embeddings with ImageClipEmbeddingRefiner...")
    print("    (Using ViT-L-14 to match aesthetic predictor requirements)")
    start = time.time()
    clip_refiner = ImageClipEmbeddingRefiner(
        model_name="ViT-L-14",
        pretrained="openai",
        normalize=True,
        inference_batch_size=8,
        use_fp16=False,
    )
    print(f"    CLIP model loaded in {time.time() - start:.2f}s")

    # Create test records
    test_cases = [
        ("Solid blue", create_test_image(512, 512, (50, 100, 200), "solid")),
        ("Gradient", create_test_image(512, 512, (0, 0, 0), "gradient")),
        ("Random noise", create_test_image(512, 512, (0, 0, 0), "noise")),
    ]
    records = [{"id": i, "name": name, "image": {"bytes": img_bytes}} for i, (name, img_bytes) in enumerate(test_cases)]

    # Compute embeddings
    start = time.time()
    clip_refiner.refine_batch(records)
    print(f"    Embeddings computed in {time.time() - start:.2f}s")

    embedding_field = clip_refiner.feature_field_name
    print(f"    Embedding field: '{embedding_field}'")
    print(f"    Embedding dim: {len(records[0][embedding_field])}")

    # Step 2: Use aesthetic refiner with pre-computed embeddings
    print("\n[2] Initializing ImageAestheticQualityRefiner with embedding_field...")
    start = time.time()
    aesthetic_refiner = ImageAestheticQualityRefiner(
        embedding_field=embedding_field,  # Reuse embeddings!
        inference_batch_size=8,
        use_fp16=False,
    )
    print(f"    Aesthetic MLP loaded in {time.time() - start:.2f}s (no CLIP loading!)")

    # Predict aesthetic scores
    print("\n[3] Predicting aesthetic scores from embeddings...")
    start = time.time()
    aesthetic_refiner.refine_batch(records)
    print(f"    Prediction took {time.time() - start:.4f}s (fast - no CLIP inference!)")

    # Print results
    print("\n[4] Results:")
    print("-" * 50)
    for record in records:
        score = record.get("image_aesthetic_score", 0.0)
        print(f"    {record['name']:20s} -> score: {score:.4f}")
    print("-" * 50)


def main():
    print("\n" + "=" * 70)
    print("ImageAestheticQualityRefiner Test Suite")
    print("=" * 70)

    # Test both modes
    test_with_precomputed_embeddings()

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print("\nRecommendation: Use Mode 1 (pre-computed embeddings) in pipelines")
    print("to avoid redundant CLIP computation when you already have embeddings.")


if __name__ == "__main__":
    main()
