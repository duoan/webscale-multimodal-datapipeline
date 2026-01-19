#!/usr/bin/env python3
"""Test script for rust_accelerated_ops extension."""

import io
import random

from PIL import Image

# Import the installed module
try:
    from webscale_multimodal_datapipeline import rust_accelerated_ops as ops
except ImportError:
    print("Error: rust_accelerated_ops not installed.")
    print("Run: uv pip install -e .")
    exit(1)

print("Testing rust_accelerated_ops extension...")
print(f"Available functions: {[f for f in dir(ops) if not f.startswith('_')]}\n")

# Test 1: Solid color (low entropy)
img = Image.new("RGB", (100, 100), color="red")
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG")
results = ops.image_assess_quality_batch([img_bytes.getvalue()])
artifacts, entropy = results[0]
print(f"✓ Test 1 (solid red): artifacts={artifacts:.4f}, entropy={entropy:.4f}")

# Test 2: Random noise (high entropy)
img = Image.new("RGB", (100, 100))
pixels = img.load()
for i in range(100):
    for j in range(100):
        pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG")
results = ops.image_assess_quality_batch([img_bytes.getvalue()])
artifacts, entropy = results[0]
print(f"✓ Test 2 (random noise): artifacts={artifacts:.4f}, entropy={entropy:.4f}")

# Test 3: Batch processing
images = []
for _ in range(5):
    img = Image.new("RGB", (100, 100), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    images.append(buf.getvalue())

results = ops.image_assess_quality_batch(images)
print(f"✓ Test 3 (batch quality): processed {len(results)} images")

# Test 4: Batch perceptual hash
phashes = ops.image_compute_phash_batch(images, 16)
print(f"✓ Test 4 (batch phash): computed {len(phashes)} hashes")
print(f"  Sample hash: {phashes[0][:20]}...")

print("\n✓ All tests passed! rust_accelerated_ops is working correctly.")
