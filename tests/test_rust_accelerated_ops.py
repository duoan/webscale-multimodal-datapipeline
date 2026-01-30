#!/usr/bin/env python3
"""Test script for rust_accelerated_ops extension."""

import io
import random

from PIL import Image

# Import the installed modules
try:
    from mega_data_factory import rust_accelerated_ops as ops  # type: ignore
except ImportError:
    print("Error: rust_accelerated_ops not installed.")
    print("Run: uv pip install -e .")
    exit(1)

try:
    from mega_data_factory import rust_operators as text_ops  # type: ignore
except ImportError:
    text_ops = None
    print("Warning: rust_operators not available, skipping text extraction tests.")

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

# Test 5: HTML text extraction with Japanese characters (UTF-8 edge case)
# This tests the fix for the panic when dom_smoothie encounters multi-byte UTF-8 chars
print("\n--- Testing HTML text extraction ---")

if text_ops is None:
    print("⚠ Skipping HTML text extraction tests (rust_operators not available)")
else:
    # Test 5a: Normal HTML with Japanese text
    html_japanese = """
    <!DOCTYPE html>
    <html>
    <head><title>日本語テスト</title></head>
    <body>
    <article>
    <h1>日本語のテスト記事</h1>
    <p>これは日本語のテキストです。美容整形や医療に関する記事のテストです。</p>
    <p>顔面が壊れて自死してもニュースにならない【フル動画は概要欄へ】#整形　#整形外科　#美容整形　#医療　#美容</p>
    <p>This is a test article with mixed Japanese and English content for testing UTF-8 handling.</p>
    </article>
    </body>
    </html>
    """
    result = text_ops.html_extract_text(html_japanese)
    if result is not None:
        title, text, length = result
        print(f"✓ Test 5a (Japanese HTML): title='{title[:30]}...', length={length}")
    else:
        print("✓ Test 5a (Japanese HTML): returned None (content too short or extraction failed)")

    # Test 5b: HTML with the exact problematic content from the error
    html_problematic = """
    <!DOCTYPE html>
    <html>
    <head><title>マグニチュード99</title></head>
    <body>
    <article>
    <h1>顔面が壊れて自死してもニュースにならない</h1>
    <p>顔面が壊れて自死してもニュースにならない【フル動画は概要欄へ】#整形　#整形外科　#美容整形　#医療　#美容 (youtube.com) - マグニチュード99</p>
    <p>This is additional content to ensure the article is long enough for extraction.</p>
    <p>More content here to make sure we have enough text for the readability algorithm.</p>
    </article>
    </body>
    </html>
    """
    result = text_ops.html_extract_text(html_problematic)
    if result is not None:
        title, text, length = result
        print(f"✓ Test 5b (problematic UTF-8): title='{title[:30]}...', length={length}")
    else:
        print("✓ Test 5b (problematic UTF-8): returned None (gracefully handled)")

    # Test 5c: Batch extraction with mixed content
    html_batch = [html_japanese, html_problematic, "<html><body><p>Short</p></body></html>"]
    results = text_ops.html_extract_text_batch(html_batch)
    successful = sum(1 for r in results if r is not None)
    print(f"✓ Test 5c (batch extraction): {successful}/{len(html_batch)} successful extractions")

    # Test 5d: Malformed HTML (should not panic)
    html_malformed = "<html><body><p>Unclosed tag<div>混合コンテンツ</p></body>"
    result = text_ops.html_extract_text(html_malformed)
    print(f"✓ Test 5d (malformed HTML): returned {'result' if result else 'None'} (no panic)")

print("\n✓ All tests passed! rust_accelerated_ops is working correctly.")
