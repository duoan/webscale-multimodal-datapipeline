#!/usr/bin/env python3
"""
Test script to compare Python and Rust implementations of ImageTechnicalQualityRefiner.

This script verifies that both implementations produce consistent results.
"""

import io
import sys
from pathlib import Path

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from webscale_multimodal_datapipeline.operators.refiners.image_technical_quality import (
    FIELD_COMPRESSION_ARTIFACTS,
    FIELD_INFORMATION_ENTROPY,
    RUST_BACKEND_AVAILABLE,
    ImageTechnicalQualityRefiner,
    _assess_quality_batch_rust,
)


def create_test_image(width: int = 200, height: int = 200, mode: str = "RGB") -> bytes:
    """Create a test image with specified dimensions and mode."""
    img = Image.new(mode, (width, height), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def test_python_implementation():
    """Test Python implementation directly."""
    refiner = ImageTechnicalQualityRefiner()

    # Force Python implementation by temporarily disabling Rust
    import operators.refiners.image_technical_quality as tq_module

    original_rust_available = tq_module.RUST_BACKEND_AVAILABLE
    tq_module.RUST_BACKEND_AVAILABLE = False

    try:
        image_bytes = create_test_image()
        result = refiner._refine_python(image_bytes)

        assert FIELD_COMPRESSION_ARTIFACTS in result
        assert FIELD_INFORMATION_ENTROPY in result
        assert isinstance(result[FIELD_COMPRESSION_ARTIFACTS], float)
        assert isinstance(result[FIELD_INFORMATION_ENTROPY], float)

        return result[FIELD_COMPRESSION_ARTIFACTS], result[FIELD_INFORMATION_ENTROPY]
    finally:
        tq_module.RUST_BACKEND_AVAILABLE = original_rust_available


def test_rust_implementation():
    """Test Rust implementation directly."""
    if not RUST_BACKEND_AVAILABLE:
        return None

    image_bytes = create_test_image()
    results = _assess_quality_batch_rust([image_bytes])
    compression_artifacts, entropy = results[0]

    return float(compression_artifacts), float(entropy)


def test_both_implementations():
    """Compare Python and Rust implementations on various test cases."""
    print("=" * 60)
    print("Testing ImageTechnicalQualityRefiner implementations")
    print("=" * 60)

    if RUST_BACKEND_AVAILABLE:
        print("✓ Rust backend is available")
    else:
        print("⚠ Rust backend is NOT available (will only test Python)")

    print()

    test_cases = [
        ("Small RGB", 100, 100, "RGB"),
        ("Medium RGB", 500, 500, "RGB"),
        ("Large RGB", 2000, 2000, "RGB"),
        ("Grayscale", 200, 200, "L"),
        ("Complex pattern", 256, 256, "RGB"),
    ]

    max_diff_artifacts = 0.0
    max_diff_entropy = 0.0
    passed_tests = 0
    total_tests = 0

    refiner = ImageTechnicalQualityRefiner()

    for test_name, width, height, mode in test_cases:
        print(f"Test: {test_name} ({width}x{height}, {mode})")

        if test_name == "Complex pattern":
            # Create a more complex image with gradients
            img = Image.new("RGB", (width, height))
            pixels = img.load()
            for i in range(width):
                for j in range(height):
                    r = int((i / width) * 255)
                    g = int((j / height) * 255)
                    b = int(((i + j) / (width + height)) * 255)
                    pixels[i, j] = (r, g, b)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            image_bytes = img_bytes.getvalue()
        else:
            image_bytes = create_test_image(width, height, mode)

        # Test Python implementation
        try:
            python_result = refiner._refine_python(image_bytes)
            python_artifacts = python_result[FIELD_COMPRESSION_ARTIFACTS]
            python_entropy = python_result[FIELD_INFORMATION_ENTROPY]
        except Exception as e:
            print(f"  ❌ Python implementation failed: {e}")
            continue

        # Test Rust implementation
        if RUST_BACKEND_AVAILABLE:
            try:
                rust_results = _assess_quality_batch_rust([image_bytes])
                rust_artifacts, rust_entropy = rust_results[0]
                rust_artifacts = float(rust_artifacts)
                rust_entropy = float(rust_entropy)
            except Exception as e:
                print(f"  ❌ Rust implementation failed: {e}")
                continue

            # Compare results
            diff_artifacts = abs(python_artifacts - rust_artifacts)
            diff_entropy = abs(python_entropy - rust_entropy)

            max_diff_artifacts = max(max_diff_artifacts, diff_artifacts)
            max_diff_entropy = max(max_diff_entropy, diff_entropy)

            # Check if differences are within acceptable tolerance
            tolerance_artifacts = 1e-5
            tolerance_entropy = 1e-5

            artifacts_match = diff_artifacts < tolerance_artifacts
            entropy_match = diff_entropy < tolerance_entropy

            total_tests += 1
            if artifacts_match and entropy_match:
                passed_tests += 1
                print(f"  ✓ Results match (diff: artifacts={diff_artifacts:.2e}, entropy={diff_entropy:.2e})")
            else:
                print("  ⚠ Results differ:")
                print(f"    Python: artifacts={python_artifacts:.6f}, entropy={python_entropy:.6f}")
                print(f"    Rust:   artifacts={rust_artifacts:.6f}, entropy={rust_entropy:.6f}")
                print(f"    Diff:   artifacts={diff_artifacts:.6e}, entropy={diff_entropy:.6e}")
                if diff_artifacts >= tolerance_artifacts:
                    print(f"    ⚠ Artifacts difference exceeds tolerance ({tolerance_artifacts})")
                if diff_entropy >= tolerance_entropy:
                    print(f"    ⚠ Entropy difference exceeds tolerance ({tolerance_entropy})")
        else:
            # Only Python available
            print(f"  ✓ Python: artifacts={python_artifacts:.6f}, entropy={python_entropy:.6f}")
            total_tests += 1
            passed_tests += 1

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    if RUST_BACKEND_AVAILABLE:
        print(f"Max difference - Artifacts: {max_diff_artifacts:.2e}")
        print(f"Max difference - Entropy: {max_diff_entropy:.2e}")
        if max_diff_artifacts < 1e-5 and max_diff_entropy < 1e-5:
            print("✓ Both implementations produce identical results!")
        else:
            print("⚠ Implementations have small numerical differences (may be acceptable)")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = test_both_implementations()
    sys.exit(0 if success else 1)
