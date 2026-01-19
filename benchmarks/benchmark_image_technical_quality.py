#!/usr/bin/env python3
"""
Performance benchmark to compare Python vs Rust implementations of ImageTechnicalQualityRefiner.

This script measures the actual performance difference between Python and Rust implementations.
"""

import io
import sys
import time
from pathlib import Path

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from webscale_multimodal_datapipeline.operators.refiners.image_technical_quality import (
    RUST_BACKEND_AVAILABLE,
    ImageTechnicalQualityRefiner,
    _assess_quality_batch_rust,
)


def create_test_image(width: int = 2000, height: int = 2000, mode: str = "RGB") -> bytes:
    """Create a test image with specified dimensions."""
    img = Image.new(mode, (width, height), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def benchmark_rust(image_bytes: bytes, num_iterations: int = 100):
    """Benchmark Rust implementation."""
    if not RUST_BACKEND_AVAILABLE:
        return None

    # Warmup
    _assess_quality_batch_rust([image_bytes])

    start = time.perf_counter()
    for _ in range(num_iterations):
        _assess_quality_batch_rust([image_bytes])
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time

    return {
        "total_time": total_time,
        "avg_time": avg_time * 1000,  # Convert to ms
        "throughput": throughput,
    }


def benchmark_python(image_bytes: bytes, num_iterations: int = 100):
    """Benchmark Python implementation."""
    import operators.refiners.image_technical_quality as tq_module

    refiner = ImageTechnicalQualityRefiner()

    # Temporarily disable Rust to force Python
    original_rust_available = tq_module.RUST_BACKEND_AVAILABLE
    tq_module.RUST_BACKEND_AVAILABLE = False

    try:
        # Warmup
        refiner._refine_python(image_bytes)

        start = time.perf_counter()
        for _ in range(num_iterations):
            refiner._refine_python(image_bytes)
        end = time.perf_counter()

        total_time = end - start
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time

        return {
            "total_time": total_time,
            "avg_time": avg_time * 1000,  # Convert to ms
            "throughput": throughput,
        }
    finally:
        tq_module.RUST_BACKEND_AVAILABLE = original_rust_available


def benchmark_auto(image_bytes: bytes, num_iterations: int = 100):
    """Benchmark auto-detection (uses Rust if available)."""
    refiner = ImageTechnicalQualityRefiner()
    records = [{"id": "test", "image": {"bytes": image_bytes}}]

    # Warmup
    refiner.refine_batch(records.copy())

    start = time.perf_counter()
    for _ in range(num_iterations):
        batch = [{"id": "test", "image": {"bytes": image_bytes}}]
        refiner.refine_batch(batch)
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time

    backend_used = "Rust" if RUST_BACKEND_AVAILABLE else "Python"

    return {
        "backend_used": backend_used,
        "total_time": total_time,
        "avg_time": avg_time * 1000,  # Convert to ms
        "throughput": throughput,
    }


def main():
    """Run performance benchmarks."""
    print("=" * 70)
    print("ImageTechnicalQualityRefiner Performance Benchmark")
    print("=" * 70)
    print()

    if RUST_BACKEND_AVAILABLE:
        print("✓ Rust backend is available")
    else:
        print("⚠ Rust backend is NOT available (will only benchmark Python)")
    print()

    # Test with different image sizes
    test_cases = [
        ("Small", 500, 500),
        ("Medium", 1000, 1000),
        ("Large", 2000, 2000),
        ("Very Large", 4000, 4000),
    ]

    num_iterations = 50

    for test_name, width, height in test_cases:
        print(f"{'=' * 70}")
        print(f"Test: {test_name} ({width}x{height}) - {num_iterations} iterations")
        print(f"{'=' * 70}")

        image_bytes = create_test_image(width, height)
        image_size_mb = len(image_bytes) / (1024 * 1024)
        print(f"Image size: {image_size_mb:.2f} MB")
        print()

        # Benchmark auto (current implementation)
        print("Auto-detection (current):")
        auto_result = benchmark_auto(image_bytes, num_iterations)
        print(f"  Backend: {auto_result['backend_used']}")
        print(f"  Total time: {auto_result['total_time']:.2f}s")
        print(f"  Avg latency: {auto_result['avg_time']:.2f}ms")
        print(f"  Throughput: {auto_result['throughput']:.2f} images/sec")
        print()

        # Benchmark Python
        print("Python implementation:")
        python_result = benchmark_python(image_bytes, num_iterations)
        print(f"  Total time: {python_result['total_time']:.2f}s")
        print(f"  Avg latency: {python_result['avg_time']:.2f}ms")
        print(f"  Throughput: {python_result['throughput']:.2f} images/sec")
        print()

        # Benchmark Rust (if available)
        if RUST_BACKEND_AVAILABLE:
            print("Rust implementation:")
            rust_result = benchmark_rust(image_bytes, num_iterations)
            if rust_result:
                print(f"  Total time: {rust_result['total_time']:.2f}s")
                print(f"  Avg latency: {rust_result['avg_time']:.2f}ms")
                print(f"  Throughput: {rust_result['throughput']:.2f} images/sec")
                print()

                # Compare
                speedup = python_result["total_time"] / rust_result["total_time"]
                print("Performance comparison:")
                print(f"  Rust is {speedup:.2f}x faster than Python")
                print(f"  Time saved: {python_result['total_time'] - rust_result['total_time']:.2f}s")
                print(
                    f"  Throughput improvement: {rust_result['throughput'] - python_result['throughput']:.2f} images/sec"
                )
        else:
            print("Rust implementation: N/A (not available)")

        print()

    print("=" * 70)
    print("Note: In real pipeline, I/O overhead (image loading, disk writes)")
    print("      may reduce the visible speedup. Pure computation should show")
    print("      significant improvement with Rust.")
    print("=" * 70)


if __name__ == "__main__":
    main()
