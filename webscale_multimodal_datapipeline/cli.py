"""
Webscale Multimodal Data Pipeline CLI

Usage:
    wmd run --config configs/z_image.yaml
    wmd run -c configs/z_image.yaml --max-samples 1000
"""

import argparse
import logging
import sys

# Import operators, loaders, and writers to register them
from webscale_multimodal_datapipeline import (
    loaders,  # noqa: F401
    operators,  # noqa: F401
    writers,  # noqa: F401
)
from webscale_multimodal_datapipeline.framework import Executor, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy logs from third-party libraries
for logger_name in [
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "datasets",
    "fsspec",
    "s3fs",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def cmd_run(args):
    """Run the data pipeline."""
    try:
        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = PipelineConfig.from_yaml(args.config)

        # Override max_samples if specified
        if args.max_samples is not None:
            config.executor.max_samples = args.max_samples

        # Override batch_size if specified
        if args.batch_size is not None:
            config.executor.batch_size = args.batch_size

        # Create executor
        print("Initializing executor...")
        executor = Executor(config)

        # Execute pipeline
        print("Starting pipeline execution...")
        print(f"  - Max samples: {config.executor.max_samples or 'unlimited'}")
        print(f"  - Batch size: {config.executor.batch_size}")
        print(f"  - Stages: {len(config.stages)}")
        print()

        total_input = 0
        total_output = 0

        for input_count, output_count in executor.execute():
            total_input += input_count
            total_output += output_count
            print(f"Progress: {total_input} input, {total_output} output (filtered: {total_input - total_output})...")

        print("\n" + "=" * 60)
        print("Pipeline completed:")
        print(f"  Input samples: {total_input}")
        print(f"  Output samples: {total_output}")
        filtered = total_input - total_output
        filtered_pct = (filtered / total_input * 100) if total_input > 0 else 0.0
        print(f"  Filtered/Deduplicated: {filtered} ({filtered_pct:.1f}%)")
        print("=" * 60)

        # Print operator performance statistics
        _print_stats(executor.get_operator_stats())

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if "executor" in locals():
            executor.shutdown()


def _print_stats(operator_stats):
    """Print operator performance statistics."""
    print("\n" + "=" * 60)
    print("Operator Performance Statistics:")
    print("=" * 60)

    if not operator_stats:
        print("  No statistics available")
        print("=" * 60)
        return

    for stage_name, stage_ops in operator_stats.items():
        print(f"\n{stage_name}:")

        # Print stage-level summary if available
        if "_stage_summary" in stage_ops:
            summary = stage_ops["_stage_summary"]
            print("  [Stage Summary]")
            print(f"    Records: {summary['total_records']}")
            print(f"    Total time: {summary['total_time']:.2f}s")
            print(f"    Throughput: {summary['throughput']:.2f} records/sec")
            print()

        # Print operator-level statistics
        for op_name, op_stats in stage_ops.items():
            if op_name == "_stage_summary":
                continue
            print(f"  {op_name}:")
            print(f"    Records: {op_stats.get('total_records', 0)}")
            print(f"    Total time: {op_stats.get('total_time', 0.0):.2f}s")
            print(f"    Avg latency: {op_stats.get('avg_latency', 0.0) * 1000:.2f}ms")
            print(
                f"    Min/Max: {op_stats.get('min_latency', 0.0) * 1000:.2f}ms / "
                f"{op_stats.get('max_latency', 0.0) * 1000:.2f}ms"
            )
            print(
                f"    P50/P95/P99: {op_stats.get('p50_latency', 0.0) * 1000:.2f}ms / "
                f"{op_stats.get('p95_latency', 0.0) * 1000:.2f}ms / "
                f"{op_stats.get('p99_latency', 0.0) * 1000:.2f}ms"
            )
            print(f"    Throughput: {op_stats.get('throughput', 0.0):.2f} records/sec")

    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="wmd",
        description="Webscale Multimodal Data Pipeline - High-performance distributed data processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    wmd run --config configs/z_image.yaml
    wmd run -c configs/z_image.yaml --max-samples 1000
    wmd run -c configs/z_image.yaml --batch-size 500
        """,
    )
    run_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to pipeline configuration YAML file",
    )
    run_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max samples to process (default: from config)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: from config)",
    )

    # Version
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
