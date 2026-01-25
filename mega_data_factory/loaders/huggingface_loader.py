"""
HuggingFace File-based Data Loader

Two-layer architecture:
1. Coordinator: Scan files and assign to workers
2. Workers: Read assigned files and yield records

No streaming parameter - the entire system is streaming by design.
"""

from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from mega_data_factory.framework import DataLoader


class HuggingFaceLoader(DataLoader):
    """File-based loader for HuggingFace datasets.

    Two-layer architecture:
    - Layer 1 (Coordinator/Executor): get_file_list() -> scan all files
    - Layer 2 (Worker): load_files() -> read assigned files

    The entire system is streaming - no "streaming" parameter needed.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
    ):
        """Initialize file-based HuggingFace loader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "jp1924/Laion400m-1")
            split: Dataset split (default: "train")
        """
        self.dataset_name = dataset_name
        self.split = split
        self._file_list = None

    def get_file_list(self) -> list[str]:
        """[Layer 1] Get sorted list of data files from HuggingFace repo.

        This is called by the coordinator to get all files before distributing
        to workers.

        Returns:
            Sorted list of file paths (e.g., ["datasets/.../data/train-00000.parquet", ...])
        """
        if self._file_list is not None:
            return self._file_list

        fs = HfFileSystem()
        repo_path = f"datasets/{self.dataset_name}"

        # List files in data directory
        try:
            files = fs.ls(f"{repo_path}/data", detail=True)
        except Exception:
            # Fallback: try root directory
            files = fs.ls(repo_path, detail=True)

        # Filter for data files (parquet, arrow, csv)
        data_extensions = (".parquet", ".arrow", ".csv", ".jsonl")
        data_files = [f["name"] for f in files if any(f["name"].endswith(ext) for ext in data_extensions)]

        # Sort by filename for consistent ordering
        self._file_list = sorted(data_files)
        print(f"[HuggingFaceLoader] Found {len(self._file_list)} data files for {self.dataset_name}/{self.split}")

        return self._file_list

    def load_files(
        self,
        file_list: list[str],
        worker_id: int | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """[Layer 2] Load assigned files and yield records.

        This is called by workers to read their assigned files.

        Args:
            file_list: List of file paths to read
            worker_id: Optional worker ID for logging
            checkpoint: Optional checkpoint with 'records_processed' (standard format)

        Yields:
            Records as dictionaries
        """
        worker_label = f"Worker {worker_id}" if worker_id is not None else "Loader"

        # Resume from checkpoint if provided
        # Standard checkpoint format uses records_processed
        skip_records = checkpoint.get("records_processed", 0) if checkpoint else 0

        if skip_records > 0:
            print(f"[{worker_label}] Resuming from record {skip_records}")

        # Process assigned files
        total_records = 0
        records_yielded = 0

        for file_idx, file_path in enumerate(file_list):
            filename = file_path.split("/")[-1]

            print(f"[{worker_label}] Loading file {file_idx + 1}/{len(file_list)}: {filename}")

            # Read parquet file
            table = pq.read_table(f"hf://{file_path}")

            # Convert to records and yield
            for batch in table.to_batches(max_chunksize=1000):
                for record in batch.to_pylist():
                    total_records += 1

                    # Skip records if resuming from checkpoint
                    if total_records <= skip_records:
                        continue

                    records_yielded += 1
                    yield record

            print(f"[{worker_label}] Completed file {file_idx + 1}/{len(file_list)}, total records processed: {total_records}")

        print(f"[{worker_label}] Completed all {len(file_list)} files (yielded {records_yielded} records)")
