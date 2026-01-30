"""CommonCrawl WARC DataLoader with streaming and Rust-accelerated text extraction."""

from __future__ import annotations

import gzip
import os
import time
from collections.abc import Iterator
from typing import Any

import requests
from warcio.archiveiterator import ArchiveIterator

from mega_data_factory.framework import DataLoader


class CommonCrawlLoader(DataLoader):
    """Streaming WARC loader with Rust text extraction.

    Uses Python warcio for streaming WARC parsing (yields records as they're parsed)
    and Rust for fast HTML text extraction. This enables true streaming where
    downstream stages can start processing before the entire file is parsed.
    """

    def __init__(
        self,
        crawl_id: str,
        base_url: str = "https://data.commoncrawl.org/",
        cache_dir: str | None = None,
        num_files: int | None = None,
    ):
        self.crawl_id = crawl_id
        self.base_url = base_url.rstrip("/") + "/"
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/commoncrawl")
        self.num_files = num_files
        self._file_list: list[str] | None = None

    def get_file_list(self, max_samples: int | None = None, num_workers: int = 1) -> list[str]:
        """Get list of WARC file paths."""
        if self._file_list is not None:
            return self._file_list

        # Calculate files needed: ~5K records per file
        num_files = self.num_files
        if num_files is None and max_samples:
            num_files = max(num_workers, max_samples // 5000 + 1)

        url = f"{self.base_url}crawl-data/{self.crawl_id}/warc.paths.gz"
        paths = []

        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        r.raw.decode_content = False

        for line in gzip.GzipFile(fileobj=r.raw):
            path = line.decode("utf-8", errors="ignore").strip()
            if path:
                paths.append(path)
                if num_files and len(paths) >= num_files:
                    break

        self._file_list = paths
        print(f"[CommonCrawl] {len(paths)} WARC files")
        return paths

    def load_files(
        self,
        file_list: list[str],
        worker_id: int | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Load WARC files and yield records with extracted text.

        TRUE STREAMING: Uses Python warcio for streaming WARC parsing, yielding
        records as they're parsed. Rust is used only for HTML text extraction.
        This enables downstream stages to start processing immediately.
        """
        from mega_data_factory.rust_operators import html_extract_text

        label = f"W{worker_id}" if worker_id is not None else "L"
        skip = checkpoint.get("records_processed", 0) if checkpoint else 0
        count = 0
        yielded = 0

        for warc_path in file_list:
            print(f"[{label}] Starting to process: {warc_path.split('/')[-1]}")
            local_path = self._download(warc_path)
            print(f"[{label}] File ready, opening: {local_path.split('/')[-1]}")

            # Stream WARC records using Python warcio - yields as it parses!
            with open(local_path, "rb") as f:
                print(f"[{label}] Starting ArchiveIterator...")
                record_count = 0
                for record in ArchiveIterator(f):
                    record_count += 1
                    if record_count == 1:
                        print(f"[{label}] First record received from ArchiveIterator")
                    # Skip non-response records
                    if record.rec_type != "response":
                        continue

                    # Get URL and date from headers
                    url = record.rec_headers.get_header("WARC-Target-URI", "")
                    warc_date = record.rec_headers.get_header("WARC-Date", "")

                    # Check content type
                    content_type = record.http_headers.get_header("Content-Type", "") if record.http_headers else ""
                    if "text/html" not in content_type.lower():
                        continue

                    # Read HTML content
                    try:
                        html_content = record.content_stream().read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    if not html_content or len(html_content) < 100:
                        continue

                    count += 1
                    if count <= skip:
                        continue

                    # 🦀 Rust: Extract text from HTML (fast!)
                    try:
                        result = html_extract_text(html_content)
                    except Exception as e:
                        print(f"Error {e}")

                    if result is None:
                        continue

                    title, text, text_length = result
                    yielded += 1

                    if yielded == 1:
                        print(f"[{label}] First record yielded!")

                    yield {
                        "crawl_id": self.crawl_id,
                        "warc_path": warc_path,
                        "url": url,
                        "warc_date": warc_date,
                        "title": title,
                        "text": text,
                        "text_length": text_length,
                    }

        print(f"[{label}] Streaming: {count} HTML records parsed, {yielded} yielded")

    def _download(self, warc_path: str) -> str:
        """Download WARC to cache."""
        os.makedirs(os.path.join(self.cache_dir, self.crawl_id), exist_ok=True)
        filename = warc_path.rsplit("/", 1)[-1]
        local_path = os.path.join(self.cache_dir, self.crawl_id, filename)

        if os.path.exists(local_path):
            return local_path

        print(f"[DL] {filename}...")
        url = f"{self.base_url}{warc_path}"

        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=300)
                r.raise_for_status()
                tmp = local_path + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(131072):
                        f.write(chunk)
                os.rename(tmp, local_path)
                print(f"[DL] {filename} done ({os.path.getsize(local_path) // 1048576}MB)")
                return local_path
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Download failed: {warc_path}") from e
                time.sleep(2**attempt)

        return local_path
