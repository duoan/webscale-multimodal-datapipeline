"""
commoncrawl_loader.py

Streaming Common Crawl WARC DataLoader with trafilatura HTML extraction.

- Input: crawl_id (e.g. CC-MAIN-2025-51)
- Streams: warc.paths.gz -> each warc.gz -> each WARC record
- Extracts: Clean text content using trafilatura
- Yields: dict records (url, text, html, metadata)

Deps:
  pip install requests warcio trafilatura
"""

from __future__ import annotations

import gzip
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests
import trafilatura
from warcio.archiveiterator import ArchiveIterator

from mega_data_factory.framework import DataLoader


@dataclass(frozen=True)
class CommonCrawlConfig:
    crawl_id: str
    base_url: str = "https://data.commoncrawl.org/"

    # Limits
    max_warc_files: int | None = None
    max_records: int | None = None

    # Networking
    request_timeout_sec: int = 300
    retry: int = 3
    retry_backoff_sec: float = 1.5

    # Filtering
    only_responses: bool = True  # only WARC-Type: response
    only_html: bool = True  # only Content-Type startswith text/html
    require_200: bool = False  # only HTTP 200

    # Trafilatura extraction
    extract_text: bool = True  # Extract clean text using trafilatura
    include_comments: bool = False  # Include comments in extracted text
    include_tables: bool = True  # Include tables in extracted text
    include_links: bool = False  # Include links in extracted text

    # Output
    include_html: bool = True  # Include raw HTML
    include_payload_bytes: bool = False  # if True, include raw bytes (careful: heavy)
    max_payload_bytes: int | None = None  # if set, skip records larger than this
    decode_fallback: str = "latin-1"  # if utf-8 decode fails


class CommonCrawlWarcStreamLoader(DataLoader):
    """
    File-based streaming WARC loader.

    Two-layer architecture:
    - Layer 1 (Coordinator/Executor): get_file_list() -> scan all WARC files
    - Layer 2 (Worker): load_files() -> read assigned WARC files

    Yields dict with keys:
      - crawl_id, warc_path, warc_record_id, warc_date
      - url, http_status, content_type
      - payload_bytes, latency_ms
      - html (optional), payload (optional)
    """

    def __init__(
        self,
        crawl_id: str,
        base_url: str = "https://data.commoncrawl.org/",
        max_warc_files: int | None = None,
        max_records: int | None = None,
        request_timeout_sec: int = 300,
        retry: int = 3,
        retry_backoff_sec: float = 1.5,
        only_responses: bool = True,
        only_html: bool = True,
        require_200: bool = False,
        extract_text: bool = True,
        include_comments: bool = False,
        include_tables: bool = True,
        include_links: bool = False,
        include_html: bool = True,
        include_payload_bytes: bool = False,
        max_payload_bytes: int | None = None,
        decode_fallback: str = "latin-1",
    ):
        """Initialize CommonCrawl WARC loader.

        Args:
            crawl_id: Common Crawl ID (e.g., "CC-MAIN-2025-51")
            base_url: Base URL for Common Crawl data
            max_warc_files: Maximum number of WARC files to process
            max_records: Maximum number of records to yield per worker
            request_timeout_sec: HTTP request timeout in seconds
            retry: Number of retries for failed requests
            retry_backoff_sec: Backoff multiplier for retries
            only_responses: Only include WARC response records
            only_html: Only include HTML content
            require_200: Only include HTTP 200 responses
            extract_text: Extract clean text using trafilatura
            include_comments: Include comments in extracted text
            include_tables: Include tables in extracted text
            include_links: Include links in extracted text
            include_html: Include decoded HTML in output
            include_payload_bytes: Include raw payload bytes in output
            max_payload_bytes: Skip records with payload larger than this
            decode_fallback: Fallback encoding if UTF-8 decode fails
        """
        self.cfg = CommonCrawlConfig(
            crawl_id=crawl_id,
            base_url=base_url,
            max_warc_files=max_warc_files,
            max_records=max_records,
            request_timeout_sec=request_timeout_sec,
            retry=retry,
            retry_backoff_sec=retry_backoff_sec,
            only_responses=only_responses,
            only_html=only_html,
            require_200=require_200,
            extract_text=extract_text,
            include_comments=include_comments,
            include_tables=include_tables,
            include_links=include_links,
            include_html=include_html,
            include_payload_bytes=include_payload_bytes,
            max_payload_bytes=max_payload_bytes,
            decode_fallback=decode_fallback,
        )
        self._file_list = None

    def get_file_list(self) -> list[str]:
        """[Layer 1] Get list of all WARC file paths from warc.paths.gz.

        This is called by the coordinator to get all files before distributing
        to workers.

        Returns:
            List of WARC file paths
        """
        if self._file_list is not None:
            return self._file_list

        warc_paths = []
        max_files = self.cfg.max_warc_files

        for path in self._iter_warc_paths(self.cfg):
            warc_paths.append(path)
            if max_files is not None and len(warc_paths) >= max_files:
                break

        self._file_list = warc_paths
        print(f"[CommonCrawlLoader] Found {len(self._file_list)} WARC files for {self.cfg.crawl_id}")

        return self._file_list

    def load_files(
        self,
        file_list: list[str],
        worker_id: int | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """[Layer 2] Load assigned WARC files and yield records.

        This is called by workers to read their assigned files.

        Args:
            file_list: List of WARC file paths to read
            worker_id: Optional worker ID for logging
            checkpoint: Optional checkpoint with 'records_processed' (standard format)

        Yields:
            Records as dictionaries
        """
        worker_label = f"Worker {worker_id}" if worker_id is not None else "Loader"

        # Resume from checkpoint if provided
        skip_records = checkpoint.get("records_processed", 0) if checkpoint else 0

        if skip_records > 0:
            print(f"[{worker_label}] Resuming from record {skip_records}")

        # Process assigned files
        total_records = 0
        records_yielded = 0

        for file_idx, warc_path in enumerate(file_list):
            filename = warc_path.split("/")[-1]
            print(f"[{worker_label}] Loading WARC {file_idx + 1}/{len(file_list)}: {filename}")

            warc_url = f"{self.cfg.base_url}{warc_path}"

            for rec in self._iter_records_from_warc(self.cfg, warc_path, warc_url):
                total_records += 1

                # Skip records if resuming from checkpoint
                if total_records <= skip_records:
                    continue

                records_yielded += 1
                yield rec

                # Check max_records limit
                if self.cfg.max_records is not None and records_yielded >= self.cfg.max_records:
                    print(f"[{worker_label}] Reached max_records limit: {self.cfg.max_records}")
                    return

            print(f"[{worker_label}] Completed WARC {file_idx + 1}/{len(file_list)}, total records processed: {total_records}")

        print(f"[{worker_label}] Completed all {len(file_list)} WARC files (yielded {records_yielded} records)")

    # -------------------- internals --------------------

    def _iter_warc_paths(self, cfg: CommonCrawlConfig) -> Iterator[str]:
        url = f"{cfg.base_url}crawl-data/{cfg.crawl_id}/warc.paths.gz"
        with self._request_stream(cfg, url) as r:
            gz = gzip.GzipFile(fileobj=r.raw)
            for line in gz:
                p = line.decode("utf-8", errors="ignore").strip()
                if p:
                    yield p

    def _iter_records_from_warc(
        self, cfg: CommonCrawlConfig, warc_path: str, warc_url: str
    ) -> Iterator[dict[str, Any]]:
        with self._request_stream(cfg, warc_url) as r:
            gz = gzip.GzipFile(fileobj=r.raw)

            for record in ArchiveIterator(gz):
                t0 = time.time()

                if cfg.only_responses and record.rec_type != "response":
                    continue

                url = self._get_header(record, "WARC-Target-URI")
                warc_date = self._get_header(record, "WARC-Date")
                warc_record_id = self._get_header(record, "WARC-Record-ID")

                http_status = self._get_http_status(record)
                content_type = self._get_http_header(record, "Content-Type")

                if cfg.require_200 and http_status is not None and http_status != 200:
                    continue

                if cfg.only_html:
                    if not content_type or not content_type.lower().startswith("text/html"):
                        continue

                payload = record.content_stream().read()
                payload_len = len(payload)

                if cfg.max_payload_bytes is not None and payload_len > cfg.max_payload_bytes:
                    continue

                # Decode HTML
                html: str | None = self._decode_html(cfg, payload)
                if not html:
                    continue  # Skip if HTML decoding failed

                # Extract clean text using trafilatura
                text: str | None = None
                if cfg.extract_text:
                    text = trafilatura.extract(
                        html,
                        include_comments=cfg.include_comments,
                        include_tables=cfg.include_tables,
                        include_links=cfg.include_links,
                        output_format="txt",
                    )
                    # Skip if text extraction failed or empty
                    if not text or len(text.strip()) == 0:
                        continue

                latency_ms = (time.time() - t0) * 1000.0

                out: dict[str, Any] = {
                    "crawl_id": cfg.crawl_id,
                    "warc_path": warc_path,
                    "warc_record_id": warc_record_id,
                    "warc_date": warc_date,
                    "url": url,
                    "http_status": http_status,
                    "content_type": content_type,
                    "payload_bytes": payload_len,
                    "latency_ms": latency_ms,
                }

                # Add extracted text
                if text:
                    out["text"] = text
                    out["text_length"] = len(text)

                # Add raw HTML if requested
                if cfg.include_html:
                    out["html"] = html

                # Add raw payload if requested
                if cfg.include_payload_bytes:
                    out["payload"] = payload  # WARNING: huge

                yield out

    def _decode_html(self, cfg: CommonCrawlConfig, payload: bytes) -> str | None:
        if not payload:
            return None
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return payload.decode(cfg.decode_fallback, errors="ignore")
            except Exception:
                return None

    def _get_header(self, record, key: str) -> str | None:
        try:
            return record.rec_headers.get_header(key)
        except Exception:
            return None

    def _get_http_status(self, record) -> int | None:
        if not record.http_headers:
            return None
        try:
            statusline = record.http_headers.statusline  # e.g. "HTTP/1.1 200 OK"
            if not statusline:
                return None
            parts = statusline.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
        except Exception:
            return None
        return None

    def _get_http_header(self, record, key: str) -> str | None:
        if not record.http_headers:
            return None
        try:
            return record.http_headers.get_header(key)
        except Exception:
            return None

    def _request_stream(self, cfg: CommonCrawlConfig, url: str) -> requests.Response:
        last_exc = None
        for i in range(cfg.retry):
            try:
                r = requests.get(url, stream=True, timeout=cfg.request_timeout_sec)
                r.raise_for_status()
                r.raw.decode_content = False
                return r
            except Exception as e:
                last_exc = e
                if i < cfg.retry - 1:
                    time.sleep(cfg.retry_backoff_sec * (2**i))
        raise RuntimeError(f"Failed to stream GET: {url}") from last_exc


# ---------------------------
# Quick smoke test
# ---------------------------
if __name__ == "__main__":
    loader = CommonCrawlWarcStreamLoader(
        crawl_id="CC-MAIN-2025-51",
        max_warc_files=1,
        only_html=True,
        require_200=True,
        extract_text=True,
        include_html=False,  # Don't need raw HTML for test
        max_payload_bytes=2_000_000,  # 2MB cap (optional)
    )

    # Get file list (Layer 1)
    file_list = loader.get_file_list()
    print(f"Found {len(file_list)} WARC files")

    # Load files (Layer 2) - limit to 5 records for smoke test
    count = 0
    for rec in loader.load_files(file_list):
        count += 1
        print(f"\n--- Record {count} ---")
        print(f"URL: {rec['url']}")
        print(f"Status: {rec['http_status']}")
        print(f"Text length: {rec.get('text_length', 0)} chars")
        if rec.get("text"):
            print(f"Text preview: {rec['text'][:200]}...")

        if count >= 5:
            break
