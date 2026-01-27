"""
Text Exact Deduplication

Deduplicates text records based on exact content hash (MD5/xxhash).
Fast and memory-efficient for removing exact duplicates from large text corpora.
"""

import hashlib
from typing import Any

from mega_data_factory.framework import Deduplicator

# Try to use xxhash for faster hashing (optional dependency)
try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False


class TextExactDeduplicator(Deduplicator):
    """Deduplicates text records based on exact content hash.

    Uses xxhash (if available) or MD5 to compute content fingerprint.
    Exact duplicates are removed, keeping only the first occurrence.

    Reference:
    - RefinedWeb (arXiv:2306.01116) Section G.3: Exact deduplication
    - FineWeb: URL + content hash deduplication
    """

    def __init__(
        self,
        text_field: str = "text",
        hash_algorithm: str = "auto",
        normalize_whitespace: bool = True,
        lowercase: bool = True,
        include_url: bool = False,
        url_field: str = "url",
    ):
        """Initialize text exact deduplicator.

        Args:
            text_field: Name of the text field to hash.
            hash_algorithm: Hash algorithm: "auto", "xxhash", "md5", "sha256".
                           "auto" uses xxhash if available, else md5.
            normalize_whitespace: Collapse multiple whitespace to single space.
            lowercase: Convert text to lowercase before hashing.
            include_url: Include URL in hash for URL+content dedup (FineWeb style).
            url_field: Name of the URL field if include_url is True.
        """
        super().__init__()
        self.text_field = text_field
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.include_url = include_url
        self.url_field = url_field

        # Select hash algorithm
        if hash_algorithm == "auto":
            self.hash_algorithm = "xxhash" if XXHASH_AVAILABLE else "md5"
        else:
            self.hash_algorithm = hash_algorithm

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing."""
        if self.normalize_whitespace:
            text = " ".join(text.split())
        if self.lowercase:
            text = text.lower()
        return text

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content string."""
        data = content.encode("utf-8", errors="ignore")

        if self.hash_algorithm == "xxhash" and XXHASH_AVAILABLE:
            return xxhash.xxh64(data).hexdigest()
        elif self.hash_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        else:  # md5 (default fallback)
            return hashlib.md5(data).hexdigest()

    def get_dedup_keys_batch(self, records: list[dict[str, Any]]) -> list[str]:
        """Compute content hashes for a batch of records.

        Returns:
            List of hash strings, one per record. Empty string for invalid records.
        """
        keys = []

        for record in records:
            text = record.get(self.text_field)

            if not text or not isinstance(text, str):
                # Use record ID or empty key for invalid records
                keys.append(record.get("id", ""))
                continue

            # Normalize text
            normalized = self._normalize_text(text)

            # Optionally include URL for URL+content dedup
            if self.include_url:
                url = record.get(self.url_field, "")
                if url:
                    normalized = f"{url}|{normalized}"

            # Compute hash
            content_hash = self._compute_hash(normalized)
            keys.append(content_hash)

        return keys
