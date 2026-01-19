"""
DedupBackend: Distributed deduplication state using Ray Actor

Provides deduplication backend with bucketing.
"""

from collections.abc import Callable

import ray


@ray.remote
class _DedupBackendBucketActor:
    """Ray Actor for a single deduplication bucket.

    Each actor maintains only one bucket's seen set.
    This distributes memory across multiple actors.
    """

    def __init__(self, bucket_id: int):
        self.bucket_id = bucket_id
        self.seen: set = set()

    def is_seen(self, key: str) -> bool:
        return key in self.seen

    def mark_seen(self, key: str) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

    def batch_mark_seen(self, keys: list[str]) -> list[bool]:
        results = []
        for key in keys:
            if key in self.seen:
                results.append(False)
            else:
                self.seen.add(key)
                results.append(True)
        return results

    def reset(self):
        self.seen.clear()


class DedupBackend:
    """Distributed deduplication backend with bucketing.

    Each bucket is maintained by a separate Ray Actor.
    Keys are routed to specific bucket actors, distributing memory across actors.

    Performance guidelines:
    - For small datasets (<1B keys): 16-64 buckets is sufficient
    - For medium datasets (1B-10B keys): 256-1000 buckets recommended
    - For large datasets (10B-100B keys): 1000-10000 buckets recommended
    - Target: Keep ~10M-100M keys per bucket for optimal performance

    Extension: For semantic deduplication, you can use cluster_id as bucket_id:
        - Cluster records first (using a refiner)
        - Pass cluster_id to get_dedup_key() or use custom bucket_id_getter
        - Each cluster bucket handles deduplication independently
    """

    def __init__(
        self,
        num_buckets: int = 2,
        name_prefix: str = "pipeline_dedup_backend",
        bucket_id_getter: Callable[[str], int] | None = None,
    ):
        """Initialize deduplication backend with bucketing.

        Args:
            num_buckets: Number of bucket actors to create (default: 16)
                        Increase for large datasets to distribute memory load.
                        See class docstring for performance guidelines.
            name_prefix: Prefix for Ray Actor names (for Ray Dashboard visibility)
            bucket_id_getter: Optional function(key: str) -> int to compute bucket_id.
                             If None, uses hash(key) % num_buckets.
        """
        self.num_buckets = num_buckets
        self.bucket_id_getter = bucket_id_getter
        self.name_prefix = name_prefix

        # Create one actor per bucket (or reuse existing ones)
        self.bucket_actors = []
        for bucket_id in range(num_buckets):
            actor_name = f"{name_prefix}_bucket_{bucket_id}"
            try:
                actor = ray.get_actor(actor_name)
            except ValueError:
                try:
                    actor = _DedupBackendBucketActor.options(name=actor_name).remote(bucket_id)
                except ray.exceptions.ActorAlreadyExistsError:
                    actor = ray.get_actor(actor_name)
            self.bucket_actors.append(actor)

    def _get_bucket_id(self, key: str) -> int:
        """Get bucket ID for a given key."""
        if self.bucket_id_getter:
            return self.bucket_id_getter(key) % self.num_buckets
        return hash(key) % self.num_buckets

    def _get_actor(self, key: str):
        """Get the bucket actor for a given key."""
        bucket_id = self._get_bucket_id(key)
        return self.bucket_actors[bucket_id]

    def is_seen(self, key: str) -> bool:
        """Check if key has been seen."""
        actor = self._get_actor(key)
        return ray.get(actor.is_seen.remote(key))

    def mark_seen(self, key: str) -> bool:
        """Mark key as seen."""
        actor = self._get_actor(key)
        return ray.get(actor.mark_seen.remote(key))

    def batch_mark_seen(self, keys: list[str]) -> list[bool]:
        """Batch mark multiple keys as seen (grouped by bucket for efficiency)."""
        # Group keys by bucket
        bucket_keys: dict[int, list[tuple]] = {}  # bucket_id -> [(original_index, key), ...]
        for idx, key in enumerate(keys):
            bucket_id = self._get_bucket_id(key)
            if bucket_id not in bucket_keys:
                bucket_keys[bucket_id] = []
            bucket_keys[bucket_id].append((idx, key))

        # Process each bucket in parallel
        futures = {}
        for bucket_id, key_list in bucket_keys.items():
            actor = self.bucket_actors[bucket_id]
            bucket_keys_only = [k for _, k in key_list]
            futures[bucket_id] = (actor.batch_mark_seen.remote(bucket_keys_only), key_list)

        # Collect results and reconstruct original order
        results = [False] * len(keys)
        for bucket_id, (future, key_list) in futures.items():
            bucket_results = ray.get(future)
            for (orig_idx, _), result in zip(key_list, bucket_results):
                results[orig_idx] = result

        return results

    def reset(self):
        """Reset all bucket states."""
        futures = [actor.reset.remote() for actor in self.bucket_actors]
        ray.get(futures)
