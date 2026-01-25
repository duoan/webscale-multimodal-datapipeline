#!/usr/bin/env python3
"""Test DataLoaderWorker performance improvements."""

import time
from mega_data_factory.loaders import HuggingFaceDataLoader
from mega_data_factory.framework.loader_worker import DataLoaderWorker
import ray

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create data loader
loader = HuggingFaceDataLoader(
    dataset_name="jp1924/Laion400m-1",
    split="train",
    streaming=True,
)

# Create worker with iterator refresh every 5 batches
print("Creating DataLoaderWorker...")
worker = DataLoaderWorker.remote(
    data_loader=loader,
    shard_id=0,
    num_shards=4,
    batch_size=100,
    checkpoint_interval=500,
    iterator_refresh_interval=5,  # Refresh every 5 batches
)

print("\nFetching 10 batches to test performance...")
print("(Watch for 'Initializing data stream' and 'Refreshing iterator' messages)\n")

start_time = time.time()
for i in range(10):
    batch_start = time.time()
    result = ray.get(worker.get_next_batch.remote(max_records=2000))
    batch_time = time.time() - batch_start

    if result["batch"]:
        print(
            f"Batch {i+1}: {len(result['batch'])} records, "
            f"time: {batch_time:.2f}s, "
            f"total: {result['records_processed']}"
        )
    else:
        print(f"Batch {i+1}: Completed")
        break

total_time = time.time() - start_time
print(f"\n✅ Total time: {total_time:.2f}s")
print(f"📊 Expected: 2 stream initializations (initial + 1 refresh after 5 batches)")

ray.shutdown()
