#!/usr/bin/env python3
"""Test CommonCrawlWarcStreamLoader file assignment logic."""

from mega_data_factory.loaders import CommonCrawlWarcStreamLoader

# Create loader
loader = CommonCrawlWarcStreamLoader(
    crawl_id="CC-MAIN-2025-51",
    max_warc_files=20,  # Scan 20 WARC files
    only_html=True,
    require_200=True,
    extract_text=True,
    include_html=False,
)

# Get file list (Layer 1)
file_list = loader.get_file_list()
print(f"✅ Total WARC files: {len(file_list)}")
print(f"   First file: {file_list[0].split('/')[-1]}")
print(f"   Last file: {file_list[-1].split('/')[-1]}")
print()

# Test file assignment for 4 workers (simulating Executor)
num_workers = 4
print(f"File assignment for {num_workers} workers:")
print()

total_files = len(file_list)
files_per_worker = total_files // num_workers
remainder = total_files % num_workers

for worker_id in range(num_workers):
    if worker_id < remainder:
        start_file = worker_id * (files_per_worker + 1)
        end_file = start_file + files_per_worker + 1
    else:
        start_file = worker_id * files_per_worker + remainder
        end_file = start_file + files_per_worker

    num_files = end_file - start_file
    print(f"Worker {worker_id}: files {start_file:3d}-{end_file - 1:3d} ({num_files:2d} files)")

# Verify no gaps or overlaps
print()
total_assigned = sum([
    (files_per_worker + 1 if i < remainder else files_per_worker)
    for i in range(num_workers)
])
print(f"✅ Total assigned: {total_assigned} files (should be {total_files})")
