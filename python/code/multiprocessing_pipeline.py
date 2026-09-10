"""Multiprocessing pipeline demo: chunk -> parallel process -> merge.

Anchor example for ../concurrency-async.md (multiprocessing section).
Demonstrates the canonical CPU-bound parallelism pattern in Python: split
work into independent chunks, farm them out to worker processes with
multiprocessing.Pool (each worker has its own interpreter and GIL, so this
is real parallelism, unlike threading), then merge the partial results back
in the parent process.

Adapted from a university image-processing lab that used this exact
chunk/pool.map/merge shape to analyze image regions in parallel. The
OpenCV/Tkinter/openpyxl scaffolding has been stripped out and the payload
replaced with a generic CPU-bound "scan a block of numbers" task so this
file has zero third-party dependencies and runs anywhere.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass


@dataclass
class ChunkResult:
    chunk_id: int
    count_above_threshold: int
    chunk_sum: float


def process_chunk(args: tuple[int, list[float], float]) -> ChunkResult:
    """CPU-bound worker function.

    Must be a top-level function (not a lambda or closure) because
    multiprocessing pickles it to send to each worker process.
    """
    chunk_id, chunk, threshold = args
    count = sum(1 for value in chunk if value > threshold)
    return ChunkResult(chunk_id, count, sum(chunk))


def split_into_chunks(data: list[float], chunk_size: int) -> list[list[float]]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def run_pipeline(
    data: list[float],
    threshold: float,
    chunk_size: int = 1000,
    processes: int | None = None,
) -> ChunkResult:
    """Chunk -> pool.map (parallel processing) -> merge partial results."""
    chunks = split_into_chunks(data, chunk_size)
    tasks = [(idx, chunk, threshold) for idx, chunk in enumerate(chunks)]

    with mp.Pool(processes=processes) as pool:
        results = pool.map(process_chunk, tasks)

    total_count = sum(r.count_above_threshold for r in results)
    total_sum = sum(r.chunk_sum for r in results)
    return ChunkResult(chunk_id=-1, count_above_threshold=total_count, chunk_sum=total_sum)


if __name__ == "__main__":
    import random

    data = [random.uniform(0, 100) for _ in range(500_000)]

    start = time.perf_counter()
    merged = run_pipeline(data, threshold=90.0, chunk_size=10_000, processes=mp.cpu_count())
    elapsed = time.perf_counter() - start

    print(f"Values above threshold: {merged.count_above_threshold}")
    print(f"Sum of all values: {merged.chunk_sum:.2f}")
    print(f"Elapsed: {elapsed:.3f}s across {mp.cpu_count()} processes")
