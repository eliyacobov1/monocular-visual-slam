"""Benchmark the async ingestion pipeline throughput and memory usage."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import numpy as np

from ingestion_pipeline import AsyncIngestionPipeline, IngestionPipelineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark async ingestion pipeline")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to decode")
    parser.add_argument("--workers", type=int, default=2, help="Decode worker count")
    parser.add_argument("--queue", type=int, default=32, help="Queue capacity")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entries = [(idx, float(idx), Path(f"frame_{idx}.png")) for idx in range(args.frames)]

    def _read_fn(path: str, _backend: int) -> np.ndarray:
        idx = int(Path(path).stem.split("_")[-1])
        return np.full((256, 256), idx % 255, dtype=np.uint8)

    pipeline = AsyncIngestionPipeline(
        entries,
        config=IngestionPipelineConfig(
            entry_queue_capacity=args.queue,
            output_queue_capacity=args.queue,
            num_decode_workers=args.workers,
            fail_fast=True,
        ),
        read_fn=_read_fn,
    )

    tracemalloc.start()
    start = time.perf_counter()
    count = 0
    for _ in pipeline:
        count += 1
    duration = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("Async ingestion benchmark")
    print(f"frames: {count}")
    print(f"duration_s: {duration:.4f}")
    print(f"throughput_fps: {count / duration:.2f}")
    print(f"memory_current_mb: {current / 1024 / 1024:.2f}")
    print(f"memory_peak_mb: {peak / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
