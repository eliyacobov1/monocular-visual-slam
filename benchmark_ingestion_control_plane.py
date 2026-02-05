"""Benchmark the adaptive ingestion control plane."""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

import numpy as np

from ingestion_pipeline import (
    AsyncIngestionPipeline,
    IngestionPipelineConfig,
    QueueTuningConfig,
    WorkerPoolConfig,
)


def _make_entries(count: int) -> Iterable[tuple[int, float, Path]]:
    return [(idx, float(idx), Path(f"frame_{idx}.png")) for idx in range(count)]


def _read_fn(path: str, _backend: int) -> np.ndarray:
    idx = int(Path(path).stem.split("_")[-1])
    return np.full((32, 32, 3), idx % 255, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ingestion control plane")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to ingest")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    args = parser.parse_args()

    durations: list[float] = []
    memory_deltas: list[int] = []

    for _ in range(args.runs):
        entries = _make_entries(args.frames)
        pipeline = AsyncIngestionPipeline(
            entries,
            config=IngestionPipelineConfig(
                entry_queue_capacity=8,
                output_queue_capacity=8,
                num_decode_workers=2,
                fail_fast=True,
                entry_queue_tuning=QueueTuningConfig(min_capacity=4, max_capacity=32, scale_step=4),
                output_queue_tuning=QueueTuningConfig(min_capacity=4, max_capacity=32, scale_step=4),
                worker_pool=WorkerPoolConfig(min_workers=2, max_workers=6, supervisor_interval_s=0.05),
            ),
            read_fn=_read_fn,
        )
        tracemalloc.start()
        start = time.perf_counter()
        for _ in pipeline:
            pass
        duration = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        durations.append(duration)
        memory_deltas.append(peak - current)

    print("Ingestion Control Plane Benchmark")
    print(f"Runs: {args.runs}")
    print(f"Frames per run: {args.frames}")
    print(f"Duration (s) avg: {statistics.mean(durations):.4f}")
    print(f"Duration (s) p95: {statistics.quantiles(durations, n=20)[-1]:.4f}")
    print(f"Memory delta (bytes) avg: {int(statistics.mean(memory_deltas))}")


if __name__ == "__main__":
    main()
