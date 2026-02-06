"""Benchmark control-plane supervisor throughput and memory delta."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np

from ingestion_pipeline import AsyncIngestionPipeline, IngestionPipelineConfig, QueueTuningConfig, WorkerPoolConfig


def _make_entries(count: int) -> list[tuple[int, float, Path]]:
    return [(idx, float(idx), Path(f"frame_{idx}.png")) for idx in range(count)]


def _read_fn(path: str, _backend: int) -> np.ndarray:
    idx = int(Path(path).stem.split("_")[-1])
    return np.full((4, 4), idx, dtype=np.uint8)


def run_benchmark(frame_count: int = 1000) -> None:
    entries = _make_entries(frame_count)
    config = IngestionPipelineConfig(
        entry_queue_capacity=16,
        output_queue_capacity=16,
        num_decode_workers=4,
        fail_fast=True,
        entry_queue_tuning=QueueTuningConfig(min_capacity=8, max_capacity=64, scale_step=4),
        output_queue_tuning=QueueTuningConfig(min_capacity=8, max_capacity=64, scale_step=4),
        worker_pool=WorkerPoolConfig(min_workers=2, max_workers=6, supervisor_interval_s=0.02),
        supervisor_ema_alpha=0.35,
    )

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    pipeline = AsyncIngestionPipeline(entries, config=config, read_fn=_read_fn)
    results = [packet.index for packet in pipeline]
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    entry_metrics = pipeline.telemetry.stage_metrics("entry")
    decode_metrics = pipeline.telemetry.stage_metrics("decode")
    output_metrics = pipeline.telemetry.stage_metrics("output")

    print("control_plane_benchmark")
    print(f"frames={frame_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"throughput_fps={frame_count / duration_s:.2f}")
    print(f"ordered={results == list(range(frame_count))}")
    print(f"entry_scale_ups={entry_metrics.queue_scale_ups}")
    print(f"decode_circuit_opens={decode_metrics.circuit_breaker_opens}")
    print(f"output_backpressure={output_metrics.backpressure_events}")


if __name__ == "__main__":
    run_benchmark()
