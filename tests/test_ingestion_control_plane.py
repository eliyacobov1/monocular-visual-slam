"""Stress and race-condition checks for ingestion control plane."""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import numpy as np

from ingestion_pipeline import AsyncIngestionPipeline, IngestionPipelineConfig, QueueTuningConfig, WorkerPoolConfig


def _make_entries(count: int) -> list[tuple[int, float, Path]]:
    return [(idx, float(idx), Path(f"frame_{idx}.png")) for idx in range(count)]


def test_parallel_consumers_stress() -> None:
    def _read_fn(path: str, _backend: int) -> np.ndarray:
        idx = int(Path(path).stem.split("_")[-1])
        time.sleep(0.0005)
        return np.full((2, 2), idx, dtype=np.uint8)

    def _run_pipeline() -> list[int]:
        entries = _make_entries(200)
        pipeline = AsyncIngestionPipeline(
            entries,
            config=IngestionPipelineConfig(
                entry_queue_capacity=6,
                output_queue_capacity=6,
                num_decode_workers=3,
                fail_fast=True,
                entry_queue_tuning=QueueTuningConfig(min_capacity=4, max_capacity=12, scale_step=2),
                output_queue_tuning=QueueTuningConfig(min_capacity=4, max_capacity=12, scale_step=2),
                worker_pool=WorkerPoolConfig(min_workers=2, max_workers=4, supervisor_interval_s=0.02),
            ),
            read_fn=_read_fn,
        )
        return [packet.index for packet in pipeline]

    results_a: list[int] = []
    results_b: list[int] = []

    def _consume(target: list[int]) -> None:
        target.extend(_run_pipeline())

    threads = [
        threading.Thread(target=_consume, args=(results_a,), daemon=True),
        threading.Thread(target=_consume, args=(results_b,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert results_a == list(range(200))
    assert results_b == list(range(200))


def test_ingestion_race_condition_resilience() -> None:
    entries = _make_entries(80)
    random.seed(11)

    def _read_fn(path: str, _backend: int) -> np.ndarray:
        idx = int(Path(path).stem.split("_")[-1])
        time.sleep(random.random() * 0.002)
        return np.full((2, 2), idx, dtype=np.uint8)

    pipeline = AsyncIngestionPipeline(
        entries,
        config=IngestionPipelineConfig(
            entry_queue_capacity=4,
            output_queue_capacity=4,
            num_decode_workers=2,
            fail_fast=True,
            entry_queue_tuning=QueueTuningConfig(min_capacity=2, max_capacity=8, scale_step=2),
            output_queue_tuning=QueueTuningConfig(min_capacity=2, max_capacity=8, scale_step=2),
            worker_pool=WorkerPoolConfig(min_workers=1, max_workers=3, supervisor_interval_s=0.03),
        ),
        read_fn=_read_fn,
    )

    indices = [packet.index for packet in pipeline]

    assert indices == list(range(80))
    assert pipeline.stats.max_entry_queue_depth >= 1
