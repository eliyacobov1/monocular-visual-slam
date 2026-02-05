"""Tests for the asynchronous ingestion pipeline."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from ingestion_pipeline import (
    AsyncIngestionPipeline,
    CircuitBreakerConfig,
    IngestionPipelineConfig,
    OrderingBufferConfig,
    QueueTuningConfig,
    RetryPolicyConfig,
    WorkerPoolConfig,
)


def _make_entries(count: int) -> list[tuple[int, float, Path]]:
    return [(idx, float(idx), Path(f"frame_{idx}.png")) for idx in range(count)]


def test_ingestion_pipeline_orders_frames() -> None:
    entries = _make_entries(5)

    def _read_fn(path: str, _backend: int) -> np.ndarray:
        idx = int(Path(path).stem.split("_")[-1])
        return np.full((2, 2), idx, dtype=np.uint8)

    pipeline = AsyncIngestionPipeline(
        entries,
        config=IngestionPipelineConfig(
            entry_queue_capacity=2,
            output_queue_capacity=2,
            num_decode_workers=2,
            fail_fast=True,
            entry_queue_tuning=QueueTuningConfig(min_capacity=1, max_capacity=4, scale_step=1),
            output_queue_tuning=QueueTuningConfig(min_capacity=1, max_capacity=4, scale_step=1),
            worker_pool=WorkerPoolConfig(min_workers=2, max_workers=2, supervisor_interval_s=0.05),
            ordering_buffer=OrderingBufferConfig(max_pending=10, forced_flush_ratio=0.5),
        ),
        read_fn=_read_fn,
    )

    indices = [packet.index for packet in pipeline]

    assert indices == [0, 1, 2, 3, 4]
    assert pipeline.stats.decoded_frames == 5
    assert pipeline.stats.dropped_frames == 0


def test_ingestion_pipeline_handles_failures() -> None:
    entries = _make_entries(3)

    def _read_fn(path: str, _backend: int) -> np.ndarray | None:
        if path.endswith("frame_1.png"):
            return None
        idx = int(Path(path).stem.split("_")[-1])
        return np.full((2, 2), idx, dtype=np.uint8)

    pipeline = AsyncIngestionPipeline(
        entries,
        config=IngestionPipelineConfig(
            entry_queue_capacity=2,
            output_queue_capacity=2,
            num_decode_workers=1,
            fail_fast=False,
            entry_queue_tuning=QueueTuningConfig(min_capacity=1, max_capacity=4, scale_step=1),
            output_queue_tuning=QueueTuningConfig(min_capacity=1, max_capacity=4, scale_step=1),
            worker_pool=WorkerPoolConfig(min_workers=1, max_workers=1, supervisor_interval_s=0.05),
            retry_policy=RetryPolicyConfig(max_attempts=1, backoff_s=0.0, jitter_s=0.0),
        ),
        read_fn=_read_fn,
    )

    indices = [packet.index for packet in pipeline]

    assert indices == [0, 2]
    assert pipeline.stats.read_failures == 1
    assert pipeline.stats.dropped_frames == 1
    assert pipeline.failures.counts["read"] == 1


def test_ingestion_pipeline_scales_workers_and_queues() -> None:
    entries = _make_entries(25)

    def _read_fn(path: str, _backend: int) -> np.ndarray:
        time.sleep(0.005)
        idx = int(Path(path).stem.split("_")[-1])
        return np.full((2, 2), idx, dtype=np.uint8)

    pipeline = AsyncIngestionPipeline(
        entries,
        config=IngestionPipelineConfig(
            entry_queue_capacity=2,
            output_queue_capacity=2,
            num_decode_workers=1,
            fail_fast=True,
            entry_queue_tuning=QueueTuningConfig(
                min_capacity=2, max_capacity=6, scale_up_ratio=0.4, scale_down_ratio=0.1, scale_step=2
            ),
            output_queue_tuning=QueueTuningConfig(
                min_capacity=2, max_capacity=6, scale_up_ratio=0.4, scale_down_ratio=0.1, scale_step=2
            ),
            worker_pool=WorkerPoolConfig(
                min_workers=1, max_workers=3, scale_up_ratio=0.4, scale_down_ratio=0.1, supervisor_interval_s=0.05
            ),
        ),
        read_fn=_read_fn,
    )

    indices = [packet.index for packet in pipeline]

    assert indices == list(range(25))
    assert pipeline.telemetry.worker_scale_ups >= 1
    assert pipeline.telemetry.queue_scale_ups >= 1


def test_ingestion_pipeline_stress_shutdown() -> None:
    entries = _make_entries(120)

    def _read_fn(path: str, _backend: int) -> np.ndarray:
        idx = int(Path(path).stem.split("_")[-1])
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
            worker_pool=WorkerPoolConfig(min_workers=2, max_workers=4, supervisor_interval_s=0.05),
        ),
        read_fn=_read_fn,
    )

    results: list[int] = []

    def _consume() -> None:
        for packet in pipeline:
            results.append(packet.index)

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert results == list(range(120))
    assert pipeline.stats.decoded_frames == 120


def test_ingestion_circuit_breaker_opens() -> None:
    entries = _make_entries(6)

    def _read_fn(_path: str, _backend: int) -> np.ndarray | None:
        return None

    pipeline = AsyncIngestionPipeline(
        entries,
        config=IngestionPipelineConfig(
            entry_queue_capacity=2,
            output_queue_capacity=2,
            num_decode_workers=1,
            fail_fast=False,
            entry_queue_tuning=QueueTuningConfig(min_capacity=1, max_capacity=4, scale_step=1),
            output_queue_tuning=QueueTuningConfig(min_capacity=1, max_capacity=4, scale_step=1),
            retry_policy=RetryPolicyConfig(max_attempts=1, backoff_s=0.0, jitter_s=0.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=1.0, half_open_successes=1),
        ),
        read_fn=_read_fn,
    )

    list(pipeline)

    assert pipeline.telemetry.circuit_breaker_opens >= 1
