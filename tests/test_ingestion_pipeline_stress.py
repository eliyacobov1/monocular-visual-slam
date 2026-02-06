"""Stress tests for the asynchronous ingestion pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ingestion_pipeline import AsyncIngestionPipeline, IngestionPipelineConfig, QueueTuningConfig, WorkerPoolConfig


def _make_entries(count: int) -> list[tuple[int, float, Path]]:
    return [(idx, float(idx), Path(f"frame_{idx}.png")) for idx in range(count)]


def _read_fn(path: str, _backend: int) -> np.ndarray:
    idx = int(Path(path).stem.split("_")[-1])
    return np.full((4, 4), idx, dtype=np.uint8)


def test_ingestion_pipeline_stress_order_and_telemetry() -> None:
    entries = _make_entries(200)
    config = IngestionPipelineConfig(
        entry_queue_capacity=8,
        output_queue_capacity=8,
        num_decode_workers=4,
        inflight_limit=8,
        fail_fast=True,
        entry_queue_tuning=QueueTuningConfig(min_capacity=4, max_capacity=16, scale_step=2),
        output_queue_tuning=QueueTuningConfig(min_capacity=4, max_capacity=16, scale_step=2),
        worker_pool=WorkerPoolConfig(min_workers=2, max_workers=4, supervisor_interval_s=0.01),
        supervisor_ema_alpha=0.4,
    )
    pipeline = AsyncIngestionPipeline(entries, config=config, read_fn=_read_fn)
    ordered = [packet.index for packet in pipeline]

    assert ordered == list(range(len(entries)))
    assert pipeline.stats.decoded_frames == len(entries)
    assert "entry" in pipeline.telemetry.stages
    assert "output" in pipeline.telemetry.stages
    assert "decode" in pipeline.telemetry.stages
