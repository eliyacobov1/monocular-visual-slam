"""Tests for the asynchronous ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ingestion_pipeline import AsyncIngestionPipeline, IngestionPipelineConfig


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
        ),
        read_fn=_read_fn,
    )

    indices = [packet.index for packet in pipeline]

    assert indices == [0, 2]
    assert pipeline.stats.read_failures == 1
    assert pipeline.stats.dropped_frames == 1
