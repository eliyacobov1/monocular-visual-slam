"""Stress tests for the tracking control plane."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_control_plane import FeatureControlConfig, FeatureControlPlane
from feature_pipeline import FeaturePipelineConfig
from tracking_control_plane import TrackingControlConfig, TrackingControlPlane


def _make_frames(count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(2024)
    return [rng.integers(0, 255, size=(120, 160), dtype=np.uint8) for _ in range(count)]


def test_tracking_control_plane_handles_burst_load() -> None:
    feature_config = FeaturePipelineConfig(nfeatures=200, deterministic_seed=11)
    feature_control = FeatureControlConfig(
        enabled=True,
        num_workers=2,
        max_inflight=4,
        result_queue_capacity=16,
        reorder_buffer_size=16,
    )
    tracking_control = TrackingControlConfig(
        enabled=True,
        max_pending_frames=16,
        frame_ttl_s=2.0,
    )
    feature_plane = FeatureControlPlane(
        feature_config=feature_config,
        control_config=feature_control,
    )
    tracking_plane = TrackingControlPlane(
        feature_plane,
        config=tracking_control,
    )

    frames = _make_frames(8)
    for idx, frame in enumerate(frames):
        tracking_plane.submit_frame(seq_id=idx, timestamp=float(idx) * 0.01, frame_gray=frame)

    results = []
    for _ in frames:
        results.append(tracking_plane.collect(timeout_s=3.0))

    assert len(results) == len(frames)
    assert [result.seq_id for result in results] == sorted(result.seq_id for result in results)

    tracking_plane.close()
