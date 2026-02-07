"""Tests for the feature control plane."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_control_plane import FeatureControlConfig, FeatureControlPlane
from feature_pipeline import FeaturePipelineConfig


def _make_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160), dtype=np.uint8)


def test_feature_control_plane_orders_results() -> None:
    feature_config = FeaturePipelineConfig(nfeatures=200, deterministic_seed=42)
    control_config = FeatureControlConfig(
        enabled=True,
        num_workers=2,
        max_inflight=2,
        result_queue_capacity=4,
        reorder_buffer_size=4,
    )
    plane = FeatureControlPlane(
        feature_config=feature_config,
        control_config=control_config,
    )
    frame_a = _make_frame(1)
    frame_b = _make_frame(2)

    plane.submit(seq_id=0, timestamp=0.0, frame=frame_a)
    plane.submit(seq_id=1, timestamp=0.1, frame=frame_b)

    result_a = plane.collect(timeout_s=2.0)
    result_b = plane.collect(timeout_s=2.0)

    assert result_a.seq_id == 0
    assert result_b.seq_id == 1
    assert result_a.keypoints is not None
    assert result_b.keypoints is not None

    plane.close()


def test_feature_control_plane_cache_hits() -> None:
    feature_config = FeaturePipelineConfig(nfeatures=200, deterministic_seed=7)
    control_config = FeatureControlConfig(
        enabled=True,
        num_workers=1,
        max_inflight=1,
        result_queue_capacity=2,
        reorder_buffer_size=2,
    )
    plane = FeatureControlPlane(
        feature_config=feature_config,
        control_config=control_config,
    )
    frame = _make_frame(3)

    plane.submit(seq_id=0, timestamp=0.0, frame=frame)
    _ = plane.collect(timeout_s=2.0)
    plane.submit(seq_id=1, timestamp=0.1, frame=frame)
    cached_result = plane.collect(timeout_s=2.0)

    assert cached_result.cache_hit is True

    plane.close()
