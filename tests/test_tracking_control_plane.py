"""Tests for the tracking control plane."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_control_plane import FeatureControlConfig, FeatureControlPlane
from feature_pipeline import FeaturePipelineConfig
from tracking_control_plane import (
    FramePayload,
    PendingFrameBuffer,
    TrackingControlConfig,
    TrackingControlPlane,
)


def _make_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160), dtype=np.uint8)


def test_pending_frame_buffer_drops_oldest() -> None:
    buffer = PendingFrameBuffer(capacity=1, frame_ttl_s=1.0, drop_policy="drop_oldest")
    now_s = time.perf_counter()
    payload_a = FramePayload(
        seq_id=0,
        timestamp=0.0,
        frame_gray=_make_frame(0),
        enqueued_at_s=now_s,
        deadline_s=now_s + 1.0,
    )
    payload_b = FramePayload(
        seq_id=1,
        timestamp=0.1,
        frame_gray=_make_frame(1),
        enqueued_at_s=now_s + 0.01,
        deadline_s=now_s + 1.01,
    )

    dropped = buffer.add(payload_a)
    assert dropped == []

    dropped = buffer.add(payload_b)
    assert len(dropped) == 1
    assert dropped[0].seq_id == 0
    assert buffer.size == 1


def test_pending_frame_buffer_expires_frames() -> None:
    buffer = PendingFrameBuffer(capacity=2, frame_ttl_s=0.05, drop_policy="drop_oldest")
    now_s = time.perf_counter()
    payload = FramePayload(
        seq_id=0,
        timestamp=0.0,
        frame_gray=_make_frame(2),
        enqueued_at_s=now_s,
        deadline_s=now_s + 0.02,
    )
    _ = buffer.add(payload)
    expired = buffer.expire(now_s + 0.03)

    assert len(expired) == 1
    assert expired[0].seq_id == 0


def test_tracking_control_plane_orders_results() -> None:
    feature_config = FeaturePipelineConfig(nfeatures=200, deterministic_seed=42)
    feature_control = FeatureControlConfig(
        enabled=True,
        num_workers=2,
        max_inflight=2,
        result_queue_capacity=4,
        reorder_buffer_size=4,
    )
    tracking_control = TrackingControlConfig(
        enabled=True,
        max_pending_frames=4,
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
    frame_a = _make_frame(3)
    frame_b = _make_frame(4)

    tracking_plane.submit_frame(seq_id=0, timestamp=0.0, frame_gray=frame_a)
    tracking_plane.submit_frame(seq_id=1, timestamp=0.1, frame_gray=frame_b)

    result_a = tracking_plane.collect(timeout_s=2.0)
    result_b = tracking_plane.collect(timeout_s=2.0)

    assert result_a.seq_id == 0
    assert result_b.seq_id == 1
    assert result_a.drop_reason is None
    assert result_b.drop_reason is None
    assert result_a.feature_result is not None
    assert result_b.feature_result is not None

    tracking_plane.close()
