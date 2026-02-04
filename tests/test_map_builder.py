from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("sklearn")

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2

from keyframe_manager import Keyframe
from map_builder import MapBuilderConfig, MapSnapshotBuilder


def _make_keyframe(frame_id: int, descriptors: np.ndarray) -> Keyframe:
    keypoints = [
        cv2.KeyPoint(float(x), float(y), 1.0)
        for x, y in zip(
            np.linspace(10, 100, descriptors.shape[0]),
            np.linspace(20, 120, descriptors.shape[0]),
        )
    ]
    pose = np.eye(4)
    pose[:3, 3] = np.array([frame_id, 0.0, 0.0])
    return Keyframe(
        frame_id=frame_id,
        pose=pose,
        keypoints=keypoints,
        descriptors=descriptors,
    )


def test_map_snapshot_builder_builds_vocab() -> None:
    rng = np.random.default_rng(42)
    descriptors_a = rng.normal(size=(120, 32)).astype(np.float32)
    descriptors_b = rng.normal(size=(90, 32)).astype(np.float32)

    builder = MapSnapshotBuilder(MapBuilderConfig(vocab_size=32, max_descriptors=150))
    snapshot, stats = builder.build_snapshot(
        [
            _make_keyframe(0, descriptors_a),
            _make_keyframe(1, descriptors_b),
        ]
    )

    assert len(snapshot.keyframes) == 2
    assert snapshot.bow_vocab.shape[1] == 32
    assert snapshot.bow_vocab.shape[0] <= 32
    assert stats.num_keyframes == 2
    assert stats.total_descriptors == 210
    assert stats.sampled_descriptors <= stats.total_descriptors
