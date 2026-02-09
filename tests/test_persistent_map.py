from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from persistent_map import (
    MapKeyframe,
    MapRelocalizer,
    PersistentMapStore,
    build_snapshot,
    compute_bow_histogram,
)


def _make_keyframe(frame_id: int, descriptors: np.ndarray) -> MapKeyframe:
    points = np.stack(
        [np.linspace(10, 100, descriptors.shape[0]), np.linspace(20, 120, descriptors.shape[0])],
        axis=1,
    )
    pose = np.eye(4)
    pose[:3, 3] = np.array([frame_id, 0.0, 0.0])
    return MapKeyframe(
        frame_id=frame_id,
        pose=pose,
        keypoints=points.astype(np.float32),
        descriptors=descriptors,
    )


def test_persistent_map_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    descriptors_a = rng.integers(0, 255, size=(64, 32), dtype=np.uint8)
    descriptors_b = rng.integers(0, 255, size=(64, 32), dtype=np.uint8)
    vocab = rng.normal(size=(32, 32)).astype(np.float32)
    snapshot = build_snapshot(
        [_make_keyframe(0, descriptors_a), _make_keyframe(1, descriptors_b)],
        vocab,
    )

    store = PersistentMapStore()
    store.save(tmp_path, snapshot)
    loaded = store.load(tmp_path)

    assert len(loaded.keyframes) == 2
    assert loaded.keyframes[0].frame_id == 0
    assert loaded.keyframes[1].frame_id == 1
    assert loaded.bow_vocab.shape == vocab.shape


def test_relocalizer_returns_best_candidate_without_geometry() -> None:
    rng = np.random.default_rng(3)
    vocab = rng.normal(size=(16, 32)).astype(np.float32)
    descriptors_ref = rng.normal(size=(50, 32)).astype(np.float32)
    descriptors_other = rng.normal(size=(50, 32)).astype(np.float32)

    keyframes = [
        _make_keyframe(10, descriptors_ref),
        _make_keyframe(20, descriptors_other),
    ]
    snapshot = build_snapshot(keyframes, vocab)

    query_hist = compute_bow_histogram(descriptors_ref, vocab)
    assert np.allclose(query_hist, snapshot.bow_hists[0])

    relocalizer = MapRelocalizer(
        snapshot,
        intrinsics=None,
        verify_geometry=False,
        score_threshold=0.1,
    )
    result = relocalizer.relocalize(None, descriptors_ref)

    assert result is not None
    assert result.frame_id == 10
    assert result.score >= 0.1
    assert result.match_count == 0


def test_relocalizer_requires_intrinsics_for_geometry() -> None:
    rng = np.random.default_rng(11)
    vocab = rng.normal(size=(8, 16)).astype(np.float32)
    descriptors = rng.normal(size=(20, 16)).astype(np.float32)
    snapshot = build_snapshot([_make_keyframe(0, descriptors)], vocab)

    with pytest.raises(ValueError):
        MapRelocalizer(snapshot, intrinsics=None, verify_geometry=True)
