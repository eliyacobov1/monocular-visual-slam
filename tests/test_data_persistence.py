"""Tests for the data persistence layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_persistence import (
    FrameDiagnosticsEntry,
    RunDataStore,
    build_frame_diagnostics_bundle,
    build_metrics_bundle,
    frame_diagnostics_artifact_path,
    load_frame_diagnostics_json,
    load_trajectory_npz,
    summarize_frame_diagnostics,
    summarize_frame_diagnostics_streaming,
    trajectory_artifact_path,
    trajectory_positions,
)


def test_run_data_store_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_id": "unit"}), encoding="utf-8")

    store = RunDataStore.create(
        base_dir=tmp_path,
        run_id="unit_run",
        config_path=config_path,
        config_hash="hash",
        use_subdir=False,
        resolved_config={"run_id": "unit_run"},
    )

    accumulator = store.create_accumulator("trajectory")
    accumulator.append(np.eye(4), 0.0, 0)
    accumulator.append(np.eye(4), 1.0, 1)
    bundle = accumulator.as_bundle()

    store.save_trajectory(bundle)
    loaded = store.load_trajectory("trajectory")
    assert loaded.poses.shape == (2, 4, 4)
    assert loaded.timestamps.tolist() == [0.0, 1.0]

    metrics_bundle = build_metrics_bundle("metrics", {"num_poses": 2})
    store.save_metrics(metrics_bundle)
    loaded_metrics = store.load_metrics("metrics")
    assert loaded_metrics.metrics["num_poses"] == 2.0

    diagnostics_bundle = build_frame_diagnostics_bundle(
        "frame_diagnostics",
        [
            FrameDiagnosticsEntry(
                frame_id=0,
                timestamp=0.0,
                match_count=0,
                inliers=0,
                method="bootstrap",
                inlier_ratio=0.0,
                median_parallax=0.0,
                score=0.0,
                status="bootstrap",
                failure_reason=None,
            ),
            FrameDiagnosticsEntry(
                frame_id=1,
                timestamp=1.0,
                match_count=10,
                inliers=8,
                method="essential",
                inlier_ratio=0.8,
                median_parallax=1.5,
                score=1.2,
                status="ok",
                failure_reason=None,
            ),
        ],
    )
    store.save_frame_diagnostics(diagnostics_bundle)
    loaded_diagnostics = store.load_frame_diagnostics("frame_diagnostics")
    assert len(loaded_diagnostics.entries) == 2
    assert loaded_diagnostics.entries[1].method == "essential"


def test_load_trajectory_npz_positions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_id": "unit"}), encoding="utf-8")

    store = RunDataStore.create(
        base_dir=tmp_path,
        run_id="unit_run",
        config_path=config_path,
        config_hash="hash",
        use_subdir=False,
        resolved_config={"run_id": "unit_run"},
    )

    accumulator = store.create_accumulator("trajectory_npz")
    pose_a = np.eye(4)
    pose_b = np.eye(4)
    pose_b[0, 3] = 2.0
    pose_b[1, 3] = 1.0
    accumulator.append(pose_a, 0.0, 0)
    accumulator.append(pose_b, 1.0, 1)
    bundle = accumulator.as_bundle()

    trajectory_path = store.save_trajectory(bundle)
    loaded = load_trajectory_npz(trajectory_path)
    positions = trajectory_positions(loaded, [0, 1])

    assert positions.shape == (2, 2)
    assert positions[1, 0] == 2.0
    assert positions[1, 1] == 1.0


def test_trajectory_artifact_path_sanitizes_name(tmp_path: Path) -> None:
    path = trajectory_artifact_path(tmp_path, "slam run/01")

    assert path.parent == tmp_path / "trajectories"
    assert path.name == "slam_run_01.npz"


def test_frame_diagnostics_roundtrip_and_summary(tmp_path: Path) -> None:
    diagnostics_path = frame_diagnostics_artifact_path(tmp_path, "frame_diagnostics")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "frame_diagnostics",
        "recorded_at": "2024-01-01T00:00:00Z",
        "entries": [
            {
                "frame_id": 0,
                "timestamp": 0.0,
                "match_count": 12,
                "inliers": 9,
                "method": "essential",
                "inlier_ratio": 0.75,
                "median_parallax": 1.2,
                "score": 0.9,
                "status": "ok",
                "failure_reason": None,
            },
            {
                "frame_id": 1,
                "timestamp": 1.0,
                "match_count": 10,
                "inliers": 6,
                "method": "homography",
                "inlier_ratio": 0.6,
                "median_parallax": 0.8,
                "score": 0.7,
                "status": "ok",
                "failure_reason": None,
            },
        ],
    }
    diagnostics_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = load_frame_diagnostics_json(diagnostics_path)
    summary = summarize_frame_diagnostics(bundle)
    streaming_summary = summarize_frame_diagnostics_streaming(diagnostics_path)

    assert bundle.name == "frame_diagnostics"
    assert summary["diag_frame_count"] == 2.0
    assert summary["diag_method_essential_count"] == 1.0
    assert summary["diag_method_homography_ratio"] == 0.5
    assert streaming_summary["diag_frame_count"] == 2.0
    assert streaming_summary["diag_method_essential_count"] == 1.0
    assert streaming_summary["diag_method_homography_ratio"] == 0.5
