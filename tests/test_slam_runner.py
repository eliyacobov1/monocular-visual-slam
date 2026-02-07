"""Tests for the SLAM runner CLI helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

from slam_runner import load_pipeline_config, run_kitti_sequence


def _write_dummy_kitti_dataset(root: Path) -> Path:
    sequence_dir = root / "sequences" / "00"
    image_dir = sequence_dir / "image_2"
    image_dir.mkdir(parents=True)
    for idx in range(2):
        frame = np.full((8, 8, 3), idx * 20, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"{idx:06d}.png"), frame)
    (sequence_dir / "times.txt").write_text("0.0\n0.1\n", encoding="utf-8")
    calib_content = "P2: 700 0 320 0 0 700 240 0 0 0 1 0\n"
    (sequence_dir / "calib.txt").write_text(calib_content, encoding="utf-8")
    return root


def test_load_pipeline_config_accepts_valid_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.json"
    payload = {
        "feature_config": {"nfeatures": 100},
        "pose_config": {"min_matches": 10},
        "feature_control": {"enabled": True, "num_workers": 1, "max_inflight": 2},
        "tracking_control": {"enabled": True, "max_pending_frames": 8},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    feature_config, pose_config, feature_control_config, tracking_control_config = load_pipeline_config(
        config_path
    )

    assert feature_config.nfeatures == 100
    assert pose_config.min_matches == 10
    assert feature_control_config is not None
    assert feature_control_config.enabled is True
    assert feature_control_config.num_workers == 1
    assert tracking_control_config is not None
    assert tracking_control_config.enabled is True
    assert tracking_control_config.max_pending_frames == 8


def test_load_pipeline_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.json"
    payload = {"feature_config": {"unknown_field": 1}}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown FeaturePipelineConfig fields"):
        load_pipeline_config(config_path)


def test_run_kitti_sequence_generates_artifacts(tmp_path: Path) -> None:
    _write_dummy_kitti_dataset(tmp_path)
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")

    result = run_kitti_sequence(
        root=tmp_path,
        sequence="00",
        camera="image_2",
        output_dir=tmp_path / "reports",
        run_id="kitti_test",
        config_path=config_path,
        use_run_subdir=False,
        max_frames=2,
    )

    assert result.trajectory_path.exists()
    assert result.metrics_path.exists()
    assert result.diagnostics_path.exists()


def test_run_kitti_sequence_streaming(tmp_path: Path) -> None:
    _write_dummy_kitti_dataset(tmp_path)
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")

    result = run_kitti_sequence(
        root=tmp_path,
        sequence="00",
        camera="image_2",
        output_dir=tmp_path / "reports",
        run_id="kitti_stream_test",
        config_path=config_path,
        use_run_subdir=False,
        max_frames=2,
        stream_frames=True,
        stream_queue_capacity=1,
    )

    assert result.trajectory_path.exists()
    assert result.metrics_path.exists()
    assert result.diagnostics_path.exists()


def test_run_kitti_sequence_async_ingestion(tmp_path: Path) -> None:
    _write_dummy_kitti_dataset(tmp_path)
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")

    result = run_kitti_sequence(
        root=tmp_path,
        sequence="00",
        camera="image_2",
        output_dir=tmp_path / "reports",
        run_id="kitti_async_test",
        config_path=config_path,
        use_run_subdir=False,
        max_frames=2,
        async_ingestion=True,
        ingestion_entry_capacity=16,
        ingestion_output_capacity=16,
        ingestion_decode_workers=1,
        ingestion_read_timeout_s=0.2,
        ingestion_decode_timeout_s=0.2,
        ingestion_fail_fast=True,
    )

    assert result.trajectory_path.exists()
    assert result.metrics_path.exists()
    assert result.diagnostics_path.exists()
