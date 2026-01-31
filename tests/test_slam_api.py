"""Tests for the high-level SLAM API wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_pipeline import FeaturePipelineConfig
from robust_pose_estimator import RobustPoseEstimatorConfig
from slam_api import SLAMSystem, SLAMSystemConfig


def test_slam_api_runs_with_blank_frames(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_id": "api_test"}), encoding="utf-8")

    config = SLAMSystemConfig(
        run_id="api_test",
        output_dir=tmp_path,
        config_path=config_path,
        config_hash="hash",
        intrinsics=np.eye(3),
        feature_config=FeaturePipelineConfig(nfeatures=200),
        pose_config=RobustPoseEstimatorConfig(min_matches=50),
        use_run_subdir=False,
    )

    slam = SLAMSystem(config)
    frames = [np.zeros((240, 320), dtype=np.uint8) for _ in range(2)]
    timestamps = [0.0, 0.1]
    result = slam.run_sequence(frames, timestamps)

    assert result.trajectory_path.exists()
    assert result.metrics_path.exists()
    assert result.diagnostics_path.exists()
    assert len(result.frame_diagnostics) == 2
