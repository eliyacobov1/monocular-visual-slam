"""Tests for the robust pose estimation suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from robust_pose_estimator import RobustPoseEstimator, RobustPoseEstimatorConfig


def _project_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformed = (rotation @ points.T).T + translation
    projected = transformed[:, :2] / transformed[:, 2:3]
    return projected.astype(np.float32)


def _make_keypoints(points: np.ndarray) -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points]


def _make_matches(count: int) -> list[cv2.DMatch]:
    return [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(count)]


def test_robust_pose_estimator_selects_model() -> None:
    rng = np.random.default_rng(0)
    points_3d = rng.uniform(-1.0, 1.0, size=(50, 3)) + np.array([0.0, 0.0, 3.0])

    intrinsics = np.eye(3)
    rotation = np.eye(3)
    translation = np.array([0.1, 0.0, 0.0])

    pts1 = _project_points(points_3d, np.eye(3), np.zeros(3))
    pts2 = _project_points(points_3d, rotation, translation)

    kp1 = _make_keypoints(pts1)
    kp2 = _make_keypoints(pts2)
    matches = _make_matches(len(kp1))

    estimator = RobustPoseEstimator(RobustPoseEstimatorConfig(min_matches=20))
    estimate = estimator.estimate_pose(kp1, kp2, matches, intrinsics)

    assert estimate.diagnostics.inliers > 0
    assert estimate.diagnostics.method in {"essential", "homography"}
