"""Tests for algorithmic stability gates and conditioning checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bundle_adjustment import BundleAdjustmentConfig, Observation, run_bundle_adjustment
from pose_graph import PoseGraph, SolverConfig


def _project_point(point: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    cam_pose = np.linalg.inv(pose)
    proj = intrinsics @ (cam_pose[:3, :3] @ point + cam_pose[:3, 3])
    return proj[:2] / proj[2]


def test_bundle_adjustment_conditioning_fallback() -> None:
    intrinsics = np.eye(3)
    poses = [np.eye(4), np.eye(4)]
    poses[1][:3, 3] = np.array([0.1, 0.0, 0.0])
    points_3d = np.array(
        [
            [0.2, -0.1, 3.0],
            [-0.1, 0.2, 3.5],
            [0.1, 0.1, 4.0],
            [-0.2, -0.1, 3.2],
        ]
    )
    observations = []
    for idx, point in enumerate(points_3d):
        uv0 = _project_point(point, poses[0], intrinsics)
        uv1 = _project_point(point, poses[1], intrinsics)
        observations.append(Observation(frame_index=0, point_index=idx, uv=uv0))
        observations.append(Observation(frame_index=1, point_index=idx, uv=uv1))

    config = BundleAdjustmentConfig(max_condition_number=1.0, min_singular_value=1.0)
    optimized_poses, optimized_points, diagnostics = run_bundle_adjustment(
        poses=poses,
        points_3d=points_3d,
        observations=observations,
        intrinsics=intrinsics,
        max_nfev=10,
        config=config,
    )

    assert diagnostics.fallback_applied is True
    for original, optimized in zip(poses, optimized_poses):
        np.testing.assert_allclose(original, optimized)
    np.testing.assert_allclose(points_3d, optimized_points)


def test_pose_graph_conditioning_gate_trips() -> None:
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=5, max_condition_number=1.0, min_diagonal=1.0),
    )
    graph.add_pose(np.eye(2), np.array([0.1, 0.0]))
    graph.add_pose(np.eye(2), np.array([0.1, 0.0]))
    _ = graph.optimize()

    assert graph.last_result is not None
    assert graph.last_result.success is False
    assert "Conditioning gate tripped" in graph.last_result.message
