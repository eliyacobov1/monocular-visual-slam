import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from homography import estimate_pose_from_orb_with_inliers


def _project_points(points, R, t, K):
    cam = (R @ points.T).T + t
    proj = (K @ cam.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def test_estimate_pose_with_inliers_accepts_geometric_match():
    rng = np.random.default_rng(42)
    num_points = 40
    points = rng.uniform(-1.0, 1.0, size=(num_points, 3))
    points[:, 2] += 4.0

    yaw = 0.05
    R = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    t = np.array([0.2, 0.0, 0.0])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    pts1 = _project_points(points, np.eye(3), np.zeros(3), K)
    pts2 = _project_points(points, R, t, K)

    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts1]
    keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts2]
    descriptors = rng.integers(0, 256, size=(num_points, 32), dtype=np.uint8)

    R_est, t_est, inliers, match_count = estimate_pose_from_orb_with_inliers(
        keypoints1,
        descriptors,
        keypoints2,
        descriptors,
        K,
        ransac_threshold=1.0,
        min_matches=20,
    )

    assert match_count == num_points
    assert len(inliers) >= int(0.8 * num_points)
    assert R_est.shape == (3, 3)
    assert t_est.shape == (3,)
