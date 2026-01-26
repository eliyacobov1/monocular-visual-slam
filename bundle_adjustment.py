"""Local bundle adjustment utilities for keyframe windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Observation:
    frame_index: int
    point_index: int
    uv: np.ndarray


def _pose_to_vec(pose: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    return np.hstack([rvec.ravel(), tvec])


def _vec_to_pose(vec: np.ndarray) -> np.ndarray:
    rvec = vec[:3]
    tvec = vec[3:6]
    rot, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = tvec
    return pose


def _projection_matrix(pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    cam_pose = np.linalg.inv(pose)
    return intrinsics @ cam_pose[:3, :]


def triangulate_points(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    intrinsics: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    proj_a = _projection_matrix(pose_a, intrinsics)
    proj_b = _projection_matrix(pose_b, intrinsics)
    homog = cv2.triangulatePoints(proj_a, proj_b, points_a.T, points_b.T)
    points_3d = (homog[:3] / homog[3]).T
    return points_3d


def run_bundle_adjustment(
    poses: Sequence[np.ndarray],
    points_3d: np.ndarray,
    observations: Iterable[Observation],
    intrinsics: np.ndarray,
    max_nfev: int = 50,
) -> tuple[list[np.ndarray], np.ndarray]:
    if points_3d.size == 0:
        raise ValueError("No points provided for bundle adjustment")

    obs_list = list(observations)
    if not obs_list:
        raise ValueError("No observations provided for bundle adjustment")

    if len(poses) < 1:
        raise ValueError("At least one pose is required for bundle adjustment")

    fixed_pose = poses[0]
    num_poses = len(poses)
    if any(obs.frame_index < 0 or obs.frame_index >= num_poses for obs in obs_list):
        raise ValueError("Observations reference poses outside the provided window")

    pose_vecs = [_pose_to_vec(pose) for pose in poses[1:]]
    pose_params = np.concatenate(pose_vecs) if pose_vecs else np.empty(0)
    points_params = points_3d.ravel()
    x0 = np.hstack([pose_params, points_params])

    num_opt_poses = len(poses) - 1
    num_points = points_3d.shape[0]

    def residuals(x: np.ndarray) -> np.ndarray:
        pose_vecs = x[: num_opt_poses * 6].reshape(num_opt_poses, 6)
        pts = x[num_opt_poses * 6 :].reshape(num_points, 3)
        res = []
        for obs in obs_list:
            if obs.frame_index == 0:
                pose = fixed_pose
            else:
                pose = _vec_to_pose(pose_vecs[obs.frame_index - 1])
            cam_pose = np.linalg.inv(pose)
            point = pts[obs.point_index]
            proj = intrinsics @ (cam_pose[:3, :3] @ point + cam_pose[:3, 3])
            uv = proj[:2] / proj[2]
            res.extend(uv - obs.uv)
        return np.array(res)

    result = least_squares(residuals, x0, loss="huber", f_scale=1.0, max_nfev=max_nfev)
    optimized_pose_vecs = result.x[: num_opt_poses * 6].reshape(num_opt_poses, 6)
    optimized_points = result.x[num_opt_poses * 6 :].reshape(num_points, 3)
    optimized_poses = [fixed_pose]
    optimized_poses.extend([_vec_to_pose(vec) for vec in optimized_pose_vecs])
    return optimized_poses, optimized_points
