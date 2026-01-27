"""Keyframe selection and local bundle adjustment management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from bundle_adjustment import Observation, run_bundle_adjustment, triangulate_points


@dataclass(frozen=True)
class Keyframe:
    frame_id: int
    pose: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass(frozen=True)
class BundleAdjustmentResult:
    frame_ids: list[int]
    poses: list[np.ndarray]


class KeyframeManager:
    def __init__(
        self,
        window_size: int = 5,
        min_translation: float = 0.1,
        min_rotation_deg: float = 5.0,
        min_match_ratio: float = 0.25,
        min_matches: int = 60,
        matcher: Callable[[np.ndarray, np.ndarray], list[cv2.DMatch]] | None = None,
    ) -> None:
        self.window_size = window_size
        self.min_translation = min_translation
        self.min_rotation_deg = min_rotation_deg
        self.min_match_ratio = min_match_ratio
        self.min_matches = min_matches
        self.matcher = matcher
        self.keyframes: list[Keyframe] = []

    def add_keyframe(
        self,
        frame_id: int,
        pose: np.ndarray,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray,
    ) -> None:
        self.keyframes.append(
            Keyframe(
                frame_id=frame_id,
                pose=pose.copy(),
                keypoints=keypoints,
                descriptors=descriptors,
            )
        )

    def should_add_keyframe(
        self,
        current_pose: np.ndarray,
        current_descriptors: np.ndarray,
    ) -> bool:
        if not self.keyframes:
            return True
        last_keyframe = self.keyframes[-1]
        rel = np.linalg.inv(last_keyframe.pose) @ current_pose
        translation = float(np.linalg.norm(rel[:3, 3]))
        rot = rel[:3, :3]
        trace = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(trace)))
        match_ratio = self._match_ratio(last_keyframe.descriptors, current_descriptors)
        return (
            translation >= self.min_translation
            or rotation_deg >= self.min_rotation_deg
            or match_ratio <= self.min_match_ratio
        )

    def run_local_bundle_adjustment(
        self,
        intrinsics: np.ndarray,
        max_nfev: int = 50,
    ) -> BundleAdjustmentResult | None:
        if len(self.keyframes) < 2:
            return None
        window = self.keyframes[-self.window_size :]
        points_3d, observations = self._build_window_observations(window, intrinsics)
        if points_3d is None or observations is None:
            return None
        if points_3d.shape[0] < 6:
            return None
        poses = [kf.pose for kf in window]
        optimized_poses, _ = run_bundle_adjustment(
            poses=poses,
            points_3d=points_3d,
            observations=observations,
            intrinsics=intrinsics,
            max_nfev=max_nfev,
        )
        return BundleAdjustmentResult(
            frame_ids=[kf.frame_id for kf in window],
            poses=optimized_poses,
        )

    def _match_ratio(self, desc_a: np.ndarray, desc_b: np.ndarray) -> float:
        if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
            return 0.0
        matcher = self.matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match
        matches = matcher(desc_a, desc_b)
        return len(matches) / float(min(len(desc_a), len(desc_b)))

    def _build_window_observations(
        self,
        window: Iterable[Keyframe],
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray | None, list[Observation] | None]:
        keyframes = list(window)
        if len(keyframes) < 2:
            return None, None
        points = []
        observations: list[Observation] = []
        point_offset = 0
        matcher = self.matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match
        for idx in range(len(keyframes) - 1):
            kf_a = keyframes[idx]
            kf_b = keyframes[idx + 1]
            if (
                kf_a.descriptors is None
                or kf_b.descriptors is None
                or len(kf_a.descriptors) == 0
                or len(kf_b.descriptors) == 0
            ):
                continue
            matches = matcher(kf_a.descriptors, kf_b.descriptors)
            if len(matches) < self.min_matches:
                continue
            matches = sorted(matches, key=lambda m: m.distance)[: self.min_matches]
            pts_a = np.array([kf_a.keypoints[m.queryIdx].pt for m in matches])
            pts_b = np.array([kf_b.keypoints[m.trainIdx].pt for m in matches])
            triangulated = triangulate_points(
                kf_a.pose,
                kf_b.pose,
                intrinsics,
                pts_a,
                pts_b,
            )
            valid = np.isfinite(triangulated).all(axis=1)
            triangulated = triangulated[valid]
            pts_a = pts_a[valid]
            pts_b = pts_b[valid]
            if triangulated.size == 0:
                continue
            points.append(triangulated)
            for local_idx, (uv_a, uv_b) in enumerate(zip(pts_a, pts_b)):
                point_index = point_offset + local_idx
                observations.append(
                    Observation(frame_index=idx, point_index=point_index, uv=uv_a)
                )
                observations.append(
                    Observation(frame_index=idx + 1, point_index=point_index, uv=uv_b)
                )
            point_offset += triangulated.shape[0]
        if not points:
            return None, None
        return np.vstack(points), observations
