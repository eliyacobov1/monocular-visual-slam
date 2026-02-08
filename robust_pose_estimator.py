"""Robust pose estimation suite combining multiple geometric models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from feature_pipeline import adaptive_ransac_threshold, matches_to_points
from homography import decompose_homography, estimate_pose_from_matches, ransac_homography

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseEstimationDiagnostics:
    """Diagnostic summary for a pose estimation attempt."""

    method: str
    match_count: int
    inliers: int
    inlier_ratio: float
    median_parallax: float
    cheirality_inliers: int
    cheirality_ratio: float
    score: float


@dataclass(frozen=True)
class PoseEstimate:
    """Pose estimate with rotation, translation, and diagnostics."""

    rotation: np.ndarray
    translation: np.ndarray
    inlier_indices: np.ndarray
    diagnostics: PoseEstimationDiagnostics


@dataclass(frozen=True)
class RobustPoseEstimatorConfig:
    """Configuration for robust pose estimation."""

    min_matches: int = 20
    min_inliers: int = 30
    base_ransac_threshold: float = 0.01
    min_ransac_threshold: float = 0.005
    max_ransac_threshold: float = 0.02
    min_inlier_ratio: float = 0.25
    homography_bias: float = 0.9
    essential_bias: float = 1.0
    min_parallax: float = 1.0
    min_cheirality_ratio: float = 0.6
    min_cheirality_inliers: int = 12

    def __post_init__(self) -> None:
        if self.min_matches <= 0:
            raise ValueError("min_matches must be positive")
        if self.min_inliers <= 0:
            raise ValueError("min_inliers must be positive")
        if self.min_inlier_ratio <= 0:
            raise ValueError("min_inlier_ratio must be positive")
        if self.min_parallax < 0:
            raise ValueError("min_parallax must be non-negative")
        if self.min_cheirality_ratio <= 0:
            raise ValueError("min_cheirality_ratio must be positive")
        if self.min_cheirality_inliers <= 0:
            raise ValueError("min_cheirality_inliers must be positive")


class PoseEstimationFailure(RuntimeError):
    """Pose estimation failure with recovery metadata."""

    def __init__(self, reason: str, recovery_action: str, metrics: dict[str, float]) -> None:
        super().__init__(f"{reason} (recovery={recovery_action})")
        self.reason = reason
        self.recovery_action = recovery_action
        self.metrics = metrics


class RobustPoseEstimator:
    """Estimate camera motion using essential + homography model selection."""

    def __init__(self, config: RobustPoseEstimatorConfig) -> None:
        self.config = config

    def estimate_pose(
        self,
        kp1: Sequence[cv2.KeyPoint],
        kp2: Sequence[cv2.KeyPoint],
        matches: Sequence[cv2.DMatch],
        intrinsics: np.ndarray,
    ) -> PoseEstimate:
        if intrinsics.shape != (3, 3):
            raise ValueError("Intrinsics must be a 3x3 matrix")
        if len(matches) < self.config.min_matches:
            raise ValueError("Not enough matches for pose estimation")
        if not kp1 or not kp2:
            raise ValueError("Keypoints must be non-empty")

        pts1, pts2 = matches_to_points(kp1, kp2, matches)
        ransac_threshold = adaptive_ransac_threshold(
            pts1,
            pts2,
            self.config.base_ransac_threshold,
            self.config.min_ransac_threshold,
            self.config.max_ransac_threshold,
        )

        essential_estimate = self._estimate_essential(
            kp1,
            kp2,
            matches,
            intrinsics,
            ransac_threshold,
        )
        homography_estimate = self._estimate_homography(
            pts1,
            pts2,
            intrinsics,
        )

        candidates = [essential_estimate, homography_estimate]
        best = max(candidates, key=lambda cand: cand.diagnostics.score)
        self._apply_stability_gates(best)
        LOGGER.info(
            "Pose estimation selected %s with %d/%d inliers",
            best.diagnostics.method,
            best.diagnostics.inliers,
            best.diagnostics.match_count,
        )
        return best

    def _estimate_essential(
        self,
        kp1: Sequence[cv2.KeyPoint],
        kp2: Sequence[cv2.KeyPoint],
        matches: Sequence[cv2.DMatch],
        intrinsics: np.ndarray,
        ransac_threshold: float,
    ) -> PoseEstimate:
        try:
            rotation, translation, inliers, match_count = estimate_pose_from_matches(
                kp1=kp1,
                kp2=kp2,
                matches=matches,
                K=intrinsics,
                ransac_threshold=ransac_threshold,
                min_matches=self.config.min_matches,
            )
        except RuntimeError as exc:
            LOGGER.exception("Essential matrix pose estimation failed")
            raise RuntimeError("Essential matrix pose estimation failed") from exc

        inlier_ratio = float(len(inliers) / max(match_count, 1))
        median_parallax = _median_parallax(kp1, kp2, matches, inliers)
        cheirality_ratio, cheirality_inliers = _cheirality_ratio(
            kp1,
            kp2,
            matches,
            inliers,
            rotation,
            translation,
            intrinsics,
        )
        score = (
            self.config.essential_bias
            * inlier_ratio
            * max(median_parallax, self.config.min_parallax)
        )
        diagnostics = PoseEstimationDiagnostics(
            method="essential",
            match_count=match_count,
            inliers=len(inliers),
            inlier_ratio=inlier_ratio,
            median_parallax=median_parallax,
            cheirality_inliers=cheirality_inliers,
            cheirality_ratio=cheirality_ratio,
            score=score,
        )
        return PoseEstimate(
            rotation=rotation,
            translation=_normalize_translation(translation),
            inlier_indices=inliers,
            diagnostics=diagnostics,
        )

    def _estimate_homography(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        intrinsics: np.ndarray,
    ) -> PoseEstimate:
        try:
            homography, inliers = ransac_homography(pts1, pts2)
            rotation, translation = decompose_homography(homography, intrinsics)
        except (RuntimeError, ValueError) as exc:
            LOGGER.exception("Homography pose estimation failed")
            raise RuntimeError("Homography pose estimation failed") from exc
        inlier_ratio = float(len(inliers) / max(len(pts1), 1))
        median_parallax = float(np.median(np.linalg.norm(pts2 - pts1, axis=1)))
        score = (
            self.config.homography_bias
            * inlier_ratio
            * max(median_parallax, self.config.min_parallax)
        )
        diagnostics = PoseEstimationDiagnostics(
            method="homography",
            match_count=len(pts1),
            inliers=len(inliers),
            inlier_ratio=inlier_ratio,
            median_parallax=median_parallax,
            cheirality_inliers=len(inliers),
            cheirality_ratio=1.0,
            score=score,
        )
        return PoseEstimate(
            rotation=rotation,
            translation=_normalize_translation(translation),
            inlier_indices=inliers,
            diagnostics=diagnostics,
        )

    def _apply_stability_gates(self, estimate: PoseEstimate) -> None:
        diag = estimate.diagnostics
        metrics = {
            "match_count": float(diag.match_count),
            "inliers": float(diag.inliers),
            "inlier_ratio": float(diag.inlier_ratio),
            "median_parallax": float(diag.median_parallax),
            "cheirality_ratio": float(diag.cheirality_ratio),
            "cheirality_inliers": float(diag.cheirality_inliers),
        }
        if diag.inliers < self.config.min_inliers:
            LOGGER.warning("Pose estimation rejected: low inlier count", extra=metrics)
            raise PoseEstimationFailure("low_inlier_count", "relocalize", metrics)
        if diag.inlier_ratio < self.config.min_inlier_ratio:
            LOGGER.warning("Pose estimation rejected: low inlier ratio", extra=metrics)
            raise PoseEstimationFailure("low_inlier_ratio", "relocalize", metrics)
        if diag.median_parallax < self.config.min_parallax:
            LOGGER.warning("Pose estimation rejected: low parallax", extra=metrics)
            raise PoseEstimationFailure("low_parallax", "relocalize", metrics)
        if diag.method == "essential":
            if diag.cheirality_inliers < self.config.min_cheirality_inliers:
                LOGGER.warning("Pose estimation rejected: cheirality inliers", extra=metrics)
                raise PoseEstimationFailure("cheirality_inliers", "relocalize", metrics)
            if diag.cheirality_ratio < self.config.min_cheirality_ratio:
                LOGGER.warning("Pose estimation rejected: cheirality ratio", extra=metrics)
                raise PoseEstimationFailure("cheirality_ratio", "relocalize", metrics)


def _median_parallax(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    inliers: np.ndarray,
) -> float:
    if len(inliers) == 0:
        return 0.0
    pts1 = np.array([kp1[matches[i].queryIdx].pt for i in inliers], dtype=np.float32)
    pts2 = np.array([kp2[matches[i].trainIdx].pt for i in inliers], dtype=np.float32)
    if pts1.size == 0 or pts2.size == 0:
        return 0.0
    return float(np.median(np.linalg.norm(pts2 - pts1, axis=1)))


def _cheirality_ratio(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    inliers: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[float, int]:
    if len(inliers) == 0:
        return 0.0, 0
    pts1 = np.array([kp1[matches[i].queryIdx].pt for i in inliers], dtype=np.float32).T
    pts2 = np.array([kp2[matches[i].trainIdx].pt for i in inliers], dtype=np.float32).T
    if pts1.size == 0 or pts2.size == 0:
        return 0.0, 0
    proj_a = intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
    proj_b = intrinsics @ np.hstack([rotation, translation.reshape(3, 1)])
    homog = cv2.triangulatePoints(proj_a, proj_b, pts1, pts2)
    pts_3d = homog[:3] / homog[3]
    depth_a = pts_3d[2]
    depth_b = (rotation @ pts_3d + translation.reshape(3, 1))[2]
    valid = np.isfinite(depth_a) & np.isfinite(depth_b)
    if not np.any(valid):
        return 0.0, 0
    positive = (depth_a > 0) & (depth_b > 0) & valid
    count = int(np.sum(positive))
    ratio = float(count / max(len(inliers), 1))
    return ratio, count


def _normalize_translation(translation: np.ndarray) -> np.ndarray:
    if translation.ndim != 1 or translation.shape[0] != 3:
        raise ValueError("Translation must be a 3D vector")
    norm = float(np.linalg.norm(translation))
    if norm == 0.0:
        raise ValueError("Translation norm is zero")
    return translation / norm
