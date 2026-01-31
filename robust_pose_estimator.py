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
    base_ransac_threshold: float = 0.01
    min_ransac_threshold: float = 0.005
    max_ransac_threshold: float = 0.02
    min_inlier_ratio: float = 0.25
    homography_bias: float = 0.9
    essential_bias: float = 1.0
    min_parallax: float = 1.0


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
        if best.diagnostics.inlier_ratio < self.config.min_inlier_ratio:
            LOGGER.warning(
                "Pose estimation rejected: low inlier ratio %.3f",
                best.diagnostics.inlier_ratio,
            )
            raise RuntimeError("Pose estimation failed due to low inlier ratio")
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
            score=score,
        )
        return PoseEstimate(
            rotation=rotation,
            translation=_normalize_translation(translation),
            inlier_indices=inliers,
            diagnostics=diagnostics,
        )


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


def _normalize_translation(translation: np.ndarray) -> np.ndarray:
    if translation.ndim != 1 or translation.shape[0] != 3:
        raise ValueError("Translation must be a 3D vector")
    norm = float(np.linalg.norm(translation))
    if norm == 0.0:
        raise ValueError("Translation norm is zero")
    return translation / norm
