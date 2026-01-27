"""Feature detection and matching pipeline utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FeaturePipelineConfig:
    name: str = "orb"
    nfeatures: int = 2000
    ratio_test: float = 0.8
    cross_check: bool = True
    max_matches: int | None = 500


@dataclass(frozen=True)
class MatchStats:
    match_count: int
    mean_distance: float
    median_distance: float


MatcherCallable = Callable[[np.ndarray, np.ndarray], list[cv2.DMatch]]


class FeaturePipeline:
    def detect_and_describe(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        raise NotImplementedError

    def match(self, desc1: np.ndarray | None, desc2: np.ndarray | None) -> list[cv2.DMatch]:
        raise NotImplementedError

    def match_stats(self, matches: Sequence[cv2.DMatch]) -> MatchStats:
        if not matches:
            return MatchStats(match_count=0, mean_distance=0.0, median_distance=0.0)
        distances = np.array([m.distance for m in matches], dtype=np.float32)
        return MatchStats(
            match_count=len(matches),
            mean_distance=float(distances.mean()),
            median_distance=float(np.median(distances)),
        )


class ORBFeaturePipeline(FeaturePipeline):
    def __init__(self, config: FeaturePipelineConfig) -> None:
        self.config = config
        self.detector = cv2.ORB_create(nfeatures=config.nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=config.cross_check)

    def detect_and_describe(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match(self, desc1: np.ndarray | None, desc2: np.ndarray | None) -> list[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        if self.config.cross_check:
            matches = list(self.matcher.match(desc1, desc2))
        else:
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            matches = []
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.config.ratio_test * n.distance:
                    matches.append(m)
        matches.sort(key=lambda m: m.distance)
        if self.config.max_matches is not None:
            matches = matches[: self.config.max_matches]
        return matches


def build_feature_pipeline(config: FeaturePipelineConfig) -> FeaturePipeline:
    if config.name.lower() == "orb":
        return ORBFeaturePipeline(config)
    raise ValueError(f"Unsupported feature pipeline: {config.name}")


def matches_to_points(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    return pts1, pts2


def adaptive_ransac_threshold(
    pts1: np.ndarray,
    pts2: np.ndarray,
    base_threshold: float,
    min_threshold: float,
    max_threshold: float,
) -> float:
    if pts1.size == 0 or pts2.size == 0:
        return float(np.clip(base_threshold, min_threshold, max_threshold))
    displacements = np.linalg.norm(pts2 - pts1, axis=1)
    if displacements.size == 0:
        return float(np.clip(base_threshold, min_threshold, max_threshold))
    median_disp = float(np.median(displacements))
    scale = float(np.clip(median_disp / 25.0, 0.5, 2.0))
    threshold = base_threshold * scale
    return float(np.clip(threshold, min_threshold, max_threshold))
