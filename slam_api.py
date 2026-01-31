"""High-level SLAM API for running pipelines with persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from data_persistence import (
    RunDataStore,
    TrajectoryAccumulator,
    build_metrics_bundle,
    summarize_trajectory,
)
from feature_pipeline import FeaturePipelineConfig, build_feature_pipeline
from robust_pose_estimator import RobustPoseEstimator, RobustPoseEstimatorConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SLAMSystemConfig:
    """Configuration for the SLAM system API."""

    run_id: str
    output_dir: Path
    config_path: Path
    config_hash: str
    intrinsics: np.ndarray
    feature_config: FeaturePipelineConfig
    pose_config: RobustPoseEstimatorConfig
    use_run_subdir: bool = True


@dataclass(frozen=True)
class FrameDiagnostics:
    """Per-frame diagnostics for monitoring pose estimation quality."""

    frame_id: int
    timestamp: float
    match_count: int
    inliers: int
    method: str


@dataclass(frozen=True)
class SLAMRunResult:
    """Result handles for a completed SLAM run."""

    run_dir: Path
    trajectory_path: Path
    metrics_path: Path
    frame_diagnostics: tuple[FrameDiagnostics, ...]


class SLAMSystem:
    """High-level SLAM pipeline wrapper with persistent artifact storage."""

    def __init__(self, config: SLAMSystemConfig) -> None:
        if config.intrinsics.shape != (3, 3):
            raise ValueError("Intrinsics must be a 3x3 matrix")
        self.config = config
        self.feature_pipeline = build_feature_pipeline(config.feature_config)
        self.pose_estimator = RobustPoseEstimator(config.pose_config)
        self.data_store = RunDataStore.create(
            base_dir=config.output_dir,
            run_id=config.run_id,
            config_path=config.config_path,
            config_hash=config.config_hash,
            use_subdir=config.use_run_subdir,
            resolved_config={
                "intrinsics": config.intrinsics.tolist(),
                "feature_config": config.feature_config.__dict__,
                "pose_config": config.pose_config.__dict__,
            },
        )
        self.trajectory = self.data_store.create_accumulator("slam_trajectory")
        self.frame_diagnostics: list[FrameDiagnostics] = []
        self._prev_frame: np.ndarray | None = None
        self._prev_kp: list[cv2.KeyPoint] | None = None
        self._prev_desc: np.ndarray | None = None
        self._current_pose = np.eye(4)
        self._frame_id = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        if frame.ndim not in (2, 3):
            raise ValueError("Frame must be a grayscale or BGR image")
        if frame.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        kp, desc = self.feature_pipeline.detect_and_describe(frame_gray)
        if self._prev_frame is None:
            self._prev_frame = frame_gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._append_pose(timestamp, method="bootstrap", match_count=0, inliers=0)
            return self._current_pose.copy()

        matches = self.feature_pipeline.match(self._prev_desc, desc)
        if len(matches) < self.config.pose_config.min_matches:
            LOGGER.warning("Frame %d rejected: not enough matches", self._frame_id)
            self._prev_frame = frame_gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._append_pose(timestamp, method="insufficient_matches", match_count=len(matches), inliers=0)
            return self._current_pose.copy()

        pose_estimate = self.pose_estimator.estimate_pose(
            kp1=self._prev_kp or [],
            kp2=kp,
            matches=matches,
            intrinsics=self.config.intrinsics,
        )
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = pose_estimate.rotation
        relative_pose[:3, 3] = pose_estimate.translation
        self._current_pose = self._current_pose @ relative_pose
        self._prev_frame = frame_gray
        self._prev_kp = kp
        self._prev_desc = desc
        self._append_pose(
            timestamp,
            method=pose_estimate.diagnostics.method,
            match_count=pose_estimate.diagnostics.match_count,
            inliers=pose_estimate.diagnostics.inliers,
        )
        return self._current_pose.copy()

    def run_sequence(self, frames: Iterable[np.ndarray], timestamps: Iterable[float]) -> SLAMRunResult:
        for frame, timestamp in zip(frames, timestamps):
            self.process_frame(frame, float(timestamp))
        return self.finalize_run()

    def finalize_run(self) -> SLAMRunResult:
        trajectory_bundle = self.trajectory.as_bundle()
        trajectory_path = self.data_store.save_trajectory(trajectory_bundle)
        metrics = summarize_trajectory(trajectory_bundle)
        metrics_bundle = build_metrics_bundle("slam_metrics", metrics)
        metrics_path = self.data_store.save_metrics(metrics_bundle)
        return SLAMRunResult(
            run_dir=self.data_store.metadata.run_dir,
            trajectory_path=trajectory_path,
            metrics_path=metrics_path,
            frame_diagnostics=tuple(self.frame_diagnostics),
        )

    def _append_pose(
        self,
        timestamp: float,
        *,
        method: str,
        match_count: int,
        inliers: int,
    ) -> None:
        self.trajectory.append(self._current_pose.copy(), timestamp, self._frame_id)
        self.frame_diagnostics.append(
            FrameDiagnostics(
                frame_id=self._frame_id,
                timestamp=float(timestamp),
                match_count=int(match_count),
                inliers=int(inliers),
                method=str(method),
            )
        )
        self._frame_id += 1
