"""High-level SLAM API for running pipelines with persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import cv2
import numpy as np

from control_plane_hub import ControlPlaneHub, ControlPlaneStageAdapter
from data_persistence import (
    FrameDiagnosticsEntry,
    RunDataStore,
    TrajectoryAccumulator,
    build_frame_diagnostics_bundle,
    build_metrics_bundle,
    summarize_trajectory,
)
from feature_control_plane import FeatureControlConfig, FeatureControlPlane
from feature_pipeline import FeaturePipelineConfig, build_feature_pipeline
from robust_pose_estimator import (
    PoseEstimationDiagnostics,
    RobustPoseEstimator,
    RobustPoseEstimatorConfig,
)
from run_telemetry import RunTelemetryRecorder, TelemetryEvent, TelemetrySink, timed_event
from keyframe_manager import KeyframeManager
from map_builder import MapBuilderConfig, MapBuildStats, MapSnapshotBuilder
from persistent_map import MapRelocalizer, PersistentMapSnapshot, PersistentMapStore
from frame_stream import FramePacket
from tracking_control_plane import (
    TrackingControlConfig,
    TrackingControlPlane,
    TrackingFrameResult,
)

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class FrameLike(Protocol):
    """Protocol describing a frame payload with timestamp."""

    frame: np.ndarray
    timestamp: float


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
    feature_control: FeatureControlConfig | None = None
    tracking_control: TrackingControlConfig | None = None
    use_run_subdir: bool = True
    enable_telemetry: bool = True
    telemetry_name: str = "slam_telemetry"
    telemetry_sink: TelemetrySink | None = None
    enable_control_plane_report: bool = True
    control_plane_report_name: str = "control_plane_report"
    keyframe_window_size: int = 5
    keyframe_min_translation: float = 0.1
    keyframe_min_rotation_deg: float = 5.0
    keyframe_min_match_ratio: float = 0.25
    keyframe_min_matches: int = 60
    map_builder: MapBuilderConfig = MapBuilderConfig()
    relocalization_min_matches: int = 80
    relocalization_min_inliers: int = 40
    relocalization_score_threshold: float = 0.75
    relocalization_ransac_threshold: float = 0.01
    relocalization_max_candidates: int = 5


@dataclass(frozen=True)
class FrameDiagnostics:
    """Per-frame diagnostics for monitoring pose estimation quality."""

    frame_id: int
    timestamp: float
    match_count: int
    inliers: int
    method: str
    inlier_ratio: float
    median_parallax: float
    score: float
    status: str
    failure_reason: str | None


@dataclass(frozen=True)
class SLAMRunResult:
    """Result handles for a completed SLAM run."""

    run_dir: Path
    trajectory_path: Path
    metrics_path: Path
    diagnostics_path: Path
    telemetry_path: Path | None
    control_plane_report_path: Path | None
    frame_diagnostics: tuple[FrameDiagnostics, ...]
    map_snapshot_path: Path | None
    map_stats: MapBuildStats | None


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
        self.telemetry = self._build_telemetry_sink()
        self.frame_diagnostics: list[FrameDiagnostics] = []
        self._prev_frame: np.ndarray | None = None
        self._prev_kp: list[cv2.KeyPoint] | None = None
        self._prev_desc: np.ndarray | None = None
        self._current_pose = np.eye(4)
        self._frame_id = 0
        self._feature_control_config = config.feature_control
        self._tracking_control_config = config.tracking_control
        self._keyframe_manager = KeyframeManager(
            window_size=config.keyframe_window_size,
            min_translation=config.keyframe_min_translation,
            min_rotation_deg=config.keyframe_min_rotation_deg,
            min_match_ratio=config.keyframe_min_match_ratio,
            min_matches=config.keyframe_min_matches,
        )
        self._map_builder = MapSnapshotBuilder(config.map_builder)
        self._last_map_snapshot: PersistentMapSnapshot | None = None
        self._last_map_stats: MapBuildStats | None = None
        self._map_dirty = False
        self._relocalizer_snapshot: PersistentMapSnapshot | None = None
        self._relocalizer: MapRelocalizer | None = None
        self._control_plane_report_path: Path | None = None

    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        if frame.ndim not in (2, 3):
            raise ValueError("Frame must be a grayscale or BGR image")
        with timed_event("frame_process", self.telemetry, {"frame_id": self._frame_id}):
            if frame.ndim == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame

            with timed_event(
                "feature_detect",
                self.telemetry,
                {"frame_id": self._frame_id},
            ):
                kp, desc = self.feature_pipeline.detect_and_describe(frame_gray)
        return self._process_frame_with_features(frame_gray, timestamp, kp, desc)

    def _process_frame_with_features(
        self,
        frame_gray: np.ndarray,
        timestamp: float,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray | None,
    ) -> np.ndarray:
        if self._prev_frame is None:
            self._prev_frame = frame_gray
            self._prev_kp = keypoints
            self._prev_desc = descriptors
            self._append_pose(
                timestamp,
                method="bootstrap",
                match_count=0,
                inliers=0,
                status="bootstrap",
                failure_reason=None,
            )
            return self._current_pose.copy()

        with timed_event("feature_match", self.telemetry, {"frame_id": self._frame_id}):
            matches = self.feature_pipeline.match(self._prev_desc, descriptors)
        if len(matches) < self.config.pose_config.min_matches:
            LOGGER.warning("Frame %d rejected: not enough matches", self._frame_id)
            if self._attempt_relocalization(keypoints, descriptors, frame_gray, timestamp):
                return self._current_pose.copy()
            self._prev_frame = frame_gray
            self._prev_kp = keypoints
            self._prev_desc = descriptors
            self._append_pose(
                timestamp,
                method="insufficient_matches",
                match_count=len(matches),
                inliers=0,
                status="skipped",
                failure_reason="min_matches",
            )
            return self._current_pose.copy()

        try:
            with timed_event(
                "pose_estimate",
                self.telemetry,
                {"frame_id": self._frame_id, "match_count": len(matches)},
            ):
                pose_estimate = self.pose_estimator.estimate_pose(
                    kp1=self._prev_kp or [],
                    kp2=keypoints,
                    matches=matches,
                    intrinsics=self.config.intrinsics,
                )
        except Exception as exc:
            LOGGER.exception("Pose estimation failed for frame %d", self._frame_id)
            if self._attempt_relocalization(keypoints, descriptors, frame_gray, timestamp):
                return self._current_pose.copy()
            self._prev_frame = frame_gray
            self._prev_kp = keypoints
            self._prev_desc = descriptors
            self._append_pose_failure(timestamp, exc)
            return self._current_pose.copy()
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = pose_estimate.rotation
        relative_pose[:3, 3] = pose_estimate.translation
        self._current_pose = self._current_pose @ relative_pose
        self._prev_frame = frame_gray
        self._prev_kp = keypoints
        self._prev_desc = descriptors
        self._append_pose_with_diagnostics(timestamp, pose_estimate.diagnostics)
        self._maybe_add_keyframe(keypoints, descriptors)
        return self._current_pose.copy()

    def inject_tracking_loss(self, reason: str | None = None) -> None:
        """Force a tracking loss by clearing frame-to-frame correspondence state."""

        if self._prev_frame is None:
            raise RuntimeError("Tracking loss injection requires at least one processed frame")
        self._prev_kp = None
        self._prev_desc = None
        event = TelemetryEvent(
            name="tracking_loss_injected",
            duration_s=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "frame_id": self._frame_id,
                "reason": reason or "unspecified",
            },
        )
        self.telemetry.record_event(event)
        LOGGER.warning(
            "Tracking loss injected at frame %d",
            self._frame_id,
            extra={"reason": reason},
        )

    def run_sequence(self, frames: Iterable[np.ndarray], timestamps: Iterable[float]) -> SLAMRunResult:
        for frame, timestamp in zip(frames, timestamps):
            self.process_frame(frame, float(timestamp))
        return self.finalize_run()

    def run_stream(self, frames: Iterable[FrameLike | tuple[np.ndarray, float]]) -> SLAMRunResult:
        """Process a stream of frames with timestamps.

        Accepts an iterable of (frame, timestamp) tuples or objects exposing
        ``frame`` and ``timestamp`` attributes (e.g., FramePacket).
        """

        if self._feature_control_config and self._feature_control_config.enabled:
            return self.run_stream_async(frames)
        for item in frames:
            if isinstance(item, tuple):
                frame, timestamp = item
            elif isinstance(item, FramePacket):
                frame, timestamp = item.frame, item.timestamp
            else:
                frame = item.frame
                timestamp = item.timestamp
            self.process_frame(frame, float(timestamp))
        return self.finalize_run()

    def run_stream_async(self, frames: Iterable[FrameLike | tuple[np.ndarray, float]]) -> SLAMRunResult:
        """Process frames with asynchronous feature extraction and deterministic ordering."""

        control_config = self._feature_control_config or FeatureControlConfig(enabled=True)
        tracking_config = self._tracking_control_config or TrackingControlConfig(enabled=True)
        feature_plane = FeatureControlPlane(
            feature_config=self.config.feature_config,
            control_config=control_config,
        )
        control_plane = TrackingControlPlane(feature_plane, config=tracking_config)
        seq_id = 0
        try:
            for item in frames:
                if isinstance(item, tuple):
                    frame, timestamp = item
                elif isinstance(item, FramePacket):
                    frame, timestamp = item.frame, item.timestamp
                else:
                    frame = item.frame
                    timestamp = item.timestamp
                if frame.ndim not in (2, 3):
                    raise ValueError("Frame must be a grayscale or BGR image")
                if frame.ndim == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame
                control_plane.submit_frame(
                    seq_id=seq_id,
                    timestamp=float(timestamp),
                    frame_gray=frame_gray,
                )
                seq_id += 1
                for result in control_plane.drain_ready():
                    self._handle_tracking_result(result)
            while control_plane.pending_frames:
                result = control_plane.collect(timeout_s=tracking_config.backpressure_timeout_s)
                self._handle_tracking_result(result)
        finally:
            control_plane.close()
            if self.config.enable_control_plane_report:
                hub = ControlPlaneHub(
                    [
                        ControlPlaneStageAdapter(
                            name="feature",
                            health_snapshot=feature_plane.health_snapshot,
                            events=feature_plane.events,
                        ),
                        ControlPlaneStageAdapter(
                            name="tracking",
                            health_snapshot=control_plane.health_snapshot,
                            events=control_plane.events,
                        ),
                    ]
                )
                report = hub.generate_report()
                self._control_plane_report_path = self.data_store.save_control_plane_report(
                    self.config.control_plane_report_name,
                    report.asdict(),
                )
        return self.finalize_run()

    def finalize_run(self) -> SLAMRunResult:
        map_snapshot_path = None
        map_stats = None
        map_snapshot = self._build_map_snapshot()
        if map_snapshot is not None:
            map_bundle = self.data_store.save_map_snapshot("slam_map", map_snapshot)
            map_snapshot_path = map_bundle.path
            map_stats = self._last_map_stats
        trajectory_bundle = self.trajectory.as_bundle()
        trajectory_path = self.data_store.save_trajectory(trajectory_bundle)
        metrics = summarize_trajectory(trajectory_bundle)
        metrics_bundle = build_metrics_bundle("slam_metrics", metrics)
        metrics_path = self.data_store.save_metrics(metrics_bundle)
        diagnostics_bundle = build_frame_diagnostics_bundle(
            "frame_diagnostics",
            (
                FrameDiagnosticsEntry(
                    frame_id=entry.frame_id,
                    timestamp=entry.timestamp,
                    match_count=entry.match_count,
                    inliers=entry.inliers,
                    method=entry.method,
                    inlier_ratio=entry.inlier_ratio,
                    median_parallax=entry.median_parallax,
                    score=entry.score,
                    status=entry.status,
                    failure_reason=entry.failure_reason,
                )
                for entry in self.frame_diagnostics
            ),
        )
        diagnostics_path = self.data_store.save_frame_diagnostics(diagnostics_bundle)
        if isinstance(self.telemetry, RunTelemetryRecorder):
            self.telemetry.flush()
        return SLAMRunResult(
            run_dir=self.data_store.metadata.run_dir,
            trajectory_path=trajectory_path,
            metrics_path=metrics_path,
            diagnostics_path=diagnostics_path,
            telemetry_path=self.telemetry.path if isinstance(self.telemetry, RunTelemetryRecorder) else None,
            control_plane_report_path=self._control_plane_report_path,
            frame_diagnostics=tuple(self.frame_diagnostics),
            map_snapshot_path=map_snapshot_path,
            map_stats=map_stats,
        )

    def _handle_tracking_result(self, result: TrackingFrameResult) -> None:
        event = TelemetryEvent(
            name="tracking_control",
            duration_s=float(result.total_wait_s),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "frame_id": result.seq_id,
                "feature_queue_wait_s": result.feature_queue_wait_s,
                "drop_reason": result.drop_reason,
            },
        )
        self.telemetry.record_event(event)
        frame_gray = result.frame_gray
        timestamp = result.timestamp
        if result.drop_reason is not None:
            LOGGER.warning(
                "Tracking frame %d dropped: %s",
                result.seq_id,
                result.drop_reason,
            )
            self._prev_frame = frame_gray
            self._prev_kp = None
            self._prev_desc = None
            self._append_pose(
                timestamp,
                method="tracking_drop",
                match_count=0,
                inliers=0,
                status="dropped",
                failure_reason=result.drop_reason,
            )
            return
        feature_result = result.feature_result
        if feature_result is None:
            LOGGER.warning("Missing feature result for frame %d", result.seq_id)
            return
        feature_event = TelemetryEvent(
            name="feature_detect_async",
            duration_s=float(feature_result.duration_s),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "frame_id": feature_result.seq_id,
                "queue_wait_s": feature_result.queue_wait_s,
                "cache_hit": feature_result.cache_hit,
                "error": feature_result.error,
            },
        )
        self.telemetry.record_event(feature_event)
        if feature_result.error:
            LOGGER.warning(
                "Feature extraction failed for frame %d: %s",
                feature_result.seq_id,
                feature_result.error,
            )
            self._prev_frame = frame_gray
            self._prev_kp = None
            self._prev_desc = None
            self._append_pose(
                timestamp,
                method="feature_failure",
                match_count=0,
                inliers=0,
                status="skipped",
                failure_reason=feature_result.error,
            )
            return
        self._process_frame_with_features(
            frame_gray,
            timestamp,
            feature_result.keypoints,
            feature_result.descriptors,
        )

    def load_map_snapshot(self, map_dir: Path) -> None:
        store = PersistentMapStore()
        snapshot = store.load(map_dir)
        self._relocalizer_snapshot = snapshot
        self._relocalizer = MapRelocalizer(
            snapshot,
            self.config.intrinsics,
            min_matches=self.config.relocalization_min_matches,
            min_inliers=self.config.relocalization_min_inliers,
            max_candidates=self.config.relocalization_max_candidates,
            score_threshold=self.config.relocalization_score_threshold,
            ransac_threshold=self.config.relocalization_ransac_threshold,
            verify_geometry=True,
        )

    def _append_pose(
        self,
        timestamp: float,
        *,
        method: str,
        match_count: int,
        inliers: int,
        status: str,
        failure_reason: str | None,
    ) -> None:
        self.trajectory.append(self._current_pose.copy(), timestamp, self._frame_id)
        inlier_ratio = 0.0 if match_count <= 0 else float(inliers) / float(match_count)
        self.frame_diagnostics.append(
            FrameDiagnostics(
                frame_id=self._frame_id,
                timestamp=float(timestamp),
                match_count=int(match_count),
                inliers=int(inliers),
                method=str(method),
                inlier_ratio=float(inlier_ratio),
                median_parallax=0.0,
                score=0.0,
                status=str(status),
                failure_reason=failure_reason,
            )
        )
        self._frame_id += 1

    def _append_pose_with_diagnostics(
        self,
        timestamp: float,
        diagnostics: PoseEstimationDiagnostics,
    ) -> None:
        self.trajectory.append(self._current_pose.copy(), timestamp, self._frame_id)
        self.frame_diagnostics.append(
            FrameDiagnostics(
                frame_id=self._frame_id,
                timestamp=float(timestamp),
                match_count=int(diagnostics.match_count),
                inliers=int(diagnostics.inliers),
                method=str(diagnostics.method),
                inlier_ratio=float(diagnostics.inlier_ratio),
                median_parallax=float(diagnostics.median_parallax),
                score=float(diagnostics.score),
                status="ok",
                failure_reason=None,
            )
        )
        self._frame_id += 1

    def _append_pose_failure(self, timestamp: float, error: Exception) -> None:
        failure_reason = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        self._append_pose(
            timestamp,
            method="pose_failure",
            match_count=0,
            inliers=0,
            status="failure",
            failure_reason=failure_reason,
        )

    def _build_telemetry_sink(self) -> TelemetrySink:
        if self.config.telemetry_sink is not None:
            return self.config.telemetry_sink
        if not self.config.enable_telemetry:
            return _NullTelemetrySink()
        telemetry_path = self.data_store.telemetry_path(self.config.telemetry_name)
        return RunTelemetryRecorder(telemetry_path)

    def _maybe_add_keyframe(
        self,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray,
    ) -> None:
        if descriptors is None or len(descriptors) == 0:
            return
        if self._keyframe_manager.should_add_keyframe(self._current_pose, descriptors):
            self._keyframe_manager.add_keyframe(
                frame_id=self._frame_id,
                pose=self._current_pose,
                keypoints=keypoints,
                descriptors=descriptors,
            )
            self._map_dirty = True

    def _build_map_snapshot(self) -> PersistentMapSnapshot | None:
        if not self._keyframe_manager.keyframes:
            return None
        with timed_event("map_snapshot_build", self.telemetry, {"keyframes": len(self._keyframe_manager.keyframes)}):
            snapshot, stats = self._map_builder.build_snapshot(self._keyframe_manager.keyframes)
        self._last_map_snapshot = snapshot
        self._last_map_stats = stats
        return snapshot

    def _ensure_relocalizer(self) -> MapRelocalizer | None:
        if self._relocalizer is not None and not self._map_dirty:
            return self._relocalizer
        if not self._keyframe_manager.keyframes:
            return None
        with timed_event(
            "map_snapshot_refresh",
            self.telemetry,
            {"keyframes": len(self._keyframe_manager.keyframes)},
        ):
            snapshot, stats = self._map_builder.build_snapshot(self._keyframe_manager.keyframes)
        self._relocalizer_snapshot = snapshot
        self._last_map_snapshot = snapshot
        self._last_map_stats = stats
        self._relocalizer = MapRelocalizer(
            snapshot,
            self.config.intrinsics,
            min_matches=self.config.relocalization_min_matches,
            min_inliers=self.config.relocalization_min_inliers,
            max_candidates=self.config.relocalization_max_candidates,
            score_threshold=self.config.relocalization_score_threshold,
            ransac_threshold=self.config.relocalization_ransac_threshold,
            verify_geometry=True,
        )
        self._map_dirty = False
        return self._relocalizer

    def _attempt_relocalization(
        self,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray,
        frame_gray: np.ndarray,
        timestamp: float,
    ) -> bool:
        relocalizer = self._ensure_relocalizer()
        if relocalizer is None:
            return False
        if descriptors is None or len(descriptors) == 0:
            return False
        with timed_event(
            "relocalization_search",
            self.telemetry,
            {"frame_id": self._frame_id},
        ):
            result = relocalizer.relocalize(keypoints, descriptors)
        if result is None:
            LOGGER.info("Relocalization failed for frame %d", self._frame_id)
            return False
        kf = self._keyframe_manager.keyframes_by_id().get(result.frame_id)
        if kf is None:
            LOGGER.warning("Relocalization keyframe %d not found", result.frame_id)
            return False
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = result.rotation
        relative_pose[:3, 3] = result.translation
        self._current_pose = kf.pose @ relative_pose
        self._prev_frame = frame_gray
        self._prev_kp = keypoints
        self._prev_desc = descriptors
        self._append_pose(
            timestamp,
            method="relocalization",
            match_count=result.inliers,
            inliers=result.inliers,
            status="relocalized",
            failure_reason=None,
        )
        LOGGER.info(
            "Relocalized frame %d against keyframe %d",
            self._frame_id,
            result.frame_id,
        )
        return True


class _NullTelemetrySink:
    def record_event(self, event: Any) -> None:
        return None
