"""Data persistence layer for SLAM runs, trajectories, metrics, and maps."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import numpy as np

from experiment_registry import create_run_artifacts, write_resolved_config

if TYPE_CHECKING:
    from persistent_map import PersistentMapSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunMetadata:
    """Metadata for a SLAM run."""

    run_id: str
    run_dir: Path
    created_at: str
    config_path: Path
    config_hash: str
    resolved_config_path: Path | None


@dataclass(frozen=True)
class TrajectoryBundle:
    """Trajectory bundle containing poses and timestamps."""

    name: str
    poses: np.ndarray
    timestamps: np.ndarray
    frame_ids: np.ndarray | None


@dataclass(frozen=True)
class MetricsBundle:
    """Metrics bundle for a run."""

    name: str
    metrics: dict[str, float]
    recorded_at: str


@dataclass(frozen=True)
class FrameDiagnosticsEntry:
    """Per-frame diagnostics entry for pose estimation."""

    frame_id: int
    timestamp: float
    match_count: int
    inliers: int
    method: str


@dataclass(frozen=True)
class FrameDiagnosticsBundle:
    """Frame diagnostics bundle for a run."""

    name: str
    entries: tuple[FrameDiagnosticsEntry, ...]
    recorded_at: str


@dataclass(frozen=True)
class MapBundle:
    """Map bundle linking to stored persistent map data."""

    name: str
    path: Path
    created_at: str


@dataclass
class TrajectoryAccumulator:
    """Mutable trajectory accumulator for incremental pipelines."""

    name: str
    poses: list[np.ndarray]
    timestamps: list[float]
    frame_ids: list[int]

    def append(self, pose: np.ndarray, timestamp: float, frame_id: int) -> None:
        if pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix")
        self.poses.append(pose.astype(np.float64, copy=False))
        self.timestamps.append(float(timestamp))
        self.frame_ids.append(int(frame_id))

    def as_bundle(self) -> TrajectoryBundle:
        if not self.poses:
            raise ValueError("Trajectory accumulator has no poses")
        poses = np.stack(self.poses)
        timestamps = np.array(self.timestamps, dtype=np.float64)
        frame_ids = np.array(self.frame_ids, dtype=np.int64)
        return TrajectoryBundle(
            name=self.name,
            poses=poses,
            timestamps=timestamps,
            frame_ids=frame_ids,
        )


class RunDataStore:
    """Persist SLAM run artifacts with validation and structured logging."""

    def __init__(self, metadata: RunMetadata) -> None:
        self.metadata = metadata
        self._trajectory_dir = metadata.run_dir / "trajectories"
        self._metrics_dir = metadata.run_dir / "metrics"
        self._maps_dir = metadata.run_dir / "maps"
        self._diagnostics_dir = metadata.run_dir / "diagnostics"
        self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._maps_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostics_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        base_dir: Path,
        run_id: str,
        config_path: Path,
        config_hash: str,
        use_subdir: bool,
        resolved_config: Mapping[str, Any] | None = None,
    ) -> "RunDataStore":
        try:
            artifacts = create_run_artifacts(
                base_dir=base_dir,
                run_id=run_id,
                config_path=config_path,
                config_hash=config_hash,
                use_subdir=use_subdir,
            )
            resolved_config_path = None
            if resolved_config is not None:
                resolved_config_path = write_resolved_config(
                    artifacts.run_dir, dict(resolved_config)
                )
        except (OSError, ValueError) as exc:
            LOGGER.exception("Failed to create run artifacts")
            raise RuntimeError("Could not create run artifacts") from exc

        metadata = RunMetadata(
            run_id=run_id,
            run_dir=artifacts.run_dir,
            created_at=artifacts.created_at,
            config_path=config_path,
            config_hash=config_hash,
            resolved_config_path=resolved_config_path,
        )
        return cls(metadata)

    def create_accumulator(self, name: str) -> TrajectoryAccumulator:
        if not name:
            raise ValueError("Trajectory name must be non-empty")
        return TrajectoryAccumulator(name=name, poses=[], timestamps=[], frame_ids=[])

    def save_trajectory(self, bundle: TrajectoryBundle) -> Path:
        self._validate_trajectory(bundle)
        filename = f"{self._sanitize_name(bundle.name)}.npz"
        path = self._trajectory_dir / filename
        try:
            np.savez_compressed(
                path,
                poses=bundle.poses,
                timestamps=bundle.timestamps,
                frame_ids=bundle.frame_ids if bundle.frame_ids is not None else np.array([], dtype=np.int64),
            )
        except OSError as exc:
            LOGGER.exception("Failed to write trajectory '%s'", bundle.name)
            raise RuntimeError("Failed to write trajectory") from exc
        LOGGER.info("Saved trajectory '%s' to %s", bundle.name, path)
        return path

    def load_trajectory(self, name: str) -> TrajectoryBundle:
        filename = f"{self._sanitize_name(name)}.npz"
        path = self._trajectory_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Trajectory '{name}' not found")
        try:
            data = np.load(path)
        except OSError as exc:
            LOGGER.exception("Failed to read trajectory '%s'", name)
            raise RuntimeError("Failed to read trajectory") from exc
        poses = data["poses"]
        timestamps = data["timestamps"]
        frame_ids = data.get("frame_ids")
        bundle = TrajectoryBundle(
            name=name,
            poses=poses,
            timestamps=timestamps,
            frame_ids=frame_ids if frame_ids is not None and len(frame_ids) > 0 else None,
        )
        self._validate_trajectory(bundle)
        return bundle

    def save_metrics(self, bundle: MetricsBundle) -> Path:
        if not bundle.metrics:
            raise ValueError("Metrics bundle must include at least one metric")
        metrics_path = self._metrics_dir / f"{self._sanitize_name(bundle.name)}.json"
        payload = {
            "name": bundle.name,
            "recorded_at": bundle.recorded_at,
            "metrics": bundle.metrics,
        }
        try:
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            LOGGER.exception("Failed to write metrics '%s'", bundle.name)
            raise RuntimeError("Failed to write metrics") from exc
        LOGGER.info("Saved metrics '%s' to %s", bundle.name, metrics_path)
        return metrics_path

    def save_frame_diagnostics(self, bundle: FrameDiagnosticsBundle) -> Path:
        if not bundle.entries:
            raise ValueError("Frame diagnostics bundle must include entries")
        diagnostics_path = (
            self._diagnostics_dir / f"{self._sanitize_name(bundle.name)}.json"
        )
        payload = {
            "name": bundle.name,
            "recorded_at": bundle.recorded_at,
            "entries": [
                {
                    "frame_id": entry.frame_id,
                    "timestamp": entry.timestamp,
                    "match_count": entry.match_count,
                    "inliers": entry.inliers,
                    "method": entry.method,
                }
                for entry in bundle.entries
            ],
        }
        try:
            diagnostics_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            LOGGER.exception("Failed to write frame diagnostics '%s'", bundle.name)
            raise RuntimeError("Failed to write frame diagnostics") from exc
        LOGGER.info("Saved frame diagnostics '%s' to %s", bundle.name, diagnostics_path)
        return diagnostics_path

    def load_metrics(self, name: str) -> MetricsBundle:
        metrics_path = self._metrics_dir / f"{self._sanitize_name(name)}.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics '{name}' not found")
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.exception("Failed to read metrics '%s'", name)
            raise RuntimeError("Failed to read metrics") from exc
        metrics = {str(k): float(v) for k, v in payload.get("metrics", {}).items()}
        if not metrics:
            raise ValueError("Metrics payload is empty")
        return MetricsBundle(
            name=str(payload.get("name", name)),
            metrics=metrics,
            recorded_at=str(payload.get("recorded_at", _timestamp())),
        )

    def load_frame_diagnostics(self, name: str) -> FrameDiagnosticsBundle:
        diagnostics_path = (
            self._diagnostics_dir / f"{self._sanitize_name(name)}.json"
        )
        if not diagnostics_path.exists():
            raise FileNotFoundError(f"Frame diagnostics '{name}' not found")
        try:
            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.exception("Failed to read frame diagnostics '%s'", name)
            raise RuntimeError("Failed to read frame diagnostics") from exc
        entries_payload = payload.get("entries", [])
        if not entries_payload:
            raise ValueError("Frame diagnostics payload is empty")
        entries = tuple(
            FrameDiagnosticsEntry(
                frame_id=int(entry.get("frame_id", 0)),
                timestamp=float(entry.get("timestamp", 0.0)),
                match_count=int(entry.get("match_count", 0)),
                inliers=int(entry.get("inliers", 0)),
                method=str(entry.get("method", "")),
            )
            for entry in entries_payload
        )
        return FrameDiagnosticsBundle(
            name=str(payload.get("name", name)),
            entries=entries,
            recorded_at=str(payload.get("recorded_at", _timestamp())),
        )

    def save_map_snapshot(self, name: str, snapshot: "PersistentMapSnapshot") -> MapBundle:
        safe_name = self._sanitize_name(name)
        map_dir = self._maps_dir / safe_name
        from persistent_map import PersistentMapStore
        store = PersistentMapStore()
        try:
            store.save(map_dir, snapshot)
        except (OSError, ValueError) as exc:
            LOGGER.exception("Failed to persist map snapshot '%s'", name)
            raise RuntimeError("Failed to persist map snapshot") from exc
        bundle = MapBundle(name=name, path=map_dir, created_at=_timestamp())
        metadata_path = map_dir / "map_bundle.json"
        try:
            metadata_path.write_text(
                json.dumps(
                    {
                        "name": name,
                        "created_at": bundle.created_at,
                        "path": str(map_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.exception("Failed to write map bundle metadata '%s'", name)
            raise RuntimeError("Failed to write map bundle metadata") from exc
        LOGGER.info("Saved map snapshot '%s' to %s", name, map_dir)
        return bundle

    def list_trajectories(self) -> list[str]:
        return sorted(path.stem for path in self._trajectory_dir.glob("*.npz"))

    def list_metrics(self) -> list[str]:
        return sorted(path.stem for path in self._metrics_dir.glob("*.json"))

    def list_map_snapshots(self) -> list[str]:
        return sorted(path.name for path in self._maps_dir.iterdir() if path.is_dir())

    @staticmethod
    def _sanitize_name(name: str) -> str:
        if not name:
            raise ValueError("Name must be non-empty")
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        return safe.strip("_") or "run"

    @staticmethod
    def _validate_trajectory(bundle: TrajectoryBundle) -> None:
        if bundle.poses.ndim != 3 or bundle.poses.shape[1:] != (4, 4):
            raise ValueError("Trajectory poses must be shaped (N,4,4)")
        if bundle.timestamps.ndim != 1:
            raise ValueError("Trajectory timestamps must be a 1D array")
        if len(bundle.timestamps) != len(bundle.poses):
            raise ValueError("Trajectory timestamps must align with poses")
        if bundle.frame_ids is not None and len(bundle.frame_ids) != len(bundle.poses):
            raise ValueError("Trajectory frame_ids must align with poses")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_metrics_bundle(name: str, metrics: Mapping[str, float]) -> MetricsBundle:
    """Build a metrics bundle with a current timestamp."""

    return MetricsBundle(
        name=name,
        metrics={str(k): float(v) for k, v in metrics.items()},
        recorded_at=_timestamp(),
    )


def build_frame_diagnostics_bundle(
    name: str,
    entries: Iterable[FrameDiagnosticsEntry],
) -> FrameDiagnosticsBundle:
    """Build a frame diagnostics bundle with a current timestamp."""

    return FrameDiagnosticsBundle(
        name=name,
        entries=tuple(entries),
        recorded_at=_timestamp(),
    )


def summarize_trajectory(trajectory: TrajectoryBundle) -> dict[str, float]:
    """Summarize trajectory statistics for quick diagnostics."""

    if len(trajectory.poses) < 2:
        raise ValueError("Trajectory must contain at least 2 poses to summarize")
    translations = trajectory.poses[:, :3, 3]
    step_lengths = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    return {
        "num_poses": float(len(trajectory.poses)),
        "total_distance": float(step_lengths.sum()),
        "mean_step": float(step_lengths.mean()),
        "max_step": float(step_lengths.max()),
    }


def load_metrics_series(directory: Path) -> list[MetricsBundle]:
    """Load all metrics bundles from a directory."""

    bundles: list[MetricsBundle] = []
    if not directory.exists():
        raise FileNotFoundError(f"Metrics directory not found: {directory}")
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.exception("Failed to parse metrics bundle %s", path)
            raise RuntimeError("Failed to parse metrics bundle") from exc
        metrics = {str(k): float(v) for k, v in payload.get("metrics", {}).items()}
        if not metrics:
            raise ValueError(f"Metrics bundle {path} is empty")
        bundles.append(
            MetricsBundle(
                name=str(payload.get("name", path.stem)),
                metrics=metrics,
                recorded_at=str(payload.get("recorded_at", _timestamp())),
            )
        )
    return bundles


def merge_metric_bundles(bundles: Iterable[MetricsBundle]) -> dict[str, list[float]]:
    """Merge metrics bundles into a series for each metric name."""

    merged: dict[str, list[float]] = {}
    for bundle in bundles:
        for key, value in bundle.metrics.items():
            merged.setdefault(key, []).append(float(value))
    return merged
