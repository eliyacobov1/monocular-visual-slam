"""Data persistence layer for SLAM runs, trajectories, metrics, and maps."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TYPE_CHECKING
from collections import Counter

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
    inlier_ratio: float
    median_parallax: float
    score: float
    status: str
    failure_reason: str | None


@dataclass(frozen=True)
class FrameDiagnosticsBundle:
    """Frame diagnostics bundle for a run."""

    name: str
    entries: tuple[FrameDiagnosticsEntry, ...]
    recorded_at: str


class P2Quantile:
    """Streaming quantile estimator using the P² algorithm."""

    def __init__(self, quantile: float) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("Quantile must be between 0 and 1")
        self.quantile = float(quantile)
        self._count = 0
        self._initial: list[float] = []
        self._positions = [0, 0, 0, 0, 0]
        self._desired = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._increments = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._heights = [0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def count(self) -> int:
        return self._count

    def update(self, value: float) -> None:
        value = float(value)
        self._count += 1
        if self._count <= 5:
            self._initial.append(value)
            if self._count == 5:
                self._initial.sort()
                self._heights = list(self._initial)
                self._positions = [1, 2, 3, 4, 5]
                q = self.quantile
                self._desired = [1.0, 1.0 + 2.0 * q, 1.0 + 4.0 * q, 3.0 + 2.0 * q, 5.0]
                self._increments = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]
            return

        if value < self._heights[0]:
            self._heights[0] = value
            k = 0
        elif value >= self._heights[4]:
            self._heights[4] = value
            k = 3
        else:
            k = 0
            for idx in range(4):
                if self._heights[idx] <= value < self._heights[idx + 1]:
                    k = idx
                    break

        for idx in range(k + 1, 5):
            self._positions[idx] += 1
        for idx in range(5):
            self._desired[idx] += self._increments[idx]

        for idx in range(1, 4):
            delta = self._desired[idx] - self._positions[idx]
            if (delta >= 1 and self._positions[idx + 1] - self._positions[idx] > 1) or (
                delta <= -1 and self._positions[idx - 1] - self._positions[idx] < -1
            ):
                step = int(np.sign(delta))
                updated = self._parabolic_update(idx, step)
                if self._heights[idx - 1] < updated < self._heights[idx + 1]:
                    self._heights[idx] = updated
                else:
                    self._heights[idx] = self._linear_update(idx, step)
                self._positions[idx] += step

    def value(self) -> float:
        if self._count == 0:
            raise ValueError("No samples available for quantile estimation")
        if self._count < 5:
            return float(np.quantile(self._initial, self.quantile))
        return float(self._heights[2])

    def _parabolic_update(self, idx: int, step: int) -> float:
        n_i = self._positions[idx]
        n_im1 = self._positions[idx - 1]
        n_ip1 = self._positions[idx + 1]
        q_i = self._heights[idx]
        q_im1 = self._heights[idx - 1]
        q_ip1 = self._heights[idx + 1]
        numerator = step * (n_i - n_im1 + step) * (q_ip1 - q_i) / (n_ip1 - n_i)
        numerator += step * (n_ip1 - n_i - step) * (q_i - q_im1) / (n_i - n_im1)
        return q_i + numerator / (n_ip1 - n_im1)

    def _linear_update(self, idx: int, step: int) -> float:
        return self._heights[idx] + step * (
            (self._heights[idx + step] - self._heights[idx])
            / (self._positions[idx + step] - self._positions[idx])
        )


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
        self._telemetry_dir = metadata.run_dir / "telemetry"
        self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._maps_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry_dir.mkdir(parents=True, exist_ok=True)

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
        filename = f"{sanitize_artifact_name(bundle.name)}.npz"
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
        filename = f"{sanitize_artifact_name(name)}.npz"
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
        metrics_path = self._metrics_dir / f"{sanitize_artifact_name(bundle.name)}.json"
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
            self._diagnostics_dir / f"{sanitize_artifact_name(bundle.name)}.json"
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
                    "inlier_ratio": entry.inlier_ratio,
                    "median_parallax": entry.median_parallax,
                    "score": entry.score,
                    "status": entry.status,
                    "failure_reason": entry.failure_reason,
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

    def save_control_plane_report(self, name: str, payload: Mapping[str, Any]) -> Path:
        if not name:
            raise ValueError("Control-plane report name must be non-empty")
        report_path = self._telemetry_dir / f"{sanitize_artifact_name(name)}.json"
        try:
            report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            LOGGER.exception("Failed to write control-plane report '%s'", name)
            raise RuntimeError("Failed to write control-plane report") from exc
        LOGGER.info("Saved control-plane report '%s' to %s", name, report_path)
        return report_path

    def load_metrics(self, name: str) -> MetricsBundle:
        metrics_path = self._metrics_dir / f"{sanitize_artifact_name(name)}.json"
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
            self._diagnostics_dir / f"{sanitize_artifact_name(name)}.json"
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
                inlier_ratio=float(entry.get("inlier_ratio", 0.0)),
                median_parallax=float(entry.get("median_parallax", 0.0)),
                score=float(entry.get("score", 0.0)),
                status=str(entry.get("status", "unknown")),
                failure_reason=entry.get("failure_reason"),
            )
            for entry in entries_payload
        )
        return FrameDiagnosticsBundle(
            name=str(payload.get("name", name)),
            entries=entries,
            recorded_at=str(payload.get("recorded_at", _timestamp())),
        )

    def save_map_snapshot(self, name: str, snapshot: "PersistentMapSnapshot") -> MapBundle:
        safe_name = sanitize_artifact_name(name)
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

    def telemetry_path(self, name: str) -> Path:
        safe_name = sanitize_artifact_name(name)
        return self._telemetry_dir / f"{safe_name}.json"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return sanitize_artifact_name(name)

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


def sanitize_artifact_name(name: str) -> str:
    """Return a filesystem-safe artifact name."""

    if not name:
        raise ValueError("Name must be non-empty")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "run"


def trajectory_artifact_path(run_dir: Path, name: str) -> Path:
    """Return the expected trajectory artifact path for a run directory."""

    safe_name = sanitize_artifact_name(name)
    return Path(run_dir) / "trajectories" / f"{safe_name}.npz"


def frame_diagnostics_artifact_path(run_dir: Path, name: str) -> Path:
    """Return the expected frame diagnostics artifact path for a run directory."""

    safe_name = sanitize_artifact_name(name)
    return Path(run_dir) / "diagnostics" / f"{safe_name}.json"


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


def load_trajectory_npz(path: Path, name: str | None = None) -> TrajectoryBundle:
    """Load a trajectory bundle from a persisted npz file."""

    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    try:
        data = np.load(path)
    except OSError as exc:
        LOGGER.exception("Failed to read trajectory npz '%s'", path)
        raise RuntimeError("Failed to read trajectory npz") from exc
    if "poses" not in data or "timestamps" not in data:
        raise ValueError("Trajectory npz must include poses and timestamps arrays")
    poses = data["poses"]
    timestamps = data["timestamps"]
    frame_ids = data.get("frame_ids")
    bundle = TrajectoryBundle(
        name=name or path.stem,
        poses=poses,
        timestamps=timestamps,
        frame_ids=frame_ids if frame_ids is not None and len(frame_ids) > 0 else None,
    )
    RunDataStore._validate_trajectory(bundle)
    return bundle


def load_frame_diagnostics_json(
    path: Path,
    name: str | None = None,
) -> FrameDiagnosticsBundle:
    """Load a frame diagnostics bundle from a persisted json file."""

    if not path.exists():
        raise FileNotFoundError(f"Frame diagnostics file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Failed to read frame diagnostics '%s'", path)
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
            inlier_ratio=float(entry.get("inlier_ratio", 0.0)),
            median_parallax=float(entry.get("median_parallax", 0.0)),
            score=float(entry.get("score", 0.0)),
            status=str(entry.get("status", "unknown")),
            failure_reason=entry.get("failure_reason"),
        )
        for entry in entries_payload
    )
    return FrameDiagnosticsBundle(
        name=str(payload.get("name", name or path.stem)),
        entries=entries,
        recorded_at=str(payload.get("recorded_at", _timestamp())),
    )


def iter_json_array_items(path: Path, key: str) -> Iterator[dict[str, Any]]:
    """Stream items from a JSON array payload without loading it fully."""

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    decoder = json.JSONDecoder()
    key_token = f"\"{key}\""
    buffer = ""
    in_array = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                if not in_array:
                    key_index = buffer.find(key_token)
                    if key_index == -1:
                        if len(buffer) > len(key_token):
                            buffer = buffer[-len(key_token):]
                        break
                    bracket_index = buffer.find("[", key_index + len(key_token))
                    if bracket_index == -1:
                        buffer = buffer[key_index:]
                        break
                    buffer = buffer[bracket_index + 1 :]
                    in_array = True
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                try:
                    item, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if isinstance(item, dict):
                    yield item
                else:
                    raise ValueError(f"Expected object items for '{key}' in {path}")
                buffer = buffer[index:]

        if not in_array:
            raise ValueError(f"Array key '{key}' not found in {path}")
        buffer = buffer.lstrip()
        while buffer:
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
                continue
            item, index = decoder.raw_decode(buffer)
            if not isinstance(item, dict):
                raise ValueError(f"Expected object items for '{key}' in {path}")
            yield item
            buffer = buffer[index:].lstrip()
        raise ValueError(f"JSON array '{key}' not terminated in {path}")


def summarize_frame_diagnostics_streaming(path: Path) -> dict[str, float]:
    """Stream frame diagnostics summaries to avoid loading all entries."""

    total = 0
    match_sum = 0.0
    inlier_sum = 0.0
    ratio_sum = 0.0
    score_sum = 0.0
    parallax_median = P2Quantile(0.5)
    method_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()

    for entry in iter_json_array_items(path, "entries"):
        total += 1
        match_sum += float(entry.get("match_count", 0.0))
        inlier_sum += float(entry.get("inliers", 0.0))
        ratio_sum += float(entry.get("inlier_ratio", 0.0))
        score_sum += float(entry.get("score", 0.0))
        parallax_median.update(float(entry.get("median_parallax", 0.0)))
        method = str(entry.get("method") or "unknown")
        status = str(entry.get("status") or "unknown")
        method_counts[method] += 1
        status_counts[status] += 1
        failure_reason = entry.get("failure_reason")
        if failure_reason:
            failure_counts[str(failure_reason)] += 1

    if total == 0:
        raise ValueError("Frame diagnostics payload is empty")

    total_float = float(total)
    metrics: dict[str, float] = {
        "diag_frame_count": total_float,
        "diag_match_mean": match_sum / total_float,
        "diag_inlier_mean": inlier_sum / total_float,
        "diag_inlier_ratio_mean": ratio_sum / total_float,
        "diag_parallax_median": parallax_median.value(),
        "diag_score_mean": score_sum / total_float,
    }
    for method, count in method_counts.items():
        safe_method = sanitize_artifact_name(method)
        metrics[f"diag_method_{safe_method}_count"] = float(count)
        metrics[f"diag_method_{safe_method}_ratio"] = float(count / total_float)
    for status, count in status_counts.items():
        safe_status = sanitize_artifact_name(status)
        metrics[f"diag_status_{safe_status}_count"] = float(count)
        metrics[f"diag_status_{safe_status}_ratio"] = float(count / total_float)
    if failure_counts:
        failures_total = float(sum(failure_counts.values()))
        for reason, count in failure_counts.items():
            safe_reason = sanitize_artifact_name(reason)
            metrics[f"diag_failure_{safe_reason}_count"] = float(count)
            metrics[f"diag_failure_{safe_reason}_ratio"] = float(count / failures_total)
    return metrics


def summarize_frame_diagnostics(bundle: FrameDiagnosticsBundle) -> dict[str, float]:
    """Summarize frame diagnostics statistics for evaluation reports."""

    if not bundle.entries:
        raise ValueError("Frame diagnostics bundle must include entries")
    match_counts = np.array([entry.match_count for entry in bundle.entries], dtype=float)
    inliers = np.array([entry.inliers for entry in bundle.entries], dtype=float)
    inlier_ratios = np.array([entry.inlier_ratio for entry in bundle.entries], dtype=float)
    parallaxes = np.array([entry.median_parallax for entry in bundle.entries], dtype=float)
    scores = np.array([entry.score for entry in bundle.entries], dtype=float)
    method_counts = Counter(entry.method or "unknown" for entry in bundle.entries)
    status_counts = Counter(entry.status or "unknown" for entry in bundle.entries)
    failure_counts = Counter(
        entry.failure_reason or "unknown"
        for entry in bundle.entries
        if entry.failure_reason
    )
    total = float(len(bundle.entries))

    metrics: dict[str, float] = {
        "diag_frame_count": total,
        "diag_match_mean": float(match_counts.mean()),
        "diag_inlier_mean": float(inliers.mean()),
        "diag_inlier_ratio_mean": float(inlier_ratios.mean()),
        "diag_parallax_median": float(np.median(parallaxes)),
        "diag_score_mean": float(scores.mean()),
    }
    for method, count in method_counts.items():
        safe_method = sanitize_artifact_name(method)
        metrics[f"diag_method_{safe_method}_count"] = float(count)
        metrics[f"diag_method_{safe_method}_ratio"] = float(count / total)
    for status, count in status_counts.items():
        safe_status = sanitize_artifact_name(status)
        metrics[f"diag_status_{safe_status}_count"] = float(count)
        metrics[f"diag_status_{safe_status}_ratio"] = float(count / total)
    if failure_counts:
        failures_total = float(sum(failure_counts.values()))
        for reason, count in failure_counts.items():
            safe_reason = sanitize_artifact_name(reason)
            metrics[f"diag_failure_{safe_reason}_count"] = float(count)
            metrics[f"diag_failure_{safe_reason}_ratio"] = float(count / failures_total)
    return metrics


def trajectory_positions(
    trajectory: TrajectoryBundle,
    columns: Sequence[int] | None = None,
) -> np.ndarray:
    """Return translation positions from a trajectory bundle."""

    if trajectory.poses.shape[1:] != (4, 4):
        raise ValueError("Trajectory poses must be shaped (N,4,4)")
    positions = trajectory.poses[:, :3, 3]
    if columns is None:
        return positions.astype(np.float64, copy=False)
    if not columns:
        raise ValueError("Trajectory columns must be non-empty when provided")
    indices = [int(idx) for idx in columns]
    if any(idx < 0 or idx > 2 for idx in indices):
        raise ValueError("Trajectory columns must be between 0 and 2")
    return positions[:, indices].astype(np.float64, copy=False)


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
