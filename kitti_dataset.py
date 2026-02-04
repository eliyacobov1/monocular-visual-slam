#!/usr/bin/env python3
"""KITTI dataset utilities for sequence loading and calibration parsing."""

from __future__ import annotations

import logging
from bisect import bisect_left
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, Mapping
import math

import numpy as np

from camera_rig import CameraRig
from run_telemetry import TelemetrySink, timed_event

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KittiFrame:
    index: int
    path: Path
    timestamp: float | None


def parse_kitti_calib_file(path: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI calibration file with `key: value` rows."""
    calib: dict[str, np.ndarray] = {}
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, values = line.split(":", 1)
        data = np.fromstring(values.strip(), sep=" ")
        if data.size == 9:
            calib[key.strip()] = data.reshape(3, 3)
        elif data.size == 12:
            calib[key.strip()] = data.reshape(3, 4)
        else:
            calib[key.strip()] = data
    return calib


def parse_kitti_timestamp(value: str) -> datetime | None:
    """Parse a KITTI timestamp string into a datetime.

    Supports ISO timestamps (as in KITTI raw) and float seconds.
    """
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except ValueError:
        pass
    try:
        normalized = raw.replace(" ", "T")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def resolve_camera_matrix(calib: dict[str, np.ndarray], camera_id: int) -> np.ndarray:
    """Return the projection matrix for a camera if present."""
    keys = [
        f"P{camera_id}",
        f"P_rect_0{camera_id}",
        f"P_rect_{camera_id}",
    ]
    for key in keys:
        if key in calib:
            matrix = calib[key]
            if matrix.shape == (3, 4):
                return matrix
    raise KeyError(f"Camera matrix not found for camera {camera_id}")


def intrinsics_from_projection(P: np.ndarray) -> np.ndarray:
    """Extract the 3x3 intrinsic matrix from a 3x4 projection matrix."""
    if P.shape != (3, 4):
        raise ValueError("Projection matrix must be 3x4")
    K = P[:3, :3]
    return K / K[2, 2]


def camera_id_from_name(camera: str) -> int:
    mapping = {
        "image_0": 0,
        "image_1": 1,
        "image_2": 2,
        "image_3": 3,
        "image_00": 0,
        "image_01": 1,
        "image_02": 2,
        "image_03": 3,
    }
    if camera not in mapping:
        raise ValueError(f"Unknown camera name: {camera}")
    return mapping[camera]


class KittiSequence:
    """Iterate through a KITTI odometry or raw sequence."""

    def __init__(self, root: Path, sequence: str, camera: str = "image_2") -> None:
        self.root = Path(root)
        self.sequence = sequence
        self.camera = camera
        self.sequence_path = self._resolve_sequence_path()
        self.image_dir = self._resolve_image_dir()
        self.timestamps = self._load_timestamps()
        self.calib = self._load_calibration()

    def _resolve_sequence_path(self) -> Path:
        candidate = self.root / "sequences" / self.sequence
        if candidate.exists():
            return candidate
        candidate = self.root / self.sequence
        if candidate.exists():
            return candidate
        raw_candidates = sorted(self.root.glob(f"*/{self.sequence}"))
        if raw_candidates:
            return raw_candidates[0]
        raise FileNotFoundError(
            f"KITTI sequence '{self.sequence}' not found under {self.root}"
        )

    def _resolve_image_dir(self) -> Path:
        candidates = []
        camera_aliases = {
            "image_0": ["image_0", "image_00"],
            "image_1": ["image_1", "image_01"],
            "image_2": ["image_2", "image_02"],
            "image_3": ["image_3", "image_03"],
        }
        aliases = camera_aliases.get(self.camera, [self.camera])
        for alias in aliases:
            candidates.append(self.sequence_path / alias)
            candidates.append(self.sequence_path / alias / "data")

        for path in candidates:
            if not path.exists():
                continue
            if any(path.glob("*.png")):
                return path
        raise FileNotFoundError(
            f"No image directory found for camera {self.camera} in {self.sequence_path}"
        )

    def _load_timestamps(self) -> list[float | None]:
        times_path = self.sequence_path / "times.txt"
        if times_path.exists():
            return [float(t) for t in times_path.read_text().splitlines() if t.strip()]
        timestamp_path = self.image_dir / "timestamps.txt"
        if not timestamp_path.exists() and self.image_dir.name == "data":
            timestamp_path = self.image_dir.parent / "timestamps.txt"
        if timestamp_path.exists():
            parsed_times: list[datetime] = []
            for line in timestamp_path.read_text().splitlines():
                parsed = parse_kitti_timestamp(line)
                if parsed is None:
                    continue
                parsed_times.append(parsed)
            if not parsed_times:
                return []
            start = parsed_times[0]
            return [float((entry - start).total_seconds()) for entry in parsed_times]
        return []

    def _load_calibration(self) -> dict[str, np.ndarray]:
        search_roots = [self.sequence_path]
        if self.sequence_path.parent != self.root:
            search_roots.append(self.sequence_path.parent)
        for root in search_roots:
            for name in ("calib.txt", "calib_cam_to_cam.txt"):
                calib_path = root / name
                if calib_path.exists():
                    return parse_kitti_calib_file(calib_path)
        LOGGER.warning("No calibration file found for %s", self.sequence_path)
        return {}

    def __len__(self) -> int:
        return len(list(self.image_dir.glob("*.png")))

    def iter_frames(self) -> Iterator[KittiFrame]:
        images = sorted(self.image_dir.glob("*.png"))
        for idx, path in enumerate(images):
            timestamp = None
            if idx < len(self.timestamps):
                timestamp = self.timestamps[idx]
            yield KittiFrame(index=idx, path=path, timestamp=timestamp)

    def camera_intrinsics(self) -> np.ndarray | None:
        if not self.calib:
            return None
        camera_id = camera_id_from_name(self.camera)
        try:
            P = resolve_camera_matrix(self.calib, camera_id)
        except KeyError:
            return None
        return intrinsics_from_projection(P)

    def camera_rig(self, camera_names: list[str] | None = None) -> CameraRig | None:
        if not self.calib:
            return None
        return CameraRig.from_kitti_calibration(
            self.calib, camera_names=camera_names, reference_camera=self.camera
        )


@dataclass(frozen=True)
class MultiCameraSyncIssue:
    level: str
    message: str
    hint: str | None = None


@dataclass
class MultiCameraSyncReport:
    reference_camera: str
    tolerance_s: float
    issues: list[MultiCameraSyncIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add_issue(self, level: str, message: str, hint: str | None = None) -> None:
        self.issues.append(MultiCameraSyncIssue(level=level, message=message, hint=hint))

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reference_camera": self.reference_camera,
            "tolerance_s": self.tolerance_s,
            "issues": [
                {"level": issue.level, "message": issue.message, "hint": issue.hint}
                for issue in self.issues
            ],
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class SyncedMultiCameraFrame:
    index: int
    timestamp: float | None
    frames: dict[str, KittiFrame]
    offsets_s: dict[str, float]


class MultiCameraKittiSequence:
    """Synchronize multiple KITTI camera streams by timestamp or index."""

    def __init__(
        self,
        root: Path,
        sequence: str,
        cameras: Iterable[str],
        reference_camera: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.sequence = sequence
        self.cameras = list(dict.fromkeys(cameras))
        if not self.cameras:
            raise ValueError("At least one camera must be provided.")
        self.reference_camera = reference_camera or self.cameras[0]
        if self.reference_camera not in self.cameras:
            raise ValueError(
                f"Reference camera {self.reference_camera} not in camera list."
            )
        self.sequences: dict[str, KittiSequence] = {
            camera: KittiSequence(self.root, self.sequence, camera=camera)
            for camera in self.cameras
        }

    def synchronize(
        self,
        tolerance_s: float = 0.002,
        telemetry_sink: TelemetrySink | None = None,
    ) -> tuple[list[SyncedMultiCameraFrame], MultiCameraSyncReport]:
        metadata = {"tolerance_s": tolerance_s, "num_cameras": len(self.cameras)}

        def _run_sync() -> tuple[list[SyncedMultiCameraFrame], MultiCameraSyncReport]:
            start = perf_counter()
            report = MultiCameraSyncReport(
                reference_camera=self.reference_camera,
                tolerance_s=tolerance_s,
            )
            report.metrics["num_cameras"] = float(len(self.cameras))
            reference_sequence = self.sequences[self.reference_camera]
            reference_frames = list(reference_sequence.iter_frames())
            report.metrics["num_reference_frames"] = float(len(reference_frames))

            per_camera_frames: dict[str, list[KittiFrame]] = {
                name: list(seq.iter_frames()) for name, seq in self.sequences.items()
            }
            per_camera_timestamps: dict[str, list[float | None]] = {
                name: [frame.timestamp for frame in frames]
                for name, frames in per_camera_frames.items()
            }
            per_camera_time_ready = {
                name: _timestamps_ready(timestamps)
                for name, timestamps in per_camera_timestamps.items()
            }

            if not per_camera_time_ready[self.reference_camera]:
                report.add_issue(
                    "warning",
                    "Reference camera timestamps are missing or non-monotonic; falling back to index sync.",
                    hint="Ensure timestamps.txt or times.txt is available for all cameras.",
                )

            synced_frames: list[SyncedMultiCameraFrame] = []
            per_camera_offsets: dict[str, list[float]] = {name: [] for name in self.cameras}
            per_camera_dropped = {name: 0 for name in self.cameras}

            for idx, ref_frame in enumerate(reference_frames):
                offsets: dict[str, float] = {self.reference_camera: 0.0}
                frames: dict[str, KittiFrame] = {self.reference_camera: ref_frame}
                reference_timestamp = ref_frame.timestamp
                frame_ok = True

                for camera in self.cameras:
                    if camera == self.reference_camera:
                        continue
                    camera_frames = per_camera_frames[camera]
                    if not camera_frames:
                        per_camera_dropped[camera] += 1
                        frame_ok = False
                        report.add_issue(
                            "error",
                            f"No frames available for camera {camera}.",
                            hint="Verify the dataset contains images for all cameras.",
                        )
                        continue

                    matched_frame, offset = _match_frame(
                        camera_frames=camera_frames,
                        camera_timestamps=per_camera_timestamps[camera],
                        reference_timestamp=reference_timestamp,
                        reference_index=idx,
                        tolerance_s=tolerance_s,
                        use_time=per_camera_time_ready[camera]
                        and per_camera_time_ready[self.reference_camera],
                    )
                    if matched_frame is None or offset is None:
                        per_camera_dropped[camera] += 1
                        frame_ok = False
                        continue

                    frames[camera] = matched_frame
                    offsets[camera] = offset
                    per_camera_offsets[camera].append(offset)

                if frame_ok:
                    synced_frames.append(
                        SyncedMultiCameraFrame(
                            index=len(synced_frames),
                            timestamp=reference_timestamp,
                            frames=frames,
                            offsets_s=offsets,
                        )
                    )
                else:
                    per_camera_dropped[self.reference_camera] += 1

            report.metrics["num_synced_frames"] = float(len(synced_frames))
            report.metrics["dropped_frames"] = float(
                report.metrics["num_reference_frames"] - report.metrics["num_synced_frames"]
            )

            drop_ratio = 0.0
            if report.metrics["num_reference_frames"]:
                drop_ratio = report.metrics["dropped_frames"] / report.metrics["num_reference_frames"]
            if drop_ratio > 0.1:
                report.add_issue(
                    "error",
                    f"High drop ratio during sync ({drop_ratio:.1%}).",
                    hint="Increase tolerance or verify timestamps alignment.",
                )
            elif drop_ratio > 0:
                report.add_issue(
                    "warning",
                    f"Dropped frames during sync ({drop_ratio:.1%}).",
                    hint="Check timestamp alignment across cameras.",
                )

            for camera, offsets in per_camera_offsets.items():
                abs_offsets = [abs(value) for value in offsets]
                mean_abs, std_abs, max_abs = _offset_stats(abs_offsets)
                report.metrics[f"{camera}_matched_frames"] = float(len(offsets))
                report.metrics[f"{camera}_dropped_frames"] = float(per_camera_dropped[camera])
                report.metrics[f"{camera}_mean_abs_offset_s"] = mean_abs
                report.metrics[f"{camera}_std_abs_offset_s"] = std_abs
                report.metrics[f"{camera}_max_abs_offset_s"] = max_abs
                if not offsets and camera != self.reference_camera:
                    report.add_issue(
                        "warning",
                        f"No synchronized frames for camera {camera}.",
                        hint="Confirm timestamps and image counts for all cameras.",
                    )

            report.metrics["sync_duration_ms"] = (perf_counter() - start) * 1000.0
            LOGGER.info(
                "Multi-camera sync complete for %s (synced=%d, dropped=%d)",
                self.sequence,
                len(synced_frames),
                int(report.metrics["dropped_frames"]),
            )
            return synced_frames, report

        if telemetry_sink is None:
            return _run_sync()

        with timed_event(
            "multi_camera_sync",
            telemetry_sink,
            metadata=metadata,
        ):
            return _run_sync()


def _timestamps_ready(timestamps: list[float | None]) -> bool:
    if not timestamps:
        return False
    if any(value is None for value in timestamps):
        return False
    return _is_monotonic([value for value in timestamps if value is not None])


def _is_monotonic(values: list[float]) -> bool:
    return all(curr >= prev for prev, curr in zip(values, values[1:]))


def _match_frame(
    camera_frames: list[KittiFrame],
    camera_timestamps: list[float | None],
    reference_timestamp: float | None,
    reference_index: int,
    tolerance_s: float,
    use_time: bool,
) -> tuple[KittiFrame | None, float | None]:
    if use_time and reference_timestamp is not None:
        indexed = [(idx, value) for idx, value in enumerate(camera_timestamps) if value is not None]
        if not indexed:
            return None, None
        idx = _nearest_timestamp_index(indexed, reference_timestamp)
        if idx is None or idx >= len(camera_frames):
            return None, None
        matched = camera_frames[idx]
        if matched.timestamp is None:
            return None, None
        offset = matched.timestamp - reference_timestamp
        if abs(offset) > tolerance_s:
            return None, None
        return matched, offset

    if reference_index < len(camera_frames):
        matched = camera_frames[reference_index]
        offset = 0.0
        if reference_timestamp is not None and matched.timestamp is not None:
            offset = matched.timestamp - reference_timestamp
        return matched, offset
    return None, None


def _nearest_timestamp_index(indexed: list[tuple[int, float]], target: float) -> int | None:
    if not indexed:
        return None
    values = [value for _, value in indexed]
    pos = bisect_left(values, target)
    candidates = []
    if pos < len(values):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return None
    best_pos = min(candidates, key=lambda idx: abs(values[idx] - target))
    return indexed[best_pos][0]


def _offset_stats(offsets: list[float]) -> tuple[float, float, float]:
    if not offsets:
        return 0.0, 0.0, 0.0
    mean_val = sum(offsets) / len(offsets)
    variance = sum((value - mean_val) ** 2 for value in offsets) / len(offsets)
    std_val = math.sqrt(variance)
    max_val = max(offsets)
    return mean_val, std_val, max_val
