#!/usr/bin/env python3
"""KITTI dataset utilities for sequence loading and calibration parsing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

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
            if path.exists():
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
