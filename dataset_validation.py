#!/usr/bin/env python3
"""Dataset validation utilities for KITTI and TUM sequences."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kitti_dataset import (
    camera_id_from_name,
    parse_kitti_calib_file,
    resolve_camera_matrix,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str
    hint: str | None = None


@dataclass
class ValidationResult:
    dataset: str
    root: Path
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.level == "warning" for issue in self.issues)

    def add_issue(self, level: str, message: str, hint: str | None = None) -> None:
        self.issues.append(ValidationIssue(level=level, message=message, hint=hint))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "root": str(self.root),
            "ok": self.ok,
            "issues": [
                {"level": issue.level, "message": issue.message, "hint": issue.hint}
                for issue in self.issues
            ],
            "metadata": self.metadata,
        }


def _resolve_kitti_sequence(root: Path, sequence: str) -> Path | None:
    candidates = [
        root / "sequences" / sequence,
        root / sequence,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raw_candidates = sorted(root.glob(f"*/{sequence}"))
    if raw_candidates:
        return raw_candidates[0]
    return None


def _resolve_kitti_image_dir(sequence_path: Path, camera: str) -> Path | None:
    camera_aliases = {
        "image_0": ["image_0", "image_00"],
        "image_1": ["image_1", "image_01"],
        "image_2": ["image_2", "image_02"],
        "image_3": ["image_3", "image_03"],
    }
    aliases = camera_aliases.get(camera, [camera])
    for alias in aliases:
        for candidate in (sequence_path / alias, sequence_path / alias / "data"):
            if candidate.exists() and any(candidate.glob("*.png")):
                return candidate
    return None


def validate_kitti(root: Path, sequence: str, camera: str = "image_2") -> ValidationResult:
    result = ValidationResult(dataset="kitti", root=root)
    if not root.exists():
        result.add_issue(
            "error",
            f"KITTI root does not exist: {root}",
            hint="Ensure the root points at the dataset base directory.",
        )
        return result

    sequence_path = _resolve_kitti_sequence(root, sequence)
    if sequence_path is None:
        result.add_issue(
            "error",
            f"Sequence '{sequence}' not found under {root}",
            hint="Check that the sequence folder exists or use the correct root.",
        )
        return result
    result.metadata["sequence_path"] = str(sequence_path)

    image_dir = _resolve_kitti_image_dir(sequence_path, camera)
    if image_dir is None:
        result.add_issue(
            "error",
            f"No image directory found for camera {camera} in {sequence_path}",
            hint="Verify the camera name (image_0/image_1/image_2/image_3).",
        )
        return result
    image_files = sorted(image_dir.glob("*.png"))
    result.metadata["image_dir"] = str(image_dir)
    result.metadata["num_frames"] = len(image_files)
    if not image_files:
        result.add_issue(
            "error",
            f"No image frames found under {image_dir}",
            hint="Confirm the dataset contains PNG images.",
        )

    timestamps = None
    times_path = sequence_path / "times.txt"
    if times_path.exists():
        timestamps = [line for line in times_path.read_text().splitlines() if line.strip()]
    else:
        timestamp_path = image_dir / "timestamps.txt"
        if not timestamp_path.exists() and image_dir.name == "data":
            timestamp_path = image_dir.parent / "timestamps.txt"
        if timestamp_path.exists():
            timestamps = [line for line in timestamp_path.read_text().splitlines() if line.strip()]
    if timestamps is None:
        result.add_issue(
            "warning",
            "No timestamps file found for sequence.",
            hint="KITTI odometry uses times.txt; raw sequences use timestamps.txt.",
        )
    else:
        result.metadata["num_timestamps"] = len(timestamps)
        if image_files and len(timestamps) != len(image_files):
            result.add_issue(
                "warning",
                "Timestamp count does not match image count.",
                hint="Check times.txt or timestamps.txt alignment.",
            )

    calib_paths = [
        sequence_path / "calib.txt",
        sequence_path / "calib_cam_to_cam.txt",
    ]
    if sequence_path.parent != root:
        calib_paths.append(sequence_path.parent / "calib_cam_to_cam.txt")
    calib_path = next((path for path in calib_paths if path.exists()), None)
    if calib_path is None:
        result.add_issue(
            "warning",
            "Calibration file not found for sequence.",
            hint="Expected calib.txt or calib_cam_to_cam.txt.",
        )
        return result
    result.metadata["calib_path"] = str(calib_path)
    try:
        calib = parse_kitti_calib_file(calib_path)
        camera_id = camera_id_from_name(camera)
        resolve_camera_matrix(calib, camera_id)
    except Exception as exc:  # pragma: no cover - defensive for parsing errors
        result.add_issue(
            "warning",
            f"Failed to parse calibration for camera {camera}: {exc}",
            hint="Ensure calibration contains the expected projection matrix.",
        )
    return result


def validate_tum(root: Path) -> ValidationResult:
    result = ValidationResult(dataset="tum", root=root)
    if not root.exists():
        result.add_issue(
            "error",
            f"TUM root does not exist: {root}",
            hint="Ensure the root points at the sequence directory.",
        )
        return result

    groundtruth = root / "groundtruth.txt"
    if not groundtruth.exists():
        result.add_issue(
            "error",
            "groundtruth.txt not found in TUM sequence.",
            hint="Download the sequence and keep groundtruth.txt at the root.",
        )
    else:
        result.metadata["groundtruth_path"] = str(groundtruth)

    rgb_dir = root / "rgb"
    if not rgb_dir.exists():
        result.add_issue(
            "error",
            "rgb directory not found in TUM sequence.",
            hint="Ensure the RGB images are under an 'rgb' folder.",
        )
        return result

    images = sorted(rgb_dir.glob("*.png"))
    result.metadata["rgb_dir"] = str(rgb_dir)
    result.metadata["num_frames"] = len(images)
    if not images:
        result.add_issue(
            "error",
            f"No PNG frames found under {rgb_dir}",
            hint="Check that the RGB images have been extracted.",
        )
    return result


def _print_human(result: ValidationResult) -> None:
    status = "OK" if result.ok else "FAILED"
    LOGGER.info("Dataset validation %s for %s", status, result.dataset.upper())
    print(f"Dataset: {result.dataset}")
    print(f"Root: {result.root}")
    print(f"Status: {status}")
    if result.metadata:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  - {key}: {value}")
    if result.issues:
        print("Issues:")
        for issue in result.issues:
            hint = f" (hint: {issue.hint})" if issue.hint else ""
            print(f"  - [{issue.level}] {issue.message}{hint}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset layouts")
    parser.add_argument("--dataset", choices=["kitti", "tum"], required=True)
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--sequence", help="Sequence name for KITTI validation")
    parser.add_argument("--camera", default="image_2", help="Camera name for KITTI")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when warnings are present.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    root = Path(args.root)
    if args.dataset == "kitti":
        if not args.sequence:
            raise SystemExit("--sequence is required for KITTI validation")
        result = validate_kitti(root, args.sequence, camera=args.camera)
    else:
        result = validate_tum(root)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human(result)

    if not result.ok or (args.strict and result.has_warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
