#!/usr/bin/env python3
"""CLI utilities for running the SLAM API on KITTI sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_validation import validate_kitti
from feature_pipeline import FeaturePipelineConfig
from kitti_dataset import KittiSequence
from robust_pose_estimator import RobustPoseEstimatorConfig
from slam_api import SLAMSystem, SLAMSystemConfig, SLAMRunResult

LOGGER = logging.getLogger(__name__)


def _hash_config(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _filter_config(payload: dict[str, Any], config_type: type) -> dict[str, Any]:
    allowed = {field.name for field in fields(config_type)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {config_type.__name__} fields: {', '.join(unknown)}")
    return {key: payload[key] for key in payload if key in allowed}


def load_pipeline_config(path: Path) -> tuple[FeaturePipelineConfig, RobustPoseEstimatorConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_payload = payload.get("feature_config", payload.get("feature", {}))
    pose_payload = payload.get("pose_config", payload.get("pose", {}))
    feature_config = FeaturePipelineConfig(
        **_filter_config(feature_payload, FeaturePipelineConfig)
    )
    pose_config = RobustPoseEstimatorConfig(
        **_filter_config(pose_payload, RobustPoseEstimatorConfig)
    )
    return feature_config, pose_config


def run_kitti_sequence(
    *,
    root: Path,
    sequence: str,
    camera: str,
    output_dir: Path,
    run_id: str,
    config_path: Path,
    use_run_subdir: bool = True,
    max_frames: int | None = None,
) -> SLAMRunResult:
    validation = validate_kitti(root, sequence, camera=camera)
    if not validation.ok:
        messages = "; ".join(issue.message for issue in validation.issues)
        raise RuntimeError(f"Dataset validation failed: {messages}")
    if validation.has_warnings:
        LOGGER.warning("Dataset validation completed with warnings")

    feature_config, pose_config = load_pipeline_config(config_path)
    config_hash = _hash_config(config_path)

    sequence_loader = KittiSequence(root, sequence, camera=camera)
    intrinsics = sequence_loader.camera_intrinsics()
    if intrinsics is None:
        raise ValueError("Camera intrinsics not found for KITTI sequence")

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    for entry in sequence_loader.iter_frames():
        if max_frames is not None and len(frames) >= max_frames:
            break
        image = cv2.imread(str(entry.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {entry.path}")
        frames.append(image)
        timestamp = entry.timestamp if entry.timestamp is not None else float(entry.index)
        timestamps.append(float(timestamp))

    if not frames:
        raise RuntimeError("No frames loaded from KITTI sequence")

    slam_config = SLAMSystemConfig(
        run_id=run_id,
        output_dir=output_dir,
        config_path=config_path,
        config_hash=config_hash,
        intrinsics=intrinsics,
        feature_config=feature_config,
        pose_config=pose_config,
        use_run_subdir=use_run_subdir,
    )
    LOGGER.info(
        "Starting KITTI SLAM run",
        extra={
            "run_id": run_id,
            "sequence": sequence,
            "camera": camera,
            "num_frames": len(frames),
        },
    )
    slam = SLAMSystem(slam_config)
    result = slam.run_sequence(frames, timestamps)
    LOGGER.info("Completed KITTI SLAM run", extra={"run_dir": str(result.run_dir)})
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLAM API on a KITTI sequence")
    parser.add_argument("--root", required=True, help="KITTI dataset root directory")
    parser.add_argument("--sequence", required=True, help="KITTI sequence id")
    parser.add_argument("--camera", default="image_2", help="Camera name (image_0-3)")
    parser.add_argument("--output_dir", default="reports", help="Output directory")
    parser.add_argument("--run_id", default="slam_run", help="Run identifier")
    parser.add_argument("--config", required=True, help="Path to pipeline config JSON")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap")
    parser.add_argument(
        "--use_run_subdir",
        action="store_true",
        help="Write outputs into a run-specific subdirectory",
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
    run_kitti_sequence(
        root=Path(args.root),
        sequence=args.sequence,
        camera=args.camera,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        config_path=Path(args.config),
        use_run_subdir=args.use_run_subdir,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
