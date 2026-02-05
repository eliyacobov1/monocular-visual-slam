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

from dataset_validation import validate_kitti
from frame_stream import FrameStream, FrameStreamConfig
from ingestion_pipeline import AsyncIngestionPipeline, IngestionPipelineConfig
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
    stream_frames: bool = False,
    stream_queue_capacity: int = 8,
    async_ingestion: bool = False,
    ingestion_entry_capacity: int = 32,
    ingestion_output_capacity: int = 16,
    ingestion_decode_workers: int = 2,
    ingestion_read_timeout_s: float | None = 0.5,
    ingestion_decode_timeout_s: float | None = 0.5,
    ingestion_fail_fast: bool = True,
) -> SLAMRunResult:
    if async_ingestion and stream_frames:
        raise ValueError("Select either stream_frames or async_ingestion, not both")
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

    frame_entries = [
        (entry.index, entry.timestamp, entry.path)
        for entry in sequence_loader.iter_frames()
    ]
    if not frame_entries:
        raise RuntimeError("No frames found in KITTI sequence")

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
    num_frames = len(frame_entries)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    LOGGER.info(
        "Starting KITTI SLAM run",
        extra={
            "run_id": run_id,
            "sequence": sequence,
            "camera": camera,
            "num_frames": num_frames,
        },
    )
    slam = SLAMSystem(slam_config)
    if async_ingestion:
        ingestion = AsyncIngestionPipeline(
            frame_entries,
            config=IngestionPipelineConfig(
                entry_queue_capacity=ingestion_entry_capacity,
                output_queue_capacity=ingestion_output_capacity,
                num_decode_workers=ingestion_decode_workers,
                read_timeout_s=ingestion_read_timeout_s,
                decode_timeout_s=ingestion_decode_timeout_s,
                max_frames=max_frames,
                fail_fast=ingestion_fail_fast,
            ),
        )
        result = slam.run_stream(ingestion)
        LOGGER.info(
            "Async ingestion stats",
            extra={
                "enqueued_entries": ingestion.stats.enqueued_entries,
                "dequeued_entries": ingestion.stats.dequeued_entries,
                "dropped_entries": ingestion.stats.dropped_entries,
                "decoded_frames": ingestion.stats.decoded_frames,
                "dropped_frames": ingestion.stats.dropped_frames,
                "read_failures": ingestion.stats.read_failures,
                "output_backpressure": ingestion.stats.output_backpressure,
                "max_entry_queue_depth": ingestion.stats.max_entry_queue_depth,
                "max_output_queue_depth": ingestion.stats.max_output_queue_depth,
                "duration_s": ingestion.stats.duration_s,
                "total_decode_s": ingestion.stats.total_decode_s,
            },
        )
    elif stream_frames:
        stream = FrameStream(
            frame_entries,
            config=FrameStreamConfig(queue_capacity=stream_queue_capacity),
            max_frames=max_frames,
        )
        result = slam.run_stream(stream)
        LOGGER.info(
            "Frame stream stats",
            extra={
                "enqueued": stream.stats.enqueued,
                "dequeued": stream.stats.dequeued,
                "dropped": stream.stats.dropped,
                "read_failures": stream.stats.read_failures,
                "max_depth": stream.stats.max_depth,
                "duration_s": stream.stats.duration_s,
                "total_read_s": stream.stats.total_read_s,
            },
        )
    else:
        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        for index, timestamp, path in frame_entries:
            if max_frames is not None and len(frames) >= max_frames:
                break
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read frame: {path}")
            frames.append(image)
            timestamp_value = timestamp if timestamp is not None else float(index)
            timestamps.append(float(timestamp_value))
        if not frames:
            raise RuntimeError("No frames loaded from KITTI sequence")
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
        "--stream_frames",
        action="store_true",
        help="Stream frames with a bounded background loader",
    )
    parser.add_argument(
        "--stream_queue_capacity",
        type=int,
        default=8,
        help="Frame stream queue capacity when streaming is enabled",
    )
    parser.add_argument(
        "--async_ingestion",
        action="store_true",
        help="Enable async ingestion with decode workers",
    )
    parser.add_argument(
        "--ingestion_entry_capacity",
        type=int,
        default=32,
        help="Async ingestion entry queue capacity",
    )
    parser.add_argument(
        "--ingestion_output_capacity",
        type=int,
        default=16,
        help="Async ingestion output queue capacity",
    )
    parser.add_argument(
        "--ingestion_decode_workers",
        type=int,
        default=2,
        help="Number of async ingestion decode workers",
    )
    parser.add_argument(
        "--ingestion_read_timeout_s",
        type=float,
        default=0.5,
        help="Timeout for async ingestion entry queue operations",
    )
    parser.add_argument(
        "--ingestion_decode_timeout_s",
        type=float,
        default=0.5,
        help="Timeout for async ingestion output queue operations",
    )
    parser.add_argument(
        "--ingestion_fail_fast",
        action="store_true",
        help="Fail fast on async ingestion decode or backpressure errors",
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
        stream_frames=args.stream_frames,
        stream_queue_capacity=args.stream_queue_capacity,
        async_ingestion=args.async_ingestion,
        ingestion_entry_capacity=args.ingestion_entry_capacity,
        ingestion_output_capacity=args.ingestion_output_capacity,
        ingestion_decode_workers=args.ingestion_decode_workers,
        ingestion_read_timeout_s=args.ingestion_read_timeout_s,
        ingestion_decode_timeout_s=args.ingestion_decode_timeout_s,
        ingestion_fail_fast=args.ingestion_fail_fast,
    )


if __name__ == "__main__":
    main()
