#!/usr/bin/env python3
"""End-to-end relocalization demo with tracking loss injection."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_validation import validate_kitti
from deterministic_registry import build_registry
from kitti_dataset import KittiSequence
from slam_api import SLAMRunResult, SLAMSystem, SLAMSystemConfig
from slam_runner import load_pipeline_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelocalizationDemoConfig:
    root: Path
    sequence: str
    camera: str
    output_dir: Path
    run_id: str
    pipeline_config: Path
    seed: int
    use_run_subdir: bool
    loss_frame: int
    max_frames: int | None


def _load_telemetry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("events", []))


def _summarize_relocalization_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    relocalization_events = [event for event in events if event.get("name") == "relocalization_search"]
    durations = [float(event.get("duration_s", 0.0)) for event in relocalization_events]
    successes = [
        bool(event.get("metadata", {}).get("success", False)) for event in relocalization_events
    ]
    if durations:
        durations_array = np.array(durations, dtype=np.float64)
        p50 = float(np.quantile(durations_array, 0.5))
        p95 = float(np.quantile(durations_array, 0.95))
        mean = float(durations_array.mean())
    else:
        p50 = 0.0
        p95 = 0.0
        mean = 0.0
    return {
        "attempts": len(relocalization_events),
        "successes": int(sum(successes)),
        "success_rate": float(sum(successes)) / float(len(successes)) if successes else 0.0,
        "latency_mean_s": mean,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
    }


def _build_demo_report(
    config: RelocalizationDemoConfig,
    result: SLAMRunResult,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    relocalized_frames = [
        {
            "frame_id": entry.frame_id,
            "timestamp": entry.timestamp,
            "inliers": entry.inliers,
            "method": entry.method,
        }
        for entry in result.frame_diagnostics
        if entry.status == "relocalized"
    ]
    injected_events = [
        event for event in events if event.get("name") == "tracking_loss_injected"
    ]
    relocalization_summary = _summarize_relocalization_events(events)
    return {
        "run_id": config.run_id,
        "sequence": config.sequence,
        "camera": config.camera,
        "loss_frame": config.loss_frame,
        "max_frames": config.max_frames,
        "run_dir": str(result.run_dir),
        "telemetry_path": str(result.telemetry_path) if result.telemetry_path else None,
        "tracking_loss_events": injected_events,
        "relocalization_summary": relocalization_summary,
        "relocalized_frames": relocalized_frames,
        "map_snapshot_path": str(result.map_snapshot_path) if result.map_snapshot_path else None,
        "map_stats": result.map_stats.to_dict() if result.map_stats else None,
    }


def run_demo(config: RelocalizationDemoConfig) -> dict[str, Any]:
    validation = validate_kitti(config.root, config.sequence, camera=config.camera)
    if not validation.ok:
        messages = "; ".join(issue.message for issue in validation.issues)
        raise RuntimeError(f"Dataset validation failed: {messages}")
    if validation.has_warnings:
        LOGGER.warning("Dataset validation completed with warnings")

    feature_config, pose_config, feature_control_config = load_pipeline_config(config.pipeline_config)
    registry = build_registry(seed=config.seed, config_path=config.pipeline_config)
    registry.apply_global_seed()
    config_hash = registry.config.config_hash

    sequence_loader = KittiSequence(config.root, config.sequence, camera=config.camera)
    intrinsics = sequence_loader.camera_intrinsics()
    if intrinsics is None:
        raise ValueError("Camera intrinsics not found for KITTI sequence")

    slam_config = SLAMSystemConfig(
        run_id=config.run_id,
        output_dir=config.output_dir,
        config_path=config.pipeline_config,
        config_hash=config_hash,
        seed=config.seed,
        intrinsics=intrinsics,
        feature_config=feature_config,
        pose_config=pose_config,
        feature_control=feature_control_config,
        use_run_subdir=config.use_run_subdir,
    )
    slam = SLAMSystem(slam_config)

    frame_count = 0
    injected = False
    for entry in sequence_loader.iter_frames():
        if config.max_frames is not None and frame_count >= config.max_frames:
            break
        image = cv2.imread(str(entry.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {entry.path}")
        if frame_count == config.loss_frame:
            slam.inject_tracking_loss(reason="demo_injection")
            injected = True
        timestamp = entry.timestamp if entry.timestamp is not None else float(entry.index)
        slam.process_frame(image, float(timestamp))
        frame_count += 1

    if not injected:
        LOGGER.warning("Tracking loss injection frame not reached; no injection performed")
    result = slam.finalize_run()
    telemetry_events: list[dict[str, Any]] = []
    if result.telemetry_path is not None:
        telemetry_events = _load_telemetry(result.telemetry_path)
    report = _build_demo_report(config, result, telemetry_events)
    report_path = result.run_dir / "relocalization_demo_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Relocalization demo report written to %s", report_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run relocalization demo with tracking loss injection")
    parser.add_argument("--root", required=True, help="KITTI dataset root directory")
    parser.add_argument("--sequence", required=True, help="KITTI sequence id")
    parser.add_argument("--camera", default="image_2", help="Camera name (image_0-3)")
    parser.add_argument("--output_dir", default="reports", help="Output directory")
    parser.add_argument("--run_id", default="relocalization_demo", help="Run identifier")
    parser.add_argument("--config", required=True, help="Path to pipeline config JSON")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument(
        "--loss_frame",
        type=int,
        default=120,
        help="Frame index at which to inject tracking loss",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional frame cap for the demo run",
    )
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
    demo_config = RelocalizationDemoConfig(
        root=Path(args.root),
        sequence=args.sequence,
        camera=args.camera,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        pipeline_config=Path(args.config),
        seed=args.seed,
        use_run_subdir=args.use_run_subdir,
        loss_frame=args.loss_frame,
        max_frames=args.max_frames,
    )
    run_demo(demo_config)


if __name__ == "__main__":
    main()
