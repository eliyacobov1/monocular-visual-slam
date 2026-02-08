"""Tests for determinism validation reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_persistence import (
    FrameDiagnosticsEntry,
    RunDataStore,
    build_frame_diagnostics_bundle,
    build_metrics_bundle,
)
from determinism_validation import build_determinism_report
from run_telemetry import RunTelemetryRecorder, TelemetryEvent


def _create_run(tmp_path: Path, run_id: str, metric_value: float) -> Path:
    config_path = tmp_path / f"{run_id}_config.json"
    config_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    store = RunDataStore.create(
        base_dir=tmp_path / run_id,
        run_id=run_id,
        config_path=config_path,
        config_hash="hash",
        seed=7,
        use_subdir=False,
        resolved_config={"run_id": run_id},
    )

    accumulator = store.create_accumulator("trajectory")
    pose_a = np.eye(4)
    pose_b = np.eye(4)
    pose_b[0, 3] = 1.0
    accumulator.append(pose_a, 0.0, 0)
    accumulator.append(pose_b, 1.0, 1)
    store.save_trajectory(accumulator.as_bundle())

    metrics_bundle = build_metrics_bundle("metrics", {"num_poses": metric_value})
    store.save_metrics(metrics_bundle)

    diagnostics_bundle = build_frame_diagnostics_bundle(
        "frame_diagnostics",
        [
            FrameDiagnosticsEntry(
                frame_id=0,
                timestamp=0.0,
                match_count=5,
                inliers=4,
                method="bootstrap",
                inlier_ratio=0.8,
                median_parallax=1.2,
                score=1.0,
                status="ok",
                failure_reason=None,
            )
        ],
    )
    store.save_frame_diagnostics(diagnostics_bundle)

    telemetry_path = store.telemetry_path("slam_telemetry")
    recorder = RunTelemetryRecorder(telemetry_path, determinism=store.determinism_payload())
    recorder.record_event(
        TelemetryEvent(
            name="tracking",
            duration_s=0.02,
            timestamp="2024-01-01T00:00:00Z",
            metadata={"success": True},
        )
    )
    recorder.flush()

    store.save_control_plane_report(
        "control_plane_report",
        {
            "status": "healthy",
            "stages": {"tracking": {"status": "healthy"}},
        },
    )

    return store.metadata.run_dir


def test_determinism_report_matches_for_identical_runs(tmp_path: Path) -> None:
    run_a = _create_run(tmp_path, "run_a", metric_value=2.0)
    run_b = _create_run(tmp_path, "run_b", metric_value=2.0)

    report = build_determinism_report(run_a, run_b)

    assert report.status == "pass"
    assert all(entry.status == "match" for entry in report.drift)


def test_determinism_report_detects_mismatch(tmp_path: Path) -> None:
    run_a = _create_run(tmp_path, "run_a", metric_value=2.0)
    run_b = _create_run(tmp_path, "run_b", metric_value=3.0)

    report = build_determinism_report(run_a, run_b)

    assert report.status == "fail"
    mismatches = [entry for entry in report.drift if entry.status == "mismatch"]
    assert mismatches
