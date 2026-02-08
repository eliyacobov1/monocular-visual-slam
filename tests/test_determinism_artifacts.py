"""Tests for determinism metadata embedded in artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_persistence import (
    FrameDiagnosticsEntry,
    RunDataStore,
    build_frame_diagnostics_bundle,
    build_metrics_bundle,
)
from run_telemetry import RunTelemetryRecorder, TelemetryEvent


def test_artifacts_include_determinism_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    store = RunDataStore.create(
        base_dir=tmp_path,
        run_id="determinism_test",
        config_path=config_path,
        config_hash="hash",
        seed=123,
        use_subdir=False,
        resolved_config={"run_id": "determinism_test"},
    )

    metrics_bundle = build_metrics_bundle("metrics", {"count": 1})
    metrics_path = store.save_metrics(metrics_bundle)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["determinism"]["seed"] == 123
    assert metrics_payload["determinism"]["config_hash"] == "hash"

    diagnostics_bundle = build_frame_diagnostics_bundle(
        "frame_diagnostics",
        [
            FrameDiagnosticsEntry(
                frame_id=0,
                timestamp=0.0,
                match_count=0,
                inliers=0,
                method="bootstrap",
                inlier_ratio=0.0,
                median_parallax=0.0,
                score=0.0,
                status="bootstrap",
                failure_reason=None,
            )
        ],
    )
    diagnostics_path = store.save_frame_diagnostics(diagnostics_bundle)
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics_payload["determinism"]["seed"] == 123

    accumulator = store.create_accumulator("trajectory")
    accumulator.append(np.eye(4), 0.0, 0)
    trajectory_path = store.save_trajectory(accumulator.as_bundle())
    trajectory_payload = np.load(trajectory_path, allow_pickle=True)
    assert "determinism" in trajectory_payload

    telemetry_path = store.telemetry_path("telemetry")
    recorder = RunTelemetryRecorder(telemetry_path, determinism=store.determinism_payload())
    recorder.record_event(
        TelemetryEvent(
            name="unit",
            duration_s=0.01,
            timestamp="2024-01-01T00:00:00Z",
            metadata={"success": True},
        )
    )
    recorder.flush()
    telemetry_payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry_payload["determinism"]["seed"] == 123
