from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control_plane_hub import ControlPlaneHub, ControlPlaneStageAdapter, StageHealthSnapshot
from readiness_report import (
    ReadinessReportConfig,
    generate_readiness_report,
    write_readiness_report,
)


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_control_plane_report(tmp_path: Path, state: str) -> Path:
    events = [
        StubEvent(
            event_type="ok",
            message="ok",
            metadata={"idx": 1},
            timestamp_s=1.0,
        )
    ]
    adapter = ControlPlaneStageAdapter(
        name="tracking",
        health_snapshot=lambda: StageHealthSnapshot(
            stage="tracking",
            state=state,
            metrics={"queue_depth_ratio": 0.1},
            counters={"events": 1},
        ),
        events=lambda: events,
    )
    report = ControlPlaneHub([adapter]).generate_report()
    payload = report.asdict()
    payload["determinism"] = {"seed": 7, "config_hash": "abc123"}
    path = tmp_path / "control_plane.json"
    _write_json(path, payload)
    return path


def test_readiness_report_pass(tmp_path: Path) -> None:
    control_plane_path = _make_control_plane_report(tmp_path, "healthy")
    evaluation_summary = {
        "run_id": "run_1",
        "dataset": "kitti",
        "aggregate_metrics": {"ate_rmse": 0.2},
        "baseline_comparison": {"status": "pass"},
        "relocalization_baseline_comparison": {"status": "pass"},
        "telemetry_drift_report": {"status": "pass"},
    }
    evaluation_path = tmp_path / "evaluation_summary.json"
    _write_json(evaluation_path, evaluation_summary)

    config = ReadinessReportConfig(
        run_id="run_1",
        output_dir=tmp_path,
        control_plane_report_path=control_plane_path,
        evaluation_summary_path=evaluation_path,
    )
    report = generate_readiness_report(config)

    assert report["status"] == "pass"
    assert report["status_breakdown"]["control_plane"] == "pass"
    assert report["status_breakdown"]["evaluation"] == "pass"
    assert report["status_breakdown"]["telemetry"] == "pass"

    output_path = tmp_path / "readiness.json"
    write_readiness_report(output_path, report)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["digest"] == report["digest"]


def test_readiness_report_warn_on_degraded_control_plane(tmp_path: Path) -> None:
    control_plane_path = _make_control_plane_report(tmp_path, "degraded")
    config = ReadinessReportConfig(
        run_id="run_2",
        output_dir=tmp_path,
        control_plane_report_path=control_plane_path,
    )
    report = generate_readiness_report(config)
    assert report["status"] == "warn"
    assert report["status_breakdown"]["control_plane"] == "warn"


def test_readiness_report_fail_on_telemetry_drift(tmp_path: Path) -> None:
    control_plane_path = _make_control_plane_report(tmp_path, "healthy")
    evaluation_summary = {
        "run_id": "run_3",
        "telemetry_drift_report": {"status": "fail"},
    }
    evaluation_path = tmp_path / "evaluation_summary.json"
    _write_json(evaluation_path, evaluation_summary)
    config = ReadinessReportConfig(
        run_id="run_3",
        output_dir=tmp_path,
        control_plane_report_path=control_plane_path,
        evaluation_summary_path=evaluation_path,
    )
    report = generate_readiness_report(config)
    assert report["status"] == "fail"
    assert report["status_breakdown"]["telemetry"] == "fail"
