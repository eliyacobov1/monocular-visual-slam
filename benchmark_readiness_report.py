"""Benchmark readiness report generation with deterministic payloads."""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from control_plane_hub import ControlPlaneHub, ControlPlaneStageAdapter, StageHealthSnapshot
from readiness_report import ReadinessReportConfig, generate_readiness_report


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def _make_control_plane_report(path: Path, stages: int, events_per_stage: int) -> None:
    adapters: list[ControlPlaneStageAdapter] = []
    for idx in range(stages):
        stage = f"stage_{idx}"
        events = [
            StubEvent(
                event_type="heartbeat",
                message="ok",
                metadata={"idx": event_idx},
                timestamp_s=event_idx * 0.01,
            )
            for event_idx in range(events_per_stage)
        ]
        adapters.append(
            ControlPlaneStageAdapter(
                name=stage,
                health_snapshot=lambda stage=stage: StageHealthSnapshot(
                    stage=stage,
                    state="healthy",
                    metrics={"queue_depth_ratio": 0.1},
                    counters={"events": events_per_stage},
                ),
                events=lambda events=events: events,
            )
        )
    report = ControlPlaneHub(adapters).generate_report()
    payload = report.asdict()
    payload["determinism"] = {"seed": 11, "config_hash": "bench"}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_evaluation_summary(path: Path) -> None:
    payload = {
        "run_id": "bench",
        "aggregate_metrics": {"ate_rmse": 0.15},
        "baseline_comparison": {"status": "pass"},
        "telemetry_drift_report": {"status": "pass"},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(stages: int = 6, events_per_stage: int = 2000) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        control_plane_path = tmp_path / "control_plane.json"
        evaluation_path = tmp_path / "evaluation_summary.json"
        _make_control_plane_report(control_plane_path, stages, events_per_stage)
        _write_evaluation_summary(evaluation_path)

        config = ReadinessReportConfig(
            run_id="bench",
            output_dir=tmp_path,
            control_plane_report_path=control_plane_path,
            evaluation_summary_path=evaluation_path,
        )

        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        start_time = time.perf_counter()
        report = generate_readiness_report(config)
        duration_s = time.perf_counter() - start_time
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
        total_memory_delta = sum(stat.size_diff for stat in memory_stats)

        print("readiness_report_benchmark")
        print(f"stages={stages}")
        print(f"events_per_stage={events_per_stage}")
        print(f"duration_s={duration_s:.4f}")
        print(f"memory_delta_bytes={total_memory_delta}")
        print(f"status={report['status']}")
        print(f"digest={report['digest']}")


if __name__ == "__main__":
    run_benchmark()
