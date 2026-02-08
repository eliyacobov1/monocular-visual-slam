"""Unit tests for control-plane supervisor state transitions."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control_plane_hub import ControlPlaneStageAdapter, StageHealthSnapshot
from control_plane_supervisor import ControlPlaneSupervisor, ControlPlaneSupervisorConfig


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def _health(stage: str, state: str) -> StageHealthSnapshot:
    return StageHealthSnapshot(stage=stage, state=state, metrics={}, counters={})


def test_supervisor_trips_on_error_events() -> None:
    events = [StubEvent("error_timeout", "timeout", {}, 1.0)]
    adapters = [
        ControlPlaneStageAdapter(
            name="ingestion",
            health_snapshot=lambda: _health("ingestion", "healthy"),
            events=lambda: events,
        )
    ]
    config = ControlPlaneSupervisorConfig(
        degrade_event_threshold=1,
        trip_event_threshold=1,
        stage_dependencies={},
    )
    supervisor = ControlPlaneSupervisor(adapters, config=config)
    report = supervisor.update()
    assert report.global_state == "tripped"
    assert report.stage_statuses[0].state == "tripped"
    assert len(report.transitions) == 1


def test_supervisor_propagates_tripped_dependencies() -> None:
    adapters = [
        ControlPlaneStageAdapter(
            name="ingestion",
            health_snapshot=lambda: _health("ingestion", "tripped"),
            events=lambda: (),
        ),
        ControlPlaneStageAdapter(
            name="feature",
            health_snapshot=lambda: _health("feature", "healthy"),
            events=lambda: (),
        ),
    ]
    config = ControlPlaneSupervisorConfig(stage_dependencies={"feature": ("ingestion",)})
    supervisor = ControlPlaneSupervisor(adapters, config=config)
    report = supervisor.update()
    stage_states = {status.stage: status.state for status in report.stage_statuses}
    assert stage_states["ingestion"] == "tripped"
    assert stage_states["feature"] == "degraded"


def test_supervisor_recovery_flow() -> None:
    now = [0.0]

    def _clock() -> float:
        return now[0]

    stage_state = {"tracking": "tripped"}

    adapters = [
        ControlPlaneStageAdapter(
            name="tracking",
            health_snapshot=lambda: _health("tracking", stage_state["tracking"]),
            events=lambda: (),
        )
    ]
    config = ControlPlaneSupervisorConfig(
        stage_dependencies={},
        recovery_cooldown_s=1.0,
        recovery_healthy_required=2,
        degraded_healthy_required=1,
        clock=_clock,
    )
    supervisor = ControlPlaneSupervisor(adapters, config=config)

    report = supervisor.update()
    assert report.stage_statuses[0].state == "tripped"

    now[0] = 1.1
    stage_state["tracking"] = "healthy"
    report = supervisor.update()
    assert report.stage_statuses[0].state == "recovering"

    now[0] = 1.2
    report = supervisor.update()
    assert report.stage_statuses[0].state == "recovering"

    now[0] = 1.3
    report = supervisor.update()
    assert report.stage_statuses[0].state == "healthy"
