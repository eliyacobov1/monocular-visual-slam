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


def test_supervisor_escalates_backpressure_and_circuit_breaker() -> None:
    adapters = [
        ControlPlaneStageAdapter(
            name="ingestion",
            health_snapshot=lambda: StageHealthSnapshot(
                stage="ingestion",
                state="healthy",
                metrics={"entry_queue_depth_ratio": 0.9},
                counters={"output_backpressure": 1},
            ),
            events=lambda: (),
        ),
        ControlPlaneStageAdapter(
            name="feature",
            health_snapshot=lambda: StageHealthSnapshot(
                stage="feature",
                state="healthy",
                metrics={"queue_depth_ratio": 0.1},
                counters={"breaker_opens": 1},
            ),
            events=lambda: (),
        ),
        ControlPlaneStageAdapter(
            name="tracking",
            health_snapshot=lambda: StageHealthSnapshot(
                stage="tracking",
                state="healthy",
                metrics={},
                counters={},
            ),
            events=lambda: (),
        ),
    ]
    config = ControlPlaneSupervisorConfig(
        stage_dependencies={"tracking": ("feature",)},
        backpressure_ratio_threshold=0.8,
        backpressure_ratio_trip_threshold=0.95,
        backpressure_counter_threshold=1,
        backpressure_counter_trip_threshold=2,
        circuit_breaker_trip_threshold=1,
    )
    supervisor = ControlPlaneSupervisor(adapters, config=config)
    report = supervisor.update()
    stage_states = {status.stage: status.state for status in report.stage_statuses}
    assert stage_states["ingestion"] == "degraded"
    assert stage_states["feature"] == "tripped"
    assert stage_states["tracking"] == "degraded"
    escalation_types = {(esc.stage, esc.escalation_type, esc.severity) for esc in report.escalations}
    assert ("ingestion", "backpressure", "degraded") in escalation_types
    assert ("feature", "circuit_breaker", "tripped") in escalation_types
    assert [action.stage for action in report.recovery_queue] == ["feature", "ingestion"]


def test_recovery_queue_bounded_capacity() -> None:
    adapters = [
        ControlPlaneStageAdapter(
            name="feature",
            health_snapshot=lambda: StageHealthSnapshot(
                stage="feature",
                state="healthy",
                metrics={"queue_depth_ratio": 0.96},
                counters={},
            ),
            events=lambda: (),
        ),
        ControlPlaneStageAdapter(
            name="ingestion",
            health_snapshot=lambda: StageHealthSnapshot(
                stage="ingestion",
                state="healthy",
                metrics={"entry_queue_depth_ratio": 0.96},
                counters={},
            ),
            events=lambda: (),
        ),
    ]
    config = ControlPlaneSupervisorConfig(
        stage_dependencies={},
        backpressure_ratio_threshold=0.8,
        backpressure_ratio_trip_threshold=0.95,
        recovery_queue_capacity=1,
    )
    supervisor = ControlPlaneSupervisor(adapters, config=config)
    report = supervisor.update()
    assert len(report.recovery_queue) == 1
    assert report.recovery_queue[0].stage == "ingestion"
