"""Tests for deterministic failure injection and chaos harness."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control_plane_supervisor import ControlPlaneSupervisor, ControlPlaneSupervisorConfig
from failure_injection import (
    FailureInjectionChaosHarness,
    FailureInjectionConfig,
    FailureInjectionHarness,
    FailureInjector,
)


def test_failure_injection_plan_is_deterministic() -> None:
    config = FailureInjectionConfig(
        seed=7,
        timeout_probability=0.4,
        dropped_frame_probability=0.2,
        solver_stall_probability=0.1,
        step_duration_s=0.1,
    )
    injector = FailureInjector(config)
    plan_a = injector.build_plan(["tracking", "optimization"], steps=8)
    plan_b = injector.build_plan(["tracking", "optimization"], steps=8)
    assert plan_a.digest == plan_b.digest
    assert plan_a.points == plan_b.points


def test_failure_injection_harness_emits_events() -> None:
    config = FailureInjectionConfig(
        seed=11,
        timeout_probability=1.0,
        dropped_frame_probability=0.0,
        solver_stall_probability=0.0,
        step_duration_s=0.1,
    )
    injector = FailureInjector(config)
    plan = injector.build_plan(["ingestion"], steps=3)
    harness = FailureInjectionHarness(plan, clock=lambda: 1.0)
    harness.advance()
    events = harness._states["ingestion"].events()
    assert events
    assert events[0].event_type == "timeout_failure"
    assert harness._states["ingestion"].snapshot().state == "tripped"


def test_failure_injection_supervisor_recovery_flow() -> None:
    clock_value = [0.0]

    def _clock() -> float:
        return clock_value[0]

    config = FailureInjectionConfig(
        seed=13,
        timeout_probability=1.0,
        dropped_frame_probability=0.0,
        solver_stall_probability=0.0,
        step_duration_s=1.0,
    )
    injector = FailureInjector(config)
    plan = injector.build_plan(["tracking"], steps=1)
    harness = FailureInjectionHarness(plan, clock=_clock)
    supervisor = ControlPlaneSupervisor(
        harness.adapters(),
        config=ControlPlaneSupervisorConfig(
            stage_dependencies={},
            degrade_event_threshold=1,
            trip_event_threshold=1,
            recovery_cooldown_s=0.5,
            recovery_healthy_required=1,
            clock=_clock,
        ),
    )
    harness.advance()
    report = supervisor.update()
    assert report.global_state == "tripped"

    clock_value[0] = 0.6
    harness.advance()
    report = supervisor.update()
    assert report.stage_statuses[0].state == "recovering"

    clock_value[0] = 0.7
    harness.advance()
    report = supervisor.update()
    assert report.stage_statuses[0].state == "healthy"


def test_failure_injection_chaos_harness_digest_is_stable() -> None:
    config = FailureInjectionConfig(
        seed=21,
        timeout_probability=0.3,
        dropped_frame_probability=0.2,
        solver_stall_probability=0.1,
        step_duration_s=0.1,
    )
    injector = FailureInjector(config)
    plan = injector.build_plan(["ingestion", "feature"], steps=6)
    chaos_first = FailureInjectionChaosHarness(plan, clock=lambda: 1.0, worker_count=2)
    chaos_second = FailureInjectionChaosHarness(plan, clock=lambda: 1.0, worker_count=2)
    digest_first = chaos_first.run(steps=2)
    digest_second = chaos_second.run(steps=2)
    assert digest_first == digest_second
