"""Unit tests for the unified control-plane hub."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control_plane_hub import ControlPlaneHub, ControlPlaneStageAdapter, StageHealthSnapshot


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def test_control_plane_hub_orders_events() -> None:
    events_a = [
        StubEvent("a_start", "start", {}, 1.0),
        StubEvent("a_mid", "mid", {}, 2.0),
    ]
    events_b = [
        StubEvent("b_start", "start", {}, 1.5),
        StubEvent("b_mid", "mid", {}, 2.0),
    ]

    def _health(stage: str) -> StageHealthSnapshot:
        return StageHealthSnapshot(stage=stage, state="healthy", metrics={}, counters={})

    hub = ControlPlaneHub(
        [
            ControlPlaneStageAdapter(name="alpha", health_snapshot=lambda: _health("alpha"), events=lambda: events_a),
            ControlPlaneStageAdapter(name="beta", health_snapshot=lambda: _health("beta"), events=lambda: events_b),
        ]
    )

    report = hub.generate_report()
    ordered = [(event.stage, event.event_type) for event in report.events]
    assert ordered == [
        ("alpha", "a_start"),
        ("beta", "b_start"),
        ("alpha", "a_mid"),
        ("beta", "b_mid"),
    ]
    assert {snapshot.stage for snapshot in report.stage_snapshots} == {"alpha", "beta"}
    assert report.digest
