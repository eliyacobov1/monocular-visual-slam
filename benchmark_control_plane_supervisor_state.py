"""Benchmark control-plane supervisor update performance."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

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


def _make_events(stage: str, count: int, base_ts: float) -> tuple[StubEvent, ...]:
    events = []
    for idx in range(count):
        event_type = "error_timeout" if idx % 11 == 0 else "info"
        events.append(StubEvent(event_type, f"{stage}-event", {"idx": idx}, base_ts + idx * 0.001))
    return tuple(events)


def run_benchmark(iterations: int = 500, event_count: int = 128) -> None:
    stage_states = {
        "ingestion": "healthy",
        "feature": "healthy",
        "tracking": "healthy",
        "optimization": "healthy",
    }
    adapters = []
    for stage in stage_states:
        events = _make_events(stage, event_count, base_ts=1.0)
        adapters.append(
            ControlPlaneStageAdapter(
                name=stage,
                health_snapshot=lambda stage_name=stage: _health(stage_name, stage_states[stage_name]),
                events=lambda events=events: events,
            )
        )
    config = ControlPlaneSupervisorConfig()
    supervisor = ControlPlaneSupervisor(adapters, config=config)

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    for _ in range(iterations):
        supervisor.update()
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("control_plane_supervisor_state_benchmark")
    print(f"iterations={iterations}")
    print(f"event_count={event_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"updates_per_s={iterations / duration_s:.2f}")


if __name__ == "__main__":
    run_benchmark()
