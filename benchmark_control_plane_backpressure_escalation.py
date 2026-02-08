"""Benchmark backpressure/circuit-breaker escalation overhead."""

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


def _health(stage: str, ratio: float, breaker_opens: int) -> StageHealthSnapshot:
    metrics = {
        "entry_queue_depth_ratio": ratio,
        "output_queue_depth_ratio": ratio,
        "queue_depth_ratio": ratio,
        "inflight_ratio": ratio,
    }
    counters = {
        "output_backpressure": int(ratio * 2),
        "circuit_breaker_opens": breaker_opens,
    }
    return StageHealthSnapshot(stage=stage, state="healthy", metrics=metrics, counters=counters)


def run_benchmark(iterations: int = 500, stage_count: int = 6) -> None:
    ratios = [0.2, 0.5, 0.82, 0.91, 0.97, 0.4]
    adapters = []
    for idx in range(stage_count):
        stage = f"stage_{idx}"
        ratio = ratios[idx % len(ratios)]
        breaker_opens = 1 if idx % 5 == 0 else 0
        adapters.append(
            ControlPlaneStageAdapter(
                name=stage,
                health_snapshot=lambda stage_name=stage, r=ratio, b=breaker_opens: _health(stage_name, r, b),
                events=lambda: (),
            )
        )
    config = ControlPlaneSupervisorConfig(
        stage_dependencies={},
        backpressure_ratio_threshold=0.8,
        backpressure_ratio_trip_threshold=0.95,
        backpressure_counter_threshold=1,
        backpressure_counter_trip_threshold=2,
        circuit_breaker_trip_threshold=1,
        recovery_queue_capacity=256,
    )
    supervisor = ControlPlaneSupervisor(adapters, config=config)

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    last_report = None
    for _ in range(iterations):
        last_report = supervisor.update()
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)
    escalations = 0 if last_report is None else len(last_report.escalations)

    print("control_plane_backpressure_escalation_benchmark")
    print(f"iterations={iterations}")
    print(f"stage_count={stage_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"updates_per_s={iterations / duration_s:.2f}")
    print(f"escalations_last_report={escalations}")


if __name__ == "__main__":
    run_benchmark()
