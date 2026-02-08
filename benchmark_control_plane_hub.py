"""Benchmark unified control-plane hub throughput and memory delta."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from control_plane_hub import ControlPlaneHub, ControlPlaneStageAdapter, StageHealthSnapshot


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def _make_events(stage: str, count: int) -> list[StubEvent]:
    return [
        StubEvent(
            event_type=f"{stage}_event",
            message="ok",
            metadata={"idx": idx},
            timestamp_s=idx * 0.001,
        )
        for idx in range(count)
    ]


def run_benchmark(stage_count: int = 4, events_per_stage: int = 1000) -> None:
    adapters: list[ControlPlaneStageAdapter] = []
    for idx in range(stage_count):
        stage = f"stage_{idx}"
        events = _make_events(stage, events_per_stage)
        adapters.append(
            ControlPlaneStageAdapter(
                name=stage,
                health_snapshot=lambda stage=stage: StageHealthSnapshot(
                    stage=stage,
                    state="healthy",
                    metrics={"queue_depth_ratio": 0.0},
                    counters={"events": events_per_stage},
                ),
                events=lambda events=events: events,
            )
        )

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    hub = ControlPlaneHub(adapters)
    report = hub.generate_report()
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("control_plane_hub_benchmark")
    print(f"stages={stage_count}")
    print(f"events_per_stage={events_per_stage}")
    print(f"total_events={len(report.events)}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"digest={report.digest}")


if __name__ == "__main__":
    run_benchmark()
