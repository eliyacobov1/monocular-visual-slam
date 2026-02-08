"""Benchmark deterministic event stream digest throughput."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from deterministic_integrity import stable_event_digest


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def _make_events(count: int) -> list[StubEvent]:
    return [
        StubEvent(
            event_type="stage_event",
            message="ok",
            metadata={"idx": idx, "bucket": idx % 5},
            timestamp_s=idx * 0.01,
        )
        for idx in range(count)
    ]


def run_benchmark(event_count: int = 10000) -> None:
    events = _make_events(event_count)
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    digest = stable_event_digest(events)
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("event_stream_digest_benchmark")
    print(f"events={event_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"digest={digest}")


if __name__ == "__main__":
    run_benchmark()
