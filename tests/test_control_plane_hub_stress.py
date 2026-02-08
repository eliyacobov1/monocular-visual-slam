"""Stress tests for the control-plane event bus."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control_plane_hub import DeterministicEventBus, StageEventEnvelope


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: int
    count: int


def _record_events(bus: DeterministicEventBus, config: WorkerConfig) -> None:
    for idx in range(config.count):
        bus.record(
            StageEventEnvelope(
                stage=f"worker_{config.worker_id}",
                event_type="tick",
                message="event",
                metadata={"idx": idx},
                timestamp_s=float(idx),
                seq_id=idx,
            )
        )


def test_event_bus_is_thread_safe() -> None:
    bus = DeterministicEventBus(capacity=10000)
    configs = [WorkerConfig(worker_id=idx, count=250) for idx in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for config in configs:
            executor.submit(_record_events, bus, config)
    snapshot = bus.snapshot()
    assert len(snapshot) == 2000
    assert bus.size == 2000
