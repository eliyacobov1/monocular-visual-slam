"""Unit tests for control-plane supervisor and primitives."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ingestion_control_plane import (
    AdaptiveBoundedQueue,
    CircuitBreaker,
    CircuitBreakerConfig,
    DeterministicEventLog,
    DynamicWorkerPool,
    IngestionTelemetry,
    MovingAverage,
    QueueTuningConfig,
    StageSupervisor,
    WorkerPoolConfig,
)


def test_moving_average_tracks_signal() -> None:
    avg = MovingAverage(alpha=0.5)
    assert avg.value is None
    assert avg.update(1.0) == 1.0
    assert avg.update(0.0) == 0.5
    assert avg.update(1.0) == 0.75


def test_circuit_breaker_state_transitions() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=0.01, half_open_successes=1))
    assert breaker.allow() is True
    assert breaker.record_failure() is False
    assert breaker.record_failure() is True
    assert breaker.state == "open"
    time.sleep(0.02)
    assert breaker.allow() is True
    assert breaker.state == "half_open"
    breaker.record_success()
    assert breaker.state == "closed"


def test_stage_supervisor_scales_queue_and_workers() -> None:
    entry_queue: AdaptiveBoundedQueue[int] = AdaptiveBoundedQueue(capacity=4, min_capacity=2, max_capacity=6)
    telemetry = IngestionTelemetry(window=32, event_log=DeterministicEventLog(capacity=32))
    worker_pool = DynamicWorkerPool(min_workers=1, max_workers=2)
    worker_pool.register_spawn()

    spawn_count = 0
    retire_count = 0

    def _spawn() -> None:
        nonlocal spawn_count
        spawn_count += 1
        worker_pool.register_spawn()

    def _retire() -> None:
        nonlocal retire_count
        retire_count += 1
        worker_pool.register_exit()

    supervisor = StageSupervisor(
        stage="entry",
        queue=entry_queue,
        tuning=QueueTuningConfig(min_capacity=2, max_capacity=6, scale_up_ratio=0.6, scale_down_ratio=0.2, scale_step=1),
        telemetry=telemetry,
        interval_s=0.01,
        worker_pool=worker_pool,
        worker_config=WorkerPoolConfig(min_workers=1, max_workers=2, scale_up_ratio=0.5, scale_down_ratio=0.1, supervisor_interval_s=0.01),
        spawn_worker=_spawn,
        retire_worker=_retire,
        inflight_count=lambda: 0,
        ema_alpha=0.5,
    )

    for idx in range(4):
        entry_queue.put(idx)

    supervisor.tick(time.perf_counter() + 1.0)

    assert telemetry.stage_metrics("entry").queue_scale_ups >= 1
    assert spawn_count >= 1
    assert entry_queue.capacity >= 5
    assert telemetry.event_log.size >= 1
    assert retire_count == 0
