"""Unit tests for control-plane supervisor and primitives."""

from __future__ import annotations

import threading
import time

from ingestion_control_plane import (
    AdaptiveBoundedQueue,
    CircuitBreaker,
    CircuitBreakerConfig,
    ControlPlaneSupervisor,
    DynamicWorkerPool,
    IngestionTelemetry,
    MovingAverage,
    QueueTuningConfig,
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


def test_control_plane_scales_queue_and_workers() -> None:
    entry_queue: AdaptiveBoundedQueue[int] = AdaptiveBoundedQueue(capacity=4, min_capacity=2, max_capacity=6)
    output_queue: AdaptiveBoundedQueue[int] = AdaptiveBoundedQueue(capacity=4, min_capacity=2, max_capacity=6)
    telemetry = IngestionTelemetry(window=32)
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

    stop_event = threading.Event()
    producer_done = threading.Event()

    supervisor = ControlPlaneSupervisor(
        entry_queue=entry_queue,
        output_queue=output_queue,
        worker_pool=worker_pool,
        entry_tuning=QueueTuningConfig(min_capacity=2, max_capacity=6, scale_up_ratio=0.6, scale_down_ratio=0.2, scale_step=1),
        output_tuning=QueueTuningConfig(min_capacity=2, max_capacity=6, scale_up_ratio=0.6, scale_down_ratio=0.2, scale_step=1),
        worker_config=WorkerPoolConfig(min_workers=1, max_workers=2, scale_up_ratio=0.5, scale_down_ratio=0.1, supervisor_interval_s=0.01),
        telemetry=telemetry,
        spawn_worker=_spawn,
        retire_worker=_retire,
        inflight_count=lambda: 0,
        stop_event=stop_event,
        producer_done=producer_done,
        ema_alpha=0.5,
    )

    for idx in range(4):
        entry_queue.put(idx)

    thread = threading.Thread(target=supervisor.run, daemon=True)
    thread.start()
    time.sleep(0.05)
    stop_event.set()
    thread.join(timeout=1.0)

    assert telemetry.queue_scale_ups >= 1
    assert spawn_count >= 1
    assert entry_queue.capacity >= 4
