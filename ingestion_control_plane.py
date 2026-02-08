"""Control-plane primitives for asynchronous ingestion."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Deque, Iterable, Literal, Protocol, TypeVar

from deterministic_integrity import stable_event_digest

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class IngestionClosed(RuntimeError):
    """Raised when consuming from a closed ingestion pipeline."""


class BackpressureTimeout(RuntimeError):
    """Raised when backpressure blocks required pipeline operations."""


class QueueLike(Protocol[T]):
    """Minimal protocol for adaptive queues used by ingestion pipeline."""

    def put(self, item: T, timeout_s: float | None = None) -> None:
        ...

    def get(self, timeout_s: float | None = None) -> T:
        ...

    def close(self) -> None:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def capacity(self) -> int:
        ...


class AdaptiveBoundedQueue(QueueLike[T]):
    """Thread-safe bounded queue with adaptive capacity and metrics."""

    def __init__(self, *, capacity: int, min_capacity: int, max_capacity: int) -> None:
        if min_capacity <= 0:
            raise ValueError("min_capacity must be positive")
        if max_capacity < min_capacity:
            raise ValueError("max_capacity must be >= min_capacity")
        if capacity < min_capacity or capacity > max_capacity:
            raise ValueError("capacity must be within min/max bounds")
        self._capacity = capacity
        self._min_capacity = min_capacity
        self._max_capacity = max_capacity
        self._items: Deque[T] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._blocked_puts = 0
        self._blocked_gets = 0

    def put(self, item: T, timeout_s: float | None = None) -> None:
        with self._condition:
            start = time.perf_counter()
            while len(self._items) >= self._capacity and not self._closed:
                self._blocked_puts += 1
                remaining = None if timeout_s is None else timeout_s - (time.perf_counter() - start)
                if remaining is not None and remaining <= 0:
                    raise BackpressureTimeout("Timed out waiting for queue space")
                self._condition.wait(timeout=remaining)
            if self._closed:
                raise IngestionClosed("Queue is closed")
            self._items.append(item)
            self._condition.notify_all()

    def get(self, timeout_s: float | None = None) -> T:
        with self._condition:
            start = time.perf_counter()
            while not self._items and not self._closed:
                self._blocked_gets += 1
                remaining = None if timeout_s is None else timeout_s - (time.perf_counter() - start)
                if remaining is not None and remaining <= 0:
                    raise BackpressureTimeout("Timed out waiting for queue data")
                self._condition.wait(timeout=remaining)
            if not self._items:
                raise IngestionClosed("Queue is closed")
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def resize(self, new_capacity: int) -> None:
        if new_capacity < self._min_capacity or new_capacity > self._max_capacity:
            raise ValueError("new_capacity must be within min/max bounds")
        with self._condition:
            self._capacity = new_capacity
            self._condition.notify_all()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def depth_ratio(self) -> float:
        with self._condition:
            if self._capacity == 0:
                return 0.0
            return len(self._items) / self._capacity

    @property
    def blocked_puts(self) -> int:
        with self._condition:
            return self._blocked_puts

    @property
    def blocked_gets(self) -> int:
        with self._condition:
            return self._blocked_gets


@dataclass(frozen=True)
class QueueTuningConfig:
    """Adaptive queue tuning parameters."""

    min_capacity: int = 16
    max_capacity: int = 128
    scale_up_ratio: float = 0.8
    scale_down_ratio: float = 0.3
    scale_step: int = 8


@dataclass(frozen=True)
class WorkerPoolConfig:
    """Dynamic worker pool tuning parameters."""

    min_workers: int = 2
    max_workers: int = 6
    scale_up_ratio: float = 0.75
    scale_down_ratio: float = 0.2
    supervisor_interval_s: float = 0.2


@dataclass(frozen=True)
class RetryPolicyConfig:
    """Retry policy for decode failures."""

    max_attempts: int = 2
    backoff_s: float = 0.02
    jitter_s: float = 0.01


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration for decode failures."""

    failure_threshold: int = 5
    recovery_timeout_s: float = 1.0
    half_open_successes: int = 2


@dataclass(frozen=True)
class OrderingBufferConfig:
    """Ordering buffer bounds for deterministic output."""

    max_pending: int = 128
    forced_flush_ratio: float = 0.75


class StageState(str, Enum):
    """Lifecycle state for a supervised ingestion stage."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TRIPPED = "tripped"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class StageEvent:
    """Structured event emitted by the control plane."""

    stage: str
    event_type: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=time.time)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class DeterministicEventLog:
    """Ring buffer for structured control-plane events."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._events: Deque[StageEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, event: StageEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[StageEvent]:
        with self._lock:
            return list(self._events)

    def digest(self) -> str:
        with self._lock:
            return stable_event_digest(self._events)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


@dataclass
class IngestionFailureEvent:
    """Structured failure event from ingestion stages."""

    seq_id: int
    stage: str
    error: str
    path: str | None
    timestamp_s: float = field(default_factory=time.time)


@dataclass
class IngestionFailureReport:
    """Aggregated failure summary for ingestion pipeline."""

    max_events: int = 256
    events: Deque[IngestionFailureEvent] = field(default_factory=deque)
    counts: Counter[str] = field(default_factory=Counter)

    def record(self, event: IngestionFailureEvent) -> None:
        self.counts[event.stage] += 1
        if len(self.events) >= self.max_events:
            self.events.popleft()
        self.events.append(event)


@dataclass(frozen=True)
class StagePressureSample:
    """Snapshot of queue pressure and worker count for a stage."""

    timestamp_s: float
    stage: str
    depth_ratio: float
    worker_count: int
    inflight_tasks: int


@dataclass(frozen=True)
class StageLatencySample:
    """Latency snapshot for decode and queue wait."""

    timestamp_s: float
    stage: str
    decode_s: float
    queue_wait_s: float


@dataclass
class StageTelemetry:
    """Per-stage telemetry samples for the ingestion control plane."""

    window: int
    samples: Deque[StagePressureSample] = field(default_factory=deque)
    latencies: Deque[StageLatencySample] = field(default_factory=deque)
    worker_scale_ups: int = 0
    worker_scale_downs: int = 0
    queue_scale_ups: int = 0
    queue_scale_downs: int = 0
    backpressure_events: int = 0
    circuit_breaker_opens: int = 0
    state_transitions: int = 0

    def record_sample(self, sample: StagePressureSample) -> None:
        if len(self.samples) >= self.window:
            self.samples.popleft()
        self.samples.append(sample)

    def record_latency(self, latency: StageLatencySample) -> None:
        if len(self.latencies) >= self.window:
            self.latencies.popleft()
        self.latencies.append(latency)


@dataclass
class IngestionTelemetry:
    """Telemetry samples for queue pressure, worker sizing, and latency."""

    window: int = 256
    stages: dict[str, StageTelemetry] = field(default_factory=dict)
    event_log: DeterministicEventLog = field(default_factory=DeterministicEventLog)
    forced_flushes: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def stage_metrics(self, stage: str) -> StageTelemetry:
        with self._lock:
            metrics = self.stages.get(stage)
            if metrics is None:
                metrics = StageTelemetry(window=self.window)
                self.stages[stage] = metrics
            return metrics

    def record_stage_sample(self, sample: StagePressureSample) -> None:
        metrics = self.stage_metrics(sample.stage)
        metrics.record_sample(sample)

    def record_stage_latency(self, latency: StageLatencySample) -> None:
        metrics = self.stage_metrics(latency.stage)
        metrics.record_latency(latency)

    def record_event(self, event: StageEvent) -> None:
        self.event_log.record(event)

    @property
    def circuit_breaker_opens(self) -> int:
        with self._lock:
            return int(sum(stage.circuit_breaker_opens for stage in self.stages.values()))


class CircuitBreaker:
    """Simple circuit breaker for decode stage isolation."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._state: Literal["closed", "open", "half_open"] = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._opened_at_s: float | None = None

    def allow(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if self._opened_at_s is None:
                    return False
                if (time.time() - self._opened_at_s) >= self._config.recovery_timeout_s:
                    self._state = "half_open"
                    self._success_count = 0
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._success_count += 1
                if self._success_count >= self._config.half_open_successes:
                    self._state = "closed"
                    self._failure_count = 0
                    self._success_count = 0

    def record_failure(self) -> bool:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._state = "open"
                self._opened_at_s = time.time()
                self._success_count = 0
                return True
            return False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


class DeterministicReorderBuffer:
    """Heap-backed ordering buffer that enforces deterministic output."""

    def __init__(self, config: OrderingBufferConfig) -> None:
        self._config = config
        self._heap: list[tuple[int, int, object]] = []
        self._forced_flushes = 0
        self._counter = 0
        self._seen: set[int] = set()

    def push(self, decoded: object) -> None:
        import heapq

        seq_id = int(getattr(decoded, "seq_id"))
        if seq_id in self._seen:
            return
        self._seen.add(seq_id)
        heapq.heappush(self._heap, (seq_id, self._counter, decoded))
        self._counter += 1

    def pop_ready(self, expected_seq: int) -> tuple[list[object], int]:
        import heapq

        ready: list[object] = []
        while self._heap and self._heap[0][0] == expected_seq:
            seq_id, _, decoded = heapq.heappop(self._heap)
            self._seen.discard(seq_id)
            ready.append(decoded)
            expected_seq += 1
        if len(self._heap) >= self._config.max_pending:
            forced = int(self._config.max_pending * self._config.forced_flush_ratio)
            for _ in range(forced):
                if not self._heap:
                    break
                seq_id, _, decoded = heapq.heappop(self._heap)
                self._seen.discard(seq_id)
                ready.append(decoded)
                expected_seq = max(expected_seq, seq_id + 1)
                self._forced_flushes += 1
        return ready, expected_seq

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def forced_flushes(self) -> int:
        return self._forced_flushes


class DynamicWorkerPool:
    """Worker pool with dynamic scaling and accounting."""

    def __init__(self, min_workers: int, max_workers: int) -> None:
        if min_workers <= 0:
            raise ValueError("min_workers must be positive")
        if max_workers < min_workers:
            raise ValueError("max_workers must be >= min_workers")
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._active_workers = 0
        self._total_spawned = 0

    @property
    def min_workers(self) -> int:
        return self._min_workers

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def active_workers(self) -> int:
        with self._lock:
            return self._active_workers

    @property
    def total_spawned(self) -> int:
        with self._lock:
            return self._total_spawned

    def register_spawn(self) -> int:
        with self._lock:
            self._active_workers += 1
            self._total_spawned += 1
            return self._total_spawned

    def register_exit(self) -> None:
        with self._lock:
            if self._active_workers > 0:
                self._active_workers -= 1


class MovingAverage:
    """Exponential moving average for smoothing supervisor signals."""

    def __init__(self, alpha: float) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._value: float | None = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = (self._alpha * value) + ((1 - self._alpha) * self._value)
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class StageSupervisor:
    """Stage-level supervisor orchestrating queue tuning and worker scaling."""

    def __init__(
        self,
        *,
        stage: str,
        queue: AdaptiveBoundedQueue[object],
        tuning: QueueTuningConfig,
        telemetry: IngestionTelemetry,
        interval_s: float,
        worker_pool: DynamicWorkerPool | None = None,
        worker_config: WorkerPoolConfig | None = None,
        spawn_worker: Callable[[], None] | None = None,
        retire_worker: Callable[[], None] | None = None,
        inflight_count: Callable[[], int] | None = None,
        ema_alpha: float = 0.25,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._stage = stage
        self._queue = queue
        self._tuning = tuning
        self._telemetry = telemetry
        self._interval_s = interval_s
        self._worker_pool = worker_pool
        self._worker_config = worker_config
        self._spawn_worker = spawn_worker
        self._retire_worker = retire_worker
        self._inflight_count = inflight_count or (lambda: 0)
        self._ema = MovingAverage(ema_alpha)
        self._last_tick_s = 0.0

    @property
    def interval_s(self) -> float:
        return self._interval_s

    def tick(self, now_s: float) -> None:
        if (now_s - self._last_tick_s) < self._interval_s:
            return
        self._last_tick_s = now_s
        depth_ratio = self._queue.depth_ratio
        inflight = self._inflight_count()
        worker_count = self._worker_pool.active_workers if self._worker_pool else 0
        self._telemetry.record_stage_sample(
            StagePressureSample(
                timestamp_s=time.time(),
                stage=self._stage,
                depth_ratio=depth_ratio,
                worker_count=worker_count,
                inflight_tasks=inflight,
            )
        )
        smoothed = self._ema.update(depth_ratio)
        self._tune_queue(smoothed)
        self._tune_workers(smoothed)

    def _tune_queue(self, depth_ratio: float) -> None:
        if depth_ratio >= self._tuning.scale_up_ratio and self._queue.capacity < self._tuning.max_capacity:
            new_capacity = min(self._queue.capacity + self._tuning.scale_step, self._tuning.max_capacity)
            self._queue.resize(new_capacity)
            self._telemetry.stage_metrics(self._stage).queue_scale_ups += 1
            self._telemetry.record_event(
                StageEvent(
                    stage=self._stage,
                    event_type="queue_scale_up",
                    message="Scaled queue up",
                    metadata={"capacity": new_capacity},
                )
            )
            return
        if depth_ratio <= self._tuning.scale_down_ratio and self._queue.capacity > self._tuning.min_capacity:
            new_capacity = max(self._queue.capacity - self._tuning.scale_step, self._tuning.min_capacity)
            self._queue.resize(new_capacity)
            self._telemetry.stage_metrics(self._stage).queue_scale_downs += 1
            self._telemetry.record_event(
                StageEvent(
                    stage=self._stage,
                    event_type="queue_scale_down",
                    message="Scaled queue down",
                    metadata={"capacity": new_capacity},
                )
            )

    def _tune_workers(self, depth_ratio: float) -> None:
        if not self._worker_pool or not self._worker_config:
            return
        if not self._spawn_worker or not self._retire_worker:
            return
        if (
            depth_ratio >= self._worker_config.scale_up_ratio
            and self._worker_pool.active_workers < self._worker_pool.max_workers
        ):
            self._spawn_worker()
            self._telemetry.stage_metrics(self._stage).worker_scale_ups += 1
            self._telemetry.record_event(
                StageEvent(
                    stage=self._stage,
                    event_type="worker_scale_up",
                    message="Scaled workers up",
                    metadata={"workers": self._worker_pool.active_workers},
                )
            )
            return
        if depth_ratio <= self._worker_config.scale_down_ratio and self._worker_pool.active_workers > self._worker_pool.min_workers:
            self._retire_worker()
            self._telemetry.stage_metrics(self._stage).worker_scale_downs += 1
            self._telemetry.record_event(
                StageEvent(
                    stage=self._stage,
                    event_type="worker_scale_down",
                    message="Scaled workers down",
                    metadata={"workers": self._worker_pool.active_workers},
                )
            )


class ControlPlaneOrchestrator:
    """Orchestrates multiple stage supervisors with deterministic cadence."""

    def __init__(
        self,
        *,
        supervisors: Iterable[StageSupervisor],
        stop_event: threading.Event,
        producer_done: threading.Event,
        drain_condition: Callable[[], bool] | None = None,
    ) -> None:
        self._supervisors = list(supervisors)
        if not self._supervisors:
            raise ValueError("At least one supervisor is required")
        self._stop_event = stop_event
        self._producer_done = producer_done
        self._drain_condition = drain_condition
        self._min_interval_s = min(s.interval_s for s in self._supervisors)

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                now_s = time.perf_counter()
                for supervisor in self._supervisors:
                    supervisor.tick(now_s)
                if self._producer_done.is_set() and self._drain_condition and self._drain_condition():
                    break
                time.sleep(self._min_interval_s / 2.0)
        except Exception as exc:  # pragma: no cover - supervisor failure
            LOGGER.exception("Control-plane orchestrator failed")
            raise exc
        finally:
            LOGGER.info("Control-plane orchestrator exiting")


__all__ = [
    "AdaptiveBoundedQueue",
    "BackpressureTimeout",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "ControlPlaneOrchestrator",
    "DeterministicEventLog",
    "DeterministicReorderBuffer",
    "DynamicWorkerPool",
    "IngestionClosed",
    "IngestionFailureEvent",
    "IngestionFailureReport",
    "IngestionTelemetry",
    "MovingAverage",
    "OrderingBufferConfig",
    "QueueLike",
    "QueueTuningConfig",
    "RetryPolicyConfig",
    "StageEvent",
    "StageLatencySample",
    "StagePressureSample",
    "StageState",
    "StageSupervisor",
    "WorkerPoolConfig",
]
