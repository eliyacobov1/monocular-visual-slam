"""Deterministic tracking control plane with supervised feature extraction."""

from __future__ import annotations

import heapq
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Literal

import numpy as np

from control_plane_hub import StageHealthSnapshot
from data_persistence import P2Quantile
from deterministic_integrity import stable_event_digest
from feature_control_plane import FeatureControlPlane, FeatureResult
from ingestion_control_plane import BackpressureTimeout, CircuitBreaker, CircuitBreakerConfig


@dataclass(frozen=True)
class TrackingControlConfig:
    """Configuration for deterministic tracking control."""

    enabled: bool = False
    max_pending_frames: int = 64
    frame_ttl_s: float = 1.0
    drop_policy: Literal["drop_oldest", "reject_new"] = "drop_oldest"
    backpressure_timeout_s: float | None = 0.5
    telemetry_quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
    event_log_capacity: int = 256
    deterministic_seed: int = 1337
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def __post_init__(self) -> None:
        if self.max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")
        if self.frame_ttl_s <= 0:
            raise ValueError("frame_ttl_s must be positive")
        if self.backpressure_timeout_s is not None and self.backpressure_timeout_s < 0:
            raise ValueError("backpressure_timeout_s must be non-negative")
        if not self.telemetry_quantiles:
            raise ValueError("telemetry_quantiles must be non-empty")
        if self.event_log_capacity <= 0:
            raise ValueError("event_log_capacity must be positive")
        if self.deterministic_seed < 0:
            raise ValueError("deterministic_seed must be non-negative")


class TrackingState(str, Enum):
    """Lifecycle state for the tracking control plane."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TRIPPED = "tripped"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class TrackingEvent:
    """Structured event emitted by the tracking control plane."""

    event_type: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=time.time)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class DeterministicEventLog:
    """Fixed-size ring buffer for tracking events."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._events: deque[TrackingEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, event: TrackingEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[TrackingEvent]:
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


class MetricTracker:
    """Streaming statistics tracker with deterministic quantiles."""

    def __init__(self, quantiles: Iterable[float]) -> None:
        self._count = 0
        self._total = 0.0
        self._total_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._quantiles = tuple(P2Quantile(q) for q in quantiles)

    def update(self, value: float) -> None:
        value = float(value)
        self._count += 1
        self._total += value
        self._total_sq += value * value
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        for estimator in self._quantiles:
            estimator.update(value)

    def summary(self) -> dict[str, float]:
        if self._count == 0:
            return {
                "count": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        mean = self._total / self._count
        variance = max(self._total_sq / self._count - mean * mean, 0.0)
        summary = {
            "count": float(self._count),
            "mean": mean,
            "std": float(variance**0.5),
            "min": float(self._min),
            "max": float(self._max),
        }
        for estimator in self._quantiles:
            key = f"p{int(estimator.quantile * 100)}"
            summary[key] = estimator.value()
        return summary


@dataclass(frozen=True)
class TrackingTelemetrySummary:
    """Aggregated telemetry summary for tracking control."""

    total_wait_s: dict[str, float]
    feature_queue_wait_s: dict[str, float]


class TrackingTelemetry:
    """Telemetry aggregator for tracking control plane."""

    def __init__(self, quantiles: Iterable[float]) -> None:
        self._total_wait = MetricTracker(quantiles)
        self._feature_queue_wait = MetricTracker(quantiles)

    def update(self, total_wait_s: float, feature_queue_wait_s: float) -> None:
        self._total_wait.update(total_wait_s)
        self._feature_queue_wait.update(feature_queue_wait_s)

    def summary(self) -> TrackingTelemetrySummary:
        return TrackingTelemetrySummary(
            total_wait_s=self._total_wait.summary(),
            feature_queue_wait_s=self._feature_queue_wait.summary(),
        )


@dataclass(frozen=True)
class FramePayload:
    """Frame payload tracked by the supervisor."""

    seq_id: int
    timestamp: float
    frame_gray: np.ndarray
    enqueued_at_s: float
    deadline_s: float


class PendingFrameBuffer:
    """Ordered frame buffer with deterministic drop and expiry semantics."""

    def __init__(
        self,
        *,
        capacity: int,
        frame_ttl_s: float,
        drop_policy: Literal["drop_oldest", "reject_new"],
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if frame_ttl_s <= 0:
            raise ValueError("frame_ttl_s must be positive")
        self._capacity = capacity
        self._frame_ttl_s = frame_ttl_s
        self._drop_policy = drop_policy
        self._frames: OrderedDict[int, FramePayload] = OrderedDict()
        self._deadlines: list[tuple[float, int]] = []

    def add(self, payload: FramePayload) -> list[FramePayload]:
        if payload.seq_id in self._frames:
            raise ValueError("Duplicate frame seq_id")
        dropped: list[FramePayload] = []
        if len(self._frames) >= self._capacity:
            if self._drop_policy == "reject_new":
                raise BackpressureTimeout("Pending frame buffer is full")
            _, dropped_payload = self._frames.popitem(last=False)
            dropped.append(dropped_payload)
        self._frames[payload.seq_id] = payload
        heapq.heappush(self._deadlines, (payload.deadline_s, payload.seq_id))
        return dropped

    def pop(self, seq_id: int) -> FramePayload | None:
        return self._frames.pop(seq_id, None)

    def expire(self, now_s: float) -> list[FramePayload]:
        expired: list[FramePayload] = []
        while self._deadlines and self._deadlines[0][0] <= now_s:
            _, seq_id = heapq.heappop(self._deadlines)
            payload = self._frames.pop(seq_id, None)
            if payload is not None:
                expired.append(payload)
        expired.sort(key=lambda item: item.seq_id)
        return expired

    @property
    def size(self) -> int:
        return len(self._frames)

    @property
    def capacity(self) -> int:
        return self._capacity


@dataclass(frozen=True)
class TrackingFrameResult:
    """Result from tracking control plane with contextual payload."""

    seq_id: int
    timestamp: float
    frame_gray: np.ndarray
    feature_result: FeatureResult | None
    drop_reason: str | None
    total_wait_s: float
    feature_queue_wait_s: float


class TrackingControlPlane:
    """Supervised tracking control plane for feature-driven tracking."""

    def __init__(
        self,
        feature_plane: FeatureControlPlane,
        *,
        config: TrackingControlConfig | None = None,
    ) -> None:
        self._feature_plane = feature_plane
        self._config = config or TrackingControlConfig()
        self._buffer = PendingFrameBuffer(
            capacity=self._config.max_pending_frames,
            frame_ttl_s=self._config.frame_ttl_s,
            drop_policy=self._config.drop_policy,
        )
        self._telemetry = TrackingTelemetry(self._config.telemetry_quantiles)
        self._event_log = DeterministicEventLog(self._config.event_log_capacity)
        self._breaker = CircuitBreaker(self._config.circuit_breaker)
        self._ready_results: deque[TrackingFrameResult] = deque()
        self._closed = False
        self._lock = threading.Lock()
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._orphan_results = 0
        self._errors = 0

    @property
    def state(self) -> TrackingState:
        state = self._breaker.state
        if state == "open":
            return TrackingState.TRIPPED
        if state == "half_open":
            return TrackingState.RECOVERING
        return TrackingState.HEALTHY

    @property
    def pending_frames(self) -> int:
        return self._buffer.size

    def _record_event(self, event_type: str, message: str, metadata: dict[str, object]) -> None:
        self._event_log.record(
            TrackingEvent(
                event_type=event_type,
                message=message,
                metadata=metadata,
            )
        )

    def _emit_drop(self, payload: FramePayload, reason: str) -> None:
        total_wait_s = time.perf_counter() - payload.enqueued_at_s
        with self._lock:
            self._dropped += 1
        self._ready_results.append(
            TrackingFrameResult(
                seq_id=payload.seq_id,
                timestamp=payload.timestamp,
                frame_gray=payload.frame_gray,
                feature_result=None,
                drop_reason=reason,
                total_wait_s=total_wait_s,
                feature_queue_wait_s=0.0,
            )
        )
        self._record_event(
            "frame_drop",
            reason,
            {"seq_id": payload.seq_id, "total_wait_s": total_wait_s},
        )

    def submit_frame(self, *, seq_id: int, timestamp: float, frame_gray: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Tracking control plane is closed")
        with self._lock:
            self._submitted += 1
        if not self._breaker.allow():
            payload = FramePayload(
                seq_id=int(seq_id),
                timestamp=float(timestamp),
                frame_gray=frame_gray,
                enqueued_at_s=time.perf_counter(),
                deadline_s=time.perf_counter() + self._config.frame_ttl_s,
            )
            self._emit_drop(payload, "circuit_breaker_open")
            return
        now_s = time.perf_counter()
        payload = FramePayload(
            seq_id=int(seq_id),
            timestamp=float(timestamp),
            frame_gray=frame_gray,
            enqueued_at_s=now_s,
            deadline_s=now_s + self._config.frame_ttl_s,
        )
        dropped = self._buffer.add(payload)
        for dropped_payload in dropped:
            self._emit_drop(dropped_payload, "buffer_overflow")
        self._feature_plane.submit(seq_id=payload.seq_id, timestamp=payload.timestamp, frame=frame_gray)

    def _expire_frames(self) -> None:
        expired = self._buffer.expire(time.perf_counter())
        for payload in expired:
            self._emit_drop(payload, "deadline_expired")

    def _consume_feature_result(self, result: FeatureResult) -> None:
        payload = self._buffer.pop(result.seq_id)
        if payload is None:
            with self._lock:
                self._orphan_results += 1
            self._record_event(
                "orphan_result",
                "Received feature result for unknown frame",
                {"seq_id": result.seq_id},
            )
            return
        total_wait_s = time.perf_counter() - payload.enqueued_at_s
        self._telemetry.update(total_wait_s, result.queue_wait_s)
        if result.error:
            self._breaker.record_failure()
            with self._lock:
                self._errors += 1
        else:
            self._breaker.record_success()
        with self._lock:
            self._processed += 1
        self._ready_results.append(
            TrackingFrameResult(
                seq_id=result.seq_id,
                timestamp=payload.timestamp,
                frame_gray=payload.frame_gray,
                feature_result=result,
                drop_reason=None,
                total_wait_s=total_wait_s,
                feature_queue_wait_s=result.queue_wait_s,
            )
        )

    def drain_ready(self) -> Iterator[TrackingFrameResult]:
        self._expire_frames()
        for result in self._feature_plane.drain():
            self._consume_feature_result(result)
        while self._ready_results:
            yield self._ready_results.popleft()

    def collect(self, timeout_s: float | None = None) -> TrackingFrameResult:
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        while True:
            self._expire_frames()
            if self._ready_results:
                return self._ready_results.popleft()
            remaining = None if deadline is None else max(deadline - time.perf_counter(), 0.0)
            if deadline is not None and remaining == 0.0:
                raise BackpressureTimeout("Timed out waiting for tracking result")
            feature_result = self._feature_plane.collect(timeout_s=remaining)
            self._consume_feature_result(feature_result)
            if self._ready_results:
                return self._ready_results.popleft()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._feature_plane.close()

    def telemetry_summary(self) -> TrackingTelemetrySummary:
        return self._telemetry.summary()

    def health_snapshot(self) -> StageHealthSnapshot:
        with self._lock:
            breaker_state = self._breaker.state
            if breaker_state == "open":
                state = "tripped"
            elif breaker_state == "half_open":
                state = "recovering"
            elif self._errors > 0:
                state = "degraded"
            else:
                state = "healthy"
            metrics = {
                "pending_frames": float(self._buffer.size),
                "buffer_capacity": float(self._buffer.capacity),
                "ready_results": float(len(self._ready_results)),
            }
            counters = {
                "submitted": self._submitted,
                "processed": self._processed,
                "dropped": self._dropped,
                "orphan_results": self._orphan_results,
                "errors": self._errors,
            }
        return StageHealthSnapshot(
            stage="tracking",
            state=state,
            metrics=metrics,
            counters=counters,
        )

    def events(self) -> tuple[TrackingEvent, ...]:
        return tuple(self._event_log.snapshot())


__all__ = [
    "TrackingControlConfig",
    "TrackingControlPlane",
    "TrackingEvent",
    "TrackingFrameResult",
    "TrackingTelemetrySummary",
    "TrackingState",
    "PendingFrameBuffer",
    "FramePayload",
]
