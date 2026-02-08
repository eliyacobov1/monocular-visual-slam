"""Deterministic, supervised feature extraction control plane."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Literal

import cv2
import numpy as np

from control_plane_hub import StageHealthSnapshot
from data_persistence import P2Quantile
from deterministic_integrity import stable_event_digest
from feature_pipeline import FeaturePipelineConfig, build_feature_pipeline
from ingestion_control_plane import (
    AdaptiveBoundedQueue,
    BackpressureTimeout,
    CircuitBreaker,
    CircuitBreakerConfig,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureControlConfig:
    """Configuration for deterministic feature extraction control plane."""

    enabled: bool = False
    executor: Literal["thread", "process"] = "thread"
    num_workers: int = 2
    max_inflight: int = 8
    result_queue_capacity: int = 32
    backpressure_timeout_s: float | None = 0.5
    cache_size: int = 64
    cache_ttl_s: float = 1.0
    reorder_buffer_size: int = 128
    telemetry_quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
    deterministic_seed: int = 1337
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def __post_init__(self) -> None:
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if self.result_queue_capacity <= 0:
            raise ValueError("Queue capacities must be positive")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.cache_ttl_s <= 0:
            raise ValueError("cache_ttl_s must be positive")
        if self.reorder_buffer_size <= 0:
            raise ValueError("reorder_buffer_size must be positive")
        if not self.telemetry_quantiles:
            raise ValueError("telemetry_quantiles must be non-empty")


@dataclass(frozen=True)
class FeatureTask:
    """Feature extraction task for a single frame."""

    seq_id: int
    timestamp: float
    frame: np.ndarray
    enqueued_at_s: float


@dataclass(frozen=True)
class FeatureWorkerResult:
    """Internal result payload from a worker."""

    seq_id: int
    timestamp: float
    keypoints_payload: np.ndarray | None
    descriptors: np.ndarray | None
    duration_s: float
    queue_wait_s: float
    cache_hit: bool
    cache_key: str | None
    error: str | None


@dataclass(frozen=True)
class FeatureResult:
    """External feature extraction result with decoded keypoints."""

    seq_id: int
    timestamp: float
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    duration_s: float
    queue_wait_s: float
    cache_hit: bool
    cache_key: str | None
    error: str | None


@dataclass(frozen=True)
class FeatureControlEvent:
    """Structured control-plane event for feature extraction."""

    event_type: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=time.time)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class MetricTracker:
    """Streaming metric tracker with deterministic quantiles."""

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
class FeatureTelemetrySummary:
    """Summary of feature extraction telemetry."""

    duration_s: dict[str, float]
    queue_wait_s: dict[str, float]


class FeatureTelemetry:
    """Telemetry aggregator for feature extraction."""

    def __init__(self, quantiles: Iterable[float]) -> None:
        self._duration = MetricTracker(quantiles)
        self._queue_wait = MetricTracker(quantiles)

    def update(self, duration_s: float, queue_wait_s: float) -> None:
        self._duration.update(duration_s)
        self._queue_wait.update(queue_wait_s)

    def summary(self) -> FeatureTelemetrySummary:
        return FeatureTelemetrySummary(
            duration_s=self._duration.summary(),
            queue_wait_s=self._queue_wait.summary(),
        )


class FeatureCache:
    """LRU cache with TTL for feature extraction payloads."""

    def __init__(self, capacity: int, ttl_s: float) -> None:
        self._capacity = capacity
        self._ttl_s = ttl_s
        self._items: OrderedDict[str, tuple[float, np.ndarray, np.ndarray | None]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray | None] | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            timestamp, keypoints, descriptors = item
            if now - timestamp > self._ttl_s:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return keypoints, descriptors

    def put(self, key: str, keypoints: np.ndarray, descriptors: np.ndarray | None) -> None:
        now = time.monotonic()
        with self._lock:
            self._items[key] = (now, keypoints, descriptors)
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)


class DeterministicReorderBuffer:
    """Ordering buffer that releases results in monotonically increasing seq_id order."""

    def __init__(self, capacity: int, initial_seq: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._pending: dict[int, FeatureWorkerResult] = {}
        self._next_seq = initial_seq

    def push(self, result: FeatureWorkerResult) -> list[FeatureWorkerResult]:
        if len(self._pending) >= self._capacity:
            raise BackpressureTimeout("Reorder buffer capacity exceeded")
        self._pending[result.seq_id] = result
        ready: list[FeatureWorkerResult] = []
        while self._next_seq in self._pending:
            ready.append(self._pending.pop(self._next_seq))
            self._next_seq += 1
        return ready


def _hash_frame(frame: np.ndarray) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(frame.tobytes())
    hasher.update(str(frame.shape).encode("utf-8"))
    hasher.update(str(frame.dtype).encode("utf-8"))
    return hasher.hexdigest()


def _serialize_keypoints(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    payload = np.zeros((len(keypoints), 7), dtype=np.float32)
    for idx, kp in enumerate(keypoints):
        payload[idx] = np.array(
            [
                kp.pt[0],
                kp.pt[1],
                kp.size,
                kp.angle,
                kp.response,
                float(kp.octave),
                float(kp.class_id),
            ],
            dtype=np.float32,
        )
    return payload


def _deserialize_keypoints(payload: np.ndarray) -> list[cv2.KeyPoint]:
    keypoints: list[cv2.KeyPoint] = []
    for row in payload:
        keypoints.append(
            cv2.KeyPoint(
                x=float(row[0]),
                y=float(row[1]),
                size=float(row[2]),
                angle=float(row[3]),
                response=float(row[4]),
                octave=int(row[5]),
                class_id=int(row[6]),
            )
        )
    return keypoints


def _process_worker_task(
    task: FeatureTask,
    feature_config: FeaturePipelineConfig,
    deterministic_seed: int,
    queue_wait_s: float,
    cache_key: str,
) -> FeatureWorkerResult:
    start = time.perf_counter()
    try:
        cv2.setRNGSeed(deterministic_seed + task.seq_id)
        pipeline = build_feature_pipeline(feature_config)
        keypoints, descriptors = pipeline.detect_and_describe(task.frame)
        keypoints_payload = _serialize_keypoints(keypoints)
        duration_s = time.perf_counter() - start
        return FeatureWorkerResult(
            seq_id=task.seq_id,
            timestamp=task.timestamp,
            keypoints_payload=keypoints_payload,
            descriptors=descriptors,
            duration_s=duration_s,
            queue_wait_s=queue_wait_s,
            cache_hit=False,
            cache_key=cache_key,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return FeatureWorkerResult(
            seq_id=task.seq_id,
            timestamp=task.timestamp,
            keypoints_payload=None,
            descriptors=None,
            duration_s=0.0,
            queue_wait_s=queue_wait_s,
            cache_hit=False,
            cache_key=cache_key,
            error=str(exc),
        )


class FeatureControlPlane:
    """Supervised feature extraction control plane with deterministic ordering."""

    def __init__(
        self,
        *,
        feature_config: FeaturePipelineConfig,
        control_config: FeatureControlConfig | None = None,
    ) -> None:
        self._feature_config = feature_config
        self._config = control_config or FeatureControlConfig()
        self._result_queue: AdaptiveBoundedQueue[FeatureWorkerResult] = AdaptiveBoundedQueue(
            capacity=self._config.result_queue_capacity,
            min_capacity=self._config.result_queue_capacity,
            max_capacity=self._config.result_queue_capacity,
        )
        self._telemetry = FeatureTelemetry(self._config.telemetry_quantiles)
        self._cache = FeatureCache(self._config.cache_size, self._config.cache_ttl_s)
        self._breaker = CircuitBreaker(self._config.circuit_breaker)
        self._reorder = DeterministicReorderBuffer(self._config.reorder_buffer_size)
        self._events: list[FeatureControlEvent] = []
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._inflight = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._errors = 0
        self._breaker_opens = 0
        self._inflight_sem = threading.Semaphore(self._config.max_inflight)
        self._executor = self._build_executor()
        self._started = False
        self._closed = False
        self._thread_local = threading.local()

    def _build_executor(self):
        if self._config.executor == "process":
            from concurrent.futures import ProcessPoolExecutor

            return ProcessPoolExecutor(max_workers=self._config.num_workers)
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(max_workers=self._config.num_workers)

    def _get_pipeline(self):
        pipeline = getattr(self._thread_local, "pipeline", None)
        if pipeline is None:
            pipeline = build_feature_pipeline(self._feature_config)
            self._thread_local.pipeline = pipeline
        return pipeline

    def _record_event(self, event_type: str, message: str, metadata: dict[str, object]) -> None:
        with self._lock:
            self._events.append(
                FeatureControlEvent(
                    event_type=event_type,
                    message=message,
                    metadata=metadata,
                )
            )

    def _record_completion(self, result: FeatureWorkerResult) -> None:
        with self._lock:
            self._completed += 1
            self._inflight = max(self._inflight - 1, 0)
            if result.cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            if result.error:
                self._errors += 1
                if result.error == "circuit_breaker_open":
                    self._breaker_opens += 1

    def submit(self, *, seq_id: int, timestamp: float, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Feature control plane is closed")
        if not self._inflight_sem.acquire(timeout=self._config.backpressure_timeout_s):
            raise BackpressureTimeout("Timed out waiting for feature slot")
        with self._lock:
            self._submitted += 1
            self._inflight += 1
        cache_key = _hash_frame(frame)
        cached = self._cache.get(cache_key)
        if cached is not None:
            keypoints_payload, descriptors = cached
            self._result_queue.put(
                FeatureWorkerResult(
                    seq_id=int(seq_id),
                    timestamp=float(timestamp),
                    keypoints_payload=keypoints_payload,
                    descriptors=descriptors,
                    duration_s=0.0,
                    queue_wait_s=0.0,
                    cache_hit=True,
                    cache_key=cache_key,
                    error=None,
                )
            )
            with self._lock:
                self._cache_hits += 1
                self._completed += 1
                self._inflight -= 1
            self._inflight_sem.release()
            return
        task = FeatureTask(
            seq_id=int(seq_id),
            timestamp=float(timestamp),
            frame=frame,
            enqueued_at_s=time.perf_counter(),
        )
        self._dispatch_task(task)

    def _dispatch_task(self, task: FeatureTask) -> None:
        from concurrent.futures import Future

        def _callback(future: Future[FeatureWorkerResult]) -> None:
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                result = FeatureWorkerResult(
                    seq_id=task.seq_id,
                    timestamp=task.timestamp,
                    keypoints_payload=None,
                    descriptors=None,
                    duration_s=0.0,
                    queue_wait_s=0.0,
                    cache_hit=False,
                    cache_key=None,
                    error=str(exc),
                )
                LOGGER.exception("Feature worker failed")
            self._record_completion(result)
            self._result_queue.put(result)
            self._inflight_sem.release()

        if self._config.executor == "process":
            future = self._executor.submit(
                _process_worker_task,
                task,
                self._feature_config,
                self._config.deterministic_seed,
                time.perf_counter() - task.enqueued_at_s,
                _hash_frame(task.frame),
            )
        else:
            future = self._executor.submit(self._execute_task, task)
        future.add_done_callback(_callback)
        if not self._started:
            self._started = True

    def _execute_task(self, task: FeatureTask) -> FeatureWorkerResult:
        queue_wait_s = time.perf_counter() - task.enqueued_at_s
        cache_key = _hash_frame(task.frame)
        cached = self._cache.get(cache_key)
        if cached is not None:
            keypoints_payload, descriptors = cached
            return FeatureWorkerResult(
                seq_id=task.seq_id,
                timestamp=task.timestamp,
                keypoints_payload=keypoints_payload,
                descriptors=descriptors,
                duration_s=0.0,
                queue_wait_s=queue_wait_s,
                cache_hit=True,
                cache_key=cache_key,
                error=None,
            )
        if not self._breaker.allow():
            return FeatureWorkerResult(
                seq_id=task.seq_id,
                timestamp=task.timestamp,
                keypoints_payload=None,
                descriptors=None,
                duration_s=0.0,
                queue_wait_s=queue_wait_s,
                cache_hit=False,
                cache_key=cache_key,
                error="circuit_breaker_open",
            )
        start = time.perf_counter()
        try:
            cv2.setRNGSeed(self._config.deterministic_seed + task.seq_id)
            pipeline = self._get_pipeline()
            keypoints, descriptors = pipeline.detect_and_describe(task.frame)
            keypoints_payload = _serialize_keypoints(keypoints)
            self._cache.put(cache_key, keypoints_payload, descriptors)
            duration_s = time.perf_counter() - start
            self._breaker.record_success()
            return FeatureWorkerResult(
                seq_id=task.seq_id,
                timestamp=task.timestamp,
                keypoints_payload=keypoints_payload,
                descriptors=descriptors,
                duration_s=duration_s,
                queue_wait_s=queue_wait_s,
                cache_hit=False,
                cache_key=cache_key,
                error=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._breaker.record_failure()
            self._record_event("feature_error", str(exc), {"seq_id": task.seq_id})
            return FeatureWorkerResult(
                seq_id=task.seq_id,
                timestamp=task.timestamp,
                keypoints_payload=None,
                descriptors=None,
                duration_s=0.0,
                queue_wait_s=queue_wait_s,
                cache_hit=False,
                cache_key=cache_key,
                error=str(exc),
            )

    def collect(self, timeout_s: float | None = None) -> FeatureResult:
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        while True:
            remaining = None if deadline is None else max(deadline - time.perf_counter(), 0.0)
            if deadline is not None and remaining == 0.0:
                raise BackpressureTimeout("Timed out waiting for feature result")
            worker_result = self._result_queue.get(timeout_s=remaining)
            ready = self._reorder.push(worker_result)
            if not ready:
                continue
            result = ready[0]
            self._telemetry.update(result.duration_s, result.queue_wait_s)
            if result.cache_key and not result.cache_hit and result.error is None:
                payload = (
                    result.keypoints_payload
                    if result.keypoints_payload is not None
                    else np.zeros((0, 7), dtype=np.float32)
                )
                self._cache.put(result.cache_key, payload, result.descriptors)
            keypoints_payload = (
                result.keypoints_payload
                if result.keypoints_payload is not None
                else np.zeros((0, 7), dtype=np.float32)
            )
            keypoints = _deserialize_keypoints(keypoints_payload)
            return FeatureResult(
                seq_id=result.seq_id,
                timestamp=result.timestamp,
                keypoints=keypoints,
                descriptors=result.descriptors,
                duration_s=result.duration_s,
                queue_wait_s=result.queue_wait_s,
                cache_hit=result.cache_hit,
                cache_key=result.cache_key,
                error=result.error,
            )

    def drain(self) -> Iterator[FeatureResult]:
        while True:
            try:
                yield self.collect(timeout_s=0.0)
            except BackpressureTimeout:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)

    def telemetry_summary(self) -> FeatureTelemetrySummary:
        return self._telemetry.summary()

    def health_snapshot(self) -> StageHealthSnapshot:
        queue_depth = self._result_queue.size
        queue_capacity = self._result_queue.capacity
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
                "queue_depth_ratio": 0.0 if queue_capacity == 0 else queue_depth / queue_capacity,
                "inflight_ratio": 0.0
                if self._config.max_inflight == 0
                else self._inflight / self._config.max_inflight,
            }
            counters = {
                "submitted": self._submitted,
                "completed": self._completed,
                "inflight": self._inflight,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "errors": self._errors,
                "breaker_opens": self._breaker_opens,
            }
        return StageHealthSnapshot(
            stage="feature",
            state=state,
            metrics=metrics,
            counters=counters,
        )

    def events(self) -> tuple[FeatureControlEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def event_digest(self) -> str:
        with self._lock:
            return stable_event_digest(self._events)


__all__ = [
    "FeatureControlConfig",
    "FeatureControlPlane",
    "FeatureControlEvent",
    "FeatureResult",
    "FeatureTelemetrySummary",
]
