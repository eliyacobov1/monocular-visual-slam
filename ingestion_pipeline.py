"""Asynchronous, bounded ingestion pipeline for frame decoding."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Iterable, Iterator, Optional, Protocol, TypeVar

import cv2
import numpy as np

from frame_stream import FramePacket

LOGGER = logging.getLogger(__name__)

_OUTPUT_SENTINEL = object()
_WORKER_STOP = object()

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
    """Thread-safe bounded queue with adaptive capacity."""

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

    def put(self, item: T, timeout_s: float | None = None) -> None:
        with self._condition:
            start = time.perf_counter()
            while len(self._items) >= self._capacity and not self._closed:
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


@dataclass(frozen=True)
class FrameSourceEntry:
    """Frame metadata input to the ingestion pipeline."""

    seq_id: int
    index: int
    timestamp: float
    path: Path


@dataclass(frozen=True)
class DecodedFrame:
    """Decoded frame payload with sequence ordering."""

    seq_id: int
    packet: FramePacket | None
    error: str | None


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
class IngestionPipelineConfig:
    """Configuration for the asynchronous ingestion pipeline."""

    entry_queue_capacity: int = 32
    output_queue_capacity: int = 16
    read_timeout_s: float | None = 0.5
    decode_timeout_s: float | None = 0.5
    num_decode_workers: int = 2
    max_frames: int | None = None
    fail_fast: bool = True
    decode_backend: int = cv2.IMREAD_COLOR
    entry_queue_tuning: QueueTuningConfig = field(default_factory=QueueTuningConfig)
    output_queue_tuning: QueueTuningConfig = field(default_factory=QueueTuningConfig)
    worker_pool: WorkerPoolConfig = field(default_factory=WorkerPoolConfig)
    telemetry_window: int = 256

    def __post_init__(self) -> None:
        if self.entry_queue_capacity <= 0 or self.output_queue_capacity <= 0:
            raise ValueError("Queue capacities must be positive")
        if self.num_decode_workers <= 0:
            raise ValueError("num_decode_workers must be positive")
        if not (self.entry_queue_tuning.min_capacity <= self.entry_queue_capacity <= self.entry_queue_tuning.max_capacity):
            raise ValueError("entry_queue_capacity must be within tuning bounds")
        if not (
            self.output_queue_tuning.min_capacity
            <= self.output_queue_capacity
            <= self.output_queue_tuning.max_capacity
        ):
            raise ValueError("output_queue_capacity must be within tuning bounds")
        min_workers = max(self.worker_pool.min_workers, self.num_decode_workers)
        max_workers = max(self.worker_pool.max_workers, min_workers)
        object.__setattr__(
            self,
            "worker_pool",
            WorkerPoolConfig(
                min_workers=min_workers,
                max_workers=max_workers,
                scale_up_ratio=self.worker_pool.scale_up_ratio,
                scale_down_ratio=self.worker_pool.scale_down_ratio,
                supervisor_interval_s=self.worker_pool.supervisor_interval_s,
            ),
        )


@dataclass
class IngestionFailureEvent:
    """Structured failure event from ingestion stages."""

    seq_id: int
    stage: str
    error: str
    path: Path | None
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


@dataclass
class QueuePressureSample:
    """Snapshot of queue pressure and worker count."""

    timestamp_s: float
    entry_depth_ratio: float
    output_depth_ratio: float
    worker_count: int


@dataclass
class IngestionTelemetry:
    """Telemetry samples for queue pressure and worker sizing."""

    window: int = 256
    samples: Deque[QueuePressureSample] = field(default_factory=deque)
    worker_scale_ups: int = 0
    worker_scale_downs: int = 0
    queue_scale_ups: int = 0
    queue_scale_downs: int = 0

    def record_sample(self, sample: QueuePressureSample) -> None:
        if len(self.samples) >= self.window:
            self.samples.popleft()
        self.samples.append(sample)


@dataclass
class IngestionPipelineStats:
    """Operational metrics for the ingestion pipeline."""

    enqueued_entries: int = 0
    dequeued_entries: int = 0
    dropped_entries: int = 0
    decoded_frames: int = 0
    dropped_frames: int = 0
    read_failures: int = 0
    output_backpressure: int = 0
    max_entry_queue_depth: int = 0
    max_output_queue_depth: int = 0
    total_decode_s: float = 0.0
    started_at_s: float | None = None
    finished_at_s: float | None = None

    def mark_start(self) -> None:
        self.started_at_s = time.perf_counter()

    def mark_finish(self) -> None:
        self.finished_at_s = time.perf_counter()

    @property
    def duration_s(self) -> float | None:
        if self.started_at_s is None or self.finished_at_s is None:
            return None
        return self.finished_at_s - self.started_at_s


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


class AsyncIngestionPipeline(Iterable[FramePacket]):
    """Multi-stage bounded ingestion pipeline with adaptive scaling."""

    def __init__(
        self,
        entries: Iterable[tuple[int, float | None, Path]],
        *,
        config: IngestionPipelineConfig | None = None,
        read_fn: Callable[[str, int], Optional[np.ndarray]] | None = None,
    ) -> None:
        self._entries = entries
        self._config = config or IngestionPipelineConfig()
        entry_tuning = self._config.entry_queue_tuning
        output_tuning = self._config.output_queue_tuning
        self._entry_queue: AdaptiveBoundedQueue[FrameSourceEntry | object] = AdaptiveBoundedQueue(
            capacity=self._config.entry_queue_capacity,
            min_capacity=entry_tuning.min_capacity,
            max_capacity=entry_tuning.max_capacity,
        )
        self._output_queue: AdaptiveBoundedQueue[DecodedFrame | object] = AdaptiveBoundedQueue(
            capacity=self._config.output_queue_capacity,
            min_capacity=output_tuning.min_capacity,
            max_capacity=output_tuning.max_capacity,
        )
        self._read_fn = read_fn or cv2.imread
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._producer_done = threading.Event()
        self._error: Exception | None = None
        self._stats = IngestionPipelineStats()
        self._telemetry = IngestionTelemetry(window=self._config.telemetry_window)
        self._failures = IngestionFailureReport()
        self._started = False
        self._worker_pool = DynamicWorkerPool(
            min_workers=self._config.worker_pool.min_workers,
            max_workers=self._config.worker_pool.max_workers,
        )
        self._retire_lock = threading.Lock()
        self._retire_requests = 0

    @property
    def stats(self) -> IngestionPipelineStats:
        return self._stats

    @property
    def telemetry(self) -> IngestionTelemetry:
        return self._telemetry

    @property
    def failures(self) -> IngestionFailureReport:
        return self._failures

    def start(self) -> None:
        if self._started:
            return
        self._stats.mark_start()
        producer = threading.Thread(target=self._run_producer, name="ingest-producer", daemon=True)
        self._threads.append(producer)
        producer.start()
        for _ in range(self._config.worker_pool.min_workers):
            self._spawn_worker()
        supervisor = threading.Thread(
            target=self._run_supervisor, name="ingest-supervisor", daemon=True
        )
        self._threads.append(supervisor)
        supervisor.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        self._entry_queue.close()
        self._output_queue.close()
        for _ in range(self._worker_pool.active_workers):
            self._safe_put_entry(_WORKER_STOP)
        for thread in self._threads:
            thread.join(timeout=1.0)

    def __enter__(self) -> "AsyncIngestionPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.stop()

    def __iter__(self) -> Iterator[FramePacket]:
        self.start()
        expected_seq = 0
        completed_workers = 0
        pending: list[tuple[int, DecodedFrame]] = []
        while True:
            if (
                self._producer_done.is_set()
                and completed_workers >= self._worker_pool.total_spawned
                and not pending
            ):
                break
            try:
                item = self._output_queue.get(timeout_s=self._config.decode_timeout_s)
            except BackpressureTimeout:
                if self._error is not None:
                    raise self._error
                if self._stop_event.is_set():
                    break
                continue
            except IngestionClosed:
                if self._error is not None:
                    raise self._error
                break
            if item is _OUTPUT_SENTINEL:
                completed_workers += 1
                continue
            decoded = item
            pending.append((decoded.seq_id, decoded))
            pending.sort(key=lambda pair: pair[0])
            while pending and pending[0][0] == expected_seq:
                _, ready = pending.pop(0)
                if ready.packet is None:
                    LOGGER.warning(
                        "Dropped frame during ingestion",
                        extra={"seq_id": ready.seq_id, "error": ready.error},
                    )
                else:
                    self._stats.dequeued_entries += 1
                    yield ready.packet
                expected_seq += 1
        if self._error is not None:
            raise self._error
        self._stats.mark_finish()

    def _spawn_worker(self) -> None:
        worker_index = self._worker_pool.register_spawn()
        worker = threading.Thread(
            target=self._run_decoder,
            name=f"ingest-decoder-{worker_index}",
            args=(worker_index,),
            daemon=True,
        )
        self._threads.append(worker)
        worker.start()
        LOGGER.info("Spawned ingestion worker", extra={"worker_id": worker_index})

    def _request_worker_retire(self) -> None:
        with self._retire_lock:
            if self._worker_pool.active_workers <= self._worker_pool.min_workers:
                return
            self._retire_requests += 1
        self._safe_put_entry(_WORKER_STOP)
        self._telemetry.worker_scale_downs += 1

    def _run_supervisor(self) -> None:
        try:
            while not self._stop_event.is_set():
                time.sleep(self._config.worker_pool.supervisor_interval_s)
                entry_ratio = self._entry_queue.depth_ratio
                output_ratio = self._output_queue.depth_ratio
                self._telemetry.record_sample(
                    QueuePressureSample(
                        timestamp_s=time.time(),
                        entry_depth_ratio=entry_ratio,
                        output_depth_ratio=output_ratio,
                        worker_count=self._worker_pool.active_workers,
                    )
                )
                self._tune_queue(self._entry_queue, self._config.entry_queue_tuning)
                self._tune_queue(self._output_queue, self._config.output_queue_tuning)
                self._tune_workers(entry_ratio)
                if self._producer_done.is_set() and self._entry_queue.size == 0:
                    break
        except Exception as exc:  # pragma: no cover - supervisor failure
            self._error = exc
            LOGGER.exception("Ingestion supervisor failed")
        finally:
            LOGGER.info("Ingestion supervisor exiting")

    def _tune_queue(self, queue: AdaptiveBoundedQueue[object], tuning: QueueTuningConfig) -> None:
        depth_ratio = queue.depth_ratio
        if depth_ratio >= tuning.scale_up_ratio and queue.capacity < tuning.max_capacity:
            new_capacity = min(queue.capacity + tuning.scale_step, tuning.max_capacity)
            queue.resize(new_capacity)
            self._telemetry.queue_scale_ups += 1
        elif depth_ratio <= tuning.scale_down_ratio and queue.capacity > tuning.min_capacity:
            new_capacity = max(queue.capacity - tuning.scale_step, tuning.min_capacity)
            queue.resize(new_capacity)
            self._telemetry.queue_scale_downs += 1

    def _tune_workers(self, entry_ratio: float) -> None:
        pool_config = self._config.worker_pool
        if entry_ratio >= pool_config.scale_up_ratio and self._worker_pool.active_workers < pool_config.max_workers:
            self._spawn_worker()
            self._telemetry.worker_scale_ups += 1
            return
        if entry_ratio <= pool_config.scale_down_ratio:
            self._request_worker_retire()

    def _run_producer(self) -> None:
        try:
            for seq_id, (index, timestamp, path) in enumerate(self._entries):
                if self._stop_event.is_set():
                    break
                if self._config.max_frames is not None and seq_id >= self._config.max_frames:
                    break
                entry = FrameSourceEntry(
                    seq_id=seq_id,
                    index=int(index),
                    timestamp=float(timestamp) if timestamp is not None else float(index),
                    path=path,
                )
                if not self._safe_put_entry(entry):
                    self._stats.dropped_entries += 1
                    self._record_failure(seq_id, "entry_queue_timeout", "entry_queue_timeout", path)
                    self._emit_drop_marker(seq_id, "entry_queue_timeout")
                    continue
                self._stats.enqueued_entries += 1
                self._stats.max_entry_queue_depth = max(
                    self._stats.max_entry_queue_depth, self._entry_queue.size
                )
        except Exception as exc:  # pragma: no cover - background failure
            self._error = exc
            LOGGER.exception("Ingestion producer failed")
        finally:
            self._producer_done.set()
            self._entry_queue.close()

    def _run_decoder(self, worker_id: int) -> None:
        try:
            while True:
                try:
                    entry = self._entry_queue.get(timeout_s=self._config.read_timeout_s)
                except BackpressureTimeout:
                    if self._stop_event.is_set() or (self._producer_done.is_set() and self._entry_queue.size == 0):
                        break
                    continue
                except IngestionClosed:
                    break
                if entry is _WORKER_STOP:
                    break
                frame_entry = entry
                start = time.perf_counter()
                frame = self._read_fn(str(frame_entry.path), self._config.decode_backend)
                self._stats.total_decode_s += time.perf_counter() - start
                if frame is None:
                    self._stats.read_failures += 1
                    error = f"Failed to read frame: {frame_entry.path}"
                    self._record_failure(frame_entry.seq_id, "read", error, frame_entry.path)
                    if self._config.fail_fast:
                        raise RuntimeError(error)
                    self._stats.dropped_frames += 1
                    self._emit_drop_marker(frame_entry.seq_id, error)
                    continue
                packet = FramePacket(
                    index=frame_entry.index,
                    timestamp=frame_entry.timestamp,
                    frame=frame,
                    path=frame_entry.path,
                )
                if not self._safe_put_output(DecodedFrame(frame_entry.seq_id, packet, None)):
                    self._stats.output_backpressure += 1
                    error = "output_queue_timeout"
                    self._record_failure(frame_entry.seq_id, "output", error, frame_entry.path)
                    if self._config.fail_fast:
                        raise BackpressureTimeout("Output queue backpressure")
                    self._stats.dropped_frames += 1
                    self._emit_drop_marker(frame_entry.seq_id, error)
                    continue
                self._stats.decoded_frames += 1
                self._stats.max_output_queue_depth = max(
                    self._stats.max_output_queue_depth, self._output_queue.size
                )
        except Exception as exc:  # pragma: no cover - background failure
            self._error = exc
            LOGGER.exception("Ingestion decoder failed", extra={"worker_id": worker_id})
        finally:
            self._worker_pool.register_exit()
            self._safe_put_output(_OUTPUT_SENTINEL)

    def _safe_put_entry(self, item: FrameSourceEntry | object) -> bool:
        try:
            self._entry_queue.put(item, timeout_s=self._config.read_timeout_s)
            return True
        except (BackpressureTimeout, IngestionClosed):
            return False

    def _safe_put_output(self, item: DecodedFrame | object) -> bool:
        try:
            self._output_queue.put(item, timeout_s=self._config.decode_timeout_s)
            return True
        except (BackpressureTimeout, IngestionClosed):
            return False

    def _emit_drop_marker(self, seq_id: int, error: str) -> None:
        if not self._safe_put_output(DecodedFrame(seq_id, None, error)):
            self._stats.output_backpressure += 1
            if self._config.fail_fast:
                self._error = BackpressureTimeout("Output queue backpressure during drop handling")
                self._stop_event.set()

    def _record_failure(self, seq_id: int, stage: str, error: str, path: Path | None = None) -> None:
        event = IngestionFailureEvent(
            seq_id=seq_id,
            stage=stage,
            error=error,
            path=path,
        )
        self._failures.record(event)
        LOGGER.warning(
            "Ingestion failure",
            extra={"seq_id": seq_id, "stage": stage, "error": error, "path": str(path) if path else None},
        )
