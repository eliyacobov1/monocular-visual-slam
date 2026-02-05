"""Asynchronous, bounded ingestion pipeline with adaptive control plane."""

from __future__ import annotations

import concurrent.futures
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Optional, TypeVar

import cv2
import numpy as np

from frame_stream import FramePacket
from ingestion_control_plane import (
    AdaptiveBoundedQueue,
    BackpressureTimeout,
    CircuitBreaker,
    CircuitBreakerConfig,
    ControlPlaneSupervisor,
    DeterministicReorderBuffer,
    DynamicWorkerPool,
    IngestionClosed,
    IngestionFailureEvent,
    IngestionFailureReport,
    IngestionTelemetry,
    LatencySample,
    OrderingBufferConfig,
    QueueTuningConfig,
    RetryPolicyConfig,
    WorkerPoolConfig,
)

LOGGER = logging.getLogger(__name__)

_OUTPUT_SENTINEL = object()
_WORKER_STOP = object()

T = TypeVar("T")


@dataclass(frozen=True)
class FrameSourceEntry:
    """Frame metadata input to the ingestion pipeline."""

    seq_id: int
    index: int
    timestamp: float
    path: Path
    enqueued_at_s: float


@dataclass(frozen=True)
class DecodedFrame:
    """Decoded frame payload with sequence ordering."""

    seq_id: int
    packet: FramePacket | None
    error: str | None
    decode_s: float
    queue_wait_s: float


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
    decode_executor: Literal["thread", "process"] = "thread"
    inflight_limit: int = 8
    entry_queue_tuning: QueueTuningConfig = field(default_factory=QueueTuningConfig)
    output_queue_tuning: QueueTuningConfig = field(default_factory=QueueTuningConfig)
    worker_pool: WorkerPoolConfig = field(default_factory=WorkerPoolConfig)
    telemetry_window: int = 256
    retry_policy: RetryPolicyConfig = field(default_factory=RetryPolicyConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    ordering_buffer: OrderingBufferConfig = field(default_factory=OrderingBufferConfig)
    supervisor_ema_alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.entry_queue_capacity <= 0 or self.output_queue_capacity <= 0:
            raise ValueError("Queue capacities must be positive")
        if self.num_decode_workers <= 0:
            raise ValueError("num_decode_workers must be positive")
        if self.inflight_limit <= 0:
            raise ValueError("inflight_limit must be positive")
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
        if self.decode_executor == "process" and self.num_decode_workers < 1:
            raise ValueError("num_decode_workers must be >= 1 for process executor")
        if not 0 < self.supervisor_ema_alpha <= 1:
            raise ValueError("supervisor_ema_alpha must be in (0, 1]")


@dataclass
class IngestionPipelineStats:
    """Operational metrics for the ingestion pipeline."""

    enqueued_entries: int = 0
    dequeued_entries: int = 0
    dropped_entries: int = 0
    decoded_frames: int = 0
    dropped_frames: int = 0
    read_failures: int = 0
    decode_exceptions: int = 0
    decode_retries: int = 0
    output_backpressure: int = 0
    ordering_forced_flushes: int = 0
    max_entry_queue_depth: int = 0
    max_output_queue_depth: int = 0
    total_decode_s: float = 0.0
    total_queue_wait_s: float = 0.0
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
        self._breaker = CircuitBreaker(self._config.circuit_breaker)
        self._ordering_buffer = DeterministicReorderBuffer(self._config.ordering_buffer)
        self._inflight_lock = threading.Lock()
        self._inflight_tasks = 0
        self._executor: concurrent.futures.Executor | None = None
        self._futures: dict[concurrent.futures.Future[np.ndarray | None], FrameSourceEntry] = {}
        self._futures_lock = threading.Lock()
        self._supervisor = ControlPlaneSupervisor(
            entry_queue=self._entry_queue,
            output_queue=self._output_queue,
            worker_pool=self._worker_pool,
            entry_tuning=self._config.entry_queue_tuning,
            output_tuning=self._config.output_queue_tuning,
            worker_config=self._config.worker_pool,
            telemetry=self._telemetry,
            spawn_worker=self._spawn_worker,
            retire_worker=self._request_worker_retire,
            inflight_count=self._inflight_count,
            stop_event=self._stop_event,
            producer_done=self._producer_done,
            ema_alpha=self._config.supervisor_ema_alpha,
        )

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
        if self._config.decode_executor == "process":
            self._start_process_executor()
        else:
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
        if self._config.decode_executor == "process":
            self._stop_process_executor()
        else:
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
        while True:
            if self._producer_done.is_set() and self._is_decode_stage_done(completed_workers):
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
            self._ordering_buffer.push(decoded)
            ready, expected_seq = self._ordering_buffer.pop_ready(expected_seq)
            if self._ordering_buffer.forced_flushes:
                self._stats.ordering_forced_flushes = self._ordering_buffer.forced_flushes
                self._telemetry.forced_flushes = self._ordering_buffer.forced_flushes
            for ready_frame in ready:
                if ready_frame.packet is None:
                    LOGGER.warning(
                        "Dropped frame during ingestion",
                        extra={"seq_id": ready_frame.seq_id, "error": ready_frame.error},
                    )
                else:
                    self._stats.dequeued_entries += 1
                    yield ready_frame.packet
        if self._error is not None:
            raise self._error
        self._stats.mark_finish()

    def _start_process_executor(self) -> None:
        if self._config.decode_executor != "process":
            return
        if self._config.num_decode_workers < 1:
            raise ValueError("num_decode_workers must be >= 1 for process executor")
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self._config.num_decode_workers
        )
        dispatcher = threading.Thread(
            target=self._run_dispatcher, name="ingest-dispatcher", daemon=True
        )
        collector = threading.Thread(
            target=self._run_collector, name="ingest-collector", daemon=True
        )
        self._threads.extend([dispatcher, collector])
        dispatcher.start()
        collector.start()

    def _stop_process_executor(self) -> None:
        for _ in range(self._config.num_decode_workers):
            self._safe_put_entry(_WORKER_STOP)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _is_decode_stage_done(self, completed_workers: int) -> bool:
        if self._config.decode_executor == "process":
            with self._futures_lock:
                inflight_empty = not self._futures
            return inflight_empty and self._producer_done.is_set() and self._entry_queue.size == 0
        return completed_workers >= self._worker_pool.total_spawned and self._ordering_buffer.size == 0

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

    def _run_supervisor(self) -> None:
        try:
            self._supervisor.run()
        except Exception as exc:  # pragma: no cover - supervisor failure
            self._error = exc
            LOGGER.exception("Ingestion supervisor failed")

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
                    enqueued_at_s=time.perf_counter(),
                )
                if not self._safe_put_entry(entry):
                    self._stats.dropped_entries += 1
                    self._record_failure(seq_id, "entry_queue_timeout", "entry_queue_timeout", path)
                    self._emit_drop_marker(seq_id, "entry_queue_timeout", 0.0, 0.0)
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
                if not self._breaker.allow():
                    error = "circuit_breaker_open"
                    self._stats.dropped_frames += 1
                    self._telemetry.circuit_breaker_opens += 1
                    self._record_failure(frame_entry.seq_id, "circuit_breaker", error, frame_entry.path)
                    self._emit_drop_marker(frame_entry.seq_id, error, 0.0, 0.0)
                    if self._config.fail_fast:
                        raise RuntimeError("Circuit breaker open")
                    continue
                decode_start = time.perf_counter()
                frame = self._decode_with_retries(frame_entry)
                decode_s = time.perf_counter() - decode_start
                self._stats.total_decode_s += decode_s
                queue_wait_s = max(0.0, decode_start - frame_entry.enqueued_at_s)
                self._stats.total_queue_wait_s += queue_wait_s
                if frame is None:
                    self._stats.read_failures += 1
                    error = f"Failed to read frame: {frame_entry.path}"
                    breaker_opened = self._breaker.record_failure()
                    if breaker_opened:
                        self._telemetry.circuit_breaker_opens += 1
                    self._record_failure(frame_entry.seq_id, "read", error, frame_entry.path)
                    if self._config.fail_fast:
                        raise RuntimeError(error)
                    self._stats.dropped_frames += 1
                    self._emit_drop_marker(frame_entry.seq_id, error, decode_s, queue_wait_s)
                    continue
                self._breaker.record_success()
                packet = FramePacket(
                    index=frame_entry.index,
                    timestamp=frame_entry.timestamp,
                    frame=frame,
                    path=frame_entry.path,
                )
                if not self._safe_put_output(DecodedFrame(frame_entry.seq_id, packet, None, decode_s, queue_wait_s)):
                    self._stats.output_backpressure += 1
                    error = "output_queue_timeout"
                    self._record_failure(frame_entry.seq_id, "output", error, frame_entry.path)
                    if self._config.fail_fast:
                        raise BackpressureTimeout("Output queue backpressure")
                    self._stats.dropped_frames += 1
                    self._emit_drop_marker(frame_entry.seq_id, error, decode_s, queue_wait_s)
                    continue
                self._stats.decoded_frames += 1
                self._telemetry.record_latency(
                    LatencySample(timestamp_s=time.time(), decode_s=decode_s, queue_wait_s=queue_wait_s)
                )
                self._stats.max_output_queue_depth = max(
                    self._stats.max_output_queue_depth, self._output_queue.size
                )
        except Exception as exc:  # pragma: no cover - background failure
            self._error = exc
            LOGGER.exception("Ingestion decoder failed", extra={"worker_id": worker_id})
        finally:
            self._worker_pool.register_exit()
            self._safe_put_output(_OUTPUT_SENTINEL)

    def _run_dispatcher(self) -> None:
        if self._executor is None:
            return
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
                if not self._breaker.allow():
                    error = "circuit_breaker_open"
                    self._stats.dropped_frames += 1
                    self._telemetry.circuit_breaker_opens += 1
                    self._record_failure(entry.seq_id, "circuit_breaker", error, entry.path)
                    self._emit_drop_marker(entry.seq_id, error, 0.0, 0.0)
                    if self._config.fail_fast:
                        raise RuntimeError("Circuit breaker open")
                    continue
                self._wait_for_inflight_slot()
                with self._futures_lock:
                    future = self._executor.submit(self._read_fn, str(entry.path), self._config.decode_backend)
                    self._futures[future] = entry
                    self._increment_inflight(1)
        except Exception as exc:  # pragma: no cover - dispatcher failure
            self._error = exc
            LOGGER.exception("Ingestion dispatcher failed")
        finally:
            self._safe_put_output(_OUTPUT_SENTINEL)

    def _run_collector(self) -> None:
        try:
            while True:
                if self._producer_done.is_set() and self._entry_queue.size == 0:
                    with self._futures_lock:
                        if not self._futures:
                            break
                done = self._poll_futures()
                if not done:
                    time.sleep(0.01)
        except Exception as exc:  # pragma: no cover - collector failure
            self._error = exc
            LOGGER.exception("Ingestion collector failed")
        finally:
            self._safe_put_output(_OUTPUT_SENTINEL)

    def _poll_futures(self) -> list[concurrent.futures.Future[np.ndarray | None]]:
        with self._futures_lock:
            futures = list(self._futures.keys())
        if not futures:
            return []
        done, _ = concurrent.futures.wait(
            futures,
            timeout=0.05,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            with self._futures_lock:
                entry = self._futures.pop(future, None)
            self._increment_inflight(-1)
            if entry is None:
                continue
            decode_start = time.perf_counter()
            try:
                frame = future.result()
            except Exception as exc:  # pragma: no cover - decode failure
                self._stats.decode_exceptions += 1
                error = f"decode_exception: {exc}"
                breaker_opened = self._breaker.record_failure()
                if breaker_opened:
                    self._telemetry.circuit_breaker_opens += 1
                self._record_failure(entry.seq_id, "decode_exception", error, entry.path)
                if self._config.fail_fast:
                    self._error = exc
                    self._stop_event.set()
                    continue
                self._stats.dropped_frames += 1
                self._emit_drop_marker(entry.seq_id, error, 0.0, 0.0)
                continue
            decode_s = time.perf_counter() - decode_start
            self._stats.total_decode_s += decode_s
            queue_wait_s = max(0.0, decode_start - entry.enqueued_at_s)
            self._stats.total_queue_wait_s += queue_wait_s
            if frame is None:
                self._stats.read_failures += 1
                error = f"Failed to read frame: {entry.path}"
                breaker_opened = self._breaker.record_failure()
                if breaker_opened:
                    self._telemetry.circuit_breaker_opens += 1
                self._record_failure(entry.seq_id, "read", error, entry.path)
                if self._config.fail_fast:
                    self._error = RuntimeError(error)
                    self._stop_event.set()
                    continue
                self._stats.dropped_frames += 1
                self._emit_drop_marker(entry.seq_id, error, decode_s, queue_wait_s)
                continue
            self._breaker.record_success()
            packet = FramePacket(
                index=entry.index,
                timestamp=entry.timestamp,
                frame=frame,
                path=entry.path,
            )
            if not self._safe_put_output(DecodedFrame(entry.seq_id, packet, None, decode_s, queue_wait_s)):
                self._stats.output_backpressure += 1
                error = "output_queue_timeout"
                self._record_failure(entry.seq_id, "output", error, entry.path)
                if self._config.fail_fast:
                    self._error = BackpressureTimeout("Output queue backpressure")
                    self._stop_event.set()
                    continue
                self._stats.dropped_frames += 1
                self._emit_drop_marker(entry.seq_id, error, decode_s, queue_wait_s)
                continue
            self._stats.decoded_frames += 1
            self._telemetry.record_latency(
                LatencySample(timestamp_s=time.time(), decode_s=decode_s, queue_wait_s=queue_wait_s)
            )
            self._stats.max_output_queue_depth = max(
                self._stats.max_output_queue_depth, self._output_queue.size
            )
        return list(done)

    def _decode_with_retries(self, entry: FrameSourceEntry) -> np.ndarray | None:
        attempt = 0
        policy = self._config.retry_policy
        while attempt < policy.max_attempts:
            frame = self._read_fn(str(entry.path), self._config.decode_backend)
            if frame is not None:
                return frame
            attempt += 1
            if attempt < policy.max_attempts:
                self._stats.decode_retries += 1
                sleep_time = policy.backoff_s + random.random() * policy.jitter_s
                time.sleep(sleep_time)
        return None

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

    def _emit_drop_marker(self, seq_id: int, error: str, decode_s: float, queue_wait_s: float) -> None:
        if not self._safe_put_output(DecodedFrame(seq_id, None, error, decode_s, queue_wait_s)):
            self._stats.output_backpressure += 1
            if self._config.fail_fast:
                self._error = BackpressureTimeout("Output queue backpressure during drop handling")
                self._stop_event.set()

    def _record_failure(self, seq_id: int, stage: str, error: str, path: Path | None = None) -> None:
        event = IngestionFailureEvent(
            seq_id=seq_id,
            stage=stage,
            error=error,
            path=str(path) if path else None,
        )
        self._failures.record(event)
        LOGGER.warning(
            "Ingestion failure",
            extra={"seq_id": seq_id, "stage": stage, "error": error, "path": str(path) if path else None},
        )

    def _wait_for_inflight_slot(self) -> None:
        while True:
            with self._inflight_lock:
                if self._inflight_tasks < self._config.inflight_limit:
                    return
            time.sleep(0.001)

    def _increment_inflight(self, delta: int) -> None:
        with self._inflight_lock:
            self._inflight_tasks += delta

    def _inflight_count(self) -> int:
        with self._inflight_lock:
            return self._inflight_tasks


__all__ = [
    "AsyncIngestionPipeline",
    "IngestionPipelineConfig",
    "IngestionPipelineStats",
    "FrameSourceEntry",
    "DecodedFrame",
    "QueueTuningConfig",
    "WorkerPoolConfig",
    "RetryPolicyConfig",
    "CircuitBreakerConfig",
    "OrderingBufferConfig",
    "BackpressureTimeout",
    "IngestionClosed",
]
