"""Asynchronous, bounded ingestion pipeline for frame decoding."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np

from frame_stream import FramePacket

LOGGER = logging.getLogger(__name__)

_ENTRY_SENTINEL = object()
_OUTPUT_SENTINEL = object()


class IngestionClosed(RuntimeError):
    """Raised when consuming from a closed ingestion pipeline."""


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


class AsyncIngestionPipeline(Iterable[FramePacket]):
    """Multi-stage bounded ingestion pipeline with decode workers."""

    def __init__(
        self,
        entries: Iterable[tuple[int, float | None, Path]],
        *,
        config: IngestionPipelineConfig | None = None,
        read_fn: Callable[[str, int], Optional[np.ndarray]] | None = None,
    ) -> None:
        self._entries = entries
        self._config = config or IngestionPipelineConfig()
        if self._config.entry_queue_capacity <= 0 or self._config.output_queue_capacity <= 0:
            raise ValueError("Queue capacities must be positive")
        if self._config.num_decode_workers <= 0:
            raise ValueError("num_decode_workers must be positive")
        self._read_fn = read_fn or cv2.imread
        self._entry_queue: Queue[FrameSourceEntry | object] = Queue(
            maxsize=self._config.entry_queue_capacity
        )
        self._output_queue: Queue[DecodedFrame | object] = Queue(
            maxsize=self._config.output_queue_capacity
        )
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._stats = IngestionPipelineStats()
        self._started = False

    @property
    def stats(self) -> IngestionPipelineStats:
        return self._stats

    def start(self) -> None:
        if self._started:
            return
        self._stats.mark_start()
        producer = threading.Thread(target=self._run_producer, name="ingest-producer", daemon=True)
        self._threads.append(producer)
        producer.start()
        for worker_id in range(self._config.num_decode_workers):
            worker = threading.Thread(
                target=self._run_decoder,
                name=f"ingest-decoder-{worker_id}",
                args=(worker_id,),
                daemon=True,
            )
            self._threads.append(worker)
            worker.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        for _ in range(self._config.num_decode_workers):
            self._safe_put_entry(_ENTRY_SENTINEL)
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
        done_workers = 0
        pending: list[tuple[int, DecodedFrame]] = []
        while True:
            if done_workers >= self._config.num_decode_workers and not pending:
                break
            try:
                item = self._output_queue.get(timeout=self._config.decode_timeout_s)
            except Empty:
                if self._error is not None:
                    raise self._error
                if self._stop_event.is_set() and done_workers >= self._config.num_decode_workers:
                    break
                continue
            if item is _OUTPUT_SENTINEL:
                done_workers += 1
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
                    self._emit_drop_marker(seq_id, "entry_queue_timeout")
                    continue
                self._stats.enqueued_entries += 1
                self._stats.max_entry_queue_depth = max(
                    self._stats.max_entry_queue_depth, self._entry_queue.qsize()
                )
        except Exception as exc:  # pragma: no cover - background failure
            self._error = exc
            LOGGER.exception("Ingestion producer failed")
        finally:
            for _ in range(self._config.num_decode_workers):
                self._safe_put_entry(_ENTRY_SENTINEL)

    def _run_decoder(self, worker_id: int) -> None:
        try:
            while True:
                try:
                    entry = self._entry_queue.get(timeout=self._config.read_timeout_s)
                except Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                if entry is _ENTRY_SENTINEL:
                    break
                frame_entry = entry
                start = time.perf_counter()
                frame = self._read_fn(str(frame_entry.path), self._config.decode_backend)
                self._stats.total_decode_s += time.perf_counter() - start
                if frame is None:
                    self._stats.read_failures += 1
                    error = f"Failed to read frame: {frame_entry.path}"
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
                    if self._config.fail_fast:
                        raise TimeoutError("Output queue backpressure")
                    self._stats.dropped_frames += 1
                    self._emit_drop_marker(frame_entry.seq_id, "output_queue_timeout")
                    continue
                self._stats.decoded_frames += 1
                self._stats.max_output_queue_depth = max(
                    self._stats.max_output_queue_depth, self._output_queue.qsize()
                )
        except Exception as exc:  # pragma: no cover - background failure
            self._error = exc
            LOGGER.exception("Ingestion decoder failed", extra={"worker_id": worker_id})
        finally:
            self._safe_put_output(_OUTPUT_SENTINEL)

    def _safe_put_entry(self, item: FrameSourceEntry | object) -> bool:
        try:
            self._entry_queue.put(item, timeout=self._config.read_timeout_s)
            return True
        except Full:
            return False

    def _safe_put_output(self, item: DecodedFrame | object) -> bool:
        try:
            self._output_queue.put(item, timeout=self._config.decode_timeout_s)
            return True
        except Full:
            return False

    def _emit_drop_marker(self, seq_id: int, error: str) -> None:
        if not self._safe_put_output(DecodedFrame(seq_id, None, error)):
            self._stats.output_backpressure += 1
            if self._config.fail_fast:
                self._error = TimeoutError("Output queue backpressure during drop handling")
                self._stop_event.set()
