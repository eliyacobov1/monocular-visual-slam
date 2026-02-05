"""Streaming frame loader with bounded buffering for SLAM pipelines."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Generic, Iterable, Iterator, Optional, TypeVar

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class StreamClosed(RuntimeError):
    """Raised when attempting to interact with a closed stream."""


@dataclass(frozen=True)
class FramePacket:
    """A single frame payload delivered by the frame stream."""

    index: int
    timestamp: float
    frame: np.ndarray
    path: Path


@dataclass
class FrameStreamStats:
    """Operational statistics for a streaming frame loader."""

    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    read_failures: int = 0
    max_depth: int = 0
    total_read_s: float = 0.0
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


class BoundedRingBuffer(Generic[T]):
    """Thread-safe bounded ring buffer with backpressure."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: Deque[T] = deque()
        self._condition = threading.Condition()
        self._closed = False

    def put(self, item: T, timeout_s: float | None = None) -> None:
        with self._condition:
            start = time.perf_counter()
            while len(self._items) >= self._capacity and not self._closed:
                remaining = None if timeout_s is None else timeout_s - (time.perf_counter() - start)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for buffer space")
                self._condition.wait(timeout=remaining)
            if self._closed:
                raise StreamClosed("Buffer is closed")
            self._items.append(item)
            self._condition.notify_all()

    def get(self, timeout_s: float | None = None) -> T:
        with self._condition:
            start = time.perf_counter()
            while not self._items and not self._closed:
                remaining = None if timeout_s is None else timeout_s - (time.perf_counter() - start)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for buffer data")
                self._condition.wait(timeout=remaining)
            if not self._items:
                raise StreamClosed("Buffer is closed")
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._items)


@dataclass(frozen=True)
class FrameStreamConfig:
    """Configuration for the streaming frame loader."""

    queue_capacity: int = 8
    read_timeout_s: float | None = None
    decode_backend: int = cv2.IMREAD_COLOR


class FrameStream(Iterable[FramePacket]):
    """Background-loading frame stream with bounded buffering."""

    def __init__(
        self,
        entries: Iterable[tuple[int, float | None, Path]],
        *,
        config: FrameStreamConfig | None = None,
        read_fn: Callable[[str, int], Optional[np.ndarray]] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self._entries = entries
        self._config = config or FrameStreamConfig()
        self._read_fn = read_fn or cv2.imread
        self._max_frames = max_frames
        self._buffer: BoundedRingBuffer[FramePacket] = BoundedRingBuffer(self._config.queue_capacity)
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._stopped = threading.Event()
        self._stats = FrameStreamStats()

    @property
    def stats(self) -> FrameStreamStats:
        return self._stats

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stats.mark_start()
        self._thread = threading.Thread(target=self._run_loader, name="frame-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._buffer.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __iter__(self) -> Iterator[FramePacket]:
        self.start()
        while True:
            try:
                packet = self._buffer.get(timeout_s=self._config.read_timeout_s)
            except StreamClosed:
                if self._error is not None:
                    raise self._error
                break
            except TimeoutError:
                LOGGER.warning("Frame stream read timed out")
                continue
            self._stats.dequeued += 1
            yield packet

    def _run_loader(self) -> None:
        try:
            for count, (index, timestamp, path) in enumerate(self._entries):
                if self._stopped.is_set():
                    break
                if self._max_frames is not None and count >= self._max_frames:
                    break
                start = time.perf_counter()
                frame = self._read_fn(str(path), self._config.decode_backend)
                self._stats.total_read_s += time.perf_counter() - start
                if frame is None:
                    self._stats.read_failures += 1
                    raise RuntimeError(f"Failed to read frame: {path}")
                packet = FramePacket(
                    index=int(index),
                    timestamp=float(timestamp) if timestamp is not None else float(index),
                    frame=frame,
                    path=path,
                )
                try:
                    self._buffer.put(packet, timeout_s=self._config.read_timeout_s)
                except TimeoutError:
                    self._stats.dropped += 1
                    LOGGER.warning(
                        "Dropping frame due to backpressure",
                        extra={"frame_index": index, "path": str(path)},
                    )
                    continue
                self._stats.enqueued += 1
                self._stats.max_depth = max(self._stats.max_depth, self._buffer.size)
        except Exception as exc:  # pragma: no cover - background failure propagation
            self._error = exc
            LOGGER.exception("Frame stream loader failed")
        finally:
            self._stats.mark_finish()
            self._buffer.close()

