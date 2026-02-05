"""Tests for the streaming frame loader."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from frame_stream import BoundedRingBuffer, FrameStream, FrameStreamConfig, StreamClosed


def test_bounded_ring_buffer_put_get() -> None:
    buffer = BoundedRingBuffer[int](capacity=2)
    buffer.put(1)
    buffer.put(2)
    assert buffer.get() == 1
    assert buffer.get() == 2


def test_bounded_ring_buffer_close_unblocks() -> None:
    buffer = BoundedRingBuffer[int](capacity=1)
    buffer.close()
    with pytest.raises(StreamClosed):
        buffer.get(timeout_s=0.01)


def test_frame_stream_reads_frames(tmp_path: Path) -> None:
    entries = [(0, 0.0, tmp_path / "frame0.png"), (1, 0.1, tmp_path / "frame1.png")]
    frames = {
        str(entries[0][2]): np.zeros((4, 4, 3), dtype=np.uint8),
        str(entries[1][2]): np.ones((4, 4, 3), dtype=np.uint8),
    }

    def read_fn(path: str, _flag: int) -> np.ndarray | None:
        return frames.get(path)

    stream = FrameStream(entries, config=FrameStreamConfig(queue_capacity=1), read_fn=read_fn)
    packets = list(stream)

    assert len(packets) == 2
    assert packets[0].timestamp == 0.0
    assert packets[1].timestamp == 0.1
    assert stream.stats.enqueued == 2
    assert stream.stats.dequeued == 2
    assert stream.stats.read_failures == 0


def test_frame_stream_propagates_read_failure(tmp_path: Path) -> None:
    entries = [(0, 0.0, tmp_path / "frame0.png")]

    def read_fn(_path: str, _flag: int) -> np.ndarray | None:
        return None

    stream = FrameStream(entries, read_fn=read_fn)
    with pytest.raises(RuntimeError, match="Failed to read frame"):
        list(stream)
