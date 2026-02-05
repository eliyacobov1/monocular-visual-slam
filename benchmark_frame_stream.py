#!/usr/bin/env python3
"""Benchmark the streaming frame loader for throughput and memory usage."""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

import numpy as np

from frame_stream import FrameStream, FrameStreamConfig

LOGGER = logging.getLogger(__name__)


def _synthetic_entries(count: int) -> list[tuple[int, float, Path]]:
    return [(idx, float(idx) * 0.1, Path(f"frame_{idx:06d}.png")) for idx in range(count)]


def _iter_image_entries(image_dir: Path) -> list[tuple[int, float, Path]]:
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG images found under {image_dir}")
    return [(idx, float(idx) * 0.1, path) for idx, path in enumerate(images)]


def _synthetic_reader(frame_shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(frame_shape, dtype=np.uint8)


def _run_stream(
    entries: Iterable[tuple[int, float, Path]],
    *,
    queue_capacity: int,
    frame_shape: tuple[int, int, int],
) -> dict[str, float]:
    frame = _synthetic_reader(frame_shape)

    def read_fn(_path: str, _flag: int) -> np.ndarray | None:
        return frame

    stream = FrameStream(
        entries,
        config=FrameStreamConfig(queue_capacity=queue_capacity),
        read_fn=read_fn,
    )
    start = time.perf_counter()
    count = 0
    for _packet in stream:
        count += 1
    duration_s = time.perf_counter() - start
    stats = stream.stats
    return {
        "frames": float(count),
        "duration_s": duration_s,
        "fps": float(count) / duration_s if duration_s > 0 else 0.0,
        "queue_capacity": float(queue_capacity),
        "max_depth": float(stats.max_depth),
        "read_failures": float(stats.read_failures),
        "dropped": float(stats.dropped),
        "total_read_s": float(stats.total_read_s),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark frame streaming")
    parser.add_argument("--frames", type=int, default=500, help="Synthetic frame count")
    parser.add_argument("--queue_capacity", type=int, default=8, help="Stream queue capacity")
    parser.add_argument("--frame_width", type=int, default=640, help="Synthetic frame width")
    parser.add_argument("--frame_height", type=int, default=480, help="Synthetic frame height")
    parser.add_argument("--image_dir", type=Path, help="Optional PNG directory for real IO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.image_dir:
        entries = _iter_image_entries(args.image_dir)
    else:
        entries = _synthetic_entries(args.frames)

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    metrics = _run_stream(
        entries,
        queue_capacity=args.queue_capacity,
        frame_shape=(args.frame_height, args.frame_width, 3),
    )
    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(start_snapshot, "lineno")
    memory_delta = sum(stat.size_diff for stat in stats)
    tracemalloc.stop()

    payload = {**metrics, "memory_delta_bytes": float(memory_delta)}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
