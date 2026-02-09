"""Benchmark relocalization regression metric summarization."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from relocalization_metrics import RelocalizationFrame, summarize_relocalized_frames


def _make_frames(count: int, seed: int = 17) -> list[RelocalizationFrame]:
    rng = np.random.default_rng(seed)
    frames: list[RelocalizationFrame] = []
    for idx in range(count):
        match_count = int(rng.integers(40, 160))
        inliers = int(rng.integers(10, match_count + 1))
        inlier_ratio = float(inliers) / float(match_count) if match_count > 0 else 0.0
        frames.append(
            RelocalizationFrame(
                frame_id=idx,
                match_count=match_count,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
            )
        )
    return frames


def run_benchmark(frame_count: int = 10000) -> None:
    frames = _make_frames(frame_count)
    loss_frame_id = frame_count // 3

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    summary = summarize_relocalized_frames(frames, loss_frame_id=loss_frame_id)
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("relocalization_regression_benchmark")
    print(f"frames={frame_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"recovery_success={summary['recovery_success']}")


if __name__ == "__main__":
    run_benchmark()
