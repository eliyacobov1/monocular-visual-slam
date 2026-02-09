"""Benchmark the loop-closure validation suite throughput and memory."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from loop_closure_validation import (
    LoopClosureVerificationDataset,
    LoopClosureVerificationSample,
    LoopClosureVerificationThresholds,
)


def _make_dataset(sample_count: int, seed: int = 13) -> LoopClosureVerificationDataset:
    rng = np.random.default_rng(seed)
    samples: list[LoopClosureVerificationSample] = []
    for idx in range(sample_count):
        match_count = int(rng.integers(20, 120))
        inlier_count = int(rng.integers(0, match_count + 1))
        reprojection_error = float(rng.uniform(0.5, 4.0))
        rotation_error = float(rng.uniform(0.0, 12.0))
        translation_error = float(rng.uniform(0.0, 1.0))
        timestamp_base = float(idx) * 0.5
        temporal_shift = float(rng.uniform(0.2, 5.0))
        expected_match = bool(rng.integers(0, 2))
        samples.append(
            LoopClosureVerificationSample(
                sample_id=f"sample-{idx:05d}",
                query_frame_id=idx,
                candidate_frame_id=idx - int(rng.integers(1, 50)),
                query_timestamp_s=timestamp_base,
                candidate_timestamp_s=timestamp_base - temporal_shift,
                match_count=match_count,
                inlier_count=inlier_count,
                mean_reprojection_error=reprojection_error,
                rotation_error_deg=rotation_error,
                translation_error=translation_error,
                expected_match=expected_match,
            )
        )
    return LoopClosureVerificationDataset(name="synthetic_loop_closure_benchmark", samples=tuple(samples))


def run_benchmark(sample_count: int = 5000) -> None:
    dataset = _make_dataset(sample_count)
    thresholds = LoopClosureVerificationThresholds()

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    report = dataset.evaluate(thresholds)
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("loop_closure_validation_benchmark")
    print(f"samples={sample_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"report_digest={report.report_digest}")


if __name__ == "__main__":
    run_benchmark()
