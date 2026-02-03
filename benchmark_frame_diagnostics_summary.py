#!/usr/bin/env python3
"""Benchmark frame diagnostics summarization for complexity validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from data_persistence import FrameDiagnosticsEntry, build_frame_diagnostics_bundle, summarize_frame_diagnostics


@dataclass(frozen=True)
class BenchmarkResult:
    size: int
    duration_s: float


def _generate_entries(size: int) -> tuple[FrameDiagnosticsEntry, ...]:
    rng = np.random.default_rng(42)
    matches = rng.integers(30, 80, size=size)
    inliers = rng.integers(10, 30, size=size)
    ratios = inliers / np.maximum(matches, 1)
    parallaxes = rng.normal(loc=1.0, scale=0.2, size=size)
    scores = ratios * np.maximum(parallaxes, 0.1)
    methods = np.where(rng.random(size=size) > 0.5, "essential", "homography")
    return tuple(
        FrameDiagnosticsEntry(
            frame_id=int(idx),
            timestamp=float(idx) * 0.1,
            match_count=int(matches[idx]),
            inliers=int(inliers[idx]),
            method=str(methods[idx]),
            inlier_ratio=float(ratios[idx]),
            median_parallax=float(parallaxes[idx]),
            score=float(scores[idx]),
            status="ok",
            failure_reason=None,
        )
        for idx in range(size)
    )


def _run_benchmark(sizes: Iterable[int]) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for size in sizes:
        entries = _generate_entries(size)
        bundle = build_frame_diagnostics_bundle("frame_diagnostics", entries)
        start = time.perf_counter()
        summarize_frame_diagnostics(bundle)
        duration = time.perf_counter() - start
        results.append(BenchmarkResult(size=size, duration_s=duration))
    return results


def main() -> None:
    sizes = [1_000, 10_000, 50_000, 100_000]
    results = _run_benchmark(sizes)
    print("size,duration_s")
    for result in results:
        print(f"{result.size},{result.duration_s:.6f}")


if __name__ == "__main__":
    main()
