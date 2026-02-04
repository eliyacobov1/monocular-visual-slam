#!/usr/bin/env python3
"""Benchmark frame diagnostics summarization for complexity validation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from data_persistence import (
    FrameDiagnosticsEntry,
    build_frame_diagnostics_bundle,
    summarize_frame_diagnostics,
    summarize_frame_diagnostics_streaming,
)


@dataclass(frozen=True)
class BenchmarkResult:
    size: int
    duration_s: float
    mode: str


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


def _write_payload(path: Path, entries: Iterable[FrameDiagnosticsEntry]) -> None:
    payload = {
        "name": "frame_diagnostics",
        "recorded_at": "benchmark",
        "entries": [
            {
                "frame_id": entry.frame_id,
                "timestamp": entry.timestamp,
                "match_count": entry.match_count,
                "inliers": entry.inliers,
                "method": entry.method,
                "inlier_ratio": entry.inlier_ratio,
                "median_parallax": entry.median_parallax,
                "score": entry.score,
                "status": entry.status,
                "failure_reason": entry.failure_reason,
            }
            for entry in entries
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_benchmark(sizes: Iterable[int], streaming: bool) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for size in sizes:
        entries = _generate_entries(size)
        if streaming:
            path = Path(f"diagnostics_benchmark_{size}.json")
            _write_payload(path, entries)
            start = time.perf_counter()
            summarize_frame_diagnostics_streaming(path)
            duration = time.perf_counter() - start
            path.unlink(missing_ok=True)
        else:
            bundle = build_frame_diagnostics_bundle("frame_diagnostics", entries)
            start = time.perf_counter()
            summarize_frame_diagnostics(bundle)
            duration = time.perf_counter() - start
        results.append(BenchmarkResult(size=size, duration_s=duration, mode="streaming" if streaming else "in_memory"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark diagnostics summarization.")
    parser.add_argument("--streaming", action="store_true", help="Use streaming summary")
    args = parser.parse_args()

    sizes = [1_000, 10_000, 50_000, 100_000]
    results = _run_benchmark(sizes, args.streaming)
    print("size,duration_s,mode")
    for result in results:
        print(f"{result.size},{result.duration_s:.6f},{result.mode}")


if __name__ == "__main__":
    main()
