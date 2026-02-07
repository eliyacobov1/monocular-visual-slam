#!/usr/bin/env python3
"""Benchmark the deterministic feature control plane."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import asdict

import numpy as np

from feature_control_plane import FeatureControlConfig, FeatureControlPlane
from feature_pipeline import FeaturePipelineConfig


def _make_frames(count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    rng = np.random.default_rng(123)
    return [rng.integers(0, 255, size=shape, dtype=np.uint8) for _ in range(count)]


def run_benchmark(frames: list[np.ndarray], workers: int, executor: str) -> dict[str, object]:
    feature_config = FeaturePipelineConfig(nfeatures=1200, deterministic_seed=42)
    control_config = FeatureControlConfig(
        enabled=True,
        num_workers=workers,
        executor=executor,
        max_inflight=max(workers * 2, 2),
        result_queue_capacity=max(len(frames), 4),
        reorder_buffer_size=max(len(frames), 4),
    )
    plane = FeatureControlPlane(feature_config=feature_config, control_config=control_config)
    tracemalloc.start()
    start = time.perf_counter()
    for idx, frame in enumerate(frames):
        plane.submit(seq_id=idx, timestamp=float(idx) * 0.01, frame=frame)
    results = []
    for _ in frames:
        results.append(plane.collect(timeout_s=5.0))
    duration_s = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plane.close()
    telemetry = plane.telemetry_summary()
    return {
        "frames": len(frames),
        "duration_s": duration_s,
        "throughput_fps": len(frames) / duration_s if duration_s > 0 else 0.0,
        "memory_current_bytes": current,
        "memory_peak_bytes": peak,
        "cache_hits": sum(1 for result in results if result.cache_hit),
        "telemetry": {
            "duration_s": telemetry.duration_s,
            "queue_wait_s": telemetry.queue_wait_s,
        },
        "config": {
            "feature": asdict(feature_config),
            "control": asdict(control_config),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark feature control plane")
    parser.add_argument("--frames", type=int, default=32, help="Number of frames")
    parser.add_argument("--workers", type=int, default=2, help="Worker count")
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="Executor backend",
    )
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    frames = _make_frames(args.frames, (args.height, args.width))
    report = run_benchmark(frames, args.workers, args.executor)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
