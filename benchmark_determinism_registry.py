"""Benchmark deterministic registry seed derivation throughput."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

from deterministic_registry import build_registry


def run_benchmark(iterations: int = 200_000, component_count: int = 8) -> None:
    config_path = Path("configs/pipeline/kitti_default.json")
    if not config_path.exists():
        raise FileNotFoundError("Pipeline config not found for benchmark")

    registry = build_registry(seed=42, config_path=config_path)
    components = [f"component_{idx}" for idx in range(component_count)]

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    checksum = 0
    for idx in range(iterations):
        checksum ^= registry.seed_for(components[idx % component_count])
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("determinism_registry_benchmark")
    print(f"iterations={iterations}")
    print(f"components={component_count}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    run_benchmark()
