"""Benchmark for block-sparse Gauss-Newton solver performance."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from pose_graph import PoseGraph, RobustLossConfig, SolverConfig


def build_graph(num_poses: int) -> PoseGraph:
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=15, linear_solver_max_iter=150),
        loss_config=RobustLossConfig(),
    )
    for _ in range(num_poses - 1):
        graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_loop(0, num_poses - 1, np.eye(2), np.array([float(num_poses - 1), 0.0]), weight=0.8)
    return graph


def run_benchmark(num_poses: int) -> dict[str, float | int]:
    graph = build_graph(num_poses)
    tracemalloc.start()
    start_time = time.perf_counter()
    graph.optimize()
    elapsed_s = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = graph.last_result
    residual_norm = float(result.residual_norm) if result is not None else float("nan")
    return {
        "num_poses": num_poses,
        "elapsed_s": elapsed_s,
        "peak_memory_mb": peak / (1024 * 1024),
        "residual_norm": residual_norm,
    }


def main() -> None:
    report = run_benchmark(num_poses=60)
    output_path = Path("artifacts") / "block_sparse_solver_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
