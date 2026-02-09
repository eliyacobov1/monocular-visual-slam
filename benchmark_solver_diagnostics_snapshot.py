"""Benchmark for solver diagnostics snapshot overhead."""

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
        solver_config=SolverConfig(
            max_iterations=12,
            linear_solver_max_iter=80,
            residual_histogram_bins=24,
            residual_histogram_range=(0.0, 6.0),
        ),
        loss_config=RobustLossConfig(),
    )
    for _ in range(num_poses - 1):
        graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_loop(0, num_poses - 1, np.eye(2), np.array([float(num_poses - 1), 0.0]), weight=0.8)
    return graph


def run_benchmark(num_poses: int) -> dict[str, float | int | str]:
    graph = build_graph(num_poses)
    tracemalloc.start()
    start_time = time.perf_counter()
    graph.optimize()
    elapsed_s = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = graph.last_report
    snapshot = report.solver_snapshot if report else None
    return {
        "num_poses": num_poses,
        "elapsed_s": elapsed_s,
        "peak_memory_mb": peak / (1024 * 1024),
        "solver": report.solver_name if report else "unknown",
        "iterations": report.result.iterations if report else 0,
        "residual_bins": len(snapshot.residual_histogram.counts) if snapshot else 0,
        "regression_status": report.regression_gate.status if report else "unknown",
    }


def main() -> None:
    report = run_benchmark(num_poses=80)
    output_path = Path("artifacts") / "solver_diagnostics_snapshot_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
