#!/usr/bin/env python3
"""Benchmark factor-graph Gauss-Newton solve time and memory delta."""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from factor_graph import FactorGraph, FactorGraphConfig, SE2BetweenFactor
from graph_optimization import PoseGraphSnapshot, RobustLossConfig, SolverConfig, get_solver_registry


def _build_graph(node_count: int, loop_every: int) -> FactorGraph:
    config = FactorGraphConfig(state_dim=3, anchor_ids=(0,))
    graph = FactorGraph(config)
    for idx in range(node_count):
        graph.add_variable(idx, np.array([float(idx), 0.0, 0.0]))
    for idx in range(node_count - 1):
        graph.add_factor(SE2BetweenFactor(idx, idx + 1, np.array([1.0, 0.0, 0.0]), weight=1.0))
    for idx in range(loop_every, node_count, loop_every):
        graph.add_factor(SE2BetweenFactor(0, idx, np.array([float(idx), 0.0, 0.0]), weight=0.5))
    return graph


def run_benchmark(node_count: int, loop_every: int, iterations: int) -> dict[str, float]:
    graph = _build_graph(node_count, loop_every)
    snapshot = PoseGraphSnapshot(
        version=1,
        solver_name="gauss_newton",
        loss_config=RobustLossConfig(),
        solver_config=SolverConfig(max_iterations=iterations),
        poses=[],
        edges=[],
        metadata={"benchmark": True},
    )
    problem, x0 = graph.build_problem(snapshot)
    solver = get_solver_registry().get("gauss_newton")

    tracemalloc.start()
    start_mem, _ = tracemalloc.get_traced_memory()
    start = time.perf_counter()
    solver.solve(problem, x0, SolverConfig(max_iterations=iterations), RobustLossConfig())
    elapsed = time.perf_counter() - start
    end_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "node_count": float(node_count),
        "loop_every": float(loop_every),
        "iterations": float(iterations),
        "elapsed_s": float(elapsed),
        "memory_delta_bytes": float(end_mem - start_mem),
        "memory_peak_bytes": float(peak_mem),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--loop-every", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    metrics = run_benchmark(args.nodes, args.loop_every, args.iterations)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
