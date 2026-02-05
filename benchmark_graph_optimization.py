"""Benchmark pose-graph optimization throughput and memory delta."""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc
from typing import Tuple

import numpy as np

from pose_graph import PoseGraph, PoseGraph3D, PoseGraphSim3D, RobustLossConfig, RobustLossType, SolverConfig

LOGGER = logging.getLogger(__name__)


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_se2_graph(rng: np.random.Generator, nodes: int, loops: int) -> PoseGraph:
    graph = PoseGraph(
        solver_name="scipy",
        solver_config=SolverConfig(max_iterations=15, max_nfev=200),
        loss_config=RobustLossConfig(loss_type=RobustLossType.HUBER, scale=1.0),
    )
    for _ in range(1, nodes):
        theta = rng.normal(0, 0.1)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = rng.normal(0, 0.2, size=2)
        graph.add_pose(R, t)
    for _ in range(loops):
        i = int(rng.integers(0, nodes - 1))
        j = int(rng.integers(i + 1, nodes))
        graph.add_loop(i, j, np.eye(2), np.zeros(2), weight=0.8)
    return graph


def _build_se3_graph(rng: np.random.Generator, nodes: int, loops: int) -> PoseGraph3D:
    import cv2

    graph = PoseGraph3D(
        solver_name="scipy",
        solver_config=SolverConfig(max_iterations=10, max_nfev=150),
        loss_config=RobustLossConfig(loss_type=RobustLossType.CAUCHY, scale=1.0),
    )
    for _ in range(1, nodes):
        rvec = rng.normal(0, 0.05, size=3)
        R, _ = cv2.Rodrigues(rvec)
        t = rng.normal(0, 0.1, size=3)
        graph.add_pose(R, t)
    for _ in range(loops):
        i = int(rng.integers(0, nodes - 1))
        j = int(rng.integers(i + 1, nodes))
        graph.add_loop(i, j, np.eye(3), np.zeros(3), weight=0.7)
    return graph


def _benchmark(graph: object) -> Tuple[float, int]:
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    start = time.perf_counter()
    _ = graph.optimize()
    elapsed = time.perf_counter() - start
    end_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return elapsed, end_mem - start_mem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=150)
    parser.add_argument("--edges", type=int, default=300)
    parser.add_argument("--solver", choices=["scipy", "gauss_newton"], default="scipy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph", choices=["se2", "se3", "sim3"], default="se2")
    args = parser.parse_args()

    rng = _seeded_rng(args.seed)
    loops = max(args.edges - (args.nodes - 1), 1)

    if args.graph == "se2":
        graph = _build_se2_graph(rng, args.nodes, loops)
    elif args.graph == "se3":
        import cv2  # local import to avoid unused dependency in se2 path

        graph = _build_se3_graph(rng, args.nodes, loops)
    else:
        graph = PoseGraphSim3D(
            solver_name="scipy",
            solver_config=SolverConfig(max_iterations=10, max_nfev=150),
            loss_config=RobustLossConfig(loss_type=RobustLossType.TUKEY, scale=1.0),
        )
        for _ in range(1, args.nodes):
            t = rng.normal(0, 0.1, size=3)
            graph.add_pose(np.eye(3), t, scale=1.0)
        for _ in range(loops):
            i = int(rng.integers(0, args.nodes - 1))
            j = int(rng.integers(i + 1, args.nodes))
            graph.add_loop(i, j, np.eye(3), np.zeros(3), 1.0, weight=0.7)

    graph.configure_solver(solver_name=args.solver)

    elapsed, mem_delta = _benchmark(graph)
    print(f"solver={args.solver} graph={args.graph} nodes={args.nodes} edges={args.edges}")
    print(f"elapsed_s={elapsed:.4f} mem_delta_bytes={mem_delta}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
