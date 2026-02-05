import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pose_graph import PoseGraph, SolverConfig


def _optimize_once(seed: int) -> float:
    rng = np.random.default_rng(seed)
    graph = PoseGraph(solver_name="gauss_newton", solver_config=SolverConfig(max_iterations=5))
    for _ in range(1, 6):
        theta = rng.normal(0.0, 0.05)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = rng.normal(0.0, 0.1, size=2)
        graph.add_pose(R, t)
    graph.add_loop(0, len(graph.poses) - 1, np.eye(2), np.zeros(2), weight=0.8)
    optimized = graph.optimize()
    return float(np.linalg.norm(optimized[-1][:2, 2]))


def test_pose_graph_solver_threaded_stress():
    seeds = [10, 11, 12, 13]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_optimize_once, seeds))
    assert all(np.isfinite(result) for result in results)
    assert len(set(np.round(results, 6))) == len(results)
