import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph_optimization import RobustLossConfig, SolverConfig
from pose_graph import PoseGraph


def _solve_graph(seed: int) -> float:
    rng = np.random.default_rng(seed)
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=8, linear_solver_max_iter=80),
        loss_config=RobustLossConfig(),
    )
    for _ in range(6):
        dx = float(rng.uniform(0.8, 1.2))
        dy = float(rng.uniform(-0.1, 0.1))
        graph.add_pose(np.eye(2), np.array([dx, dy]))
    graph.add_loop(0, 5, np.eye(2), np.array([5.0, 0.0]), weight=0.9)
    graph.optimize()
    result = graph.last_result
    assert result is not None
    assert result.diagnostics is not None
    return float(result.residual_norm)


def test_gauss_newton_solver_is_thread_safe_under_stress():
    seeds = list(range(6))
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_solve_graph, seeds))
    assert len(results) == len(seeds)
    assert all(result >= 0 for result in results)
