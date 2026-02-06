import concurrent.futures

import numpy as np

from factor_graph import FactorGraph, FactorGraphConfig, SE2BetweenFactor
from graph_optimization import PoseGraphSnapshot, RobustLossConfig, SolverConfig, get_solver_registry


def _build_graph(node_count: int = 8) -> FactorGraph:
    config = FactorGraphConfig(state_dim=3, anchor_ids=(0,))
    graph = FactorGraph(config)
    for idx in range(node_count):
        graph.add_variable(idx, np.array([float(idx), 0.0, 0.0]))
    for idx in range(node_count - 1):
        measurement = np.array([1.0, 0.0, 0.0])
        graph.add_factor(SE2BetweenFactor(idx, idx + 1, measurement, weight=1.0))
    return graph


def _solve_once(graph: FactorGraph) -> np.ndarray:
    snapshot = PoseGraphSnapshot(
        version=1,
        solver_name="gauss_newton",
        loss_config=RobustLossConfig(),
        solver_config=SolverConfig(max_iterations=5),
        poses=[],
        edges=[],
        metadata={"stress": True},
    )
    problem, x0 = graph.build_problem(snapshot)
    solver = get_solver_registry().get("gauss_newton")
    x_opt, _ = solver.solve(problem, x0, SolverConfig(max_iterations=5), RobustLossConfig())
    return x_opt


def test_factor_graph_parallel_solve_is_deterministic():
    graph = _build_graph()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: _solve_once(graph), range(4)))
    for result in results[1:]:
        assert np.allclose(result, results[0])
