import numpy as np

from factor_graph import FactorGraph, FactorGraphConfig, SE2BetweenFactor
from graph_optimization import PoseGraphSnapshot, RobustLossConfig, SolverConfig


def _numeric_jacobian(func, vec, eps=1e-6):
    vec = np.asarray(vec, dtype=float)
    base = func(vec)
    jac = np.zeros((base.size, vec.size), dtype=float)
    for idx in range(vec.size):
        step = np.zeros_like(vec)
        step[idx] = eps
        forward = func(vec + step)
        backward = func(vec - step)
        jac[:, idx] = (forward - backward) / (2 * eps)
    return jac


def test_se2_between_factor_jacobian_matches_numeric():
    xi = np.array([1.2, -0.3, 0.5])
    xj = np.array([2.0, 0.4, -0.2])
    measurement = np.array([0.7, 0.2, 0.1])
    factor = SE2BetweenFactor(0, 1, measurement, weight=1.0)

    jac_i, jac_j = factor.jacobians(xi, xj)
    numeric_i = _numeric_jacobian(lambda v: factor.residual(v, xj), xi)
    numeric_j = _numeric_jacobian(lambda v: factor.residual(xi, v), xj)

    assert np.allclose(jac_i, numeric_i, atol=1e-5)
    assert np.allclose(jac_j, numeric_j, atol=1e-5)


def test_factor_graph_anchor_linearization():
    config = FactorGraphConfig(state_dim=3, anchor_ids=(0,))
    graph = FactorGraph(config)
    graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
    graph.add_variable(1, np.array([1.0, 0.0, 0.0]))
    factor = SE2BetweenFactor(0, 1, np.array([1.0, 0.0, 0.0]))
    graph.add_factor(factor)

    snapshot = PoseGraphSnapshot(
        version=1,
        solver_name="gauss_newton",
        loss_config=RobustLossConfig(),
        solver_config=SolverConfig(),
        poses=[np.eye(3).tolist(), np.eye(3).tolist()],
        edges=[],
        metadata={},
    )
    problem, x0 = graph.build_problem(snapshot)
    linearized = list(problem.linearize_fn(x0))

    assert len(linearized) == 1
    block = linearized[0]
    assert block.j is None
