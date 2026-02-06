import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph_optimization import (
    BlockSparseMatrix,
    ConjugateGradientSolver,
    RobustLossConfig,
    SolverConfig,
)
from pose_graph import PoseGraph


def test_block_sparse_matrix_matvec():
    matrix = BlockSparseMatrix(block_size=2, num_blocks=2)
    matrix.add_block(0, 0, np.array([[2.0, 0.0], [0.0, 2.0]]))
    matrix.add_block(0, 1, np.array([[1.0, 0.0], [0.0, 1.0]]))
    matrix.add_block(1, 0, np.array([[1.0, 0.0], [0.0, 1.0]]))
    matrix.add_block(1, 1, np.array([[3.0, 0.0], [0.0, 3.0]]))
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    result = matrix.matvec(vec)
    expected = np.array([
        2.0 * 1.0 + 1.0 * 3.0,
        2.0 * 2.0 + 1.0 * 4.0,
        1.0 * 1.0 + 3.0 * 3.0,
        1.0 * 2.0 + 3.0 * 4.0,
    ])
    assert np.allclose(result, expected)


def test_conjugate_gradient_solver_solves_system():
    matrix = BlockSparseMatrix(block_size=1, num_blocks=2)
    matrix.add_block(0, 0, np.array([[4.0]]))
    matrix.add_block(0, 1, np.array([[1.0]]))
    matrix.add_block(1, 0, np.array([[1.0]]))
    matrix.add_block(1, 1, np.array([[3.0]]))
    rhs = np.array([1.0, 2.0])
    solver = ConjugateGradientSolver()
    result = solver.solve(matrix, rhs, max_iter=50, tol=1e-10)
    dense = matrix.to_dense()
    expected = np.linalg.solve(dense, rhs)
    assert result.converged
    assert np.allclose(result.solution, expected, atol=1e-6)


def test_gauss_newton_solver_emits_diagnostics():
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=5, linear_solver_max_iter=50),
        loss_config=RobustLossConfig(),
    )
    graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_loop(0, 2, np.eye(2), np.array([2.0, 0.0]), weight=1.0)
    graph.optimize()
    result = graph.last_result
    assert result is not None
    assert result.diagnostics is not None
    assert len(result.diagnostics.iterations) >= 1
