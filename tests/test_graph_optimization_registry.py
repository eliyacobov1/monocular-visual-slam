import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph_optimization import get_solver_registry
from pose_graph import PoseGraph, RobustLossConfig, RobustLossType, SolverConfig


def test_solver_registry_contains_default_backends():
    registry = get_solver_registry()
    available = registry.available()
    assert "scipy" in available
    assert "gauss_newton" in available


def test_pose_graph_snapshot_digest_is_deterministic():
    def build_graph() -> PoseGraph:
        graph = PoseGraph(
            solver_name="scipy",
            solver_config=SolverConfig(max_iterations=5, max_nfev=50),
            loss_config=RobustLossConfig(loss_type=RobustLossType.HUBER, scale=1.0),
        )
        graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
        graph.add_pose(np.eye(2), np.array([0.0, 1.0]))
        graph.add_loop(0, 2, np.eye(2), np.zeros(2), weight=0.9)
        return graph

    graph_a = build_graph()
    graph_b = build_graph()

    graph_a.optimize()
    graph_b.optimize()

    snapshot_a = graph_a.last_snapshot
    snapshot_b = graph_b.last_snapshot

    assert snapshot_a is not None
    assert snapshot_b is not None
    assert snapshot_a.digest() == snapshot_b.digest()
