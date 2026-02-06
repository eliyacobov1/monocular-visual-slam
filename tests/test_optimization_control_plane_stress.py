"""Stress tests for optimization control plane determinism."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pose_graph import PoseGraph, RobustLossConfig, SolverConfig


def _run_graph() -> str:
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=6, linear_solver_max_iter=30),
        loss_config=RobustLossConfig(),
    )
    for _ in range(4):
        graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_loop(0, len(graph.poses) - 1, np.eye(2), np.array([4.0, 0.0]), weight=0.8)
    graph.optimize()
    report = graph.last_report
    assert report is not None
    return report.snapshot_digest


def test_optimization_control_plane_determinism() -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        digests = list(executor.map(lambda _: _run_graph(), range(4)))
    assert len(set(digests)) == 1

