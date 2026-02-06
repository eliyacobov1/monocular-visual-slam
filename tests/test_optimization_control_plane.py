"""Tests for the optimization control plane supervisor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pose_graph import PoseGraph, RobustLossConfig, SolverConfig


def test_optimization_control_plane_report() -> None:
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=8, linear_solver_max_iter=40),
        loss_config=RobustLossConfig(),
    )
    for _ in range(3):
        graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_loop(0, len(graph.poses) - 1, np.eye(2), np.array([3.0, 0.0]), weight=0.7)

    graph.optimize()

    report = graph.last_report
    assert report is not None
    assert report.attempts >= 1
    assert report.solver_name == "gauss_newton"
    assert report.telemetry.residual_norm["count"] >= 0
    assert report.events
    assert report.snapshot_digest

