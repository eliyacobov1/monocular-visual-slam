"""Tests for solver diagnostics snapshots and persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_persistence import RunDataStore, load_solver_diagnostics_json
from pose_graph import PoseGraph, RobustLossConfig, SolverConfig


def test_solver_snapshot_and_persistence(tmp_path: Path) -> None:
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(
            max_iterations=6,
            linear_solver_max_iter=30,
            residual_histogram_bins=12,
            residual_histogram_range=(0.0, 4.0),
        ),
        loss_config=RobustLossConfig(),
    )
    for _ in range(4):
        graph.add_pose(np.eye(2), np.array([1.0, 0.0]))
    graph.add_loop(0, len(graph.poses) - 1, np.eye(2), np.array([4.0, 0.0]), weight=0.7)

    graph.optimize()

    report = graph.last_report
    assert report is not None
    snapshot = report.solver_snapshot
    assert snapshot.solver_name == "gauss_newton"
    assert snapshot.iteration_diagnostics
    assert snapshot.residual_histogram.counts
    assert len(snapshot.residual_histogram.counts) == 12
    assert snapshot.convergence.status in {"converged", "failed"}
    assert all(entry.residual_histogram is not None for entry in snapshot.iteration_diagnostics)
    assert report.regression_gate.status in {"baseline", "pass", "regressed"}

    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    store = RunDataStore.create(
        base_dir=tmp_path,
        run_id="solver-diagnostics",
        config_path=config_path,
        config_hash="abc123",
        seed=1,
        use_subdir=False,
    )
    report_path = store.save_solver_diagnostics_report(
        "solver_diagnostics",
        report.asdict(),
    )
    loaded = load_solver_diagnostics_json(report_path)
    assert loaded["solver_snapshot"]["snapshot_digest"] == report.snapshot_digest
