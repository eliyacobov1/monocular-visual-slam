from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation_harness import load_config, run_evaluation
from data_persistence import (
    FrameDiagnosticsEntry,
    build_frame_diagnostics_bundle,
    frame_diagnostics_artifact_path,
    trajectory_artifact_path,
)


def _write_traj(path: Path, points: list[tuple[float, float]]) -> None:
    lines = [f"{x} {y}" for x, y in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_evaluation_harness_runs(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    est_path = tmp_path / "est.txt"

    gt_points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    est_points = [(0.0, 0.0), (1.1, 0.0), (2.1, 0.0)]

    _write_traj(gt_path, gt_points)
    _write_traj(est_path, est_points)

    config = {
        "run_id": "unit_test",
        "dataset": "custom",
        "seed": 0,
        "output_dir": str(tmp_path / "reports"),
        "trajectories": [
            {
                "name": "synthetic",
                "gt_path": str(gt_path),
                "est_path": str(est_path),
                "format": "xy",
                "cols": "0,1",
                "est_cols": "0,1",
                "rpe_delta": 1,
            }
        ],
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    eval_config = load_config(config_path)
    summary = run_evaluation(eval_config)

    assert summary["run_id"] == "unit_test"
    assert "aggregate_metrics" in summary
    assert "synthetic" in summary["per_sequence"]
    assert "config_hash" in summary
    assert summary["config_hash"]
    assert summary["config_path"].endswith("config.json")

    summary_path = Path(config["output_dir"]) / "summary.json"
    assert summary_path.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "aggregate_metrics" in summary_payload


def test_evaluation_harness_with_experiment_schema(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    est_path = tmp_path / "est.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0)])
    _write_traj(est_path, [(0.0, 0.0), (1.1, 0.0)])

    config = {
        "run": {
            "run_id": "exp_schema",
            "dataset": "custom",
            "seed": 42,
            "output_dir": str(tmp_path / "reports"),
            "use_run_subdir": False,
        },
        "pipeline": {"feature_type": "orb", "motion_ransac_threshold": 0.01},
        "evaluation": {
            "trajectories": [
                {
                    "name": "synthetic",
                    "gt_path": str(gt_path),
                    "est_path": str(est_path),
                    "format": "xy",
                    "cols": "0,1",
                    "est_cols": "0,1",
                    "rpe_delta": 1,
                }
            ]
        },
    }

    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    eval_config = load_config(config_path)
    summary = run_evaluation(eval_config)

    resolved_path = Path(summary["run_dir"]) / "resolved_config.json"
    assert resolved_path.exists()
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved["run_id"] == "exp_schema"
    assert resolved["pipeline"]["feature_type"] == "orb"


def test_evaluation_harness_with_slam_npz(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)])

    est_path = tmp_path / "slam_trajectory.npz"
    poses = np.repeat(np.eye(4)[None, ...], 3, axis=0)
    poses[1, 0, 3] = 1.0
    poses[2, 0, 3] = 2.0
    poses[2, 1, 3] = 1.0
    np.savez_compressed(
        est_path,
        poses=poses,
        timestamps=np.array([0.0, 1.0, 2.0]),
        frame_ids=np.array([0, 1, 2]),
    )

    config = {
        "run_id": "unit_npz",
        "dataset": "custom",
        "seed": 0,
        "output_dir": str(tmp_path / "reports"),
        "trajectories": [
            {
                "name": "slam_run",
                "gt_path": str(gt_path),
                "est_path": str(est_path),
                "format": "xy",
                "cols": "0,1",
                "est_format": "slam_npz",
                "rpe_delta": 1,
            }
        ],
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    eval_config = load_config(config_path)
    summary = run_evaluation(eval_config)

    assert summary["run_id"] == "unit_npz"
    assert "slam_run" in summary["per_sequence"]


def test_evaluation_harness_with_run_dir_npz(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0)])

    run_dir = tmp_path / "slam_run"
    trajectory_path = trajectory_artifact_path(run_dir, "slam trajectory 01")
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 1.0
    np.savez_compressed(
        trajectory_path,
        poses=poses,
        timestamps=np.array([0.0, 1.0]),
        frame_ids=np.array([0, 1]),
    )

    config = {
        "run_id": "unit_run_dir",
        "dataset": "custom",
        "seed": 0,
        "output_dir": str(tmp_path / "reports"),
        "trajectories": [
            {
                "name": "slam_run",
                "gt_path": str(gt_path),
                "est_run_dir": str(run_dir),
                "est_trajectory": "slam trajectory 01",
                "format": "xy",
                "cols": "0,1",
                "rpe_delta": 1,
            }
        ],
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    eval_config = load_config(config_path)
    summary = run_evaluation(eval_config)

    assert summary["run_id"] == "unit_run_dir"
    assert "slam_run" in summary["per_sequence"]


def test_evaluation_harness_with_diagnostics_summary(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0)])

    run_dir = tmp_path / "slam_run"
    trajectory_path = trajectory_artifact_path(run_dir, "slam_run")
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 1.0
    np.savez_compressed(
        trajectory_path,
        poses=poses,
        timestamps=np.array([0.0, 1.0]),
        frame_ids=np.array([0, 1]),
    )
    diagnostics_bundle = build_frame_diagnostics_bundle(
        "frame_diagnostics",
        [
            FrameDiagnosticsEntry(
                frame_id=0,
                timestamp=0.0,
                match_count=12,
                inliers=9,
                method="essential",
                inlier_ratio=0.75,
                median_parallax=1.2,
                score=0.9,
                status="ok",
                failure_reason=None,
            ),
            FrameDiagnosticsEntry(
                frame_id=1,
                timestamp=1.0,
                match_count=10,
                inliers=6,
                method="homography",
                inlier_ratio=0.6,
                median_parallax=0.8,
                score=0.7,
                status="ok",
                failure_reason=None,
            ),
        ],
    )
    diagnostics_path = frame_diagnostics_artifact_path(run_dir, diagnostics_bundle.name)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(
            {
                "name": diagnostics_bundle.name,
                "recorded_at": diagnostics_bundle.recorded_at,
                "entries": [
                    {
                        "frame_id": entry.frame_id,
                        "timestamp": entry.timestamp,
                        "match_count": entry.match_count,
                        "inliers": entry.inliers,
                        "method": entry.method,
                        "inlier_ratio": entry.inlier_ratio,
                        "median_parallax": entry.median_parallax,
                        "score": entry.score,
                        "status": entry.status,
                        "failure_reason": entry.failure_reason,
                    }
                    for entry in diagnostics_bundle.entries
                ],
            }
        ),
        encoding="utf-8",
    )

    config = {
        "run_id": "unit_diagnostics",
        "dataset": "custom",
        "seed": 0,
        "output_dir": str(tmp_path / "reports"),
        "trajectories": [
            {
                "name": "slam_run",
                "gt_path": str(gt_path),
                "est_run_dir": str(run_dir),
                "format": "xy",
                "cols": "0,1",
                "rpe_delta": 1,
            }
        ],
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    eval_config = load_config(config_path)
    summary = run_evaluation(eval_config)

    per_sequence = summary["per_sequence"]["slam_run"]
    assert "diagnostics_summary" in per_sequence
    assert per_sequence["metrics"]["diag_frame_count"] == 2.0


def test_evaluation_harness_with_relocalization_report(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0)])

    run_dir = tmp_path / "slam_run"
    trajectory_path = trajectory_artifact_path(run_dir, "slam_run")
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 1.0
    np.savez_compressed(
        trajectory_path,
        poses=poses,
        timestamps=np.array([0.0, 1.0]),
        frame_ids=np.array([0, 1]),
    )

    report_payload = {
        "run_id": "demo",
        "sequence": "00",
        "relocalization_summary": {
            "attempts": 2,
            "successes": 1,
            "success_rate": 0.5,
            "latency_mean_s": 0.2,
            "latency_p50_s": 0.2,
            "latency_p95_s": 0.3,
        },
    }
    report_path = run_dir / "relocalization_demo_report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    config = {
        "run": {
            "run_id": "unit_relocalization",
            "dataset": "custom",
            "seed": 0,
            "output_dir": str(tmp_path / "reports"),
        },
        "evaluation": {
            "trajectories": [
                {
                    "name": "slam_run",
                    "gt_path": str(gt_path),
                    "est_run_dir": str(run_dir),
                    "format": "xy",
                    "cols": "0,1",
                    "rpe_delta": 1,
                }
            ]
        },
        "baseline": {"relocalization": {"report_name": "relocalization_demo_report.json"}},
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    eval_config = load_config(config_path)
    summary = run_evaluation(eval_config)

    assert summary["relocalization_metrics"]["relocalization_success_rate"] == 0.5
    per_sequence = summary["per_sequence"]["slam_run"]
    assert per_sequence["relocalization_metrics"]["relocalization_attempts"] == 2.0
