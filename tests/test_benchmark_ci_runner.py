from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark_ci_runner import execute_benchmark, load_benchmark_config
from evaluation_harness import load_config, run_evaluation


def _write_traj(path: Path, points: list[tuple[float, float]]) -> None:
    lines = [f"{x} {y}" for x, y in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_eval_config(
    path: Path,
    gt_path: Path,
    est_path: Path,
    baseline_store: Path,
    baseline_key: str,
    thresholds: dict[str, float],
    write_baseline: bool,
) -> None:
    config = {
        "run": {
            "run_id": path.stem,
            "dataset": "custom",
            "seed": 0,
            "output_dir": str(path.parent / "reports"),
            "use_run_subdir": False,
        },
        "baseline": {
            "store_path": str(baseline_store),
            "key": baseline_key,
            "thresholds": thresholds,
            "write": write_baseline,
        },
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
    path.write_text(json.dumps(config), encoding="utf-8")


def test_ci_benchmark_runner_scores_regression(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    est_baseline = tmp_path / "est_baseline.txt"
    est_regressed = tmp_path / "est_regressed.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    _write_traj(est_baseline, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    _write_traj(est_regressed, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)])

    baseline_store = tmp_path / "baselines.json"
    baseline_key = "synthetic"
    baseline_config = tmp_path / "baseline.json"
    _write_eval_config(
        baseline_config,
        gt_path,
        est_baseline,
        baseline_store,
        baseline_key,
        {"ATE_RMSE": 0.01, "RPE_RMSE": 0.01},
        True,
    )

    eval_config = load_config(baseline_config)
    run_evaluation(eval_config)

    regressed_config = tmp_path / "regressed.json"
    _write_eval_config(
        regressed_config,
        gt_path,
        est_regressed,
        baseline_store,
        baseline_key,
        {"ATE_RMSE": 0.01, "RPE_RMSE": 0.01},
        False,
    )

    benchmark_config_path = tmp_path / "benchmark.json"
    benchmark_payload = {
        "runs": [{"name": "synthetic_benchmark", "config_path": str(regressed_config)}],
        "output_path": str(tmp_path / "benchmark_summary.json"),
        "fail_fast": True,
        "max_workers": 1,
        "severity": {"metric_weights": {"ATE_RMSE": 2.0}},
    }
    benchmark_config_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")

    benchmark_config = load_benchmark_config(benchmark_config_path)
    summary = asyncio.run(execute_benchmark(benchmark_config))

    assert summary["status"] == "regressed"
    run_summary = summary["runs"][0]
    assert run_summary["status"] == "regressed"
    assert run_summary["severity"]["score"] > 0.0
    assert "ATE_RMSE" in run_summary["severity"]["metrics"]["contributions"]


def test_ci_benchmark_runner_flags_missing_baseline(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    est_path = tmp_path / "est.txt"
    _write_traj(gt_path, [(0.0, 0.0), (1.0, 0.0)])
    _write_traj(est_path, [(0.0, 0.0), (1.0, 0.0)])

    baseline_store = tmp_path / "baselines.json"
    baseline_key = "missing"
    eval_config_path = tmp_path / "eval.json"
    _write_eval_config(
        eval_config_path,
        gt_path,
        est_path,
        baseline_store,
        baseline_key,
        {"ATE_RMSE": 0.1},
        False,
    )

    benchmark_config_path = tmp_path / "benchmark_missing.json"
    benchmark_payload = {
        "runs": [
            {
                "name": "missing_baseline",
                "config_path": str(eval_config_path),
                "require_baseline": True,
            }
        ],
        "output_path": str(tmp_path / "benchmark_summary.json"),
        "fail_fast": False,
        "max_workers": 1,
    }
    benchmark_config_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")

    benchmark_config = load_benchmark_config(benchmark_config_path)
    summary = asyncio.run(execute_benchmark(benchmark_config))

    assert summary["status"] == "regressed"
    assert summary["runs"][0]["status"] == "missing_baseline"
