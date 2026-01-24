from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation_harness import load_config, run_evaluation


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

    summary_path = Path(config["output_dir"]) / "summary.json"
    assert summary_path.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "aggregate_metrics" in summary_payload
