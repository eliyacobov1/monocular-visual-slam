from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regression_baselines import compare_metrics, load_baseline_store, upsert_baseline


def test_upsert_and_compare_baseline(tmp_path: Path) -> None:
    store_path = tmp_path / "baselines.json"
    metrics = {"ate": 0.5, "rpe": 0.2}

    baseline = upsert_baseline(
        store_path,
        key="kitti_00",
        metrics=metrics,
        config_hash="abc123",
        metadata={"dataset": "kitti"},
    )

    store = load_baseline_store(store_path)
    assert "kitti_00" in store["baselines"]
    assert baseline["metrics"]["ate"] == 0.5

    comparison = compare_metrics(
        "kitti_00",
        current={"ate": 0.6, "rpe": 0.25},
        baseline=baseline,
        thresholds={"ate": 0.2, "rpe": 0.1},
    )

    assert comparison.status == "pass"
    assert comparison.per_metric["ate"]["status"] == "pass"
    assert comparison.stats["evaluated_metrics"] == 2.0
    assert comparison.stats["comparison_time_ms"] >= 0.0


def test_compare_supports_directional_thresholds() -> None:
    baseline = {"metrics": {"inlier_ratio": 0.6, "ate": 0.3}}

    comparison = compare_metrics(
        "kitti_00",
        current={"inlier_ratio": 0.52, "ate": 0.33},
        baseline=baseline,
        thresholds={
            "inlier_ratio": {"direction": "higher", "tolerance": 0.05},
            "ate": {"direction": "lower", "tolerance": 0.02},
        },
    )

    assert comparison.per_metric["inlier_ratio"]["status"] == "regressed"
    assert comparison.per_metric["ate"]["status"] == "regressed"
    assert comparison.status == "regressed"
