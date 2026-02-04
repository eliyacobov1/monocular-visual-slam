"""Tests for telemetry baseline configuration and metrics extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation_harness import _telemetry_metrics_from_summary, load_config


def test_telemetry_metrics_from_summary() -> None:
    summary = {
        "event_count": 3,
        "total_duration_s": 0.3,
        "mean_duration_s": 0.1,
        "per_stage": {
            "Feature Detect": {
                "count": 2,
                "mean_duration_s": 0.05,
                "p95_duration_s": 0.08,
                "max_duration_s": 0.1,
            }
        },
    }

    metrics = _telemetry_metrics_from_summary(summary)

    assert metrics["telemetry_event_count"] == 3.0
    assert metrics["telemetry_total_duration_s"] == pytest.approx(0.3)
    assert metrics["telemetry_mean_duration_s"] == pytest.approx(0.1)
    assert metrics["telemetry_stage_feature_detect_count"] == 2.0
    assert metrics["telemetry_stage_feature_detect_mean_duration_s"] == pytest.approx(0.05)
    assert metrics["telemetry_stage_feature_detect_p95_duration_s"] == pytest.approx(0.08)
    assert metrics["telemetry_stage_feature_detect_max_duration_s"] == pytest.approx(0.1)


def test_telemetry_baseline_default_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    payload = {
        "run": {
            "run_id": "demo",
            "dataset": "custom",
            "output_dir": "reports",
        },
        "baseline": {
            "store_path": "reports/baselines.json",
            "key": "demo_baseline",
            "thresholds": {"ate": {"direction": "lower", "tolerance": 0.1}},
            "telemetry": {
                "thresholds": {"telemetry_mean_duration_s": 0.01}
            },
        },
        "evaluation": {
            "trajectories": [
                {
                    "name": "demo",
                    "gt_path": "gt.txt",
                    "est_path": "est.txt",
                    "format": "xy",
                }
            ]
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.telemetry_baseline_key == "demo_baseline_telemetry"
    assert config.telemetry_baseline_thresholds == {"telemetry_mean_duration_s": 0.01}
