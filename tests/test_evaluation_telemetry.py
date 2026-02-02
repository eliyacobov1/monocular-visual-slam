"""Tests for telemetry aggregation in the evaluation harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation_harness import _load_telemetry_events, _summarize_telemetry_events


def test_summarize_telemetry_events() -> None:
    events = [
        {"name": "feature_detect", "duration_s": 0.1},
        {"name": "feature_detect", "duration_s": 0.2},
        {"name": "pose_estimate", "duration_s": 0.4},
    ]
    summary = _summarize_telemetry_events(events)

    assert summary["event_count"] == 3
    assert np.isclose(summary["total_duration_s"], 0.7)
    assert "feature_detect" in summary["per_stage"]
    assert summary["per_stage"]["feature_detect"]["count"] == 2
    assert np.isclose(summary["per_stage"]["feature_detect"]["mean_duration_s"], 0.15)


def test_load_telemetry_events(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    payload = {
        "recorded_at": "2024-01-01T00:00:00Z",
        "events": [
            {"name": "frame_process", "duration_s": 0.3},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    events = _load_telemetry_events(path)
    assert events == payload["events"]
