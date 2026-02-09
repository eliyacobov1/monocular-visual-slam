"""Tests for telemetry intelligence summaries and drift evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telemetry_intelligence import (
    TelemetryDigest,
    TelemetryDriftThresholds,
    compare_telemetry_summaries,
    load_telemetry_events,
    summarize_telemetry_events,
)


def test_telemetry_digest_summary() -> None:
    digest = TelemetryDigest()
    digest.update("feature", 0.1)
    digest.update("feature", 0.2)
    digest.update("track", 0.3)

    summary = digest.summarize()

    assert summary["event_count"] == 3
    assert summary["per_stage"]["feature"]["count"] == 2
    assert summary["per_stage"]["feature"]["mean_duration_s"] == pytest.approx(0.15)
    assert summary["per_stage"]["track"]["max_duration_s"] == pytest.approx(0.3)


def test_telemetry_digest_memory_and_correlation() -> None:
    digest = TelemetryDigest()
    digest.update("optimize", 0.1, memory_delta_bytes=128.0, correlation_id="abc")
    digest.update("optimize", 0.2, memory_delta_bytes=256.0, correlation_id="abc")
    digest.update("track", 0.1, correlation_id="def")

    summary = digest.summarize()

    assert summary["memory_event_count"] == 2
    assert summary["total_memory_delta_bytes"] == pytest.approx(384.0)
    assert summary["per_stage"]["optimize"]["memory_delta_count"] == 2
    assert summary["per_stage_correlation_ids"]["optimize"] == "abc"
    assert summary["per_stage_correlation_ids"]["track"] == "def"


def test_compare_telemetry_summaries() -> None:
    baseline = summarize_telemetry_events(
        [
            {"name": "track", "duration_s": 0.1},
            {"name": "track", "duration_s": 0.2},
        ]
    )
    current = summarize_telemetry_events(
        [
            {"name": "track", "duration_s": 0.2},
            {"name": "track", "duration_s": 0.3},
        ]
    )

    thresholds = TelemetryDriftThresholds(relative_increase=0.2, absolute_increase_s=0.05)
    report = compare_telemetry_summaries(current, baseline, thresholds)

    assert report["status"] in {"pass", "warn", "fail"}
    assert "track" in report["per_stage"]
    assert report["per_stage"]["track"]["findings"]


def test_load_telemetry_events(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    payload = {
        "recorded_at": "2024-01-01T00:00:00Z",
        "events": [
            {"name": "frame", "duration_s": 0.4},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    events = load_telemetry_events(path)
    assert events == payload["events"]
