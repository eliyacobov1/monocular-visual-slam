"""Tests for run telemetry utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_telemetry import (
    RunTelemetryRecorder,
    TelemetryCorrelationConfig,
    TelemetryCorrelationRegistry,
    timed_event,
)


def test_run_telemetry_records_events(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    recorder = RunTelemetryRecorder(
        path,
        correlation_registry=TelemetryCorrelationRegistry(
            TelemetryCorrelationConfig(seed=7, config_hash="hash", run_id="run")
        ),
    )

    with timed_event("operation", recorder, {"stage": "unit"}, track_memory=True):
        pass

    recorder.flush()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["events"][0]["name"] == "operation"
    assert payload["events"][0]["metadata"]["stage"] == "unit"
    assert payload["events"][0]["metadata"]["success"] is True
    assert payload["events"][0]["metadata"]["correlation_id"]
    assert "memory_delta_bytes" in payload["events"][0]["metadata"]


def test_run_telemetry_captures_failure(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    recorder = RunTelemetryRecorder(
        path,
        correlation_registry=TelemetryCorrelationRegistry(
            TelemetryCorrelationConfig(seed=8, config_hash="hash", run_id="run")
        ),
    )

    with pytest.raises(ValueError):
        with timed_event("explode", recorder, {"stage": "unit"}):
            raise ValueError("boom")

    recorder.flush()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["events"][0]["metadata"]["success"] is False
    assert payload["events"][0]["metadata"]["error"] == "ValueError"
