from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark_governance import execute_governance, load_governance_config


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_governance_flags_budget_exceeded(tmp_path: Path) -> None:
    config_path = tmp_path / "governance.json"
    payload = {
        "benchmarks": [
            {
                "name": "fast_command",
                "command": ["python", "-c", "pass"],
                "runtime_budget_s": 0.0,
                "memory_budget_bytes": 0,
                "require_baseline": False,
            }
        ],
        "output_path": str(tmp_path / "summary.json"),
        "baseline_store": str(tmp_path / "baselines.json"),
        "write_baseline": False,
    }
    _write_config(config_path, payload)
    config = load_governance_config(config_path)
    summary = asyncio.run(execute_governance(config))

    assert summary["status"] == "regressed"
    run_summary = summary["runs"][0]
    assert run_summary["status"] == "budget_exceeded"
    assert run_summary["budget_status"]["runtime"] == "over"
    assert run_summary["budget_status"]["memory"] == "over"


def test_governance_detects_regression_against_baseline(tmp_path: Path) -> None:
    baseline_config_path = tmp_path / "baseline.json"
    baseline_payload = {
        "benchmarks": [
            {
                "name": "sleepy",
                "command": ["python", "-c", "pass"],
                "runtime_budget_s": 5.0,
                "memory_budget_bytes": 1000000,
                "require_baseline": False,
                "baseline_key": "sleepy",
            }
        ],
        "output_path": str(tmp_path / "baseline_summary.json"),
        "baseline_store": str(tmp_path / "baselines.json"),
        "write_baseline": True,
        "thresholds": {"runtime_s": {"max_delta": 0.0}},
    }
    _write_config(baseline_config_path, baseline_payload)
    baseline_config = load_governance_config(baseline_config_path)
    asyncio.run(execute_governance(baseline_config))

    regressed_config_path = tmp_path / "regressed.json"
    regressed_payload = {
        "benchmarks": [
            {
                "name": "sleepy",
                "command": ["python", "-c", "import time; time.sleep(0.02)"],
                "runtime_budget_s": 5.0,
                "memory_budget_bytes": 1000000,
                "require_baseline": True,
                "baseline_key": "sleepy",
            }
        ],
        "output_path": str(tmp_path / "regressed_summary.json"),
        "baseline_store": str(tmp_path / "baselines.json"),
        "write_baseline": False,
        "thresholds": {"runtime_s": {"max_delta": 0.0}},
    }
    _write_config(regressed_config_path, regressed_payload)
    regressed_config = load_governance_config(regressed_config_path)
    summary = asyncio.run(execute_governance(regressed_config))

    assert summary["status"] == "regressed"
    run_summary = summary["runs"][0]
    assert run_summary["status"] == "regressed"
    assert run_summary["baseline_comparison"]["status"] == "regressed"
