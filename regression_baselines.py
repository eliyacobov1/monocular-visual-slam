"""Baseline store utilities for regression tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineComparison:
    key: str
    status: str
    per_metric: dict[str, dict[str, float | str]]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_baseline_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "baselines": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def upsert_baseline(
    path: Path,
    key: str,
    metrics: dict[str, float],
    config_hash: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = load_baseline_store(path)
    baselines = store.setdefault("baselines", {})
    baselines[key] = {
        "metrics": metrics,
        "config_hash": config_hash,
        "updated_at": _timestamp(),
        "metadata": metadata or {},
    }
    save_baseline_store(path, store)
    return baselines[key]


def compare_metrics(
    key: str,
    current: dict[str, float],
    baseline: dict[str, Any],
    thresholds: dict[str, float] | None,
) -> BaselineComparison:
    thresholds = thresholds or {}
    per_metric: dict[str, dict[str, float | str]] = {}
    status = "pass"
    baseline_metrics = baseline.get("metrics", {})

    for metric, current_value in current.items():
        baseline_value = float(baseline_metrics.get(metric, float("nan")))
        threshold = thresholds.get(metric)
        delta = current_value - baseline_value
        metric_status = "pass"
        if threshold is not None and delta > threshold:
            metric_status = "regressed"
            status = "regressed"
        per_metric[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta": delta,
            "threshold": threshold if threshold is not None else float("nan"),
            "status": metric_status,
        }

    return BaselineComparison(key=key, status=status, per_metric=per_metric)
