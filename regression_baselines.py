"""Baseline store utilities for regression tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any
from time import perf_counter


@dataclass(frozen=True)
class BaselineComparison:
    key: str
    status: str
    per_metric: dict[str, dict[str, float | str]]
    stats: dict[str, float]


@dataclass(frozen=True)
class MetricThreshold:
    max_delta: float | None = None
    min_delta: float | None = None
    max_ratio: float | None = None
    min_ratio: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "max_delta": self.max_delta,
            "min_delta": self.min_delta,
            "max_ratio": self.max_ratio,
            "min_ratio": self.min_ratio,
        }


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
    thresholds: dict[str, float | dict[str, Any]] | None,
) -> BaselineComparison:
    thresholds = thresholds or {}
    per_metric: dict[str, dict[str, float | str]] = {}
    status = "pass"
    baseline_metrics = baseline.get("metrics", {})
    start_time = perf_counter()

    for metric, current_value in current.items():
        baseline_value = float(baseline_metrics.get(metric, float("nan")))
        threshold_spec = _parse_threshold_spec(thresholds.get(metric))
        delta = current_value - baseline_value
        ratio = _safe_ratio(delta, baseline_value)
        metric_status = _evaluate_metric_status(
            baseline_value,
            threshold_spec,
            delta,
            ratio,
        )
        if metric_status not in {"pass"}:
            status = "regressed"
        per_metric[metric] = _metric_payload(
            baseline_value,
            current_value,
            delta,
            ratio,
            threshold_spec,
            metric_status,
        )

    elapsed_ms = (perf_counter() - start_time) * 1000.0
    stats = {
        "evaluated_metrics": float(len(per_metric)),
        "comparison_time_ms": float(elapsed_ms),
    }
    return BaselineComparison(key=key, status=status, per_metric=per_metric, stats=stats)


def _parse_threshold_spec(raw: float | dict[str, Any] | None) -> MetricThreshold | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return MetricThreshold(max_delta=float(raw))
    if not isinstance(raw, dict):
        raise TypeError("Threshold entry must be a number or a mapping")

    max_delta = _coerce_optional_float(raw.get("max_delta"))
    min_delta = _coerce_optional_float(raw.get("min_delta"))
    max_ratio = _coerce_optional_float(raw.get("max_ratio"))
    min_ratio = _coerce_optional_float(raw.get("min_ratio"))
    direction = raw.get("direction")
    tolerance = _coerce_optional_float(raw.get("tolerance"))

    if direction and tolerance is not None:
        if direction == "lower":
            max_delta = tolerance
        elif direction == "higher":
            min_delta = -tolerance
        else:
            raise ValueError(f"Unknown direction '{direction}' in threshold spec")

    if all(value is None for value in (max_delta, min_delta, max_ratio, min_ratio)):
        return None

    return MetricThreshold(
        max_delta=max_delta,
        min_delta=min_delta,
        max_ratio=max_ratio,
        min_ratio=min_ratio,
    )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_ratio(delta: float, baseline_value: float) -> float:
    if not math.isfinite(baseline_value) or baseline_value == 0.0:
        return float("nan")
    return delta / abs(baseline_value)


def _evaluate_metric_status(
    baseline_value: float,
    threshold_spec: MetricThreshold | None,
    delta: float,
    ratio: float,
) -> str:
    if threshold_spec is None:
        return "pass"
    if not math.isfinite(baseline_value):
        return "missing_baseline"
    regressions = []
    if threshold_spec.max_delta is not None and delta > threshold_spec.max_delta:
        regressions.append("max_delta")
    if threshold_spec.min_delta is not None and delta < threshold_spec.min_delta:
        regressions.append("min_delta")
    if threshold_spec.max_ratio is not None:
        if math.isnan(ratio) or ratio > threshold_spec.max_ratio:
            regressions.append("max_ratio")
    if threshold_spec.min_ratio is not None:
        if math.isnan(ratio) or ratio < threshold_spec.min_ratio:
            regressions.append("min_ratio")
    return "regressed" if regressions else "pass"


def _metric_payload(
    baseline_value: float,
    current_value: float,
    delta: float,
    ratio: float,
    threshold_spec: MetricThreshold | None,
    status: str,
) -> dict[str, float | str]:
    payload: dict[str, float | str] = {
        "baseline": baseline_value,
        "current": current_value,
        "delta": delta,
        "ratio": ratio,
        "status": status,
    }
    if threshold_spec is None:
        payload["thresholds"] = "none"
    else:
        payload.update({f"threshold_{key}": value for key, value in threshold_spec.as_dict().items()})
    return payload
