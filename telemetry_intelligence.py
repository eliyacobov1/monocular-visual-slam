"""Telemetry intelligence layer with streaming summaries and drift analysis."""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from data_persistence import P2Quantile, iter_json_array_items

LOGGER = logging.getLogger(__name__)


class TelemetryEventLike(Protocol):
    name: str
    duration_s: float


@dataclass(frozen=True)
class TelemetryDriftThresholds:
    """Thresholds for drift evaluation between telemetry summaries."""

    relative_increase: float = 0.1
    absolute_increase_s: float = 0.01
    metrics: tuple[str, ...] = ("mean_duration_s", "p95_duration_s")

    def __post_init__(self) -> None:
        if self.relative_increase < 0:
            raise ValueError("relative_increase must be non-negative")
        if self.absolute_increase_s < 0:
            raise ValueError("absolute_increase_s must be non-negative")
        if not self.metrics:
            raise ValueError("metrics must be non-empty")


@dataclass(frozen=True)
class DriftFinding:
    """Single drift finding for a stage metric."""

    stage: str
    metric: str
    current: float
    baseline: float
    delta: float
    relative: float | None
    threshold: float
    status: str


class DriftPolicy(Protocol):
    """Protocol for drift policies."""

    def evaluate(
        self,
        stage: str,
        metric: str,
        current: float,
        baseline: float,
    ) -> DriftFinding | None:
        ...


@dataclass(frozen=True)
class RelativeIncreasePolicy:
    """Relative drift policy for percentage increases."""

    threshold: float

    def evaluate(
        self,
        stage: str,
        metric: str,
        current: float,
        baseline: float,
    ) -> DriftFinding | None:
        if baseline <= 0:
            relative = None
        else:
            relative = (current - baseline) / baseline
        delta = current - baseline
        status = "pass"
        threshold = self.threshold
        if relative is None:
            if delta > 0:
                status = "warn"
        elif relative > threshold:
            status = "fail"
        return DriftFinding(
            stage=stage,
            metric=metric,
            current=current,
            baseline=baseline,
            delta=delta,
            relative=relative,
            threshold=threshold,
            status=status,
        )


@dataclass(frozen=True)
class AbsoluteIncreasePolicy:
    """Absolute drift policy for raw increases in seconds."""

    threshold_s: float

    def evaluate(
        self,
        stage: str,
        metric: str,
        current: float,
        baseline: float,
    ) -> DriftFinding | None:
        delta = current - baseline
        status = "pass"
        if delta > self.threshold_s:
            status = "fail"
        return DriftFinding(
            stage=stage,
            metric=metric,
            current=current,
            baseline=baseline,
            delta=delta,
            relative=None,
            threshold=self.threshold_s,
            status=status,
        )


class TelemetryDriftEvaluator:
    """Evaluate telemetry drift using policy composition."""

    def __init__(self, policies: Iterable[DriftPolicy]) -> None:
        self._policies = tuple(policies)
        if not self._policies:
            raise ValueError("At least one drift policy is required")

    def evaluate(
        self,
        current: Mapping[str, Any],
        baseline: Mapping[str, Any],
        *,
        thresholds: TelemetryDriftThresholds,
    ) -> dict[str, Any]:
        per_stage_current = current.get("per_stage", {})
        per_stage_baseline = baseline.get("per_stage", {})
        results: dict[str, Any] = {}
        overall_status = "pass"

        for stage, baseline_stats in per_stage_baseline.items():
            current_stats = per_stage_current.get(stage)
            if current_stats is None:
                continue
            stage_findings: list[dict[str, Any]] = []
            stage_status = "pass"
            for metric in thresholds.metrics:
                baseline_value = float(baseline_stats.get(metric, 0.0))
                current_value = float(current_stats.get(metric, 0.0))
                for policy in self._policies:
                    finding = policy.evaluate(stage, metric, current_value, baseline_value)
                    if finding is None:
                        continue
                    stage_findings.append(
                        {
                            "metric": finding.metric,
                            "current": finding.current,
                            "baseline": finding.baseline,
                            "delta": finding.delta,
                            "relative": finding.relative,
                            "threshold": finding.threshold,
                            "status": finding.status,
                            "policy": type(policy).__name__,
                        }
                    )
                    if finding.status == "fail":
                        stage_status = "fail"
                        overall_status = "fail"
                    elif finding.status == "warn" and stage_status != "fail":
                        stage_status = "warn"
                        if overall_status == "pass":
                            overall_status = "warn"
            if stage_findings:
                results[stage] = {
                    "status": stage_status,
                    "findings": stage_findings,
                }

        return {
            "status": overall_status,
            "per_stage": results,
        }


class _StageStats:
    def __init__(self, quantiles: Iterable[float]) -> None:
        self.count = 0
        self.total_duration = 0.0
        self.min_duration = float("inf")
        self.max_duration = float("-inf")
        self.mean = 0.0
        self.m2 = 0.0
        self._quantiles = {float(q): P2Quantile(float(q)) for q in quantiles}

    def update(self, duration: float) -> None:
        duration = float(duration)
        self.count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        delta = duration - self.mean
        self.mean += delta / self.count
        delta2 = duration - self.mean
        self.m2 += delta * delta2
        for quantile in self._quantiles.values():
            quantile.update(duration)

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "total_duration_s": 0.0,
                "mean_duration_s": 0.0,
                "min_duration_s": 0.0,
                "max_duration_s": 0.0,
                "variance_duration_s": 0.0,
                "stdev_duration_s": 0.0,
            }
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        summary: dict[str, float | int] = {
            "count": int(self.count),
            "total_duration_s": float(self.total_duration),
            "mean_duration_s": float(self.mean),
            "min_duration_s": float(self.min_duration),
            "max_duration_s": float(self.max_duration),
            "variance_duration_s": float(variance),
            "stdev_duration_s": float(math.sqrt(variance)),
        }
        for quantile, estimator in self._quantiles.items():
            key = f"p{int(round(quantile * 100))}_duration_s"
            summary[key] = float(estimator.value())
        return summary


class TelemetryDigest:
    """Thread-safe telemetry digest for streaming ingestion."""

    def __init__(
        self,
        *,
        quantiles: Iterable[float] = (0.5, 0.9, 0.95, 0.99),
        thread_safe: bool = False,
    ) -> None:
        self._quantiles = tuple(float(q) for q in quantiles)
        self._lock = threading.Lock() if thread_safe else None
        self.total_events = 0
        self.total_duration = 0.0
        self.per_stage: dict[str, _StageStats] = {}

    def update(self, name: str, duration: float) -> None:
        if self._lock is None:
            self._update_locked(name, duration)
        else:
            with self._lock:
                self._update_locked(name, duration)

    def ingest_event(self, event: Mapping[str, Any] | TelemetryEventLike) -> None:
        name = str(getattr(event, "name", None) or event.get("name") or "unknown")
        duration = float(getattr(event, "duration_s", None) or event.get("duration_s", 0.0))
        self.update(name, duration)

    def ingest_events(self, events: Iterable[Mapping[str, Any] | TelemetryEventLike]) -> None:
        for event in events:
            self.ingest_event(event)

    def summarize(self) -> dict[str, Any]:
        if self._lock is None:
            return self._summarize_locked()
        with self._lock:
            return self._summarize_locked()

    def _update_locked(self, name: str, duration: float) -> None:
        self.total_events += 1
        self.total_duration += float(duration)
        stats = self.per_stage.get(name)
        if stats is None:
            stats = _StageStats(self._quantiles)
            self.per_stage[name] = stats
        stats.update(duration)

    def _summarize_locked(self) -> dict[str, Any]:
        per_stage = {name: stats.summary() for name, stats in self.per_stage.items()}
        mean_duration = self.total_duration / self.total_events if self.total_events else 0.0
        return {
            "event_count": int(self.total_events),
            "total_duration_s": float(self.total_duration),
            "mean_duration_s": float(mean_duration),
            "per_stage": per_stage,
        }


def _normalize_metric_name(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "unknown"


def load_telemetry_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"Telemetry payload at {path} is missing 'events' list")
    return events


def load_telemetry_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Telemetry summary at {path} is invalid")
    return payload


def write_telemetry_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize_telemetry_streaming(
    path: Path,
    *,
    global_digest: TelemetryDigest | None = None,
) -> dict[str, Any]:
    local_digest = TelemetryDigest()
    for event in iter_json_array_items(path, "events"):
        local_digest.ingest_event(event)
        if global_digest is not None:
            global_digest.ingest_event(event)
    return local_digest.summarize()


def summarize_telemetry_events(
    events: Iterable[Mapping[str, Any]],
    *,
    global_digest: TelemetryDigest | None = None,
) -> dict[str, Any]:
    local_digest = TelemetryDigest()
    local_digest.ingest_events(events)
    if global_digest is not None:
        global_digest.ingest_events(events)
    return local_digest.summarize()


def telemetry_metrics_from_summary(summary: Mapping[str, Any]) -> dict[str, float]:
    if not summary:
        return {}
    metrics: dict[str, float] = {
        "telemetry_event_count": float(summary.get("event_count", 0.0)),
        "telemetry_total_duration_s": float(summary.get("total_duration_s", 0.0)),
        "telemetry_mean_duration_s": float(summary.get("mean_duration_s", 0.0)),
    }
    per_stage = summary.get("per_stage", {})
    if isinstance(per_stage, dict):
        for stage, stats in per_stage.items():
            prefix = f"telemetry_stage_{_normalize_metric_name(str(stage))}"
            metrics[f"{prefix}_count"] = float(stats.get("count", 0.0))
            metrics[f"{prefix}_mean_duration_s"] = float(stats.get("mean_duration_s", 0.0))
            metrics[f"{prefix}_p50_duration_s"] = float(stats.get("p50_duration_s", 0.0))
            metrics[f"{prefix}_p90_duration_s"] = float(stats.get("p90_duration_s", 0.0))
            metrics[f"{prefix}_p95_duration_s"] = float(stats.get("p95_duration_s", 0.0))
            metrics[f"{prefix}_p99_duration_s"] = float(stats.get("p99_duration_s", 0.0))
            metrics[f"{prefix}_max_duration_s"] = float(stats.get("max_duration_s", 0.0))
            metrics[f"{prefix}_stdev_duration_s"] = float(stats.get("stdev_duration_s", 0.0))
    return metrics


def compare_telemetry_summaries(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    thresholds: TelemetryDriftThresholds,
) -> dict[str, Any]:
    evaluator = TelemetryDriftEvaluator(
        policies=[
            RelativeIncreasePolicy(thresholds.relative_increase),
            AbsoluteIncreasePolicy(thresholds.absolute_increase_s),
        ]
    )
    return evaluator.evaluate(current, baseline, thresholds=thresholds)


__all__ = [
    "AbsoluteIncreasePolicy",
    "DriftFinding",
    "DriftPolicy",
    "RelativeIncreasePolicy",
    "TelemetryDigest",
    "TelemetryDriftEvaluator",
    "TelemetryDriftThresholds",
    "TelemetryEventLike",
    "compare_telemetry_summaries",
    "load_telemetry_events",
    "load_telemetry_summary",
    "summarize_telemetry_events",
    "summarize_telemetry_streaming",
    "telemetry_metrics_from_summary",
    "write_telemetry_summary",
]
