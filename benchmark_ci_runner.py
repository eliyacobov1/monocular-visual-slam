#!/usr/bin/env python3
"""CI benchmark runner with severity scoring for regression summaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from evaluation_harness import load_config as load_eval_config
from evaluation_harness import run_evaluation

LOGGER = logging.getLogger(__name__)

EPSILON = 1e-12


@dataclass(frozen=True)
class BenchmarkRun:
    name: str
    config_path: Path
    require_baseline: bool


@dataclass(frozen=True)
class SeverityWeights:
    default_weight: float
    metric_weights: dict[str, float]
    telemetry_weights: dict[str, float]

    def weight_for(self, metric: str, telemetry: bool) -> float:
        weights = self.telemetry_weights if telemetry else self.metric_weights
        return float(weights.get(metric, self.default_weight))


@dataclass(frozen=True)
class BenchmarkConfig:
    runs: tuple[BenchmarkRun, ...]
    output_path: Path
    max_workers: int
    fail_fast: bool
    severity: SeverityWeights


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float") from exc


def _validate_non_negative(value: float, field_name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _parse_weights(raw: dict[str, Any], field_name: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key, value in raw.items():
        weight = _coerce_float(value, f"{field_name}.{key}")
        weights[str(key)] = _validate_non_negative(weight, f"{field_name}.{key}")
    return weights


def _parse_severity_config(raw: dict[str, Any] | None) -> SeverityWeights:
    raw = raw or {}
    default_weight = _coerce_float(raw.get("default_weight", 1.0), "severity.default_weight")
    default_weight = _validate_non_negative(default_weight, "severity.default_weight")
    metric_weights = _parse_weights(raw.get("metric_weights", {}), "severity.metric_weights")
    telemetry_weights = _parse_weights(raw.get("telemetry_weights", {}), "severity.telemetry_weights")
    return SeverityWeights(
        default_weight=default_weight,
        metric_weights=metric_weights,
        telemetry_weights=telemetry_weights,
    )


def load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    runs_raw = raw.get("runs")
    if not runs_raw:
        raise ValueError("Benchmark config must include non-empty 'runs'")

    base_dir = config_path.parent
    runs = tuple(
        BenchmarkRun(
            name=str(run.get("name", Path(run["config_path"]).stem)),
            config_path=_resolve_path(run["config_path"], base_dir),
            require_baseline=bool(run.get("require_baseline", True)),
        )
        for run in runs_raw
    )

    output_path = _resolve_path(raw.get("output_path", "reports/ci_benchmark_summary.json"), base_dir)
    max_workers = int(raw.get("max_workers", 1))
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    fail_fast = bool(raw.get("fail_fast", True))
    severity = _parse_severity_config(raw.get("severity"))
    return BenchmarkConfig(
        runs=runs,
        output_path=output_path,
        max_workers=max_workers,
        fail_fast=fail_fast,
        severity=severity,
    )


def _safe_penalty(value: float, threshold: float, direction: str) -> float:
    if not math.isfinite(value):
        return 1.0
    denom = max(abs(threshold), EPSILON)
    if direction == "max":
        if value > threshold:
            return (value - threshold) / denom
        return 0.0
    if direction == "min":
        if value < threshold:
            return (threshold - value) / denom
        return 0.0
    raise ValueError(f"Unknown penalty direction '{direction}'")


def _metric_severity(payload: dict[str, Any]) -> float:
    if payload.get("status") == "missing_baseline":
        return 1.0

    delta = float(payload.get("delta", 0.0))
    ratio = float(payload.get("ratio", float("nan")))
    penalties: list[float] = []

    max_delta = payload.get("threshold_max_delta")
    if max_delta is not None:
        penalties.append(_safe_penalty(delta, float(max_delta), "max"))

    min_delta = payload.get("threshold_min_delta")
    if min_delta is not None:
        penalties.append(_safe_penalty(delta, float(min_delta), "min"))

    max_ratio = payload.get("threshold_max_ratio")
    if max_ratio is not None:
        penalties.append(_safe_penalty(ratio, float(max_ratio), "max"))

    min_ratio = payload.get("threshold_min_ratio")
    if min_ratio is not None:
        penalties.append(_safe_penalty(ratio, float(min_ratio), "min"))

    if not penalties:
        return 0.0
    return math.sqrt(sum(penalty * penalty for penalty in penalties))


def _severity_penalty(
    comparison: dict[str, Any] | None,
    weights: SeverityWeights,
    telemetry: bool,
) -> dict[str, Any]:
    if not comparison:
        return {
            "penalty_sum": 0.0,
            "score": 0.0,
            "total_weight": 0.0,
            "contributions": {},
        }

    per_metric = comparison.get("per_metric", {})
    penalty_sum = 0.0
    total_weight = 0.0
    contributions: dict[str, float] = {}

    for metric, payload in per_metric.items():
        severity = _metric_severity(payload)
        weight = weights.weight_for(str(metric), telemetry)
        total_weight += weight
        penalty_sum += weight * severity * severity
        contributions[str(metric)] = severity

    score = math.sqrt(penalty_sum)
    return {
        "penalty_sum": penalty_sum,
        "score": score,
        "total_weight": total_weight,
        "contributions": contributions,
    }


def _compute_severity(
    baseline_comparison: dict[str, Any] | None,
    telemetry_comparison: dict[str, Any] | None,
    weights: SeverityWeights,
) -> dict[str, Any]:
    metrics_penalty = _severity_penalty(baseline_comparison, weights, telemetry=False)
    telemetry_penalty = _severity_penalty(telemetry_comparison, weights, telemetry=True)

    total_penalty = metrics_penalty["penalty_sum"] + telemetry_penalty["penalty_sum"]
    score = math.sqrt(total_penalty)
    return {
        "score": score,
        "metrics": metrics_penalty,
        "telemetry": telemetry_penalty,
    }


def _determine_status(
    run: BenchmarkRun,
    baseline_comparison: dict[str, Any] | None,
    telemetry_comparison: dict[str, Any] | None,
    baseline_key: str | None,
    telemetry_baseline_key: str | None,
) -> str:
    if run.require_baseline and baseline_key and baseline_comparison is None:
        return "missing_baseline"
    if run.require_baseline and telemetry_baseline_key and telemetry_comparison is None:
        return "missing_baseline"
    if baseline_comparison and baseline_comparison.get("status") == "regressed":
        return "regressed"
    if telemetry_comparison and telemetry_comparison.get("status") == "regressed":
        return "regressed"
    return "pass"


def _benchmark_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    runtimes = [float(result.get("runtime_s", 0.0)) for result in results]
    if not runtimes:
        return {"run_count": 0.0, "total_runtime_s": 0.0}
    total = float(sum(runtimes))
    return {
        "run_count": float(len(runtimes)),
        "total_runtime_s": total,
        "mean_runtime_s": total / len(runtimes),
        "max_runtime_s": float(max(runtimes)),
        "min_runtime_s": float(min(runtimes)),
    }


async def _run_single(run: BenchmarkRun, severity: SeverityWeights) -> dict[str, Any]:
    start = perf_counter()
    try:
        LOGGER.info("Benchmark run started", extra={"run": run.name, "config": str(run.config_path)})
        eval_config = await asyncio.to_thread(load_eval_config, run.config_path)
        summary = await asyncio.to_thread(run_evaluation, eval_config)
        baseline_comparison = summary.get("baseline_comparison")
        telemetry_comparison = summary.get("telemetry_baseline_comparison")
        baseline_key = summary.get("baseline_key")
        telemetry_baseline_key = summary.get("telemetry_baseline_key")
        status = _determine_status(
            run,
            baseline_comparison,
            telemetry_comparison,
            baseline_key,
            telemetry_baseline_key,
        )
        severity_payload = _compute_severity(baseline_comparison, telemetry_comparison, severity)
        error_info = None
    except Exception as exc:  # noqa: BLE001 - boundary for CI runs
        LOGGER.exception("Benchmark run failed", extra={"run": run.name})
        status = "error"
        summary = {}
        baseline_comparison = None
        telemetry_comparison = None
        severity_payload = {"score": 0.0, "metrics": {}, "telemetry": {}}
        error_info = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }

    duration = perf_counter() - start
    result = {
        "name": run.name,
        "status": status,
        "run_dir": summary.get("run_dir"),
        "config_path": str(run.config_path),
        "config_hash": summary.get("config_hash"),
        "aggregate_metrics": summary.get("aggregate_metrics", {}),
        "telemetry_metrics": summary.get("telemetry_metrics", {}),
        "baseline_comparison": baseline_comparison,
        "telemetry_baseline_comparison": telemetry_comparison,
        "severity": severity_payload,
        "runtime_s": float(duration),
    }
    if error_info:
        result["error"] = error_info
    LOGGER.info("Benchmark run finished", extra={"run": run.name, "status": status})
    return result


async def execute_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, config.max_workers))
    results: list[dict[str, Any]] = []
    start_time = _timestamp()

    async def _bounded_run(run: BenchmarkRun) -> dict[str, Any]:
        async with semaphore:
            return await _run_single(run, config.severity)

    tasks = [asyncio.create_task(_bounded_run(run)) for run in config.runs]
    pending = set(tasks)
    fail_fast_triggered = False

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for completed in done:
            result = completed.result()
            results.append(result)
            if config.fail_fast and result["status"] != "pass":
                fail_fast_triggered = True
                for task in pending:
                    task.cancel()
                pending.clear()
                break

    if fail_fast_triggered:
        LOGGER.warning("Fail-fast triggered: pending benchmark runs cancelled")

    if any(result["status"] == "error" for result in results):
        status = "error"
    elif all(result["status"] == "pass" for result in results):
        status = "pass"
    else:
        status = "regressed"

    summary = {
        "status": status,
        "started_at": start_time,
        "finished_at": _timestamp(),
        "fail_fast": config.fail_fast,
        "benchmark_metrics": _benchmark_stats(results),
        "runs": results,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("CI benchmark summary written to %s", config.output_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CI benchmark suite")
    parser.add_argument("--config", required=True, help="Path to CI benchmark JSON config")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_benchmark_config(Path(args.config))
    summary = asyncio.run(execute_benchmark(config))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
