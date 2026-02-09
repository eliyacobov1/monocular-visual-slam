#!/usr/bin/env python3
"""Benchmark governance runner with runtime/memory budgets and regression gating."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import resource
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CompletedProcess, run
from time import perf_counter
from typing import Any, Sequence

from deterministic_integrity import stable_hash
from deterministic_registry import hash_config_path
from regression_baselines import (
    compare_metrics,
    load_baseline_store,
    upsert_baseline,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetThresholds:
    runtime_budget_s: float
    memory_budget_bytes: int


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    command: tuple[str, ...]
    budgets: BudgetThresholds
    baseline_key: str
    require_baseline: bool
    thresholds: dict[str, float | dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkGovernanceConfig:
    benchmarks: tuple[BenchmarkSpec, ...]
    output_path: Path
    baseline_store: Path
    write_baseline: bool
    max_workers: int
    fail_fast: bool
    config_hash: str


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


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an int") from exc


def _validate_non_negative(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _parse_thresholds(raw: Any) -> dict[str, float | dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("thresholds must be a mapping")
    parsed: dict[str, float | dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            parsed[str(key)] = dict(value)
        else:
            parsed[str(key)] = float(value)
    return parsed


def _parse_budgets(raw: dict[str, Any], field_prefix: str) -> BudgetThresholds:
    runtime_budget_s = _coerce_float(raw.get("runtime_budget_s", 0.0), f"{field_prefix}.runtime_budget_s")
    memory_budget_bytes = _coerce_int(
        raw.get("memory_budget_bytes", 0),
        f"{field_prefix}.memory_budget_bytes",
    )
    _validate_non_negative(runtime_budget_s, f"{field_prefix}.runtime_budget_s")
    _validate_non_negative(memory_budget_bytes, f"{field_prefix}.memory_budget_bytes")
    return BudgetThresholds(runtime_budget_s=runtime_budget_s, memory_budget_bytes=memory_budget_bytes)


def load_governance_config(config_path: Path) -> BenchmarkGovernanceConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    benchmarks_raw = raw.get("benchmarks")
    if not benchmarks_raw:
        raise ValueError("Governance config must include non-empty 'benchmarks'")

    base_dir = config_path.parent
    output_path = _resolve_path(raw.get("output_path", "reports/benchmark_governance_summary.json"), base_dir)
    baseline_store = _resolve_path(raw.get("baseline_store", "reports/benchmark_governance_baselines.json"), base_dir)
    write_baseline = bool(raw.get("write_baseline", False))
    max_workers = int(raw.get("max_workers", 1))
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    fail_fast = bool(raw.get("fail_fast", True))
    default_thresholds = _parse_thresholds(raw.get("thresholds"))

    benchmarks: list[BenchmarkSpec] = []
    for idx, benchmark in enumerate(benchmarks_raw):
        if not isinstance(benchmark, dict):
            raise ValueError("Each benchmark entry must be a mapping")
        name = str(benchmark.get("name", f"benchmark_{idx}"))
        command_raw = benchmark.get("command")
        if not isinstance(command_raw, Sequence) or isinstance(command_raw, (str, bytes)):
            raise ValueError(f"command for benchmark '{name}' must be a list of strings")
        command = tuple(str(part) for part in command_raw)
        if not command:
            raise ValueError(f"command for benchmark '{name}' must be non-empty")
        budgets = _parse_budgets(benchmark, f"benchmarks[{name}]")
        thresholds = _parse_thresholds(benchmark.get("thresholds", default_thresholds))
        baseline_key = str(benchmark.get("baseline_key", name))
        require_baseline = bool(benchmark.get("require_baseline", True))
        benchmarks.append(
            BenchmarkSpec(
                name=name,
                command=command,
                budgets=budgets,
                baseline_key=baseline_key,
                require_baseline=require_baseline,
                thresholds=thresholds,
            )
        )

    return BenchmarkGovernanceConfig(
        benchmarks=tuple(benchmarks),
        output_path=output_path,
        baseline_store=baseline_store,
        write_baseline=write_baseline,
        max_workers=max_workers,
        fail_fast=fail_fast,
        config_hash=hash_config_path(config_path),
    )


def _rss_to_bytes(rss: int) -> int:
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def _run_command(command: Sequence[str]) -> tuple[CompletedProcess[str], float, int]:
    start_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start_time = perf_counter()
    result = run(command, capture_output=True, text=True, check=False)
    duration_s = perf_counter() - start_time
    end_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    memory_delta = max(0, _rss_to_bytes(end_rss - start_rss))
    return result, duration_s, memory_delta


def _budget_status(runtime_s: float, memory_bytes: int, budgets: BudgetThresholds) -> dict[str, str]:
    return {
        "runtime": "over" if runtime_s > budgets.runtime_budget_s else "pass",
        "memory": "over" if memory_bytes > budgets.memory_budget_bytes else "pass",
    }


def _budget_exceeded(budget_status: dict[str, str]) -> bool:
    return any(value == "over" for value in budget_status.values())


def _result_status(
    *,
    budget_status: dict[str, str],
    baseline_status: str | None,
    require_baseline: bool,
    baseline_missing: bool,
    return_code: int,
) -> str:
    if return_code != 0:
        return "error"
    if _budget_exceeded(budget_status):
        return "budget_exceeded"
    if require_baseline and baseline_missing:
        return "missing_baseline"
    if baseline_status == "regressed":
        return "regressed"
    return "pass"


def _aggregate_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    runtimes = [float(result["metrics"]["runtime_s"]) for result in results]
    memories = [float(result["metrics"]["memory_delta_bytes"]) for result in results]
    if not runtimes:
        return {
            "run_count": 0.0,
            "total_runtime_s": 0.0,
            "total_memory_delta_bytes": 0.0,
        }
    return {
        "run_count": float(len(runtimes)),
        "total_runtime_s": float(sum(runtimes)),
        "total_memory_delta_bytes": float(sum(memories)),
        "mean_runtime_s": float(sum(runtimes) / len(runtimes)),
        "mean_memory_delta_bytes": float(sum(memories) / len(memories)),
        "max_runtime_s": float(max(runtimes)),
        "max_memory_delta_bytes": float(max(memories)),
    }


def _output_digest(stdout: str, stderr: str, return_code: int) -> str:
    payload = {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
    }
    return stable_hash(payload)


def _benchmark_config_hash(spec: BenchmarkSpec) -> str:
    return stable_hash(
        {
            "name": spec.name,
            "command": list(spec.command),
            "budgets": {
                "runtime_budget_s": spec.budgets.runtime_budget_s,
                "memory_budget_bytes": spec.budgets.memory_budget_bytes,
            },
            "baseline_key": spec.baseline_key,
            "require_baseline": spec.require_baseline,
            "thresholds": spec.thresholds,
        }
    )


def _load_baseline_entry(store_path: Path, key: str) -> dict[str, Any] | None:
    store = load_baseline_store(store_path)
    baselines = store.get("baselines", {})
    baseline = baselines.get(key)
    if baseline is None:
        return None
    if not isinstance(baseline, dict):
        return None
    return baseline


async def _run_single(
    spec: BenchmarkSpec,
    config: BenchmarkGovernanceConfig,
) -> dict[str, Any]:
    LOGGER.info("Benchmark governance run started", extra={"benchmark": spec.name})
    result, runtime_s, memory_delta = await asyncio.to_thread(_run_command, spec.command)
    budget_status = _budget_status(runtime_s, memory_delta, spec.budgets)
    metrics = {
        "runtime_s": float(runtime_s),
        "memory_delta_bytes": int(memory_delta),
    }
    baseline_entry = _load_baseline_entry(config.baseline_store, spec.baseline_key)
    baseline_missing = baseline_entry is None
    baseline_comparison = None
    if baseline_entry is not None:
        baseline_comparison = compare_metrics(
            spec.baseline_key,
            metrics,
            baseline_entry,
            spec.thresholds,
        )

    if config.write_baseline:
        upsert_baseline(
            config.baseline_store,
            spec.baseline_key,
            metrics,
            _benchmark_config_hash(spec),
            metadata={
                "command": list(spec.command),
                "budgets": {
                    "runtime_budget_s": spec.budgets.runtime_budget_s,
                    "memory_budget_bytes": spec.budgets.memory_budget_bytes,
                },
            },
        )

    status = _result_status(
        budget_status=budget_status,
        baseline_status=baseline_comparison.status if baseline_comparison else None,
        require_baseline=spec.require_baseline,
        baseline_missing=baseline_missing,
        return_code=result.returncode,
    )

    run_payload = {
        "name": spec.name,
        "status": status,
        "command": list(spec.command),
        "return_code": result.returncode,
        "metrics": metrics,
        "budgets": {
            "runtime_budget_s": spec.budgets.runtime_budget_s,
            "memory_budget_bytes": spec.budgets.memory_budget_bytes,
        },
        "budget_status": budget_status,
        "baseline_key": spec.baseline_key,
        "baseline_comparison": baseline_comparison.__dict__ if baseline_comparison else None,
        "output_digest": _output_digest(result.stdout, result.stderr, result.returncode),
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "config_hash": _benchmark_config_hash(spec),
    }
    if result.stdout:
        run_payload["stdout_preview"] = result.stdout[:500]
    if result.stderr:
        run_payload["stderr_preview"] = result.stderr[:500]

    LOGGER.info(
        "Benchmark governance run finished",
        extra={"benchmark": spec.name, "status": status},
    )
    return run_payload


async def execute_governance(config: BenchmarkGovernanceConfig) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, config.max_workers))
    results: list[dict[str, Any]] = []
    start_time = _timestamp()

    async def _bounded_run(spec: BenchmarkSpec) -> dict[str, Any]:
        async with semaphore:
            return await _run_single(spec, config)

    tasks = [asyncio.create_task(_bounded_run(spec)) for spec in config.benchmarks]
    pending = set(tasks)
    fail_fast_triggered = False

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for completed in done:
            result = completed.result()
            results.append(result)
            if config.fail_fast and result["status"] not in {"pass"}:
                fail_fast_triggered = True
                for task in pending:
                    task.cancel()
                pending.clear()
                break

    if fail_fast_triggered:
        LOGGER.warning("Fail-fast triggered: pending benchmark governance runs cancelled")

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
        "benchmark_metrics": _aggregate_stats(results),
        "baseline_store": str(config.baseline_store),
        "config_hash": config.config_hash,
        "runs": results,
    }
    summary["digest"] = stable_hash(summary, exclude_keys=("started_at", "finished_at"))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Benchmark governance summary written to %s", config.output_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark governance with budgets and regression gates")
    parser.add_argument("--config", required=True, help="Path to benchmark governance JSON config")
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
    config = load_governance_config(Path(args.config))
    summary = asyncio.run(execute_governance(config))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
