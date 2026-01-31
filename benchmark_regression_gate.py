#!/usr/bin/env python3
"""Benchmark regression gate for evaluation configs."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_harness import load_config as load_eval_config
from evaluation_harness import run_evaluation

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateRun:
    name: str
    config_path: Path
    require_baseline: bool


@dataclass(frozen=True)
class GateConfig:
    runs: tuple[GateRun, ...]
    output_path: Path
    max_workers: int
    fail_fast: bool


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def load_gate_config(config_path: Path) -> GateConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    runs_raw = raw.get("runs")
    if not runs_raw:
        raise ValueError("Gate config must include non-empty 'runs'")

    base_dir = config_path.parent
    runs = tuple(
        GateRun(
            name=str(run.get("name", Path(run["config_path"]).stem)),
            config_path=_resolve_path(run["config_path"], base_dir),
            require_baseline=bool(run.get("require_baseline", True)),
        )
        for run in runs_raw
    )

    output_path = _resolve_path(raw.get("output_path", "reports/regression_gate_summary.json"), base_dir)
    max_workers = int(raw.get("max_workers", 1))
    fail_fast = bool(raw.get("fail_fast", True))
    return GateConfig(runs=runs, output_path=output_path, max_workers=max_workers, fail_fast=fail_fast)


async def _run_single(run: GateRun) -> dict[str, Any]:
    LOGGER.info("Gate run started", extra={"run": run.name, "config": str(run.config_path)})

    eval_config = await asyncio.to_thread(load_eval_config, run.config_path)
    summary = await asyncio.to_thread(run_evaluation, eval_config)

    baseline_comparison = summary.get("baseline_comparison")
    if baseline_comparison is None and run.require_baseline:
        status = "missing_baseline"
    elif baseline_comparison and baseline_comparison.get("status") == "regressed":
        status = "regressed"
    else:
        status = "pass"

    result = {
        "name": run.name,
        "status": status,
        "run_dir": summary.get("run_dir"),
        "config_path": str(run.config_path),
        "config_hash": summary.get("config_hash"),
        "aggregate_metrics": summary.get("aggregate_metrics", {}),
        "baseline_comparison": baseline_comparison,
    }
    LOGGER.info("Gate run finished", extra={"run": run.name, "status": status})
    return result


async def execute_gate(config: GateConfig) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, config.max_workers))
    results: list[dict[str, Any]] = []
    start_time = _timestamp()

    async def _bounded_run(run: GateRun) -> dict[str, Any]:
        async with semaphore:
            return await _run_single(run)

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
        LOGGER.warning("Fail-fast triggered: pending gate runs cancelled")

    status = "pass" if all(result["status"] == "pass" for result in results) else "regressed"
    summary = {
        "status": status,
        "started_at": start_time,
        "finished_at": _timestamp(),
        "fail_fast": config.fail_fast,
        "runs": results,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Regression gate summary written to %s", config.output_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regression gate for evaluation configs")
    parser.add_argument("--config", required=True, help="Path to regression gate JSON config")
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
    config = load_gate_config(Path(args.config))
    summary = asyncio.run(execute_gate(config))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
