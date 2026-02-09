"""Benchmark the governance runner overhead with lightweight commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from benchmark_governance import (
    BenchmarkGovernanceConfig,
    BenchmarkSpec,
    BudgetThresholds,
    execute_governance,
)
from deterministic_integrity import stable_hash

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GovernanceBenchmarkResult:
    run_count: int
    duration_s: float
    digest: str


def _build_config(run_count: int) -> BenchmarkGovernanceConfig:
    benchmarks = []
    for idx in range(run_count):
        benchmarks.append(
            BenchmarkSpec(
                name=f"noop_{idx}",
                command=("python", "-c", "pass"),
                budgets=BudgetThresholds(runtime_budget_s=5.0, memory_budget_bytes=50_000_000),
                baseline_key=f"noop_{idx}",
                require_baseline=False,
                thresholds={},
            )
        )
    return BenchmarkGovernanceConfig(
        benchmarks=tuple(benchmarks),
        output_path=Path("reports/governance_benchmark_summary.json"),
        baseline_store=Path("reports/governance_benchmark_baselines.json"),
        write_baseline=False,
        max_workers=1,
        fail_fast=False,
        config_hash=stable_hash({"run_count": run_count}),
    )


def run_benchmark(run_count: int = 10) -> GovernanceBenchmarkResult:
    config = _build_config(run_count)
    start = perf_counter()
    summary = __import__("asyncio").run(execute_governance(config))
    duration_s = perf_counter() - start
    digest = stable_hash({"status": summary["status"], "runs": run_count})
    return GovernanceBenchmarkResult(run_count=run_count, duration_s=duration_s, digest=digest)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run_benchmark()
    LOGGER.info(
        "governance_benchmark",
        extra={
            "runs": result.run_count,
            "duration_s": result.duration_s,
            "digest": result.digest,
        },
    )


if __name__ == "__main__":
    main()
