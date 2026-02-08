"""Benchmark deterministic failure injection and supervisor stress."""

from __future__ import annotations

import logging
import time
import tracemalloc

from control_plane_supervisor import ControlPlaneSupervisor, ControlPlaneSupervisorConfig
from failure_injection import FailureInjectionChaosHarness, FailureInjectionConfig, FailureInjector

LOGGER = logging.getLogger(__name__)


def run_benchmark(
    *,
    stage_count: int = 16,
    steps: int = 50,
    worker_count: int = 4,
) -> None:
    stages = [f"stage_{idx}" for idx in range(stage_count)]
    config = FailureInjectionConfig(
        seed=42,
        timeout_probability=0.1,
        dropped_frame_probability=0.05,
        solver_stall_probability=0.03,
        step_duration_s=0.05,
    )
    injector = FailureInjector(config)
    plan = injector.build_plan(stages, steps=steps)
    chaos = FailureInjectionChaosHarness(plan, worker_count=worker_count)
    supervisor = ControlPlaneSupervisor(
        chaos.harness.adapters(),
        config=ControlPlaneSupervisorConfig(stage_dependencies={}),
    )

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    digest = chaos.run(steps=steps)
    report = supervisor.update()
    duration_s = time.perf_counter() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
    total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    LOGGER.info(
        "failure_injection_benchmark stages=%s steps=%s workers=%s duration_s=%.4f memory_delta_bytes=%s "
        "global_state=%s digest=%s",
        stage_count,
        steps,
        worker_count,
        duration_s,
        total_memory_delta,
        report.global_state,
        digest,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_benchmark()
