"""Optimization control plane with deterministic telemetry and supervision."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from control_plane_hub import StageHealthSnapshot
from data_persistence import P2Quantile
from deterministic_integrity import stable_event_digest
from graph_optimization import (
    ConditioningDiagnostics,
    IterationDiagnostics,
    PoseGraphProblem,
    PoseGraphSnapshot,
    PoseGraphSolver,
    RobustLossConfig,
    SolverConfig,
    SolverResult,
    compute_conditioning_diagnostics,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationControlConfig:
    """Configuration for supervised optimization control plane."""

    max_attempts: int = 3
    backoff_s: float = 0.05
    jitter_s: float = 0.02
    seed: int = 1337
    loss_scale_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    damping_multiplier: float = 2.0
    telemetry_quantiles: tuple[float, ...] = (0.5, 0.95, 0.99)
    event_log_capacity: int = 256

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.backoff_s < 0:
            raise ValueError("backoff_s must be non-negative")
        if self.jitter_s < 0:
            raise ValueError("jitter_s must be non-negative")
        if not self.loss_scale_multipliers:
            raise ValueError("loss_scale_multipliers must be non-empty")
        if self.damping_multiplier <= 0:
            raise ValueError("damping_multiplier must be positive")
        if not self.telemetry_quantiles:
            raise ValueError("telemetry_quantiles must be non-empty")
        if self.event_log_capacity <= 0:
            raise ValueError("event_log_capacity must be positive")


@dataclass(frozen=True)
class OptimizationStageEvent:
    """Structured event emitted by the optimization control plane."""

    stage: str
    event_type: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=time.time)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class DeterministicEventLog:
    """Thread-safe ring buffer for optimization events."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._events: list[OptimizationStageEvent] = []
        self._lock = threading.Lock()

    def record(self, event: OptimizationStageEvent) -> None:
        with self._lock:
            if len(self._events) >= self._capacity:
                self._events.pop(0)
            self._events.append(event)

    def snapshot(self) -> list[OptimizationStageEvent]:
        with self._lock:
            return list(self._events)

    def digest(self) -> str:
        with self._lock:
            return stable_event_digest(self._events)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


class MetricTracker:
    """Streaming statistics tracker with quantile estimation."""

    def __init__(self, quantiles: Iterable[float]) -> None:
        self._count = 0
        self._total = 0.0
        self._total_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._quantiles = tuple(P2Quantile(q) for q in quantiles)

    def update(self, value: float) -> None:
        value = float(value)
        self._count += 1
        self._total += value
        self._total_sq += value * value
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        for estimator in self._quantiles:
            estimator.update(value)

    def summary(self) -> dict[str, float]:
        if self._count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        mean = self._total / self._count
        variance = max(self._total_sq / self._count - mean * mean, 0.0)
        summary = {
            "count": float(self._count),
            "mean": mean,
            "std": float(variance**0.5),
            "min": float(self._min),
            "max": float(self._max),
        }
        for estimator in self._quantiles:
            key = f"p{int(estimator.quantile * 100)}"
            summary[key] = estimator.value()
        return summary


@dataclass(frozen=True)
class OptimizationTelemetrySummary:
    """Aggregated telemetry summary for optimization iterations."""

    residual_norm: dict[str, float]
    step_norm: dict[str, float]
    linear_solver_residual: dict[str, float]


@dataclass(frozen=True)
class OptimizationRunReport:
    """Run report emitted by the optimization supervisor."""

    snapshot_digest: str
    solver_name: str
    attempts: int
    success: bool
    result: SolverResult
    telemetry: OptimizationTelemetrySummary
    events: tuple[OptimizationStageEvent, ...]
    started_at_s: float
    finished_at_s: float

    @property
    def duration_s(self) -> float:
        return self.finished_at_s - self.started_at_s


class OptimizationSupervisor:
    """Supervises solver execution with deterministic retries and telemetry."""

    def __init__(self, config: OptimizationControlConfig | None = None) -> None:
        self._config = config or OptimizationControlConfig()
        self._event_log = DeterministicEventLog(self._config.event_log_capacity)
        self._random = random.Random(self._config.seed)
        self._lock = threading.Lock()
        self._run_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._last_success: bool | None = None
        self._last_attempts = 0
        self._last_duration_s: float | None = None
        self._last_snapshot_digest: str | None = None
        self._last_solver_name: str | None = None
        self._last_cost: float | None = None

    @property
    def event_log(self) -> DeterministicEventLog:
        return self._event_log

    def run(
        self,
        *,
        solver: PoseGraphSolver,
        problem: PoseGraphProblem,
        x0: list[float] | tuple[float, ...] | Iterable[float],
        solver_config: SolverConfig,
        loss_config: RobustLossConfig,
        snapshot: PoseGraphSnapshot,
        solver_name: str,
    ) -> tuple[list[float], SolverResult, OptimizationRunReport]:
        started_at_s = time.perf_counter()
        attempts = 0
        last_result: SolverResult | None = None
        diagnostics: list[IterationDiagnostics] = []
        x_opt = np.asarray(list(x0), dtype=float)

        conditioning = self._check_conditioning(problem, x_opt, solver_config, loss_config)
        if conditioning is not None:
            residual = problem.residual_fn(x_opt)
            last_result = SolverResult(
                success=False,
                status=-2,
                cost=float(0.5 * np.dot(residual, residual)),
                residual_norm=float(np.linalg.norm(residual)),
                iterations=0,
                message=conditioning.message,
                diagnostics=None,
            )
            telemetry = _summarize_telemetry((), self._config.telemetry_quantiles)
            finished_at_s = time.perf_counter()
            events = tuple(self._event_log.snapshot())
            report = OptimizationRunReport(
                snapshot_digest=snapshot.digest(),
                solver_name=solver_name,
                attempts=attempts,
                success=False,
                result=last_result,
                telemetry=telemetry,
                events=events,
                started_at_s=started_at_s,
                finished_at_s=finished_at_s,
            )
            with self._lock:
                self._run_count += 1
                self._failure_count += 1
                self._last_success = False
                self._last_attempts = attempts
                self._last_duration_s = report.duration_s
                self._last_snapshot_digest = report.snapshot_digest
                self._last_solver_name = solver_name
                self._last_cost = last_result.cost
            return list(x_opt), last_result, report

        for attempt in range(self._config.max_attempts):
            attempts += 1
            loss_scale = loss_config.scale * self._config.loss_scale_multipliers[
                min(attempt, len(self._config.loss_scale_multipliers) - 1)
            ]
            adjusted_loss = RobustLossConfig(loss_type=loss_config.loss_type, scale=loss_scale)
            damping = solver_config.damping * (self._config.damping_multiplier ** attempt)
            adjusted_solver = SolverConfig(
                max_iterations=solver_config.max_iterations,
                max_nfev=solver_config.max_nfev,
                damping=damping,
                step_scale=solver_config.step_scale,
                xtol=solver_config.xtol,
                ftol=solver_config.ftol,
                gtol=solver_config.gtol,
                linear_solver_max_iter=solver_config.linear_solver_max_iter,
                linear_solver_tol=solver_config.linear_solver_tol,
                max_condition_number=solver_config.max_condition_number,
                min_diagonal=solver_config.min_diagonal,
            )
            self._record_event(
                stage="solver",
                event_type="attempt_start",
                message="Solver attempt started",
                metadata={
                    "attempt": attempt + 1,
                    "loss_scale": loss_scale,
                    "damping": damping,
                },
            )
            try:
                x_opt, result = solver.solve(
                    problem,
                    np.asarray(list(x0), dtype=float),
                    adjusted_solver,
                    adjusted_loss,
                )
                last_result = result
                if result.diagnostics is not None:
                    diagnostics = list(result.diagnostics.iterations)
                if result.success:
                    self._record_event(
                        stage="solver",
                        event_type="attempt_success",
                        message="Solver attempt succeeded",
                        metadata={"attempt": attempt + 1, "status": result.message},
                    )
                    break
                self._record_event(
                    stage="solver",
                    event_type="attempt_failed",
                    message="Solver attempt failed",
                    metadata={"attempt": attempt + 1, "status": result.message},
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Solver attempt failed")
                last_result = SolverResult(
                    success=False,
                    status=0,
                    cost=float("nan"),
                    residual_norm=float("nan"),
                    iterations=0,
                    message=f"Solver exception: {exc}",
                    diagnostics=None,
                )
                self._record_event(
                    stage="solver",
                    event_type="attempt_exception",
                    message="Solver attempt raised exception",
                    metadata={"attempt": attempt + 1, "error": str(exc)},
                )
            if attempt < self._config.max_attempts - 1:
                sleep_s = self._config.backoff_s + self._random.random() * self._config.jitter_s
                time.sleep(sleep_s)

        if last_result is None:
            last_result = SolverResult(
                success=False,
                status=0,
                cost=float("nan"),
                residual_norm=float("nan"),
                iterations=0,
                message="Solver did not run",
                diagnostics=None,
            )
        telemetry = _summarize_telemetry(diagnostics, self._config.telemetry_quantiles)
        finished_at_s = time.perf_counter()
        events = tuple(self._event_log.snapshot())
        report = OptimizationRunReport(
            snapshot_digest=snapshot.digest(),
            solver_name=solver_name,
            attempts=attempts,
            success=last_result.success,
            result=last_result,
            telemetry=telemetry,
            events=events,
            started_at_s=started_at_s,
            finished_at_s=finished_at_s,
        )
        with self._lock:
            self._run_count += 1
            if last_result.success:
                self._success_count += 1
            else:
                self._failure_count += 1
            self._last_success = last_result.success
            self._last_attempts = attempts
            self._last_duration_s = report.duration_s
            self._last_snapshot_digest = report.snapshot_digest
            self._last_solver_name = solver_name
            self._last_cost = last_result.cost
        return list(x_opt), last_result, report

    def _check_conditioning(
        self,
        problem: PoseGraphProblem,
        x0: np.ndarray,
        solver_config: SolverConfig,
        loss_config: RobustLossConfig,
    ) -> ConditioningDiagnostics | None:
        linearized = list(problem.linearize_fn(x0))
        if not linearized:
            return None
        num_blocks = problem.parameter_size // problem.block_size
        conditioning = compute_conditioning_diagnostics(
            linearized,
            block_size=problem.block_size,
            num_blocks=num_blocks,
            loss_config=loss_config,
            damping=solver_config.damping,
        )
        metadata = {
            "condition_number": conditioning.condition_number,
            "min_diagonal": conditioning.min_diagonal,
            "max_diagonal": conditioning.max_diagonal,
            "status": conditioning.status,
        }
        self._record_event(
            stage="conditioning",
            event_type="check",
            message=conditioning.message,
            metadata=metadata,
        )
        if (
            conditioning.status != "ok"
            or conditioning.condition_number > solver_config.max_condition_number
            or conditioning.min_diagonal < solver_config.min_diagonal
        ):
            message = "Conditioning gate tripped; fallback to prior state"
            self._record_event(
                stage="conditioning",
                event_type="tripped",
                message=message,
                metadata=metadata,
            )
            return ConditioningDiagnostics(
                condition_number=conditioning.condition_number,
                min_diagonal=conditioning.min_diagonal,
                max_diagonal=conditioning.max_diagonal,
                status="tripped",
                message=message,
            )
        return None

    def _record_event(self, *, stage: str, event_type: str, message: str, metadata: dict[str, object]) -> None:
        event = OptimizationStageEvent(
            stage=stage,
            event_type=event_type,
            message=message,
            metadata=metadata,
        )
        self._event_log.record(event)

    def health_snapshot(self) -> StageHealthSnapshot:
        with self._lock:
            if self._last_success is None:
                state = "idle"
            elif self._last_success:
                state = "healthy"
            elif self._last_attempts >= self._config.max_attempts:
                state = "tripped"
            else:
                state = "degraded"
            metrics = {
                "last_duration_s": float(self._last_duration_s or 0.0),
                "last_cost": float(self._last_cost or 0.0),
                "last_attempts": float(self._last_attempts),
            }
            counters = {
                "runs": self._run_count,
                "successes": self._success_count,
                "failures": self._failure_count,
            }
        return StageHealthSnapshot(
            stage="optimization",
            state=state,
            metrics=metrics,
            counters=counters,
        )


def _summarize_telemetry(
    diagnostics: Iterable[IterationDiagnostics],
    quantiles: Iterable[float],
) -> OptimizationTelemetrySummary:
    residual_tracker = MetricTracker(quantiles)
    step_tracker = MetricTracker(quantiles)
    linear_tracker = MetricTracker(quantiles)
    for entry in diagnostics:
        residual_tracker.update(entry.residual_norm)
        step_tracker.update(entry.step_norm)
        linear_tracker.update(entry.linear_solver_residual)
    return OptimizationTelemetrySummary(
        residual_norm=residual_tracker.summary(),
        step_norm=step_tracker.summary(),
        linear_solver_residual=linear_tracker.summary(),
    )


__all__ = [
    "DeterministicEventLog",
    "MetricTracker",
    "OptimizationControlConfig",
    "OptimizationRunReport",
    "OptimizationStageEvent",
    "OptimizationSupervisor",
    "OptimizationTelemetrySummary",
]
