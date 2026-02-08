"""Deterministic failure injection and chaos harness utilities."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from control_plane_hub import ControlPlaneStageAdapter, StageHealthSnapshot
from deterministic_integrity import stable_event_digest, stable_hash

LOGGER = logging.getLogger(__name__)


_FAILURE_TYPES: tuple[str, ...] = ("timeout", "dropped_frame", "solver_stall")


@dataclass(frozen=True)
class FailureInjectionConfig:
    """Configuration for deterministic failure schedules."""

    seed: int
    timeout_probability: float = 0.05
    dropped_frame_probability: float = 0.03
    solver_stall_probability: float = 0.02
    step_duration_s: float = 0.05
    max_stall_duration_s: float = 0.25
    timeout_duration_s: float = 0.2
    allow_multiple_per_step: bool = False

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.step_duration_s <= 0:
            raise ValueError("step_duration_s must be positive")
        if self.max_stall_duration_s <= 0:
            raise ValueError("max_stall_duration_s must be positive")
        if self.timeout_duration_s <= 0:
            raise ValueError("timeout_duration_s must be positive")
        for name, value in (
            ("timeout_probability", self.timeout_probability),
            ("dropped_frame_probability", self.dropped_frame_probability),
            ("solver_stall_probability", self.solver_stall_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True)
class FailureInjectionPoint:
    """Single failure injection trigger."""

    stage: str
    failure_type: str
    step: int
    timestamp_s: float
    severity: str
    metadata: Mapping[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FailureInjectionPlan:
    """Deterministic schedule of injected failures."""

    seed: int
    stages: tuple[str, ...]
    steps: int
    points: tuple[FailureInjectionPoint, ...]
    digest: str

    def asdict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "stages": self.stages,
            "points": [point.asdict() for point in self.points],
            "digest": self.digest,
        }

    def points_for_stage(self, stage: str) -> tuple[FailureInjectionPoint, ...]:
        return tuple(point for point in self.points if point.stage == stage)


@dataclass(frozen=True)
class FailureInjectionEvent:
    """Event emitted by the failure injection harness."""

    event_type: str
    message: str
    metadata: Mapping[str, object]
    timestamp_s: float


class FailureInjector:
    """Build deterministic failure schedules."""

    def __init__(self, config: FailureInjectionConfig) -> None:
        self._config = config

    @property
    def config(self) -> FailureInjectionConfig:
        return self._config

    def build_plan(self, stages: Sequence[str], steps: int) -> FailureInjectionPlan:
        if steps <= 0:
            raise ValueError("steps must be positive")
        stage_list = tuple(sorted({str(stage) for stage in stages if stage}))
        if not stage_list:
            raise ValueError("at least one stage is required")
        rng = random.Random(self._config.seed)
        points: list[FailureInjectionPoint] = []
        for stage in stage_list:
            for step in range(steps):
                timestamp_s = step * self._config.step_duration_s
                points.extend(self._failures_for_step(rng, stage, step, timestamp_s))
        points = sorted(points, key=_failure_sort_key)
        digest = stable_hash(
            [point.asdict() for point in points],
            exclude_keys=("timestamp_s",),
        )
        return FailureInjectionPlan(
            seed=self._config.seed,
            stages=stage_list,
            steps=steps,
            points=tuple(points),
            digest=digest,
        )

    def _failures_for_step(
        self,
        rng: random.Random,
        stage: str,
        step: int,
        timestamp_s: float,
    ) -> list[FailureInjectionPoint]:
        failures: list[FailureInjectionPoint] = []
        candidate_order = (
            ("timeout", self._config.timeout_probability),
            ("dropped_frame", self._config.dropped_frame_probability),
            ("solver_stall", self._config.solver_stall_probability),
        )
        for failure_type, probability in candidate_order:
            if rng.random() < probability:
                metadata = _failure_metadata(failure_type, rng, self._config)
                severity = "tripped" if failure_type != "dropped_frame" else "degraded"
                failures.append(
                    FailureInjectionPoint(
                        stage=stage,
                        failure_type=failure_type,
                        step=step,
                        timestamp_s=timestamp_s,
                        severity=severity,
                        metadata=metadata,
                    )
                )
                if not self._config.allow_multiple_per_step:
                    break
        return failures


@dataclass
class StageFailureState:
    """Mutable state for failure-injected stages."""

    stage: str
    plan_points: deque[FailureInjectionPoint]
    base_state: str = "healthy"
    metrics: dict[str, float] = field(default_factory=dict)
    counters: Counter[str] = field(default_factory=Counter)
    last_snapshot: StageHealthSnapshot | None = None
    last_events: tuple[FailureInjectionEvent, ...] = ()
    last_step: int = -1
    lock: threading.Lock = field(default_factory=threading.Lock)

    def advance(self, step: int, now_s: float) -> None:
        with self.lock:
            if step == self.last_step:
                return
            events: list[FailureInjectionEvent] = []
            state = self.base_state
            while self.plan_points and self.plan_points[0].step == step:
                point = self.plan_points.popleft()
                state = point.severity
                self.counters[f"{point.failure_type}_events"] += 1
                event_type = _failure_event_type(point.failure_type)
                message = _failure_message(point.failure_type)
                metadata = dict(point.metadata)
                metadata["stage"] = point.stage
                metadata["failure_type"] = point.failure_type
                metadata["severity"] = point.severity
                events.append(
                    FailureInjectionEvent(
                        event_type=event_type,
                        message=message,
                        metadata=metadata,
                        timestamp_s=point.timestamp_s,
                    )
                )
                if point.failure_type == "solver_stall":
                    self.metrics["stall_duration_s"] = float(point.metadata.get("stall_duration_s", 0.0))
                if point.failure_type == "timeout":
                    self.metrics["timeout_duration_s"] = float(point.metadata.get("timeout_duration_s", 0.0))
            snapshot = StageHealthSnapshot(
                stage=self.stage,
                state=state,
                metrics=dict(self.metrics),
                counters=dict(self.counters),
                updated_at_s=now_s,
            )
            self.last_snapshot = snapshot
            self.last_events = tuple(sorted(events, key=_event_sort_key))
            self.last_step = step

    def snapshot(self) -> StageHealthSnapshot:
        with self.lock:
            if self.last_snapshot is None:
                return StageHealthSnapshot(
                    stage=self.stage,
                    state=self.base_state,
                    metrics=dict(self.metrics),
                    counters=dict(self.counters),
                )
            return self.last_snapshot

    def events(self) -> tuple[FailureInjectionEvent, ...]:
        with self.lock:
            return self.last_events


class FailureInjectionHarness:
    """Harness to emit failure-injected events and health snapshots."""

    def __init__(
        self,
        plan: FailureInjectionPlan,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._plan = plan
        self._clock = clock or time.time
        self._step = 0
        self._states: dict[str, StageFailureState] = {}
        stage_points: dict[str, deque[FailureInjectionPoint]] = {
            stage: deque(plan.points_for_stage(stage))
            for stage in plan.stages
        }
        for stage in plan.stages:
            self._states[stage] = StageFailureState(stage=stage, plan_points=stage_points[stage])
        self._last_digest = ""

    @property
    def step(self) -> int:
        return self._step

    @property
    def plan(self) -> FailureInjectionPlan:
        return self._plan

    def adapters(self) -> list[ControlPlaneStageAdapter]:
        return [
            ControlPlaneStageAdapter(
                name=stage,
                health_snapshot=self._states[stage].snapshot,
                events=self._states[stage].events,
            )
            for stage in self._plan.stages
        ]

    def advance(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            now_s = float(self._clock())
            for stage in self._plan.stages:
                self._states[stage].advance(self._step, now_s)
            self._step += 1
        self._last_digest = self._digest_state()

    def digest_state(self) -> str:
        if not self._last_digest:
            self._last_digest = self._digest_state()
        return self._last_digest

    def summary(self) -> dict[str, object]:
        counters = {
            stage: dict(state.counters)
            for stage, state in self._states.items()
        }
        return {
            "seed": self._plan.seed,
            "steps": self._plan.steps,
            "stages": list(self._plan.stages),
            "failure_counts": counters,
            "state_digest": self.digest_state(),
        }

    def _digest_state(self) -> str:
        payload = {
            "step": self._step,
            "states": {
                stage: {
                    "snapshot": state.snapshot().state,
                    "metrics": state.snapshot().metrics,
                    "counters": state.snapshot().counters,
                    "events": [asdict(event) for event in state.events()],
                }
                for stage, state in self._states.items()
            },
        }
        return stable_hash(payload, exclude_keys=("timestamp_s", "updated_at_s"))


class FailureInjectionChaosHarness:
    """Chaos harness that exercises failure injection under concurrent access."""

    def __init__(
        self,
        plan: FailureInjectionPlan,
        *,
        clock: Callable[[], float] | None = None,
        worker_count: int = 4,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self._harness = FailureInjectionHarness(plan, clock=clock)
        self._worker_count = worker_count
        self._lock = threading.Lock()
        self._events_digest = ""

    @property
    def harness(self) -> FailureInjectionHarness:
        return self._harness

    def run(self, steps: int) -> str:
        if steps <= 0:
            raise ValueError("steps must be positive")
        threads: list[threading.Thread] = []
        for _ in range(self._worker_count):
            thread = threading.Thread(target=self._advance_loop, args=(steps,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        self._events_digest = self._digest_events()
        return self._events_digest

    def _advance_loop(self, steps: int) -> None:
        for _ in range(steps):
            with self._lock:
                self._harness.advance(steps=1)

    def _digest_events(self) -> str:
        events: list[dict[str, object]] = []
        for stage in self._harness.plan.stages:
            for event in self._harness._states[stage].events():
                events.append(asdict(event))
        return stable_event_digest(events, exclude_keys=("timestamp_s",))


def _failure_metadata(
    failure_type: str,
    rng: random.Random,
    config: FailureInjectionConfig,
) -> dict[str, object]:
    if failure_type == "solver_stall":
        duration = rng.uniform(config.step_duration_s, config.max_stall_duration_s)
        return {"stall_duration_s": float(duration)}
    if failure_type == "timeout":
        duration = rng.uniform(config.step_duration_s, config.timeout_duration_s)
        return {"timeout_duration_s": float(duration)}
    return {}


def _failure_event_type(failure_type: str) -> str:
    if failure_type == "timeout":
        return "timeout_failure"
    if failure_type == "solver_stall":
        return "solver_failure_stall"
    return "dropped_frame"


def _failure_message(failure_type: str) -> str:
    if failure_type == "timeout":
        return "timeout_failure_detected"
    if failure_type == "solver_stall":
        return "solver_stall_failure"
    return "dropped_frame_failure"


def _failure_sort_key(point: FailureInjectionPoint) -> tuple[str, int, str, float]:
    return (point.stage, point.step, point.failure_type, point.timestamp_s)


def _event_sort_key(event: FailureInjectionEvent) -> tuple[str, str]:
    return (event.event_type, event.message)


__all__ = [
    "FailureInjectionChaosHarness",
    "FailureInjectionConfig",
    "FailureInjectionEvent",
    "FailureInjectionHarness",
    "FailureInjectionPlan",
    "FailureInjectionPoint",
    "FailureInjector",
]
