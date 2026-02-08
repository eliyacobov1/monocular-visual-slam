"""Unified control-plane supervisor with deterministic state transitions."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Mapping

from control_plane_hub import ControlPlaneHub, ControlPlaneReport, ControlPlaneStageAdapter, StageHealthSnapshot
from deterministic_integrity import stable_event_digest, stable_hash

LOGGER = logging.getLogger(__name__)


DEFAULT_STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "feature": ("ingestion",),
    "tracking": ("ingestion", "feature"),
    "optimization": ("tracking",),
}

DEFAULT_ERROR_KEYWORDS = (
    "error",
    "failure",
    "timeout",
    "exception",
    "circuit",
    "tripped",
    "dropped",
)


@dataclass(frozen=True)
class ControlPlaneSupervisorConfig:
    """Configuration for control-plane supervision and recovery policies."""

    stage_dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_STAGE_DEPENDENCIES))
    degrade_event_threshold: int = 2
    trip_event_threshold: int = 4
    event_window: int = 64
    recovery_cooldown_s: float = 0.5
    recovery_healthy_required: int = 2
    degraded_healthy_required: int = 2
    propagate_degraded: bool = True
    propagate_recovering: bool = True
    propagate_tripped: bool = True
    error_keywords: tuple[str, ...] = DEFAULT_ERROR_KEYWORDS
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if self.degrade_event_threshold < 0:
            raise ValueError("degrade_event_threshold must be non-negative")
        if self.trip_event_threshold < 0:
            raise ValueError("trip_event_threshold must be non-negative")
        if self.event_window <= 0:
            raise ValueError("event_window must be positive")
        if self.recovery_cooldown_s < 0:
            raise ValueError("recovery_cooldown_s must be non-negative")
        if self.recovery_healthy_required <= 0:
            raise ValueError("recovery_healthy_required must be positive")
        if self.degraded_healthy_required <= 0:
            raise ValueError("degraded_healthy_required must be positive")


@dataclass
class StageRuntimeState:
    """Mutable runtime state for a supervised stage."""

    state: str = "healthy"
    last_transition_s: float = field(default_factory=time.time)
    consecutive_healthy: int = 0
    last_raw_state: str = "healthy"
    last_error_events: int = 0


@dataclass(frozen=True)
class StageTransition:
    """Stage transition emitted by the supervisor."""

    stage: str
    previous_state: str
    next_state: str
    reason: str
    timestamp_s: float


@dataclass(frozen=True)
class SupervisorStageStatus:
    """Current status for a supervised stage."""

    stage: str
    state: str
    raw_state: str
    metrics: Mapping[str, float]
    counters: Mapping[str, int]
    error_events: int
    last_transition_s: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControlPlaneSupervisorReport:
    """Supervisor report combining stage states with hub telemetry."""

    generated_at_s: float
    global_state: str
    stage_statuses: tuple[SupervisorStageStatus, ...]
    transitions: tuple[StageTransition, ...]
    hub_report: ControlPlaneReport
    digest: str

    def asdict(self) -> dict[str, object]:
        return {
            "generated_at_s": self.generated_at_s,
            "global_state": self.global_state,
            "digest": self.digest,
            "hub_report": self.hub_report.asdict(),
            "stages": [status.asdict() for status in self.stage_statuses],
            "transitions": [asdict(transition) for transition in self.transitions],
        }


class ControlPlaneSupervisor:
    """Supervises cross-stage health with deterministic transitions."""

    def __init__(
        self,
        adapters: Iterable[ControlPlaneStageAdapter],
        *,
        config: ControlPlaneSupervisorConfig | None = None,
    ) -> None:
        self._config = config or ControlPlaneSupervisorConfig()
        self._hub = ControlPlaneHub(adapters)
        self._stage_states: dict[str, StageRuntimeState] = {}

    def update(self) -> ControlPlaneSupervisorReport:
        hub_report = self._hub.generate_report()
        now_s = float(self._config.clock())
        stage_snapshots = {snapshot.stage: snapshot for snapshot in hub_report.stage_snapshots}
        error_counts = self._count_error_events(hub_report)
        stage_states = self._initial_stage_states(stage_snapshots, error_counts)
        propagated_states = self._propagate_dependencies(stage_states)
        statuses, transitions = self._apply_recovery(stage_snapshots, propagated_states, error_counts, now_s)
        global_state = self._global_state(statuses)
        digest = self._digest(hub_report, statuses, transitions, global_state)
        return ControlPlaneSupervisorReport(
            generated_at_s=now_s,
            global_state=global_state,
            stage_statuses=tuple(statuses),
            transitions=tuple(transitions),
            hub_report=hub_report,
            digest=digest,
        )

    def _count_error_events(self, report: ControlPlaneReport) -> dict[str, int]:
        windowed: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=self._config.event_window))
        for event in report.events:
            stage = event.stage
            windowed[stage].append(self._is_error_event(event.event_type, event.message))
        return {stage: sum(1 for flagged in flags if flagged) for stage, flags in windowed.items()}

    def _initial_stage_states(
        self,
        snapshots: Mapping[str, StageHealthSnapshot],
        error_counts: Mapping[str, int],
    ) -> dict[str, str]:
        stage_states: dict[str, str] = {}
        for stage in sorted(snapshots.keys()):
            raw_state = self._normalize_state(snapshots[stage].state)
            state = raw_state
            error_count = error_counts.get(stage, 0)
            if error_count >= self._config.trip_event_threshold and self._config.trip_event_threshold > 0:
                state = "tripped"
            elif error_count >= self._config.degrade_event_threshold and self._config.degrade_event_threshold > 0:
                if state not in {"tripped", "recovering"}:
                    state = "degraded"
            stage_states[stage] = state
        return stage_states

    def _propagate_dependencies(self, stage_states: dict[str, str]) -> dict[str, str]:
        propagated = dict(stage_states)
        stages = sorted(stage_states.keys())
        for _ in range(len(stages)):
            for stage in stages:
                dependencies = self._config.stage_dependencies.get(stage, ())
                if not dependencies:
                    continue
                upstream_states = [propagated.get(dep, "healthy") for dep in dependencies]
                if self._config.propagate_tripped and any(state == "tripped" for state in upstream_states):
                    if propagated[stage] != "tripped":
                        propagated[stage] = "degraded"
                    continue
                if self._config.propagate_recovering and any(state == "recovering" for state in upstream_states):
                    if propagated[stage] == "healthy":
                        propagated[stage] = "degraded"
                    continue
                if self._config.propagate_degraded and any(state == "degraded" for state in upstream_states):
                    if propagated[stage] == "healthy":
                        propagated[stage] = "degraded"
        return propagated

    def _apply_recovery(
        self,
        snapshots: Mapping[str, StageHealthSnapshot],
        proposed_states: Mapping[str, str],
        error_counts: Mapping[str, int],
        now_s: float,
    ) -> tuple[list[SupervisorStageStatus], list[StageTransition]]:
        statuses: list[SupervisorStageStatus] = []
        transitions: list[StageTransition] = []
        for stage in sorted(snapshots.keys()):
            snapshot = snapshots[stage]
            runtime = self._stage_states.get(stage)
            if runtime is None:
                runtime = StageRuntimeState(state="healthy", last_transition_s=now_s)
                self._stage_states[stage] = runtime
            raw_state = self._normalize_state(snapshot.state)
            desired_state = proposed_states.get(stage, raw_state)
            reason = "snapshot"
            if runtime.state == "tripped" and desired_state in {"healthy", "degraded"}:
                if (now_s - runtime.last_transition_s) >= self._config.recovery_cooldown_s:
                    desired_state = "recovering"
                    reason = "cooldown_elapsed"
            if runtime.state == "recovering":
                if desired_state == "healthy":
                    runtime.consecutive_healthy += 1
                    if runtime.consecutive_healthy >= self._config.recovery_healthy_required:
                        desired_state = "healthy"
                        reason = "recovered"
                    else:
                        desired_state = "recovering"
                        reason = "awaiting_stability"
                elif desired_state == "tripped":
                    runtime.consecutive_healthy = 0
                    reason = "retrip"
                else:
                    runtime.consecutive_healthy = 0
                    desired_state = "recovering"
                    reason = "hold_recovering"
            if runtime.state == "degraded" and desired_state == "healthy":
                runtime.consecutive_healthy += 1
                if runtime.consecutive_healthy < self._config.degraded_healthy_required:
                    desired_state = "degraded"
                    reason = "awaiting_stability"
                else:
                    reason = "stabilized"
            if desired_state in {"tripped", "degraded"}:
                runtime.consecutive_healthy = 0
            if runtime.state != desired_state:
                transitions.append(
                    StageTransition(
                        stage=stage,
                        previous_state=runtime.state,
                        next_state=desired_state,
                        reason=reason,
                        timestamp_s=now_s,
                    )
                )
                runtime.state = desired_state
                runtime.last_transition_s = now_s
            runtime.last_raw_state = raw_state
            runtime.last_error_events = error_counts.get(stage, 0)
            statuses.append(
                SupervisorStageStatus(
                    stage=stage,
                    state=runtime.state,
                    raw_state=raw_state,
                    metrics=snapshot.metrics,
                    counters=snapshot.counters,
                    error_events=runtime.last_error_events,
                    last_transition_s=runtime.last_transition_s,
                )
            )
        return statuses, transitions

    def _global_state(self, statuses: Iterable[SupervisorStageStatus]) -> str:
        state_set = {status.state for status in statuses}
        if "tripped" in state_set:
            return "tripped"
        if "degraded" in state_set:
            return "degraded"
        if "recovering" in state_set:
            return "recovering"
        return "healthy"

    def _digest(
        self,
        hub_report: ControlPlaneReport,
        statuses: Iterable[SupervisorStageStatus],
        transitions: Iterable[StageTransition],
        global_state: str,
    ) -> str:
        payload = {
            "global_state": global_state,
            "hub_report_digest": hub_report.digest,
            "stages": [status.asdict() for status in statuses],
            "transitions": [asdict(transition) for transition in transitions],
            "transition_digest": stable_event_digest(transitions),
        }
        return stable_hash(payload, exclude_keys=("timestamp_s", "last_transition_s", "generated_at_s"))

    def _normalize_state(self, raw_state: str) -> str:
        normalized = raw_state.lower().strip()
        if normalized == "idle":
            return "healthy"
        return normalized

    def _is_error_event(self, event_type: str, message: str) -> bool:
        combined = f"{event_type} {message}".lower()
        return any(keyword in combined for keyword in self._config.error_keywords)


__all__ = [
    "ControlPlaneSupervisor",
    "ControlPlaneSupervisorConfig",
    "ControlPlaneSupervisorReport",
    "StageTransition",
    "SupervisorStageStatus",
]
