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

DEFAULT_BACKPRESSURE_RATIO_KEYS = (
    "entry_queue_depth_ratio",
    "output_queue_depth_ratio",
    "queue_depth_ratio",
    "inflight_ratio",
)

DEFAULT_BACKPRESSURE_COUNTER_KEYS = (
    "output_backpressure",
    "backpressure_events",
)

DEFAULT_CIRCUIT_BREAKER_COUNTER_KEYS = (
    "circuit_breaker_opens",
    "breaker_opens",
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
    backpressure_ratio_threshold: float = 0.8
    backpressure_ratio_trip_threshold: float = 0.95
    backpressure_counter_threshold: int = 1
    backpressure_counter_trip_threshold: int = 3
    backpressure_ratio_keys: tuple[str, ...] = DEFAULT_BACKPRESSURE_RATIO_KEYS
    backpressure_counter_keys: tuple[str, ...] = DEFAULT_BACKPRESSURE_COUNTER_KEYS
    circuit_breaker_counter_keys: tuple[str, ...] = DEFAULT_CIRCUIT_BREAKER_COUNTER_KEYS
    circuit_breaker_trip_threshold: int = 1
    recovery_queue_capacity: int = 128
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
        if not 0.0 <= self.backpressure_ratio_threshold <= 1.0:
            raise ValueError("backpressure_ratio_threshold must be within [0, 1]")
        if not 0.0 <= self.backpressure_ratio_trip_threshold <= 1.0:
            raise ValueError("backpressure_ratio_trip_threshold must be within [0, 1]")
        if self.backpressure_ratio_trip_threshold < self.backpressure_ratio_threshold:
            raise ValueError("backpressure_ratio_trip_threshold must be >= backpressure_ratio_threshold")
        if self.backpressure_counter_threshold < 0:
            raise ValueError("backpressure_counter_threshold must be non-negative")
        if self.backpressure_counter_trip_threshold < 0:
            raise ValueError("backpressure_counter_trip_threshold must be non-negative")
        if self.backpressure_counter_trip_threshold < self.backpressure_counter_threshold:
            raise ValueError("backpressure_counter_trip_threshold must be >= backpressure_counter_threshold")
        if self.circuit_breaker_trip_threshold < 0:
            raise ValueError("circuit_breaker_trip_threshold must be non-negative")
        if self.recovery_queue_capacity <= 0:
            raise ValueError("recovery_queue_capacity must be positive")


@dataclass
class StageRuntimeState:
    """Mutable runtime state for a supervised stage."""

    state: str = "healthy"
    last_transition_s: float = field(default_factory=time.time)
    consecutive_healthy: int = 0
    last_raw_state: str = "healthy"
    last_error_events: int = 0
    last_backpressure_count: int = 0
    last_circuit_breaker_count: int = 0


@dataclass(frozen=True)
class StageTransition:
    """Stage transition emitted by the supervisor."""

    stage: str
    previous_state: str
    next_state: str
    reason: str
    timestamp_s: float


@dataclass(frozen=True)
class StageEscalation:
    """Escalation derived from backpressure or circuit-breaker signals."""

    stage: str
    escalation_type: str
    severity: str
    reason: str
    metrics: Mapping[str, float]
    counters: Mapping[str, int]
    timestamp_s: float


@dataclass(frozen=True)
class RecoveryAction:
    """Deterministic recovery action queued for supervision."""

    stage: str
    action_type: str
    severity: str
    reason: str
    queued_at_s: float
    seq_id: int


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
    escalations: tuple[StageEscalation, ...]
    recovery_queue: tuple[RecoveryAction, ...]
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
            "escalations": [asdict(escalation) for escalation in self.escalations],
            "recovery_queue": [asdict(action) for action in self.recovery_queue],
        }


class RecoveryQueue:
    """Deterministic recovery queue with bounded memory."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._queue: deque[RecoveryAction] = deque()

    def enqueue(self, action: RecoveryAction) -> None:
        if len(self._queue) >= self._capacity:
            self._queue.popleft()
        self._queue.append(action)

    def ordered(self) -> tuple[RecoveryAction, ...]:
        return tuple(sorted(self._queue, key=_recovery_sort_key))

    def drain(self) -> tuple[RecoveryAction, ...]:
        ordered = self.ordered()
        self._queue.clear()
        return ordered

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._queue)


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
        self._recovery_queue = RecoveryQueue(self._config.recovery_queue_capacity)
        self._recovery_seq = 0

    def update(self) -> ControlPlaneSupervisorReport:
        hub_report = self._hub.generate_report()
        now_s = float(self._config.clock())
        stage_snapshots = {snapshot.stage: snapshot for snapshot in hub_report.stage_snapshots}
        error_counts = self._count_error_events(hub_report)
        stage_states = self._initial_stage_states(stage_snapshots, error_counts)
        stage_states, escalations = self._apply_escalations(stage_snapshots, stage_states, now_s)
        propagated_states = self._propagate_dependencies(stage_states)
        statuses, transitions = self._apply_recovery(stage_snapshots, propagated_states, error_counts, now_s)
        global_state = self._global_state(statuses)
        recovery_queue = self._recovery_queue.ordered()
        digest = self._digest(hub_report, statuses, transitions, escalations, recovery_queue, global_state)
        return ControlPlaneSupervisorReport(
            generated_at_s=now_s,
            global_state=global_state,
            stage_statuses=tuple(statuses),
            transitions=tuple(transitions),
            escalations=tuple(escalations),
            recovery_queue=recovery_queue,
            hub_report=hub_report,
            digest=digest,
        )

    def drain_recovery_queue(self) -> tuple[RecoveryAction, ...]:
        return self._recovery_queue.drain()

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

    def _apply_escalations(
        self,
        snapshots: Mapping[str, StageHealthSnapshot],
        stage_states: dict[str, str],
        now_s: float,
    ) -> tuple[dict[str, str], list[StageEscalation]]:
        escalations: list[StageEscalation] = []
        updated_states = dict(stage_states)
        for stage in sorted(snapshots.keys()):
            snapshot = snapshots[stage]
            runtime = self._stage_states.get(stage)
            if runtime is None:
                runtime = StageRuntimeState(state="healthy", last_transition_s=now_s)
                self._stage_states[stage] = runtime
            backpressure = self._evaluate_backpressure(snapshot, runtime, now_s)
            circuit_breaker = self._evaluate_circuit_breaker(snapshot, runtime, now_s)
            for escalation in (backpressure, circuit_breaker):
                if escalation is None:
                    continue
                escalations.append(escalation)
                updated_states[stage] = _merge_state(updated_states.get(stage, "healthy"), escalation.severity)
                action = self._queue_recovery_action(escalation)
                LOGGER.info(
                    "control_plane_escalation stage=%s type=%s severity=%s reason=%s action_seq=%s",
                    escalation.stage,
                    escalation.escalation_type,
                    escalation.severity,
                    escalation.reason,
                    action.seq_id,
                )
        return updated_states, escalations

    def _evaluate_backpressure(
        self,
        snapshot: StageHealthSnapshot,
        runtime: StageRuntimeState,
        now_s: float,
    ) -> StageEscalation | None:
        ratio = _max_metric(snapshot.metrics, self._config.backpressure_ratio_keys)
        counter = _sum_counters(snapshot.counters, self._config.backpressure_counter_keys)
        delta = max(counter - runtime.last_backpressure_count, 0)
        runtime.last_backpressure_count = counter
        severity = None
        reason_parts: list[str] = []
        if ratio is not None:
            if ratio >= self._config.backpressure_ratio_trip_threshold:
                severity = "tripped"
                reason_parts.append(f"ratio={ratio:.3f}")
            elif ratio >= self._config.backpressure_ratio_threshold:
                severity = "degraded"
                reason_parts.append(f"ratio={ratio:.3f}")
        if delta >= self._config.backpressure_counter_trip_threshold > 0:
            severity = "tripped"
            reason_parts.append(f"delta={delta}")
        elif delta >= self._config.backpressure_counter_threshold > 0 and severity is None:
            severity = "degraded"
            reason_parts.append(f"delta={delta}")
        if severity is None:
            return None
        reason = "backpressure(" + ",".join(reason_parts) + ")"
        return StageEscalation(
            stage=snapshot.stage,
            escalation_type="backpressure",
            severity=severity,
            reason=reason,
            metrics=dict(snapshot.metrics),
            counters=dict(snapshot.counters),
            timestamp_s=now_s,
        )

    def _evaluate_circuit_breaker(
        self,
        snapshot: StageHealthSnapshot,
        runtime: StageRuntimeState,
        now_s: float,
    ) -> StageEscalation | None:
        counter = _sum_counters(snapshot.counters, self._config.circuit_breaker_counter_keys)
        delta = max(counter - runtime.last_circuit_breaker_count, 0)
        runtime.last_circuit_breaker_count = counter
        if self._config.circuit_breaker_trip_threshold <= 0:
            return None
        if delta < self._config.circuit_breaker_trip_threshold:
            return None
        reason = f"circuit_breaker(delta={delta})"
        return StageEscalation(
            stage=snapshot.stage,
            escalation_type="circuit_breaker",
            severity="tripped",
            reason=reason,
            metrics=dict(snapshot.metrics),
            counters=dict(snapshot.counters),
            timestamp_s=now_s,
        )

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
        escalations: Iterable[StageEscalation],
        recovery_queue: Iterable[RecoveryAction],
        global_state: str,
    ) -> str:
        payload = {
            "global_state": global_state,
            "hub_report_digest": hub_report.digest,
            "stages": [status.asdict() for status in statuses],
            "transitions": [asdict(transition) for transition in transitions],
            "escalations": [asdict(escalation) for escalation in escalations],
            "recovery_queue": [asdict(action) for action in recovery_queue],
            "transition_digest": stable_event_digest(transitions),
        }
        return stable_hash(
            payload,
            exclude_keys=("timestamp_s", "last_transition_s", "generated_at_s", "queued_at_s"),
        )

    def _normalize_state(self, raw_state: str) -> str:
        normalized = raw_state.lower().strip()
        if normalized == "idle":
            return "healthy"
        return normalized

    def _is_error_event(self, event_type: str, message: str) -> bool:
        combined = f"{event_type} {message}".lower()
        return any(keyword in combined for keyword in self._config.error_keywords)

    def _queue_recovery_action(self, escalation: StageEscalation) -> RecoveryAction:
        self._recovery_seq += 1
        action = RecoveryAction(
            stage=escalation.stage,
            action_type=escalation.escalation_type,
            severity=escalation.severity,
            reason=escalation.reason,
            queued_at_s=escalation.timestamp_s,
            seq_id=self._recovery_seq,
        )
        self._recovery_queue.enqueue(action)
        return action


def _max_metric(metrics: Mapping[str, float], keys: Iterable[str]) -> float | None:
    values = [float(metrics[key]) for key in keys if key in metrics]
    if not values:
        return None
    return max(values)


def _sum_counters(counters: Mapping[str, int], keys: Iterable[str]) -> int:
    return int(sum(int(counters.get(key, 0)) for key in keys))


def _merge_state(current: str, incoming: str) -> str:
    rank = {"healthy": 0, "recovering": 1, "degraded": 2, "tripped": 3}
    return current if rank.get(current, 0) >= rank.get(incoming, 0) else incoming


def _recovery_sort_key(action: RecoveryAction) -> tuple[int, float, str, int]:
    severity_rank = 0 if action.severity == "tripped" else 1
    return (severity_rank, action.queued_at_s, action.stage, action.seq_id)


__all__ = [
    "ControlPlaneSupervisor",
    "ControlPlaneSupervisorConfig",
    "ControlPlaneSupervisorReport",
    "RecoveryAction",
    "RecoveryQueue",
    "StageEscalation",
    "StageTransition",
    "SupervisorStageStatus",
]
