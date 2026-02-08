"""Unified control-plane orchestration and deterministic reporting."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

from deterministic_integrity import stable_event_digest, stable_hash

LOGGER = logging.getLogger(__name__)


class ControlPlaneEventLike(Protocol):
    """Protocol for control-plane events emitted by stages."""

    event_type: str
    message: str
    metadata: Mapping[str, object]
    timestamp_s: float


@dataclass(frozen=True)
class StageHealthSnapshot:
    """Generic health snapshot for a control-plane stage."""

    stage: str
    state: str
    metrics: Mapping[str, float]
    counters: Mapping[str, int]
    updated_at_s: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StageEventEnvelope:
    """Normalized event envelope for cross-stage ordering."""

    stage: str
    event_type: str
    message: str
    metadata: Mapping[str, object]
    timestamp_s: float
    seq_id: int

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlPlaneReport:
    """Aggregated report for multi-stage control-plane supervision."""

    stage_snapshots: tuple[StageHealthSnapshot, ...]
    events: tuple[StageEventEnvelope, ...]
    digest: str
    event_stream_digest: str
    stage_snapshot_digest: str
    generated_at_s: float

    def asdict(self) -> dict[str, Any]:
        return {
            "generated_at_s": self.generated_at_s,
            "digest": self.digest,
            "event_stream_digest": self.event_stream_digest,
            "stage_snapshot_digest": self.stage_snapshot_digest,
            "stages": [asdict(snapshot) for snapshot in self.stage_snapshots],
            "events": [event.asdict() for event in self.events],
        }


@dataclass(frozen=True)
class ControlPlaneStageAdapter:
    """Adapter for integrating stage-specific telemetry into the hub."""

    name: str
    health_snapshot: Callable[[], StageHealthSnapshot]
    events: Callable[[], Iterable[ControlPlaneEventLike]]


class DeterministicEventBus:
    """Thread-safe event buffer with deterministic ordering."""

    def __init__(self, capacity: int = 2048) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._events: list[StageEventEnvelope] = []
        self._lock = threading.Lock()

    def record(self, event: StageEventEnvelope) -> None:
        with self._lock:
            if len(self._events) >= self._capacity:
                self._events.pop(0)
            self._events.append(event)

    def snapshot(self) -> list[StageEventEnvelope]:
        with self._lock:
            return list(self._events)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


class ControlPlaneHub:
    """Aggregates stage telemetry with deterministic ordering."""

    def __init__(self, adapters: Iterable[ControlPlaneStageAdapter]) -> None:
        self._adapters = list(adapters)
        if not self._adapters:
            raise ValueError("At least one adapter is required")
        self._event_bus = DeterministicEventBus()
        self._lock = threading.Lock()

    def generate_report(self) -> ControlPlaneReport:
        with self._lock:
            adapters = sorted(self._adapters, key=lambda adapter: adapter.name)
            stage_snapshots = tuple(
                sorted(
                    (adapter.health_snapshot() for adapter in adapters),
                    key=lambda snapshot: snapshot.stage,
                )
            )
            merged_events = self._merge_events(adapters)
            generated_at_s = time.time()
            stage_snapshot_digest = stable_hash(stage_snapshots, exclude_keys=("updated_at_s",))
            event_stream_digest = stable_event_digest(merged_events)
            digest = self._digest(stage_snapshots, merged_events)
            return ControlPlaneReport(
                stage_snapshots=stage_snapshots,
                events=tuple(merged_events),
                digest=digest,
                event_stream_digest=event_stream_digest,
                stage_snapshot_digest=stage_snapshot_digest,
                generated_at_s=generated_at_s,
            )

    def _merge_events(
        self,
        adapters: list[ControlPlaneStageAdapter],
    ) -> list[StageEventEnvelope]:
        streams: list[list[StageEventEnvelope]] = []
        for adapter in adapters:
            events = list(adapter.events())
            ordered = sorted(
                events,
                key=lambda event: (
                    float(event.timestamp_s),
                    str(event.event_type),
                    str(event.message),
                    stable_hash(getattr(event, "metadata", {})),
                ),
            )
            envelopes = []
            for seq_id, event in enumerate(ordered):
                envelopes.append(
                    StageEventEnvelope(
                        stage=adapter.name,
                        event_type=event.event_type,
                        message=event.message,
                        metadata=dict(event.metadata),
                        timestamp_s=float(event.timestamp_s),
                        seq_id=seq_id,
                    )
                )
            streams.append(envelopes)

        import heapq

        heap: list[tuple[float, str, int, int]] = []
        for stream_idx, stream in enumerate(streams):
            if stream:
                event = stream[0]
                heapq.heappush(heap, (event.timestamp_s, event.stage, event.seq_id, stream_idx))

        merged: list[StageEventEnvelope] = []
        indices = [0 for _ in streams]
        while heap:
            _, _, _, stream_idx = heapq.heappop(heap)
            index = indices[stream_idx]
            event = streams[stream_idx][index]
            merged.append(event)
            indices[stream_idx] += 1
            next_index = indices[stream_idx]
            if next_index < len(streams[stream_idx]):
                next_event = streams[stream_idx][next_index]
                heapq.heappush(
                    heap,
                    (
                        next_event.timestamp_s,
                        next_event.stage,
                        next_event.seq_id,
                        stream_idx,
                    ),
                )
        for event in merged:
            self._event_bus.record(event)
        return merged

    def _digest(
        self,
        stage_snapshots: Iterable[StageHealthSnapshot],
        events: Iterable[StageEventEnvelope],
    ) -> str:
        payload = {
            "stages": [asdict(snapshot) for snapshot in stage_snapshots],
            "events": [event.asdict() for event in events],
        }
        return stable_hash(payload, exclude_keys=("updated_at_s", "timestamp_s"))


__all__ = [
    "ControlPlaneEventLike",
    "ControlPlaneHub",
    "ControlPlaneReport",
    "ControlPlaneStageAdapter",
    "DeterministicEventBus",
    "StageEventEnvelope",
    "StageHealthSnapshot",
]
