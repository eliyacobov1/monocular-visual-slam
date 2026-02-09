"""Telemetry utilities for tracking SLAM run performance."""

from __future__ import annotations

import json
import threading
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping, Protocol

from deterministic_integrity import stable_hash


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TelemetryEvent:
    """Telemetry event capturing a timed operation."""

    name: str
    duration_s: float
    timestamp: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TelemetryCorrelationConfig:
    """Configuration for deterministic telemetry correlation IDs."""

    seed: int
    config_hash: str
    run_id: str | None = None
    salt: str = "telemetry"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.config_hash:
            raise ValueError("config_hash must be non-empty")
        if not self.salt:
            raise ValueError("salt must be non-empty")


class TelemetryCorrelationRegistry:
    """Deterministic registry that maps stages to correlation IDs."""

    def __init__(self, config: TelemetryCorrelationConfig) -> None:
        self._config = config
        self._cache: dict[str, str] = {}

    def correlation_id(self, stage: str) -> str:
        if not stage:
            raise ValueError("stage must be non-empty")
        cached = self._cache.get(stage)
        if cached is not None:
            return cached
        payload = {
            "stage": stage,
            "seed": self._config.seed,
            "config_hash": self._config.config_hash,
            "run_id": self._config.run_id,
            "salt": self._config.salt,
        }
        correlation_id = stable_hash(payload)
        self._cache[stage] = correlation_id
        return correlation_id


class TelemetrySink(Protocol):
    """Interface for telemetry sinks."""

    def record_event(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""


class RunTelemetryRecorder:
    """Record telemetry events to a JSON file for a run."""

    def __init__(
        self,
        path: Path,
        determinism: Mapping[str, object] | None = None,
        correlation_registry: TelemetryCorrelationRegistry | None = None,
    ) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._events: list[TelemetryEvent] = []
        self._determinism = dict(determinism or {})
        self._correlation_registry = correlation_registry

    @property
    def path(self) -> Path:
        return self._path

    def record_event(self, event: TelemetryEvent) -> None:
        if self._correlation_registry is not None:
            metadata = dict(event.metadata)
            metadata.setdefault("stage", event.name)
            metadata.setdefault(
                "correlation_id",
                self._correlation_registry.correlation_id(str(metadata["stage"])),
            )
            if metadata is not event.metadata:
                event = TelemetryEvent(
                    name=event.name,
                    duration_s=event.duration_s,
                    timestamp=event.timestamp,
                    metadata=metadata,
                )
        with self._lock:
            self._events.append(event)

    def flush(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        payload = {
            "recorded_at": _timestamp(),
            "determinism": dict(self._determinism),
            "events": [
                {
                    "name": evt.name,
                    "duration_s": evt.duration_s,
                    "timestamp": evt.timestamp,
                    "metadata": evt.metadata,
                }
                for evt in self._events
            ],
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@contextmanager
def timed_event(
    name: str,
    sink: TelemetrySink,
    metadata: Mapping[str, Any] | None = None,
    *,
    track_memory: bool = False,
) -> Iterator[None]:
    """Context manager to record timing telemetry around an operation."""

    start = perf_counter()
    memory_before: tuple[int, int] | None = None
    if track_memory:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        memory_before = tracemalloc.get_traced_memory()
    success = True
    error: str | None = None
    try:
        yield
    except Exception as exc:
        success = False
        error = type(exc).__name__
        raise
    finally:
        duration = perf_counter() - start
        memory_after = None
        if memory_before is not None:
            memory_after = tracemalloc.get_traced_memory()
        event_metadata = dict(metadata or {})
        event_metadata["success"] = success
        if error:
            event_metadata["error"] = error
        if memory_before is not None and memory_after is not None:
            event_metadata["memory_delta_bytes"] = int(memory_after[0] - memory_before[0])
            event_metadata["memory_current_bytes"] = int(memory_after[0])
            event_metadata["memory_peak_bytes"] = int(memory_after[1])
        sink.record_event(
            TelemetryEvent(
                name=name,
                duration_s=duration,
                timestamp=_timestamp(),
                metadata=event_metadata,
            )
        )
