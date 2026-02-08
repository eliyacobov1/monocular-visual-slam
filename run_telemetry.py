"""Telemetry utilities for tracking SLAM run performance."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping, Protocol


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TelemetryEvent:
    """Telemetry event capturing a timed operation."""

    name: str
    duration_s: float
    timestamp: str
    metadata: dict[str, Any]


class TelemetrySink(Protocol):
    """Interface for telemetry sinks."""

    def record_event(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""


class RunTelemetryRecorder:
    """Record telemetry events to a JSON file for a run."""

    def __init__(self, path: Path, determinism: Mapping[str, object] | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._events: list[TelemetryEvent] = []
        self._determinism = dict(determinism or {})

    @property
    def path(self) -> Path:
        return self._path

    def record_event(self, event: TelemetryEvent) -> None:
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
) -> Iterator[None]:
    """Context manager to record timing telemetry around an operation."""

    start = perf_counter()
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
        event_metadata = dict(metadata or {})
        event_metadata["success"] = success
        if error:
            event_metadata["error"] = error
        sink.record_event(
            TelemetryEvent(
                name=name,
                duration_s=duration,
                timestamp=_timestamp(),
                metadata=event_metadata,
            )
        )
