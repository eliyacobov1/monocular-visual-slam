"""Deterministic ordering and stable hashing utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def stable_hash(payload: Any, *, exclude_keys: Iterable[str] = ()) -> str:
    """Compute a stable hash for a payload with optional key exclusion."""

    normalized = _normalize_payload(payload)
    stripped = _strip_keys(normalized, set(exclude_keys))
    encoded = json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_event_digest(
    events: Iterable[Any],
    *,
    exclude_keys: Iterable[str] = ("timestamp_s",),
) -> str:
    """Compute a stable digest for a sequence of event payloads."""

    payloads = [_normalize_payload(_event_payload(event)) for event in events]
    return stable_hash(payloads, exclude_keys=exclude_keys)


def _event_payload(event: Any) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    if is_dataclass(event):
        return asdict(event)
    asdict_method = getattr(event, "asdict", None)
    if callable(asdict_method):
        return dict(asdict_method())
    data = getattr(event, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    return {"value": event}


def _normalize_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return _normalize_payload(asdict(payload))
    if isinstance(payload, Mapping):
        return {str(key): _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_normalize_payload(value) for value in payload]
    if isinstance(payload, set):
        return sorted((_normalize_payload(value) for value in payload), key=str)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, bytes):
        return payload.hex()
    return payload


def _strip_keys(payload: Any, excluded: set[str]) -> Any:
    if isinstance(payload, Mapping):
        return {
            key: _strip_keys(value, excluded)
            for key, value in payload.items()
            if key not in excluded
        }
    if isinstance(payload, list):
        return [_strip_keys(value, excluded) for value in payload]
    return payload


__all__ = ["stable_hash", "stable_event_digest"]
