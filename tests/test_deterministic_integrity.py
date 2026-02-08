"""Tests for deterministic hashing utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deterministic_integrity import stable_event_digest, stable_hash


@dataclass(frozen=True)
class StubEvent:
    event_type: str
    message: str
    metadata: dict[str, object]
    timestamp_s: float


def test_stable_hash_excludes_keys() -> None:
    payload = {"a": 1, "b": {"c": 2, "skip": 3}}
    digest_full = stable_hash(payload)
    digest_stripped = stable_hash(payload, exclude_keys=("skip",))
    assert digest_full != digest_stripped


def test_event_digest_ignores_timestamp() -> None:
    events = [
        StubEvent("stage_start", "ok", {"seq": 1}, 1.0),
        StubEvent("stage_start", "ok", {"seq": 1}, 2.0),
    ]
    digest_a = stable_event_digest(events[:1])
    digest_b = stable_event_digest(events[1:])
    assert digest_a == digest_b
