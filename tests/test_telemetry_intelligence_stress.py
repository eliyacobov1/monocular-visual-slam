"""Stress tests for telemetry intelligence concurrency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from telemetry_intelligence import TelemetryDigest


def _emit_events(digest: TelemetryDigest, count: int) -> None:
    for _ in range(count):
        digest.update("stage", 0.001)


def test_thread_safe_digest_concurrency() -> None:
    digest = TelemetryDigest(thread_safe=True)
    per_thread = 5000
    threads = 4

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for _ in range(threads):
            executor.submit(_emit_events, digest, per_thread)

    summary = digest.summarize()
    assert summary["event_count"] == per_thread * threads
    assert summary["per_stage"]["stage"]["count"] == per_thread * threads
