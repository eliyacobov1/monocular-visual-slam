#!/usr/bin/env python3
"""Benchmark telemetry intelligence summaries and drift evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from telemetry_intelligence import (
    TelemetryDigest,
    TelemetryDriftThresholds,
    compare_telemetry_summaries,
    summarize_telemetry_events,
    telemetry_metrics_from_summary,
)


def _build_events(count: int, stages: int) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(42)
    stage_names = [f"stage_{idx}" for idx in range(stages)]
    durations = rng.uniform(0.001, 0.05, size=count)
    names = rng.choice(stage_names, size=count)
    return [
        {"name": str(name), "duration_s": float(duration)}
        for name, duration in zip(names, durations)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark telemetry intelligence layer.")
    parser.add_argument("--events", type=int, default=200_000, help="Number of events")
    parser.add_argument("--stages", type=int, default=8, help="Number of stages")
    parser.add_argument("--output", type=Path, default=Path("telemetry_intelligence_benchmark.json"))
    args = parser.parse_args()

    events = _build_events(args.events, args.stages)

    start = perf_counter()
    summary = summarize_telemetry_events(events)
    duration_summary = perf_counter() - start

    digest = TelemetryDigest()
    start = perf_counter()
    digest.ingest_events(events)
    digest_summary = digest.summarize()
    duration_digest = perf_counter() - start

    start = perf_counter()
    metrics = telemetry_metrics_from_summary(summary)
    duration_metrics = perf_counter() - start

    thresholds = TelemetryDriftThresholds(relative_increase=0.05, absolute_increase_s=0.002)
    start = perf_counter()
    drift_report = compare_telemetry_summaries(digest_summary, summary, thresholds)
    duration_drift = perf_counter() - start

    payload = {
        "event_count": args.events,
        "stages": args.stages,
        "durations_s": {
            "summary": duration_summary,
            "digest": duration_digest,
            "metrics": duration_metrics,
            "drift": duration_drift,
        },
        "summary": summary,
        "digest_summary": digest_summary,
        "drift_status": drift_report.get("status"),
        "metrics": metrics,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Telemetry intelligence benchmark written to {args.output}")


if __name__ == "__main__":
    main()
