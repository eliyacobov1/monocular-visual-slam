#!/usr/bin/env python3
"""Benchmark telemetry aggregation performance for evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation_harness import _summarize_telemetry_events


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
    parser = argparse.ArgumentParser(description="Benchmark telemetry aggregation.")
    parser.add_argument("--events", type=int, default=100_000, help="Number of events to aggregate")
    parser.add_argument("--stages", type=int, default=6, help="Number of distinct stages")
    parser.add_argument("--output", type=Path, default=Path("telemetry_aggregation_benchmark.json"))
    args = parser.parse_args()

    events = _build_events(args.events, args.stages)
    start = perf_counter()
    summary = _summarize_telemetry_events(events)
    duration = perf_counter() - start

    payload = {
        "event_count": args.events,
        "stages": args.stages,
        "elapsed_s": duration,
        "summary": summary,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Aggregated {args.events} events in {duration:.4f}s -> {args.output}")


if __name__ == "__main__":
    main()
