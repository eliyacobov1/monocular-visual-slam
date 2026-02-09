"""Benchmark telemetry correlation and resource summary aggregation."""

from __future__ import annotations

import argparse
import json
import random
import tracemalloc
from pathlib import Path
from time import perf_counter

from telemetry_intelligence import TelemetryDigest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark telemetry correlation summaries.")
    parser.add_argument("--events", type=int, default=50000)
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=Path("telemetry_correlation_summary_benchmark.json")
    )
    return parser


def _run(events: int, stages: int) -> dict[str, object]:
    rng = random.Random(42)
    stage_names = [f"stage_{idx}" for idx in range(stages)]
    digest = TelemetryDigest()
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    start = perf_counter()
    for idx in range(events):
        stage = stage_names[idx % stages]
        duration = rng.random() * 0.02
        memory_delta = rng.randint(-2048, 4096)
        digest.update(
            stage,
            duration,
            memory_delta_bytes=float(memory_delta),
            correlation_id=f"cid-{stage}",
        )
    summary = digest.summarize()
    duration_s = perf_counter() - start
    end_mem = tracemalloc.get_traced_memory()[0]
    return {
        "events": events,
        "stages": stages,
        "duration_s": duration_s,
        "memory_delta_bytes": float(end_mem - start_mem),
        "summary": summary,
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.events <= 0:
        raise SystemExit("events must be positive")
    if args.stages <= 0:
        raise SystemExit("stages must be positive")
    payload = _run(args.events, args.stages)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
