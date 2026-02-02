"""Benchmark telemetry recording overhead."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from run_telemetry import RunTelemetryRecorder, TelemetryEvent


def benchmark(recorder: RunTelemetryRecorder, iterations: int) -> float:
    start = perf_counter()
    for idx in range(iterations):
        recorder.record_event(
            TelemetryEvent(
                name="benchmark",
                duration_s=0.0,
                timestamp="0",
                metadata={"idx": idx},
            )
        )
    recorder.flush()
    return perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark telemetry recorder.")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("telemetry_benchmark.json"))
    args = parser.parse_args()

    recorder = RunTelemetryRecorder(args.output)
    duration = benchmark(recorder, args.iterations)
    per_event = duration / max(args.iterations, 1)
    print(f"iterations={args.iterations}")
    print(f"duration_s={duration:.6f}")
    print(f"per_event_s={per_event:.9f}")


if __name__ == "__main__":
    main()
