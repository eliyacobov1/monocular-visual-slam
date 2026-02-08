"""Benchmark determinism validation digest throughput."""

from __future__ import annotations

import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from data_persistence import RunDataStore, build_metrics_bundle
from determinism_validation import build_run_digest


def _populate_run(run_dir: Path, trajectories: int, metrics: int) -> None:
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps({"run_id": "benchmark"}), encoding="utf-8")
    store = RunDataStore.create(
        base_dir=run_dir,
        run_id="benchmark",
        config_path=config_path,
        config_hash="hash",
        seed=9,
        use_subdir=False,
        resolved_config={"run_id": "benchmark"},
    )

    rng = np.random.default_rng(0)
    for idx in range(trajectories):
        accumulator = store.create_accumulator(f"trajectory_{idx}")
        for frame_id in range(10):
            pose = np.eye(4)
            pose[0, 3] = float(rng.random())
            pose[1, 3] = float(rng.random())
            accumulator.append(pose, float(frame_id), frame_id)
        store.save_trajectory(accumulator.as_bundle())

    for idx in range(metrics):
        bundle = build_metrics_bundle(f"metrics_{idx}", {"num_poses": float(idx)})
        store.save_metrics(bundle)


def run_benchmark(trajectories: int = 20, metrics: int = 20) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        _populate_run(run_dir, trajectories=trajectories, metrics=metrics)

        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        start_time = time.perf_counter()
        digest = build_run_digest(run_dir)
        duration_s = time.perf_counter() - start_time
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        memory_stats = end_snapshot.compare_to(start_snapshot, "lineno")
        total_memory_delta = sum(stat.size_diff for stat in memory_stats)

    print("determinism_validation_benchmark")
    print(f"trajectories={trajectories}")
    print(f"metrics={metrics}")
    print(f"duration_s={duration_s:.4f}")
    print(f"memory_delta_bytes={total_memory_delta}")
    print(f"artifact_count={len(digest.artifacts)}")


if __name__ == "__main__":
    run_benchmark()
