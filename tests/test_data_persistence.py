"""Tests for the data persistence layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_persistence import RunDataStore, build_metrics_bundle


def test_run_data_store_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_id": "unit"}), encoding="utf-8")

    store = RunDataStore.create(
        base_dir=tmp_path,
        run_id="unit_run",
        config_path=config_path,
        config_hash="hash",
        use_subdir=False,
        resolved_config={"run_id": "unit_run"},
    )

    accumulator = store.create_accumulator("trajectory")
    accumulator.append(np.eye(4), 0.0, 0)
    accumulator.append(np.eye(4), 1.0, 1)
    bundle = accumulator.as_bundle()

    store.save_trajectory(bundle)
    loaded = store.load_trajectory("trajectory")
    assert loaded.poses.shape == (2, 4, 4)
    assert loaded.timestamps.tolist() == [0.0, 1.0]

    metrics_bundle = build_metrics_bundle("metrics", {"num_poses": 2})
    store.save_metrics(metrics_bundle)
    loaded_metrics = store.load_metrics("metrics")
    assert loaded_metrics.metrics["num_poses"] == 2.0
