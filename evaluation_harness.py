#!/usr/bin/env python3
"""Configuration-driven evaluation harness for trajectory metrics."""

from __future__ import annotations

import argparse
import json
import logging
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluate_trajectory import (
    build_report_payload,
    compute_additional_metrics,
    load_traj,
    resolve_columns,
    write_csv_report,
    write_json_report,
    write_report,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryEntry:
    name: str
    gt_path: Path
    est_path: Path
    format: str
    cols: str | None
    est_cols: str | None
    rpe_delta: int


@dataclass(frozen=True)
class EvaluationConfig:
    run_id: str
    dataset: str
    output_dir: Path
    seed: int
    trajectories: tuple[TrajectoryEntry, ...]
    config_path: Path
    config_hash: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _hash_config(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _build_entry_from_mapping(mapping: dict[str, Any], base_dir: Path) -> TrajectoryEntry:
    return TrajectoryEntry(
        name=str(mapping["name"]),
        gt_path=_resolve_path(mapping["gt_path"], base_dir),
        est_path=_resolve_path(mapping["est_path"], base_dir),
        format=mapping.get("format", "xy"),
        cols=mapping.get("cols"),
        est_cols=mapping.get("est_cols"),
        rpe_delta=int(mapping.get("rpe_delta", 1)),
    )


def _build_kitti_entries(config: dict[str, Any], base_dir: Path) -> list[TrajectoryEntry]:
    sequences = config.get("sequences")
    if not sequences:
        raise ValueError("KITTI config must include non-empty 'sequences'")

    kitti_root = _resolve_path(config["kitti_root"], base_dir)
    gt_root = _resolve_path(config.get("gt_root", kitti_root / "poses"), base_dir)
    est_root = _resolve_path(config["est_root"], base_dir)
    gt_pattern = config.get("gt_pattern", "{sequence}.txt")
    est_pattern = config.get("estimate_pattern", "{sequence}.txt")
    rpe_delta = int(config.get("rpe_delta", 1))

    entries: list[TrajectoryEntry] = []
    for seq in sequences:
        name = str(seq)
        entries.append(
            TrajectoryEntry(
                name=name,
                gt_path=_resolve_path(gt_pattern.format(sequence=name), gt_root),
                est_path=_resolve_path(est_pattern.format(sequence=name), est_root),
                format=config.get("format", "kitti_odom"),
                cols=None,
                est_cols=None,
                rpe_delta=rpe_delta,
            )
        )
    return entries


def load_config(config_path: Path) -> EvaluationConfig:
    base_dir = config_path.parent
    raw = _load_json(config_path)
    config_hash = _hash_config(config_path)

    run_id = raw.get("run_id", config_path.stem)
    dataset = raw.get("dataset", "custom")
    output_dir = _resolve_path(raw.get("output_dir", "reports"), base_dir)
    seed = int(raw.get("seed", 0))

    trajectories_data = raw.get("trajectories")
    if trajectories_data is None and dataset.lower() == "kitti":
        trajectories = tuple(_build_kitti_entries(raw, base_dir))
    elif trajectories_data:
        trajectories = tuple(
            _build_entry_from_mapping(entry, base_dir) for entry in trajectories_data
        )
    else:
        raise ValueError(
            "Config must include 'trajectories' or specify dataset='kitti' with sequences"
        )

    return EvaluationConfig(
        run_id=run_id,
        dataset=str(dataset),
        output_dir=output_dir,
        seed=seed,
        trajectories=trajectories,
        config_path=config_path,
        config_hash=config_hash,
    )


def _evaluate_entry(entry: TrajectoryEntry) -> dict[str, Any]:
    cols = resolve_columns(entry.format, entry.cols)
    est_cols = cols if entry.est_cols is None else [int(c) for c in entry.est_cols.split(",")]
    gt = load_traj(str(entry.gt_path), cols)
    est = load_traj(str(entry.est_path), est_cols)

    metrics = compute_additional_metrics(gt, est, delta=entry.rpe_delta)
    metadata = {
        "sequence": entry.name,
        "format": entry.format,
        "rpe_delta": entry.rpe_delta,
        "gt_path": str(entry.gt_path),
        "est_path": str(entry.est_path),
    }
    payload = build_report_payload(metrics, metadata)
    return payload


def _aggregate_metrics(per_sequence: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_sequence:
        return {}
    metrics = {}
    keys = list(next(iter(per_sequence.values())).keys())
    for key in keys:
        values = [seq_metrics[key] for seq_metrics in per_sequence.values()]
        metrics[key] = float(np.mean(values))
    return metrics


def _write_summary_csv(path: Path, per_sequence: dict[str, dict[str, float]], aggregate: dict[str, float]) -> None:
    lines = ["sequence,metric,value"]
    for sequence, metrics in per_sequence.items():
        for metric, value in metrics.items():
            lines.append(f"{sequence},{metric},{value}")
    if aggregate:
        for metric, value in aggregate.items():
            lines.append(f"aggregate,{metric},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    _set_deterministic_seed(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sequence_metrics: dict[str, dict[str, float]] = {}
    per_sequence_reports: dict[str, dict[str, Any]] = {}

    LOGGER.info("Starting evaluation run %s", config.run_id)
    for entry in config.trajectories:
        LOGGER.info("Evaluating sequence %s", entry.name)
        payload = _evaluate_entry(entry)
        per_sequence_reports[entry.name] = payload
        metrics = payload["metrics"]
        per_sequence_metrics[entry.name] = metrics

        report_base = output_dir / f"{entry.name}_metrics"
        write_report(report_base.with_suffix(".txt"), [f"{k} {v:.4f}" for k, v in metrics.items()])
        write_json_report(report_base.with_suffix(".json"), payload)
        write_csv_report(report_base.with_suffix(".csv"), payload)

    aggregate = _aggregate_metrics(per_sequence_metrics)
    summary = {
        "run_id": config.run_id,
        "dataset": config.dataset,
        "seed": config.seed,
        "timestamp": _timestamp(),
        "config_path": str(config.config_path),
        "config_hash": config.config_hash,
        "aggregate_metrics": aggregate,
        "per_sequence": per_sequence_reports,
    }

    write_json_report(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", per_sequence_metrics, aggregate)
    LOGGER.info("Evaluation run %s complete", config.run_id)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory evaluation from config")
    parser.add_argument("--config", required=True, help="Path to evaluation JSON config")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(Path(args.config))
    run_evaluation(config)


if __name__ == "__main__":
    main()
