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

from experiment_registry import create_run_artifacts, write_resolved_config
from regression_baselines import compare_metrics, load_baseline_store, upsert_baseline
from data_persistence import (
    P2Quantile,
    frame_diagnostics_artifact_path,
    iter_json_array_items,
    load_trajectory_npz,
    load_frame_diagnostics_json,
    summarize_frame_diagnostics,
    summarize_frame_diagnostics_streaming,
    trajectory_artifact_path,
    trajectory_positions,
)
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
    est_format: str | None
    est_run_dir: Path | None
    est_trajectory: str | None
    rpe_delta: int


@dataclass(frozen=True)
class EvaluationConfig:
    run_id: str
    dataset: str
    output_dir: Path
    seed: int
    use_run_subdir: bool
    trajectories: tuple[TrajectoryEntry, ...]
    config_path: Path
    config_hash: str
    resolved_config: dict[str, Any]
    pipeline_config: dict[str, Any] | None
    telemetry_name: str
    baseline_store: Path | None
    baseline_key: str | None
    baseline_thresholds: dict[str, float] | None
    write_baseline: bool


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
    est_run_dir = mapping.get("est_run_dir")
    est_trajectory = mapping.get("est_trajectory")
    est_format = mapping.get("est_format")
    if est_run_dir and mapping.get("est_path"):
        raise ValueError("Use either est_path or est_run_dir, not both")
    if est_run_dir:
        if est_trajectory is None:
            est_trajectory = str(mapping["name"])
        run_dir = _resolve_path(est_run_dir, base_dir)
        est_path = trajectory_artifact_path(run_dir, str(est_trajectory))
        if est_format is None:
            est_format = "slam_npz"
    else:
        est_path = _resolve_path(mapping["est_path"], base_dir)
    return TrajectoryEntry(
        name=str(mapping["name"]),
        gt_path=_resolve_path(mapping["gt_path"], base_dir),
        est_path=est_path,
        format=mapping.get("format", "xy"),
        cols=mapping.get("cols"),
        est_cols=mapping.get("est_cols"),
        est_format=est_format,
        est_run_dir=_resolve_path(est_run_dir, base_dir) if est_run_dir else None,
        est_trajectory=str(est_trajectory) if est_trajectory is not None else None,
        rpe_delta=int(mapping.get("rpe_delta", 1)),
    )


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    if any(key in raw for key in ("run", "evaluation", "pipeline", "baseline")):
        run_section = raw.get("run", {})
        eval_section = raw.get("evaluation", {})
        dataset = run_section.get(
            "dataset",
            raw.get("dataset", eval_section.get("dataset", "custom")),
        )
        output_dir = run_section.get("output_dir", raw.get("output_dir", "reports"))
        seed = run_section.get("seed", raw.get("seed", 0))
        run_id = run_section.get("run_id", raw.get("run_id"))
        use_run_subdir = run_section.get(
            "use_run_subdir",
            raw.get("use_run_subdir", False),
        )
        telemetry_name = run_section.get(
            "telemetry_name",
            raw.get("telemetry_name", eval_section.get("telemetry_name", "slam_telemetry")),
        )
        merged = dict(eval_section)
        merged.update(
            {
                "dataset": dataset,
                "output_dir": output_dir,
                "seed": seed,
                "run_id": run_id,
                "use_run_subdir": use_run_subdir,
                "telemetry_name": telemetry_name,
            }
        )
        merged["pipeline"] = raw.get("pipeline")
        merged["baseline"] = raw.get("baseline")
        return merged
    return raw


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
                est_format=None,
                est_run_dir=None,
                est_trajectory=None,
            )
        )
    return entries


def load_config(config_path: Path) -> EvaluationConfig:
    base_dir = config_path.parent
    raw = _load_json(config_path)
    normalized = _normalize_config(raw)
    config_hash = _hash_config(config_path)

    run_id = normalized.get("run_id", config_path.stem)
    dataset = normalized.get("dataset", "custom")
    output_dir = _resolve_path(normalized.get("output_dir", "reports"), base_dir)
    seed = int(normalized.get("seed", 0))
    use_run_subdir = bool(normalized.get("use_run_subdir", False))

    trajectories_data = normalized.get("trajectories")
    if trajectories_data is None and dataset.lower() == "kitti":
        trajectories = tuple(_build_kitti_entries(normalized, base_dir))
    elif trajectories_data:
        trajectories = tuple(
            _build_entry_from_mapping(entry, base_dir) for entry in trajectories_data
        )
    else:
        raise ValueError(
            "Config must include 'trajectories' or specify dataset='kitti' with sequences"
        )

    resolved_trajectories = [
        {
            "name": entry.name,
            "gt_path": str(entry.gt_path),
            "est_path": str(entry.est_path),
            "format": entry.format,
            "cols": entry.cols,
            "est_cols": entry.est_cols,
            "est_format": entry.est_format,
            "est_run_dir": str(entry.est_run_dir) if entry.est_run_dir else None,
            "est_trajectory": entry.est_trajectory,
            "rpe_delta": entry.rpe_delta,
        }
        for entry in trajectories
    ]
    telemetry_name = str(normalized.get("telemetry_name", "slam_telemetry"))
    resolved_config = {
        "run_id": run_id,
        "dataset": dataset,
        "output_dir": str(output_dir),
        "seed": seed,
        "use_run_subdir": use_run_subdir,
        "trajectories": resolved_trajectories,
        "config_hash": config_hash,
        "telemetry_name": telemetry_name,
    }
    pipeline_config = normalized.get("pipeline")
    if pipeline_config:
        resolved_config["pipeline"] = pipeline_config
    baseline_config = normalized.get("baseline") or {}
    if baseline_config:
        resolved_config["baseline"] = baseline_config

    baseline_store = baseline_config.get("store_path") if baseline_config else None
    baseline_key = baseline_config.get("key") if baseline_config else None
    baseline_thresholds = baseline_config.get("thresholds") if baseline_config else None
    write_baseline = bool(baseline_config.get("write", False)) if baseline_config else False

    return EvaluationConfig(
        run_id=run_id,
        dataset=str(dataset),
        output_dir=output_dir,
        seed=seed,
        use_run_subdir=use_run_subdir,
        trajectories=trajectories,
        config_path=config_path,
        config_hash=config_hash,
        resolved_config=resolved_config,
        pipeline_config=pipeline_config,
        telemetry_name=telemetry_name,
        baseline_store=_resolve_path(baseline_store, base_dir) if baseline_store else None,
        baseline_key=str(baseline_key) if baseline_key else None,
        baseline_thresholds=baseline_thresholds,
        write_baseline=write_baseline,
    )


def _parse_columns(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(c) for c in value.split(",")]


def _load_estimated_trajectory(
    entry: TrajectoryEntry, gt_cols: list[int]
) -> np.ndarray:
    if entry.est_format == "slam_npz":
        est_cols = _parse_columns(entry.est_cols)
        if est_cols is None:
            est_cols = list(range(len(gt_cols)))
        bundle = load_trajectory_npz(entry.est_path)
        return trajectory_positions(bundle, est_cols)
    est_cols = gt_cols if entry.est_cols is None else _parse_columns(entry.est_cols)
    if est_cols is None:
        raise ValueError("Estimated trajectory columns could not be resolved")
    return load_traj(str(entry.est_path), est_cols)


def _evaluate_entry(entry: TrajectoryEntry) -> dict[str, Any]:
    cols = resolve_columns(entry.format, entry.cols)
    gt = load_traj(str(entry.gt_path), cols)
    est = _load_estimated_trajectory(entry, cols)
    if gt.shape[1] != est.shape[1]:
        raise ValueError(
            f"Trajectory dimension mismatch: gt has {gt.shape[1]} dims, est has {est.shape[1]}"
        )

    metrics = compute_additional_metrics(gt, est, delta=entry.rpe_delta)
    metadata = {
        "sequence": entry.name,
        "format": entry.format,
        "est_format": entry.est_format or entry.format,
        "rpe_delta": entry.rpe_delta,
        "gt_path": str(entry.gt_path),
        "est_path": str(entry.est_path),
    }
    if entry.est_run_dir:
        metadata["est_run_dir"] = str(entry.est_run_dir)
        metadata["est_trajectory"] = entry.est_trajectory or entry.name
    payload = build_report_payload(metrics, metadata)
    return payload


def _aggregate_metrics(per_sequence: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_sequence:
        return {}
    metrics: dict[str, float] = {}
    keys: set[str] = set()
    for seq_metrics in per_sequence.values():
        keys.update(seq_metrics.keys())
    for key in sorted(keys):
        values = [seq_metrics[key] for seq_metrics in per_sequence.values() if key in seq_metrics]
        if not values:
            continue
        metrics[key] = float(np.mean(values))
    return metrics


def _load_telemetry_events(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"Telemetry payload at {path} is missing 'events' list")
    return events


class _TelemetryStageStats:
    def __init__(self) -> None:
        self.count = 0
        self.total_duration = 0.0
        self.min_duration = float("inf")
        self.max_duration = float("-inf")
        self.p95 = P2Quantile(0.95)

    def update(self, duration: float) -> None:
        self.count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.p95.update(duration)

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "total_duration_s": 0.0,
                "mean_duration_s": 0.0,
                "min_duration_s": 0.0,
                "max_duration_s": 0.0,
                "p95_duration_s": 0.0,
            }
        return {
            "count": int(self.count),
            "total_duration_s": float(self.total_duration),
            "mean_duration_s": float(self.total_duration / self.count),
            "min_duration_s": float(self.min_duration),
            "max_duration_s": float(self.max_duration),
            "p95_duration_s": float(self.p95.value()),
        }


class _TelemetryAccumulator:
    def __init__(self) -> None:
        self.total_events = 0
        self.total_duration = 0.0
        self.per_stage: dict[str, _TelemetryStageStats] = {}

    def update(self, name: str, duration: float) -> None:
        self.total_events += 1
        self.total_duration += duration
        stats = self.per_stage.get(name)
        if stats is None:
            stats = _TelemetryStageStats()
            self.per_stage[name] = stats
        stats.update(duration)

    def summarize(self) -> dict[str, Any]:
        per_stage = {name: stats.summary() for name, stats in self.per_stage.items()}
        mean_duration = self.total_duration / self.total_events if self.total_events else 0.0
        return {
            "event_count": int(self.total_events),
            "total_duration_s": float(self.total_duration),
            "mean_duration_s": float(mean_duration),
            "per_stage": per_stage,
        }


def _summarize_telemetry_streaming(
    path: Path,
    global_accumulator: _TelemetryAccumulator | None = None,
) -> dict[str, Any]:
    local_accumulator = _TelemetryAccumulator()
    for event in iter_json_array_items(path, "events"):
        name = str(event.get("name") or "unknown")
        duration = float(event.get("duration_s", 0.0))
        local_accumulator.update(name, duration)
        if global_accumulator is not None:
            global_accumulator.update(name, duration)
    return local_accumulator.summarize()


def _summarize_telemetry_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    durations: dict[str, list[float]] = {}
    total_events = 0
    total_duration = 0.0
    for event in events:
        name = str(event.get("name", "unknown"))
        duration = float(event.get("duration_s", 0.0))
        durations.setdefault(name, []).append(duration)
        total_events += 1
        total_duration += duration

    per_stage: dict[str, dict[str, float]] = {}
    for name, values in durations.items():
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            continue
        per_stage[name] = {
            "count": int(arr.size),
            "total_duration_s": float(np.sum(arr)),
            "mean_duration_s": float(np.mean(arr)),
            "min_duration_s": float(np.min(arr)),
            "max_duration_s": float(np.max(arr)),
            "p95_duration_s": float(np.percentile(arr, 95)),
        }

    mean_duration = total_duration / total_events if total_events > 0 else 0.0
    return {
        "event_count": total_events,
        "total_duration_s": float(total_duration),
        "mean_duration_s": float(mean_duration),
        "per_stage": per_stage,
    }


def _resolve_telemetry_path(run_dir: Path, telemetry_name: str) -> Path:
    return run_dir / "telemetry" / f"{telemetry_name}.json"


def _resolve_diagnostics_path(run_dir: Path, name: str) -> Path:
    return frame_diagnostics_artifact_path(run_dir, name)


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
    artifacts = create_run_artifacts(
        config.output_dir,
        config.run_id,
        config.config_path,
        config.config_hash,
        config.use_run_subdir,
    )
    output_dir = artifacts.run_dir
    write_resolved_config(output_dir, config.resolved_config)

    per_sequence_metrics: dict[str, dict[str, float]] = {}
    per_sequence_reports: dict[str, dict[str, Any]] = {}
    per_sequence_telemetry: dict[str, dict[str, Any]] = {}
    telemetry_accumulator = _TelemetryAccumulator()

    LOGGER.info("Starting evaluation run %s", config.run_id)
    for entry in config.trajectories:
        LOGGER.info("Evaluating sequence %s", entry.name)
        payload = _evaluate_entry(entry)
        if entry.est_run_dir:
            telemetry_path = _resolve_telemetry_path(entry.est_run_dir, config.telemetry_name)
            if telemetry_path.exists():
                try:
                    per_sequence_telemetry[entry.name] = _summarize_telemetry_streaming(
                        telemetry_path,
                        telemetry_accumulator,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    LOGGER.warning(
                        "Streaming telemetry summary failed for %s (%s); falling back to in-memory",
                        entry.name,
                        exc,
                    )
                    events = _load_telemetry_events(telemetry_path)
                    per_sequence_telemetry[entry.name] = _summarize_telemetry_events(events)
                    for event in events:
                        name = str(event.get("name") or "unknown")
                        duration = float(event.get("duration_s", 0.0))
                        telemetry_accumulator.update(name, duration)
            else:
                LOGGER.warning("Telemetry file not found for %s at %s", entry.name, telemetry_path)
            diagnostics_path = _resolve_diagnostics_path(entry.est_run_dir, "frame_diagnostics")
            if diagnostics_path.exists():
                try:
                    diagnostics_summary = summarize_frame_diagnostics_streaming(diagnostics_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    LOGGER.warning(
                        "Streaming diagnostics summary failed for %s (%s); falling back to in-memory",
                        entry.name,
                        exc,
                    )
                    diagnostics_bundle = load_frame_diagnostics_json(diagnostics_path)
                    diagnostics_summary = summarize_frame_diagnostics(diagnostics_bundle)
                payload["diagnostics_summary"] = diagnostics_summary
                payload["metrics"] = {**payload["metrics"], **diagnostics_summary}
            else:
                LOGGER.warning(
                    "Frame diagnostics not found for %s at %s",
                    entry.name,
                    diagnostics_path,
                )
        if entry.name in per_sequence_telemetry:
            payload["telemetry"] = per_sequence_telemetry[entry.name]
        per_sequence_reports[entry.name] = payload
        metrics = payload["metrics"]
        per_sequence_metrics[entry.name] = metrics

        report_base = output_dir / f"{entry.name}_metrics"
        write_report(report_base.with_suffix(".txt"), [f"{k} {v:.4f}" for k, v in metrics.items()])
        write_json_report(report_base.with_suffix(".json"), payload)
        write_csv_report(report_base.with_suffix(".csv"), payload)

    aggregate = _aggregate_metrics(per_sequence_metrics)
    telemetry_summary = telemetry_accumulator.summarize() if telemetry_accumulator.total_events else {}
    summary = {
        "run_id": config.run_id,
        "dataset": config.dataset,
        "seed": config.seed,
        "timestamp": _timestamp(),
        "config_path": str(config.config_path),
        "config_hash": config.config_hash,
        "run_dir": str(output_dir),
        "run_metadata": str(artifacts.metadata_path),
        "aggregate_metrics": aggregate,
        "telemetry_summary": telemetry_summary,
        "telemetry_per_sequence": per_sequence_telemetry,
        "per_sequence": per_sequence_reports,
    }

    baseline_comparison = None
    if config.baseline_store and config.baseline_key:
        store = load_baseline_store(config.baseline_store)
        baselines = store.get("baselines", {})
        baseline = baselines.get(config.baseline_key)
        if baseline:
            comparison = compare_metrics(
                config.baseline_key,
                aggregate,
                baseline,
                config.baseline_thresholds,
            )
            baseline_comparison = {
                "key": comparison.key,
                "status": comparison.status,
                "per_metric": comparison.per_metric,
            }
            summary["baseline_comparison"] = baseline_comparison
            write_json_report(output_dir / "baseline_comparison.json", baseline_comparison)
            LOGGER.info(
                "Baseline comparison for %s: %s",
                config.baseline_key,
                comparison.status,
            )
        else:
            LOGGER.warning("Baseline key '%s' not found in %s", config.baseline_key, config.baseline_store)
    if config.baseline_store and config.baseline_key and config.write_baseline:
        upsert_baseline(
            config.baseline_store,
            config.baseline_key,
            aggregate,
            config.config_hash,
            metadata={"dataset": config.dataset, "run_id": config.run_id},
        )
        LOGGER.info("Baseline '%s' updated at %s", config.baseline_key, config.baseline_store)

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
