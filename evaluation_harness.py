#!/usr/bin/env python3
"""Configuration-driven evaluation harness for trajectory metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from deterministic_registry import build_registry
from experiment_registry import create_run_artifacts, write_resolved_config
from regression_baselines import compare_metrics, load_baseline_store, upsert_baseline
from data_persistence import (
    frame_diagnostics_artifact_path,
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
from telemetry_intelligence import (
    TelemetryDigest,
    TelemetryDriftThresholds,
    compare_telemetry_summaries,
    load_telemetry_events,
    load_telemetry_summary,
    summarize_telemetry_events,
    summarize_telemetry_streaming,
    telemetry_metrics_from_summary,
    write_telemetry_summary,
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
    baseline_thresholds: dict[str, float | dict[str, Any]] | None
    write_baseline: bool
    telemetry_baseline_key: str | None
    telemetry_baseline_thresholds: dict[str, float | dict[str, Any]] | None
    write_telemetry_baseline: bool
    telemetry_drift_baseline_path: Path | None
    telemetry_drift_thresholds: dict[str, Any] | None
    telemetry_drift_report_name: str
    write_telemetry_drift_baseline: bool
    relocalization_baseline_key: str | None
    relocalization_baseline_thresholds: dict[str, float | dict[str, Any]] | None
    write_relocalization_baseline: bool
    relocalization_report_name: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    telemetry_config = baseline_config.get("telemetry", {}) if baseline_config else {}
    telemetry_baseline_key = telemetry_config.get("key") if telemetry_config else None
    telemetry_baseline_thresholds = telemetry_config.get("thresholds") if telemetry_config else None
    write_telemetry_baseline = bool(telemetry_config.get("write", False)) if telemetry_config else False
    telemetry_drift_config = telemetry_config.get("drift", {}) if telemetry_config else {}
    telemetry_drift_baseline = telemetry_drift_config.get("baseline_path") if telemetry_drift_config else None
    telemetry_drift_thresholds = telemetry_drift_config.get("thresholds") if telemetry_drift_config else None
    telemetry_drift_report_name = telemetry_drift_config.get(
        "report_name",
        "telemetry_drift_report.json",
    )
    write_telemetry_drift_baseline = bool(telemetry_drift_config.get("write", False)) if telemetry_drift_config else False
    if telemetry_baseline_thresholds and telemetry_baseline_key is None and baseline_key:
        telemetry_baseline_key = f"{baseline_key}_telemetry"
    relocalization_config = baseline_config.get("relocalization", {}) if baseline_config else {}
    relocalization_baseline_key = (
        relocalization_config.get("key") if relocalization_config else None
    )
    relocalization_baseline_thresholds = (
        relocalization_config.get("thresholds") if relocalization_config else None
    )
    write_relocalization_baseline = (
        bool(relocalization_config.get("write", False)) if relocalization_config else False
    )
    if relocalization_baseline_thresholds and relocalization_baseline_key is None and baseline_key:
        relocalization_baseline_key = f"{baseline_key}_relocalization"
    relocalization_report_name = str(
        relocalization_config.get("report_name", "relocalization_demo_report.json")
    )

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
        telemetry_baseline_key=str(telemetry_baseline_key) if telemetry_baseline_key else None,
        telemetry_baseline_thresholds=telemetry_baseline_thresholds,
        write_telemetry_baseline=write_telemetry_baseline,
        telemetry_drift_baseline_path=_resolve_path(telemetry_drift_baseline, base_dir)
        if telemetry_drift_baseline
        else None,
        telemetry_drift_thresholds=telemetry_drift_thresholds,
        telemetry_drift_report_name=str(telemetry_drift_report_name),
        write_telemetry_drift_baseline=write_telemetry_drift_baseline,
        relocalization_baseline_key=str(relocalization_baseline_key)
        if relocalization_baseline_key
        else None,
        relocalization_baseline_thresholds=relocalization_baseline_thresholds,
        write_relocalization_baseline=write_relocalization_baseline,
        relocalization_report_name=relocalization_report_name,
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
    return load_telemetry_events(path)


def _load_relocalization_report(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Relocalization report at {path} is invalid")
    return payload


def _relocalization_metrics_from_report(report: dict[str, Any]) -> dict[str, float]:
    summary = report.get("relocalization_summary")
    if not isinstance(summary, dict):
        raise ValueError("Relocalization report missing 'relocalization_summary'")
    return {
        "relocalization_attempts": float(summary.get("attempts", 0.0)),
        "relocalization_successes": float(summary.get("successes", 0.0)),
        "relocalization_success_rate": float(summary.get("success_rate", 0.0)),
        "relocalization_latency_mean_s": float(summary.get("latency_mean_s", 0.0)),
        "relocalization_latency_p50_s": float(summary.get("latency_p50_s", 0.0)),
        "relocalization_latency_p95_s": float(summary.get("latency_p95_s", 0.0)),
        "relocalization_match_count_mean": float(summary.get("match_count_mean", 0.0)),
        "relocalization_match_count_p50": float(summary.get("match_count_p50", 0.0)),
        "relocalization_match_count_p95": float(summary.get("match_count_p95", 0.0)),
        "relocalization_inlier_ratio_mean": float(summary.get("inlier_ratio_mean", 0.0)),
        "relocalization_inlier_ratio_p50": float(summary.get("inlier_ratio_p50", 0.0)),
        "relocalization_inlier_ratio_p95": float(summary.get("inlier_ratio_p95", 0.0)),
        "relocalization_recovery_success": float(summary.get("recovery_success", 0.0)),
        "relocalization_recovery_frame_gap": float(summary.get("recovery_frame_gap", 0.0)),
    }


def _telemetry_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    return telemetry_metrics_from_summary(summary)


def _summarize_telemetry_streaming(
    path: Path,
    global_accumulator: TelemetryDigest | None = None,
) -> dict[str, Any]:
    return summarize_telemetry_streaming(path, global_digest=global_accumulator)


def _summarize_telemetry_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return summarize_telemetry_events(events)


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
    registry = build_registry(seed=config.seed, config_path=config.config_path, config_hash=config.config_hash)
    registry.apply_global_seed()
    artifacts = create_run_artifacts(
        config.output_dir,
        config.run_id,
        config.config_path,
        config.config_hash,
        config.seed,
        config.use_run_subdir,
    )
    output_dir = artifacts.run_dir
    write_resolved_config(output_dir, config.resolved_config)

    per_sequence_metrics: dict[str, dict[str, float]] = {}
    per_sequence_reports: dict[str, dict[str, Any]] = {}
    per_sequence_telemetry: dict[str, dict[str, Any]] = {}
    per_sequence_relocalization_metrics: dict[str, dict[str, float]] = {}
    telemetry_accumulator = TelemetryDigest()

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
                    telemetry_accumulator.ingest_events(events)
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
            relocalization_path = entry.est_run_dir / config.relocalization_report_name
            if relocalization_path.exists():
                try:
                    relocalization_report = _load_relocalization_report(relocalization_path)
                    relocalization_metrics = _relocalization_metrics_from_report(
                        relocalization_report
                    )
                    payload["relocalization_summary"] = relocalization_report.get(
                        "relocalization_summary", {}
                    )
                    payload["relocalization_metrics"] = relocalization_metrics
                    per_sequence_relocalization_metrics[entry.name] = relocalization_metrics
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    LOGGER.warning(
                        "Failed to load relocalization report for %s (%s)",
                        entry.name,
                        exc,
                    )
            else:
                LOGGER.info(
                    "Relocalization report not found for %s at %s",
                    entry.name,
                    relocalization_path,
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
    relocalization_metrics = _aggregate_metrics(per_sequence_relocalization_metrics)
    telemetry_summary = telemetry_accumulator.summarize() if telemetry_accumulator.total_events else {}
    telemetry_metrics = _telemetry_metrics_from_summary(telemetry_summary)
    telemetry_drift_report = None
    if telemetry_summary and config.telemetry_drift_baseline_path:
        try:
            baseline_summary = load_telemetry_summary(config.telemetry_drift_baseline_path)
            thresholds = TelemetryDriftThresholds(
                relative_increase=float(
                    config.telemetry_drift_thresholds.get("relative_increase", 0.1)
                    if config.telemetry_drift_thresholds
                    else 0.1
                ),
                absolute_increase_s=float(
                    config.telemetry_drift_thresholds.get("absolute_increase_s", 0.01)
                    if config.telemetry_drift_thresholds
                    else 0.01
                ),
                metrics=tuple(
                    config.telemetry_drift_thresholds.get("metrics", ("mean_duration_s", "p95_duration_s"))
                    if config.telemetry_drift_thresholds
                    else ("mean_duration_s", "p95_duration_s")
                ),
            )
            telemetry_drift_report = compare_telemetry_summaries(
                telemetry_summary,
                baseline_summary,
                thresholds,
            )
            write_telemetry_summary(
                output_dir / config.telemetry_drift_report_name,
                telemetry_drift_report,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to evaluate telemetry drift (%s)", exc)
    if telemetry_summary and config.telemetry_drift_baseline_path and config.write_telemetry_drift_baseline:
        try:
            write_telemetry_summary(config.telemetry_drift_baseline_path, telemetry_summary)
            LOGGER.info(
                "Telemetry drift baseline updated at %s",
                config.telemetry_drift_baseline_path,
            )
        except OSError as exc:
            LOGGER.warning("Failed to write telemetry drift baseline (%s)", exc)
    summary = {
        "run_id": config.run_id,
        "dataset": config.dataset,
        "seed": config.seed,
        "timestamp": _timestamp(),
        "config_path": str(config.config_path),
        "config_hash": config.config_hash,
        "run_dir": str(output_dir),
        "run_metadata": str(artifacts.metadata_path),
        "baseline_key": config.baseline_key,
        "telemetry_baseline_key": config.telemetry_baseline_key,
        "relocalization_baseline_key": config.relocalization_baseline_key,
        "aggregate_metrics": aggregate,
        "relocalization_metrics": relocalization_metrics,
        "telemetry_summary": telemetry_summary,
        "telemetry_metrics": telemetry_metrics,
        "telemetry_per_sequence": per_sequence_telemetry,
        "telemetry_drift_report": telemetry_drift_report,
        "relocalization_per_sequence": per_sequence_relocalization_metrics,
        "per_sequence": per_sequence_reports,
    }

    baseline_comparison = None
    telemetry_baseline_comparison = None
    relocalization_baseline_comparison = None
    store = None
    baselines: dict[str, Any] = {}
    if config.baseline_store:
        store = load_baseline_store(config.baseline_store)
        baselines = store.get("baselines", {})
    if config.baseline_store and config.baseline_key:
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
                "stats": comparison.stats,
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

    if config.baseline_store and config.telemetry_baseline_key:
        baseline = baselines.get(config.telemetry_baseline_key) if baselines else None
        if baseline:
            comparison = compare_metrics(
                config.telemetry_baseline_key,
                telemetry_metrics,
                baseline,
                config.telemetry_baseline_thresholds,
            )
            telemetry_baseline_comparison = {
                "key": comparison.key,
                "status": comparison.status,
                "per_metric": comparison.per_metric,
                "stats": comparison.stats,
            }
            summary["telemetry_baseline_comparison"] = telemetry_baseline_comparison
            write_json_report(
                output_dir / "telemetry_baseline_comparison.json",
                telemetry_baseline_comparison,
            )
            LOGGER.info(
                "Telemetry baseline comparison for %s: %s",
                config.telemetry_baseline_key,
                comparison.status,
            )
        else:
            LOGGER.warning(
                "Telemetry baseline key '%s' not found in %s",
                config.telemetry_baseline_key,
                config.baseline_store,
            )
    if config.baseline_store and config.telemetry_baseline_key and config.write_telemetry_baseline:
        upsert_baseline(
            config.baseline_store,
            config.telemetry_baseline_key,
            telemetry_metrics,
            config.config_hash,
            metadata={"dataset": config.dataset, "run_id": config.run_id, "telemetry": True},
        )
        LOGGER.info(
            "Telemetry baseline '%s' updated at %s",
            config.telemetry_baseline_key,
            config.baseline_store,
        )

    if config.baseline_store and config.relocalization_baseline_key:
        baseline = baselines.get(config.relocalization_baseline_key) if baselines else None
        if baseline:
            comparison = compare_metrics(
                config.relocalization_baseline_key,
                relocalization_metrics,
                baseline,
                config.relocalization_baseline_thresholds,
            )
            relocalization_baseline_comparison = {
                "key": comparison.key,
                "status": comparison.status,
                "per_metric": comparison.per_metric,
                "stats": comparison.stats,
            }
            summary["relocalization_baseline_comparison"] = relocalization_baseline_comparison
            write_json_report(
                output_dir / "relocalization_baseline_comparison.json",
                relocalization_baseline_comparison,
            )
            LOGGER.info(
                "Relocalization baseline comparison for %s: %s",
                config.relocalization_baseline_key,
                comparison.status,
            )
        else:
            LOGGER.warning(
                "Relocalization baseline key '%s' not found in %s",
                config.relocalization_baseline_key,
                config.baseline_store,
            )
    if (
        config.baseline_store
        and config.relocalization_baseline_key
        and config.write_relocalization_baseline
    ):
        upsert_baseline(
            config.baseline_store,
            config.relocalization_baseline_key,
            relocalization_metrics,
            config.config_hash,
            metadata={"dataset": config.dataset, "run_id": config.run_id, "relocalization": True},
        )
        LOGGER.info(
            "Relocalization baseline '%s' updated at %s",
            config.relocalization_baseline_key,
            config.baseline_store,
        )

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
