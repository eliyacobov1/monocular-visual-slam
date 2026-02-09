"""Generate deterministic readiness reports for interview-grade SLAM runs."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from data_persistence import sanitize_artifact_name
from deterministic_integrity import stable_hash

LOGGER = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required report not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read JSON report: {path}") from exc


def _coerce_float_map(payload: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items()}


def _coerce_int_map(payload: Mapping[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in payload.items()}


@dataclass(frozen=True)
class ReadinessReportConfig:
    """Configuration for generating readiness reports."""

    run_id: str
    output_dir: Path
    control_plane_report_path: Path
    evaluation_summary_path: Path | None = None
    telemetry_summary_path: Path | None = None
    report_name: str = "readiness_report"
    seed: int | None = None
    config_hash: str | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not str(self.control_plane_report_path):
            raise ValueError("control_plane_report_path must be provided")
        if not self.report_name:
            raise ValueError("report_name must be non-empty")


def load_readiness_config(path: Path) -> ReadinessReportConfig:
    raw = _load_json(path)
    config = raw.get("readiness", raw)
    run_section = raw.get("run", {})
    run_id = str(config.get("run_id") or run_section.get("run_id") or "")
    output_dir = Path(config.get("output_dir") or run_section.get("output_dir") or "")
    control_plane_path = Path(config.get("control_plane_report_path") or "")
    evaluation_path = config.get("evaluation_summary_path")
    telemetry_path = config.get("telemetry_summary_path")
    report_name = str(config.get("report_name", "readiness_report"))
    seed = config.get("seed", run_section.get("seed"))
    config_hash = config.get("config_hash", run_section.get("config_hash"))
    config_path = config.get("config_path", run_section.get("config_path"))

    if not str(output_dir):
        raise ValueError("output_dir must be provided")
    if not str(control_plane_path):
        raise ValueError("control_plane_report_path must be provided")

    return ReadinessReportConfig(
        run_id=run_id,
        output_dir=output_dir,
        control_plane_report_path=control_plane_path,
        evaluation_summary_path=Path(evaluation_path) if evaluation_path else None,
        telemetry_summary_path=Path(telemetry_path) if telemetry_path else None,
        report_name=report_name,
        seed=int(seed) if seed is not None else None,
        config_hash=str(config_hash) if config_hash else None,
        config_path=Path(config_path) if config_path else None,
    )


def _control_plane_status(state_counts: Mapping[str, int]) -> str:
    fail_states = {"tripped", "error", "failed", "halted"}
    warn_states = {"degraded", "recovering"}
    normalized = {state.lower(): count for state, count in state_counts.items()}
    if any(state in fail_states for state in normalized):
        return "fail"
    if any(state in warn_states for state in normalized):
        return "warn"
    if any(state not in fail_states | warn_states | {"healthy", "ok"} for state in normalized):
        return "unknown"
    return "pass"


def _summarize_control_plane(report: Mapping[str, Any]) -> dict[str, Any]:
    stages_payload = report.get("stages", [])
    stages: list[dict[str, Any]] = []
    for stage in stages_payload:
        stage_name = str(stage.get("stage", "unknown"))
        stages.append(
            {
                "stage": stage_name,
                "state": str(stage.get("state", "unknown")),
                "metrics": _coerce_float_map(stage.get("metrics", {})),
                "counters": _coerce_int_map(stage.get("counters", {})),
            }
        )
    stages = sorted(stages, key=lambda item: item["stage"])
    state_counts = Counter(stage["state"] for stage in stages)
    return {
        "status": _control_plane_status(state_counts),
        "state_counts": dict(sorted(state_counts.items())),
        "stages": stages,
        "event_count": int(len(report.get("events", []) or [])),
        "event_stream_digest": report.get("event_stream_digest"),
        "stage_snapshot_digest": report.get("stage_snapshot_digest"),
        "digest": report.get("digest"),
    }


def _status_from_payload(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return "unknown"
    status = payload.get("status")
    if isinstance(status, str):
        normalized = status.lower()
        if normalized in {"pass", "warn", "fail"}:
            return normalized
    return "unknown"


def _summarize_telemetry(
    telemetry_summary: Mapping[str, Any] | None,
    evaluation_summary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    drift_report = None
    if evaluation_summary:
        drift_report = evaluation_summary.get("telemetry_drift_report")
    telemetry_baseline = None
    if evaluation_summary:
        telemetry_baseline = evaluation_summary.get("telemetry_baseline_comparison")

    status = _status_from_payload(drift_report) if drift_report else "unknown"
    if status == "unknown" and telemetry_baseline:
        status = _status_from_payload(telemetry_baseline)

    summary = {
        "status": status,
        "summary": telemetry_summary or {},
        "drift_report": drift_report,
        "baseline_comparison": telemetry_baseline,
    }
    return summary, status


def _summarize_evaluation(
    evaluation_summary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, str]:
    if not evaluation_summary:
        return {}, "unknown", "unknown"
    baseline_comparison = evaluation_summary.get("baseline_comparison")
    relocalization_comparison = evaluation_summary.get("relocalization_baseline_comparison")
    eval_status = _status_from_payload(baseline_comparison)
    relocalization_status = _status_from_payload(relocalization_comparison)
    summary = {
        "run_id": evaluation_summary.get("run_id"),
        "dataset": evaluation_summary.get("dataset"),
        "aggregate_metrics": evaluation_summary.get("aggregate_metrics", {}),
        "baseline_comparison": baseline_comparison,
        "relocalization_metrics": evaluation_summary.get("relocalization_metrics", {}),
        "relocalization_baseline_comparison": relocalization_comparison,
        "telemetry_metrics": evaluation_summary.get("telemetry_metrics", {}),
        "telemetry_baseline_comparison": evaluation_summary.get("telemetry_baseline_comparison"),
    }
    return summary, eval_status, relocalization_status


def _merge_determinism(
    control_plane: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any] | None,
    config: ReadinessReportConfig,
) -> dict[str, Any]:
    determinism: dict[str, Any] = {}
    control_det = control_plane.get("determinism", {})
    if isinstance(control_det, Mapping):
        determinism.update(control_det)
    if evaluation:
        for key in ("seed", "config_hash", "config_path"):
            if key in evaluation:
                determinism.setdefault(key, evaluation[key])
    if telemetry:
        telemetry_det = telemetry.get("determinism", {})
        if isinstance(telemetry_det, Mapping):
            for key, value in telemetry_det.items():
                determinism.setdefault(key, value)
    if config.seed is not None:
        determinism.setdefault("seed", config.seed)
    if config.config_hash:
        determinism.setdefault("config_hash", config.config_hash)
    if config.config_path:
        determinism.setdefault("config_path", str(config.config_path))
    return determinism


def _aggregate_status(statuses: Mapping[str, str]) -> str:
    normalized = [status for status in statuses.values()]
    if any(status == "fail" for status in normalized):
        return "fail"
    if any(status == "warn" for status in normalized):
        return "warn"
    if any(status == "unknown" for status in normalized):
        return "unknown"
    if normalized and all(status == "pass" for status in normalized):
        return "pass"
    return "unknown"


def generate_readiness_report(config: ReadinessReportConfig) -> dict[str, Any]:
    control_plane = _load_json(config.control_plane_report_path)
    evaluation = _load_json(config.evaluation_summary_path) if config.evaluation_summary_path else None
    telemetry = _load_json(config.telemetry_summary_path) if config.telemetry_summary_path else None

    if telemetry is None and evaluation:
        telemetry_summary = evaluation.get("telemetry_summary")
    else:
        telemetry_summary = telemetry

    control_plane_summary = _summarize_control_plane(control_plane)
    telemetry_summary_payload, telemetry_status = _summarize_telemetry(
        telemetry_summary, evaluation
    )
    evaluation_summary_payload, evaluation_status, relocalization_status = _summarize_evaluation(
        evaluation
    )

    status_breakdown = {
        "control_plane": control_plane_summary["status"],
        "telemetry": telemetry_status,
        "evaluation": evaluation_status,
        "relocalization": relocalization_status,
    }
    overall_status = _aggregate_status(status_breakdown)

    base_payload = {
        "report_type": "senior_interview_readiness",
        "generated_at": _timestamp(),
        "run_id": config.run_id,
        "artifacts": {
            "control_plane_report": str(config.control_plane_report_path),
            "evaluation_summary": str(config.evaluation_summary_path)
            if config.evaluation_summary_path
            else None,
            "telemetry_summary": str(config.telemetry_summary_path)
            if config.telemetry_summary_path
            else None,
        },
        "determinism": _merge_determinism(control_plane, evaluation, telemetry, config),
        "control_plane": control_plane_summary,
        "telemetry": telemetry_summary_payload,
        "evaluation": evaluation_summary_payload,
        "status_breakdown": status_breakdown,
        "status": overall_status,
    }
    digest = stable_hash(
        base_payload,
        exclude_keys=("generated_at", "recorded_at", "timestamp", "timestamp_s", "updated_at_s"),
    )
    payload = dict(base_payload)
    payload["digest"] = digest
    return payload


def write_readiness_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write readiness report: {path}") from exc


def run_readiness_report(config: ReadinessReportConfig) -> dict[str, Any]:
    report = generate_readiness_report(config)
    report_path = config.output_dir / f"{sanitize_artifact_name(config.report_name)}.json"
    write_readiness_report(report_path, report)
    LOGGER.info("Readiness report saved to %s", report_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a senior interview readiness report")
    parser.add_argument("--config", required=True, help="Path to readiness report config JSON")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = load_readiness_config(Path(args.config))
    run_readiness_report(config)


if __name__ == "__main__":
    _main()
