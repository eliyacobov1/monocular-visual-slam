"""Determinism validation suite for run artifact digests and drift checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from data_persistence import load_trajectory_npz
from deterministic_integrity import stable_hash

LOGGER = logging.getLogger(__name__)

_JSON_EXCLUDE_KEYS = ("recorded_at", "created_at")


@dataclass(frozen=True)
class ArtifactDigest:
    """Digest for a single artifact."""

    artifact_type: str
    name: str
    digest: str


@dataclass(frozen=True)
class RunDigest:
    """Deterministic digest summary for a run directory."""

    run_dir: Path
    run_id: str | None
    seed: int | None
    config_hash: str | None
    artifacts: tuple[ArtifactDigest, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "artifacts": [
                {
                    "type": artifact.artifact_type,
                    "name": artifact.name,
                    "digest": artifact.digest,
                }
                for artifact in self.artifacts
            ],
        }

    def artifact_map(self) -> dict[str, dict[str, str]]:
        grouped: dict[str, dict[str, str]] = {}
        for artifact in self.artifacts:
            grouped.setdefault(artifact.artifact_type, {})[artifact.name] = artifact.digest
        return grouped


@dataclass(frozen=True)
class DriftEntry:
    """Drift entry for a single artifact comparison."""

    artifact_type: str
    name: str
    status: str
    run_a_digest: str | None
    run_b_digest: str | None


@dataclass(frozen=True)
class DeterminismReport:
    """Determinism report payload with digest and drift checks."""

    status: str
    run_a: RunDigest
    run_b: RunDigest
    drift: tuple[DriftEntry, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_a": self.run_a.as_dict(),
            "run_b": self.run_b.as_dict(),
            "drift": [
                {
                    "type": entry.artifact_type,
                    "name": entry.name,
                    "status": entry.status,
                    "run_a_digest": entry.run_a_digest,
                    "run_b_digest": entry.run_b_digest,
                }
                for entry in self.drift
            ],
        }


def build_run_digest(run_dir: Path) -> RunDigest:
    """Compute deterministic digests for artifacts in a run directory."""

    run_dir = Path(run_dir)
    metadata = _load_run_metadata(run_dir)
    artifacts = _collect_artifacts(run_dir)
    return RunDigest(
        run_dir=run_dir,
        run_id=metadata.get("run_id"),
        seed=metadata.get("seed"),
        config_hash=metadata.get("config_hash"),
        artifacts=tuple(artifacts),
    )


def compare_run_digests(run_a: RunDigest, run_b: RunDigest) -> DeterminismReport:
    """Compare two run digests and produce a determinism report."""

    drift: list[DriftEntry] = []
    status = "pass"
    artifacts_a = run_a.artifact_map()
    artifacts_b = run_b.artifact_map()
    artifact_types = sorted(set(artifacts_a) | set(artifacts_b))

    for artifact_type in artifact_types:
        names = sorted(set(artifacts_a.get(artifact_type, {})) | set(artifacts_b.get(artifact_type, {})))
        for name in names:
            digest_a = artifacts_a.get(artifact_type, {}).get(name)
            digest_b = artifacts_b.get(artifact_type, {}).get(name)
            if digest_a is None or digest_b is None:
                drift.append(
                    DriftEntry(
                        artifact_type=artifact_type,
                        name=name,
                        status="missing",
                        run_a_digest=digest_a,
                        run_b_digest=digest_b,
                    )
                )
                status = "fail"
                continue
            if digest_a != digest_b:
                drift.append(
                    DriftEntry(
                        artifact_type=artifact_type,
                        name=name,
                        status="mismatch",
                        run_a_digest=digest_a,
                        run_b_digest=digest_b,
                    )
                )
                status = "fail"
            else:
                drift.append(
                    DriftEntry(
                        artifact_type=artifact_type,
                        name=name,
                        status="match",
                        run_a_digest=digest_a,
                        run_b_digest=digest_b,
                    )
                )

    return DeterminismReport(status=status, run_a=run_a, run_b=run_b, drift=tuple(drift))


def write_determinism_report(path: Path, report: DeterminismReport) -> Path:
    """Write a determinism report to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote determinism report to %s", path)
    return path


def build_determinism_report(run_dir_a: Path, run_dir_b: Path) -> DeterminismReport:
    """Build a determinism report by comparing two run directories."""

    run_a = build_run_digest(run_dir_a)
    run_b = build_run_digest(run_dir_b)
    return compare_run_digests(run_a, run_b)


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Failed to read run metadata from %s", metadata_path)
        raise RuntimeError("Failed to load run metadata") from exc
    return {
        "run_id": payload.get("run_id"),
        "seed": payload.get("seed"),
        "config_hash": payload.get("config_hash"),
    }


def _collect_artifacts(run_dir: Path) -> list[ArtifactDigest]:
    artifacts: list[ArtifactDigest] = []
    artifacts.extend(_collect_trajectories(run_dir / "trajectories"))
    artifacts.extend(_collect_json_artifacts(run_dir / "metrics", "metrics"))
    artifacts.extend(_collect_json_artifacts(run_dir / "diagnostics", "diagnostics"))
    artifacts.extend(_collect_telemetry(run_dir / "telemetry"))
    artifacts.extend(_collect_map_snapshots(run_dir / "maps"))
    return artifacts


def _collect_trajectories(path: Path) -> list[ArtifactDigest]:
    if not path.exists():
        return []
    artifacts: list[ArtifactDigest] = []
    for file_path in sorted(path.glob("*.npz")):
        bundle = load_trajectory_npz(file_path)
        payload = {
            "poses": bundle.poses,
            "timestamps": bundle.timestamps,
            "frame_ids": bundle.frame_ids,
        }
        digest = stable_hash(payload)
        artifacts.append(
            ArtifactDigest(artifact_type="trajectory", name=bundle.name, digest=digest)
        )
    return artifacts


def _collect_json_artifacts(path: Path, artifact_type: str) -> list[ArtifactDigest]:
    if not path.exists():
        return []
    artifacts: list[ArtifactDigest] = []
    for file_path in sorted(path.glob("*.json")):
        payload = _load_json(file_path)
        digest = stable_hash(payload, exclude_keys=_JSON_EXCLUDE_KEYS)
        artifacts.append(
            ArtifactDigest(
                artifact_type=artifact_type,
                name=file_path.stem,
                digest=digest,
            )
        )
    return artifacts


def _collect_telemetry(path: Path) -> list[ArtifactDigest]:
    if not path.exists():
        return []
    artifacts: list[ArtifactDigest] = []
    for file_path in sorted(path.glob("*.json")):
        payload = _load_json(file_path)
        if "events" in payload:
            normalized = _normalize_telemetry(payload)
            digest = stable_hash(normalized, exclude_keys=_JSON_EXCLUDE_KEYS)
        else:
            digest = stable_hash(payload, exclude_keys=_JSON_EXCLUDE_KEYS)
        artifacts.append(
            ArtifactDigest(
                artifact_type="telemetry",
                name=file_path.stem,
                digest=digest,
            )
        )
    return artifacts


def _collect_map_snapshots(path: Path) -> list[ArtifactDigest]:
    if not path.exists():
        return []
    artifacts: list[ArtifactDigest] = []
    for map_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        digest = _hash_map_directory(map_dir)
        artifacts.append(
            ArtifactDigest(
                artifact_type="map",
                name=map_dir.name,
                digest=digest,
            )
        )
    return artifacts


def _hash_map_directory(path: Path) -> str:
    entries: list[dict[str, Any]] = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel_path = str(file_path.relative_to(path))
        if file_path.name == "map_bundle.json":
            payload = _load_json(file_path)
            payload = {
                key: value
                for key, value in payload.items()
                if key not in _JSON_EXCLUDE_KEYS
            }
            entries.append({"path": rel_path, "payload": payload})
        else:
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            entries.append({"path": rel_path, "sha256": digest})
    return stable_hash(entries)


def _normalize_telemetry(payload: Mapping[str, Any]) -> dict[str, Any]:
    events = payload.get("events", [])
    normalized_events = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_payload = {
            key: value
            for key, value in event.items()
            if key
            not in (
                "timestamp",
                "memory_delta_bytes",
                "memory_current_bytes",
                "memory_peak_bytes",
            )
        }
        normalized_events.append(event_payload)
    normalized = dict(payload)
    normalized["events"] = normalized_events
    return normalized


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Failed to read json from %s", path)
        raise RuntimeError("Failed to read json payload") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Determinism validation suite")
    parser.add_argument("--run-dir-a", required=True, help="First run directory")
    parser.add_argument("--run-dir-b", required=True, help="Second run directory")
    parser.add_argument("--output", required=True, help="Output report path")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = build_determinism_report(Path(args.run_dir_a), Path(args.run_dir_b))
    write_determinism_report(Path(args.output), report)
    if report.status != "pass":
        LOGGER.error("Determinism validation failed")
        return 1
    LOGGER.info("Determinism validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
