"""Run artifact registry for reproducible evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    created_at: str
    metadata_path: Path


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_artifacts(
    base_dir: Path,
    run_id: str,
    config_path: Path,
    config_hash: str,
    use_subdir: bool,
) -> RunArtifacts:
    created_at = _timestamp()
    run_dir = base_dir
    if use_subdir:
        safe_stamp = created_at.replace(":", "").replace("-", "").replace("+", "Z")
        run_dir = base_dir / f"{run_id}_{safe_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "created_at": created_at,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "run_dir": str(run_dir),
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        created_at=created_at,
        metadata_path=metadata_path,
    )


def write_resolved_config(run_dir: Path, resolved_config: dict) -> Path:
    """Write the resolved experiment configuration into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    return resolved_path
