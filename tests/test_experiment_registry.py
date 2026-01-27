from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment_registry import create_run_artifacts


def test_create_run_artifacts_with_subdir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    artifacts = create_run_artifacts(
        base_dir=tmp_path / "reports",
        run_id="unit_test",
        config_path=config_path,
        config_hash="abc123",
        use_subdir=True,
    )

    assert artifacts.run_dir.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.run_id == "unit_test"
