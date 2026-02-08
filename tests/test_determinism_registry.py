"""Tests for deterministic seed registry behavior."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from deterministic_registry import build_registry


def test_registry_seed_is_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    registry = build_registry(seed=42, config_path=config_path)

    seed_first = registry.seed_for("feature_pipeline")
    seed_second = registry.seed_for("feature_pipeline")
    seed_other = registry.seed_for("tracking")

    assert seed_first == seed_second
    assert seed_first != seed_other
    assert 0 <= seed_first < 2**31
