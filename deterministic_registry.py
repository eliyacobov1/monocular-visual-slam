"""Deterministic seed and config registry for cross-stage SLAM runs."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

_MAX_CV2_SEED = 2**31 - 1


@dataclass(frozen=True)
class DeterminismConfig:
    """Configuration payload for deterministic execution."""

    seed: int
    config_path: Path
    config_hash: str

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.config_hash:
            raise ValueError("config_hash must be non-empty")


class DeterminismRegistry:
    """Registry for consistent deterministic seeds and metadata."""

    def __init__(self, config: DeterminismConfig) -> None:
        self._config = config

    @property
    def config(self) -> DeterminismConfig:
        return self._config

    def apply_global_seed(self) -> None:
        """Apply the global seed across RNG providers."""

        random.seed(self._config.seed)
        np.random.seed(self._config.seed)
        cv2.setRNGSeed(self._bounded_seed(self._config.seed))
        LOGGER.info(
            "Applied global deterministic seed",
            extra={
                "seed": self._config.seed,
                "config_hash": self._config.config_hash,
            },
        )

    def seed_for(self, component: str) -> int:
        """Derive a deterministic seed for a named component."""

        if not component:
            raise ValueError("component must be non-empty")
        digest = hashlib.sha256(
            f"{self._config.seed}:{component}".encode("utf-8")
        ).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return self._bounded_seed(value)

    def metadata(self) -> dict[str, object]:
        return {
            "seed": self._config.seed,
            "config_hash": self._config.config_hash,
        }

    @staticmethod
    def _bounded_seed(value: int) -> int:
        return int(value % _MAX_CV2_SEED)


def hash_config_path(config_path: Path) -> str:
    """Compute a deterministic hash for a config file."""

    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def build_registry(
    *,
    seed: int,
    config_path: Path,
    config_hash: str | None = None,
) -> DeterminismRegistry:
    """Create a deterministic registry from config path + seed."""

    config_hash_value = config_hash or hash_config_path(config_path)
    return DeterminismRegistry(
        DeterminismConfig(
            seed=int(seed),
            config_path=config_path,
            config_hash=config_hash_value,
        )
    )


def enrich_payload_with_determinism(
    payload: Mapping[str, object],
    determinism: Mapping[str, object],
) -> dict[str, object]:
    """Return a copy of payload enriched with determinism metadata."""

    merged = dict(payload)
    merged["determinism"] = dict(determinism)
    return merged
