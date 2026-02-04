"""Persistent map snapshot builder utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from keyframe_manager import Keyframe
from persistent_map import MapKeyframe, PersistentMapSnapshot, build_snapshot, keypoints_to_array

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MapBuilderConfig:
    """Configuration for building BoW vocabularies from keyframes."""

    vocab_size: int = 64
    max_descriptors: int = 5000
    rng_seed: int = 13
    kmeans_max_iters: int = 50
    kmeans_epsilon: float = 1e-3


@dataclass(frozen=True)
class MapBuildStats:
    """Summary statistics for a map build."""

    num_keyframes: int
    total_descriptors: int
    sampled_descriptors: int
    vocab_size: int


class MapSnapshotBuilder:
    """Build persistent map snapshots from selected keyframes."""

    def __init__(self, config: MapBuilderConfig) -> None:
        if config.vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive")
        if config.max_descriptors <= 0:
            raise ValueError("Max descriptors must be positive")
        self._config = config
        self._rng = np.random.default_rng(config.rng_seed)

    def build_snapshot(
        self,
        keyframes: Sequence[Keyframe],
    ) -> tuple[PersistentMapSnapshot, MapBuildStats]:
        if not keyframes:
            raise ValueError("At least one keyframe is required to build a map")
        map_keyframes: list[MapKeyframe] = []
        descriptors_list: list[np.ndarray] = []
        for kf in keyframes:
            if kf.descriptors is None or len(kf.descriptors) == 0:
                raise ValueError("Keyframe descriptors must be non-empty")
            map_keyframes.append(
                MapKeyframe(
                    frame_id=int(kf.frame_id),
                    pose=kf.pose,
                    keypoints=keypoints_to_array(kf.keypoints),
                    descriptors=kf.descriptors,
                )
            )
            descriptors_list.append(kf.descriptors)

        descriptors = np.vstack(descriptors_list)
        total_descriptors = int(descriptors.shape[0])
        if total_descriptors == 0:
            raise ValueError("No descriptors available for vocabulary")

        sampled = self._sample_descriptors(descriptors)
        vocab = self._build_vocab(sampled)
        stats = MapBuildStats(
            num_keyframes=len(map_keyframes),
            total_descriptors=total_descriptors,
            sampled_descriptors=int(sampled.shape[0]),
            vocab_size=int(vocab.shape[0]),
        )
        LOGGER.info(
            "Built BoW vocabulary: keyframes=%d descriptors=%d sampled=%d vocab=%d",
            stats.num_keyframes,
            stats.total_descriptors,
            stats.sampled_descriptors,
            stats.vocab_size,
        )
        return build_snapshot(map_keyframes, vocab), stats

    def _sample_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.ndim != 2:
            raise ValueError("Descriptors must be a 2D array")
        max_count = min(self._config.max_descriptors, len(descriptors))
        if len(descriptors) <= max_count:
            return descriptors.astype(np.float32, copy=False)
        indices = self._rng.choice(len(descriptors), size=max_count, replace=False)
        return descriptors[indices].astype(np.float32, copy=False)

    def _build_vocab(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.ndim != 2:
            raise ValueError("Descriptors must be a 2D array for k-means")
        if descriptors.shape[0] < 2:
            raise ValueError("At least two descriptors are required for k-means")
        k = min(self._config.vocab_size, descriptors.shape[0])
        if k < self._config.vocab_size:
            LOGGER.warning(
                "Reducing vocab size from %d to %d due to descriptor count",
                self._config.vocab_size,
                k,
            )
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self._config.kmeans_max_iters,
            self._config.kmeans_epsilon,
        )
        _, _, centers = cv2.kmeans(
            descriptors.astype(np.float32, copy=False),
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        return centers.astype(np.float32, copy=False)
