"""Persistent map storage and relocalization utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min

from deterministic_integrity import stable_hash
from homography import estimate_pose_from_matches

logger = logging.getLogger(__name__)

MAP_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MapKeyframe:
    frame_id: int
    pose: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class PersistentMapSnapshot:
    keyframes: tuple[MapKeyframe, ...]
    bow_vocab: np.ndarray
    bow_hists: np.ndarray
    bow_frame_ids: np.ndarray

    def digest(self) -> str:
        payload = {
            "keyframes": [
                {
                    "frame_id": int(kf.frame_id),
                    "pose": kf.pose,
                    "keypoints": kf.keypoints,
                    "descriptors": kf.descriptors,
                }
                for kf in self.keyframes
            ],
            "bow_vocab": self.bow_vocab,
            "bow_hists": self.bow_hists,
            "bow_frame_ids": self.bow_frame_ids,
        }
        return stable_hash(payload)


@dataclass(frozen=True)
class RelocalizationResult:
    frame_id: int
    score: float
    match_count: int
    inliers: int
    rotation: np.ndarray
    translation: np.ndarray


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_keyframe(kf: MapKeyframe) -> None:
    if kf.pose.shape != (4, 4):
        raise ValueError("Keyframe pose must be 4x4")
    if kf.keypoints.ndim != 2 or kf.keypoints.shape[1] != 2:
        raise ValueError("Keyframe keypoints must be (N,2)")
    if kf.descriptors.ndim != 2:
        raise ValueError("Keyframe descriptors must be (N,D)")
    if len(kf.keypoints) != len(kf.descriptors):
        raise ValueError("Keyframe keypoints and descriptors must align")


def compute_bow_histogram(descriptors: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab.shape[0], dtype=np.float32)
    if descriptors.ndim != 2:
        raise ValueError("Descriptors must be a 2D array")
    if vocab.ndim != 2:
        raise ValueError("Vocabulary must be a 2D array")
    if descriptors.shape[1] != vocab.shape[1]:
        raise ValueError("Descriptor dimensionality must match vocabulary")
    desc = descriptors.astype(np.float32, copy=False)
    words, _ = pairwise_distances_argmin_min(desc, vocab.astype(np.float32, copy=False))
    hist = np.bincount(words, minlength=vocab.shape[0]).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def build_snapshot(
    keyframes: Sequence[MapKeyframe],
    bow_vocab: np.ndarray,
) -> PersistentMapSnapshot:
    if not keyframes:
        raise ValueError("At least one keyframe is required")
    keyframes = sorted(keyframes, key=lambda kf: int(kf.frame_id))
    if bow_vocab.ndim != 2 or bow_vocab.shape[0] == 0:
        raise ValueError("BoW vocabulary must be a non-empty 2D array")
    for kf in keyframes:
        _validate_keyframe(kf)
    bow_hists = np.vstack(
        [compute_bow_histogram(kf.descriptors, bow_vocab) for kf in keyframes]
    )
    bow_frame_ids = np.array([kf.frame_id for kf in keyframes], dtype=np.int64)
    return PersistentMapSnapshot(
        keyframes=tuple(keyframes),
        bow_vocab=bow_vocab,
        bow_hists=bow_hists,
        bow_frame_ids=bow_frame_ids,
    )


class PersistentMapStore:
    """Save/load persistent maps with fail-fast validation."""

    def save(self, directory: Path, snapshot: PersistentMapSnapshot) -> None:
        directory = Path(directory)
        if directory.exists() and not directory.is_dir():
            raise ValueError("Map path must be a directory")
        directory.mkdir(parents=True, exist_ok=True)
        if snapshot.bow_hists.shape[0] != len(snapshot.keyframes):
            raise ValueError("BoW histograms must match keyframe count")
        for kf in snapshot.keyframes:
            _validate_keyframe(kf)
        arrays = {
            "poses": np.stack([kf.pose for kf in snapshot.keyframes]),
            "frame_ids": np.array([kf.frame_id for kf in snapshot.keyframes], dtype=np.int64),
            "bow_vocab": snapshot.bow_vocab,
            "bow_hists": snapshot.bow_hists,
            "bow_frame_ids": snapshot.bow_frame_ids,
        }
        for idx, kf in enumerate(snapshot.keyframes):
            arrays[f"keypoints_{idx}"] = kf.keypoints
            arrays[f"descriptors_{idx}"] = kf.descriptors
        np.savez_compressed(directory / "map_arrays.npz", **arrays)

        metadata = {
            "schema_version": MAP_SCHEMA_VERSION,
            "created_at": _timestamp(),
            "num_keyframes": len(snapshot.keyframes),
            "snapshot_digest": snapshot.digest(),
        }
        (directory / "map_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved persistent map with %d keyframes", len(snapshot.keyframes))

    def load(self, directory: Path) -> PersistentMapSnapshot:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Map directory not found: {directory}")
        metadata_path = directory / "map_metadata.json"
        arrays_path = directory / "map_arrays.npz"
        if not metadata_path.exists() or not arrays_path.exists():
            raise FileNotFoundError("Map metadata or arrays missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != MAP_SCHEMA_VERSION:
            raise ValueError("Unsupported map schema version")
        arrays = np.load(arrays_path)
        num_keyframes = int(metadata.get("num_keyframes", 0))
        if num_keyframes <= 0:
            raise ValueError("Map metadata must include a positive keyframe count")
        poses = arrays["poses"]
        frame_ids = arrays["frame_ids"]
        keyframes: list[MapKeyframe] = []
        for idx in range(num_keyframes):
            keypoints = arrays[f"keypoints_{idx}"]
            descriptors = arrays[f"descriptors_{idx}"]
            keyframes.append(
                MapKeyframe(
                    frame_id=int(frame_ids[idx]),
                    pose=poses[idx],
                    keypoints=keypoints,
                    descriptors=descriptors,
                )
            )
        snapshot = PersistentMapSnapshot(
            keyframes=tuple(keyframes),
            bow_vocab=arrays["bow_vocab"],
            bow_hists=arrays["bow_hists"],
            bow_frame_ids=arrays["bow_frame_ids"],
        )
        logger.info("Loaded persistent map with %d keyframes", len(snapshot.keyframes))
        return snapshot


class MapRelocalizer:
    """Relocalize against a persistent map using BoW + geometric checks."""

    def __init__(
        self,
        snapshot: PersistentMapSnapshot,
        intrinsics: np.ndarray | None,
        *,
        min_matches: int = 60,
        min_inliers: int = 30,
        max_candidates: int = 5,
        score_threshold: float = 0.75,
        ransac_threshold: float = 0.01,
        verify_geometry: bool = True,
    ) -> None:
        if snapshot.bow_hists.size == 0:
            raise ValueError("Persistent map has no BoW histograms")
        if verify_geometry and intrinsics is None:
            raise ValueError("Intrinsics are required for geometric verification")
        self.snapshot = snapshot
        self.intrinsics = intrinsics
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.max_candidates = max_candidates
        self.score_threshold = score_threshold
        self.ransac_threshold = ransac_threshold
        self.verify_geometry = verify_geometry
        self._frame_lookup = {kf.frame_id: kf for kf in snapshot.keyframes}

    def relocalize(
        self,
        keypoints: Sequence[cv2.KeyPoint] | None,
        descriptors: np.ndarray,
    ) -> RelocalizationResult | None:
        if descriptors is None or len(descriptors) == 0:
            raise ValueError("Descriptors are required for relocalization")
        hist = compute_bow_histogram(descriptors, self.snapshot.bow_vocab)
        scores = cosine_similarity([hist], self.snapshot.bow_hists)[0]
        ranked = sorted(
            range(len(scores)),
            key=lambda idx: (
                -float(scores[idx]),
                int(self.snapshot.bow_frame_ids[idx]),
            ),
        )
        candidate_indices = ranked[: self.max_candidates]
        best: RelocalizationResult | None = None
        for idx in candidate_indices:
            score = float(scores[idx])
            if score < self.score_threshold:
                continue
            frame_id = int(self.snapshot.bow_frame_ids[idx])
            kf = self._frame_lookup.get(frame_id)
            if kf is None:
                logger.warning("BoW frame id %d missing from keyframes", frame_id)
                continue
            if not self.verify_geometry:
                logger.info("Relocalization candidate %d score=%.3f", frame_id, score)
                return RelocalizationResult(
                    frame_id=frame_id,
                    score=score,
                    match_count=0,
                    inliers=0,
                    rotation=np.eye(3),
                    translation=np.zeros(3),
                )
            if keypoints is None:
                raise ValueError("Keypoints required for geometric verification")
            matcher = _build_matcher(kf.descriptors, descriptors)
            matches = matcher.match(kf.descriptors, descriptors)
            if len(matches) < self.min_matches:
                logger.debug("Candidate %d rejected: only %d matches", frame_id, len(matches))
                continue
            matches = sorted(matches, key=lambda m: m.distance)
            try:
                rotation, translation, inliers, _ = estimate_pose_from_matches(
                    kp1=_to_keypoints(kf.keypoints),
                    kp2=list(keypoints),
                    matches=matches,
                    K=self.intrinsics,
                    ransac_threshold=self.ransac_threshold,
                    min_matches=self.min_matches,
                )
            except RuntimeError as exc:
                logger.debug("Candidate %d rejected: %s", frame_id, exc)
                continue
            inlier_count = int(len(inliers))
            if inlier_count < self.min_inliers:
                logger.debug(
                    "Candidate %d rejected: %d inliers < %d",
                    frame_id,
                    inlier_count,
                    self.min_inliers,
                )
                continue
            result = RelocalizationResult(
                frame_id=frame_id,
                score=score,
                match_count=len(matches),
                inliers=inlier_count,
                rotation=rotation,
                translation=translation,
            )
            if best is None or (
                result.inliers,
                result.score,
                -result.frame_id,
            ) > (
                best.inliers,
                best.score,
                -best.frame_id,
            ):
                best = result
        if best:
            logger.info(
                "Relocalized against frame %d (score=%.3f inliers=%d)",
                best.frame_id,
                best.score,
                best.inliers,
            )
        else:
            logger.info("Relocalization failed: no candidates passed thresholds")
        return best


def _to_keypoints(points: np.ndarray) -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points]


def _build_matcher(desc_a: np.ndarray, desc_b: np.ndarray) -> cv2.BFMatcher:
    if desc_a.dtype == np.uint8 and desc_b.dtype == np.uint8:
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=True)


def keypoints_to_array(keypoints: Iterable[cv2.KeyPoint]) -> np.ndarray:
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)
