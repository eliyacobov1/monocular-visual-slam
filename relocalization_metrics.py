"""Relocalization metrics for regression gating and recovery validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelocalizationFrame:
    frame_id: int
    match_count: int
    inliers: int
    inlier_ratio: float
    timestamp: float | None = None
    method: str | None = None


def summarize_relocalization_events(events: Iterable[Mapping[str, object]]) -> dict[str, float]:
    relocalization_events = [event for event in events if event.get("name") == "relocalization_search"]
    durations = [float(event.get("duration_s", 0.0)) for event in relocalization_events]
    successes = [
        bool(event.get("metadata", {}).get("success", False)) for event in relocalization_events
    ]
    if durations:
        durations_array = np.array(durations, dtype=np.float64)
        p50 = float(np.quantile(durations_array, 0.5))
        p95 = float(np.quantile(durations_array, 0.95))
        mean = float(durations_array.mean())
    else:
        p50 = 0.0
        p95 = 0.0
        mean = 0.0
    return {
        "attempts": float(len(relocalization_events)),
        "successes": float(sum(successes)),
        "success_rate": float(sum(successes)) / float(len(successes)) if successes else 0.0,
        "latency_mean_s": mean,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
    }


def summarize_relocalized_frames(
    frames: Sequence[RelocalizationFrame],
    *,
    loss_frame_id: int | None,
) -> dict[str, float]:
    match_counts = [float(frame.match_count) for frame in frames]
    inlier_ratios = [float(frame.inlier_ratio) for frame in frames]

    if match_counts:
        match_counts_arr = np.array(match_counts, dtype=np.float64)
        match_mean = float(match_counts_arr.mean())
        match_p50 = float(np.quantile(match_counts_arr, 0.5))
        match_p95 = float(np.quantile(match_counts_arr, 0.95))
    else:
        match_mean = 0.0
        match_p50 = 0.0
        match_p95 = 0.0

    if inlier_ratios:
        inlier_ratios_arr = np.array(inlier_ratios, dtype=np.float64)
        inlier_mean = float(inlier_ratios_arr.mean())
        inlier_p50 = float(np.quantile(inlier_ratios_arr, 0.5))
        inlier_p95 = float(np.quantile(inlier_ratios_arr, 0.95))
    else:
        inlier_mean = 0.0
        inlier_p50 = 0.0
        inlier_p95 = 0.0

    recovery_success = 0.0
    recovery_frame_gap = 0.0
    if loss_frame_id is not None:
        recovered_frames = [frame for frame in frames if frame.frame_id > loss_frame_id]
        if recovered_frames:
            recovery_success = 1.0
            recovery_frame_gap = float(
                min(frame.frame_id for frame in recovered_frames) - loss_frame_id
            )

    return {
        "relocalized_frame_count": float(len(frames)),
        "match_count_mean": match_mean,
        "match_count_p50": match_p50,
        "match_count_p95": match_p95,
        "inlier_ratio_mean": inlier_mean,
        "inlier_ratio_p50": inlier_p50,
        "inlier_ratio_p95": inlier_p95,
        "recovery_success": recovery_success,
        "recovery_frame_gap": recovery_frame_gap,
    }
