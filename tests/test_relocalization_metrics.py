from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relocalization_metrics import RelocalizationFrame, summarize_relocalized_frames


def test_summarize_relocalized_frames_with_recovery() -> None:
    frames = [
        RelocalizationFrame(frame_id=10, match_count=100, inliers=80, inlier_ratio=0.8),
        RelocalizationFrame(frame_id=12, match_count=50, inliers=25, inlier_ratio=0.5),
    ]
    summary = summarize_relocalized_frames(frames, loss_frame_id=9)

    assert summary["relocalized_frame_count"] == 2.0
    assert summary["match_count_mean"] == 75.0
    assert summary["match_count_p50"] == 75.0
    assert summary["match_count_p95"] == 97.5
    assert summary["inlier_ratio_mean"] == 0.65
    assert summary["inlier_ratio_p50"] == 0.65
    assert summary["recovery_success"] == 1.0
    assert summary["recovery_frame_gap"] == 1.0


def test_summarize_relocalized_frames_without_recovery() -> None:
    summary = summarize_relocalized_frames([], loss_frame_id=None)

    assert summary["relocalized_frame_count"] == 0.0
    assert summary["match_count_mean"] == 0.0
    assert summary["inlier_ratio_mean"] == 0.0
    assert summary["recovery_success"] == 0.0
