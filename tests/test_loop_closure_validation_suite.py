from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from loop_closure_validation import (
    LoopClosureVerificationDataset,
    LoopClosureVerificationSample,
    LoopClosureVerificationThresholds,
)


def test_loop_closure_validation_accepts_true_loop() -> None:
    thresholds = LoopClosureVerificationThresholds(
        min_match_count=20,
        min_inlier_ratio=0.6,
        max_reprojection_error=2.0,
        min_temporal_separation_s=1.0,
        min_geometric_score=0.7,
        min_temporal_score=0.6,
        min_combined_score=0.75,
    )
    sample = LoopClosureVerificationSample(
        sample_id="good-loop",
        query_frame_id=10,
        candidate_frame_id=1,
        query_timestamp_s=12.0,
        candidate_timestamp_s=4.0,
        match_count=60,
        inlier_count=45,
        mean_reprojection_error=1.0,
        rotation_error_deg=5.0,
        translation_error=0.2,
        expected_match=True,
    )
    dataset = LoopClosureVerificationDataset(name="single", samples=(sample,))

    report = dataset.evaluate(thresholds)

    assert report.accepted_count == 1
    assert report.true_positive == 1
    assert report.sample_results[0].accepted
    assert report.sample_results[0].classification == "true_positive"


def test_loop_closure_validation_rejects_false_positive() -> None:
    thresholds = LoopClosureVerificationThresholds(
        min_match_count=30,
        min_inlier_ratio=0.7,
        max_reprojection_error=2.0,
        min_temporal_separation_s=2.0,
        min_geometric_score=0.8,
        min_temporal_score=0.7,
        min_combined_score=0.8,
    )
    sample = LoopClosureVerificationSample(
        sample_id="false-loop",
        query_frame_id=20,
        candidate_frame_id=19,
        query_timestamp_s=20.0,
        candidate_timestamp_s=19.2,
        match_count=40,
        inlier_count=20,
        mean_reprojection_error=3.0,
        rotation_error_deg=12.0,
        translation_error=0.8,
        expected_match=False,
    )
    dataset = LoopClosureVerificationDataset(name="single", samples=(sample,))

    report = dataset.evaluate(thresholds)

    assert report.accepted_count == 0
    assert report.true_negative == 1
    result = report.sample_results[0]
    assert not result.accepted
    assert "temporal_separation" in result.rejection_reasons
    assert result.classification == "true_negative"


def test_loop_closure_validation_digest_is_deterministic() -> None:
    thresholds = LoopClosureVerificationThresholds()
    samples = (
        LoopClosureVerificationSample(
            sample_id="sample-a",
            query_frame_id=1,
            candidate_frame_id=100,
            query_timestamp_s=10.0,
            candidate_timestamp_s=0.0,
            match_count=55,
            inlier_count=40,
            mean_reprojection_error=1.5,
            rotation_error_deg=6.0,
            translation_error=0.3,
            expected_match=True,
        ),
        LoopClosureVerificationSample(
            sample_id="sample-b",
            query_frame_id=2,
            candidate_frame_id=101,
            query_timestamp_s=12.0,
            candidate_timestamp_s=1.0,
            match_count=35,
            inlier_count=10,
            mean_reprojection_error=2.8,
            rotation_error_deg=11.0,
            translation_error=0.9,
            expected_match=False,
        ),
    )
    dataset = LoopClosureVerificationDataset(name="digest", samples=samples)

    report_first = dataset.evaluate(thresholds)
    report_second = dataset.evaluate(thresholds)

    assert report_first.report_digest == report_second.report_digest
    assert report_first.report_digest
