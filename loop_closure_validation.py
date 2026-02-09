"""Loop-closure validation suite with deterministic geometric and temporal scoring."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from deterministic_integrity import stable_hash

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopClosureVerificationThresholds:
    """Deterministic thresholds for loop-closure acceptance."""

    min_match_count: int = 30
    min_inlier_ratio: float = 0.55
    max_reprojection_error: float = 2.5
    min_temporal_separation_s: float = 1.0
    max_temporal_separation_s: float | None = None
    max_rotation_error_deg: float | None = 10.0
    max_translation_error: float | None = 0.5
    min_geometric_score: float = 0.7
    min_temporal_score: float = 0.6
    min_combined_score: float = 0.75
    geometric_weight: float = 0.7
    temporal_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.min_match_count <= 0:
            raise ValueError("min_match_count must be positive")
        if not 0.0 < self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in (0, 1]")
        if self.max_reprojection_error <= 0:
            raise ValueError("max_reprojection_error must be positive")
        if self.min_temporal_separation_s < 0:
            raise ValueError("min_temporal_separation_s must be non-negative")
        if self.max_temporal_separation_s is not None and self.max_temporal_separation_s <= 0:
            raise ValueError("max_temporal_separation_s must be positive when provided")
        if self.max_rotation_error_deg is not None and self.max_rotation_error_deg <= 0:
            raise ValueError("max_rotation_error_deg must be positive when provided")
        if self.max_translation_error is not None and self.max_translation_error <= 0:
            raise ValueError("max_translation_error must be positive when provided")
        if not 0.0 <= self.min_geometric_score <= 1.0:
            raise ValueError("min_geometric_score must be in [0, 1]")
        if not 0.0 <= self.min_temporal_score <= 1.0:
            raise ValueError("min_temporal_score must be in [0, 1]")
        if not 0.0 <= self.min_combined_score <= 1.0:
            raise ValueError("min_combined_score must be in [0, 1]")
        if self.geometric_weight < 0 or self.temporal_weight < 0:
            raise ValueError("weights must be non-negative")
        if self.geometric_weight + self.temporal_weight <= 0:
            raise ValueError("geometric_weight + temporal_weight must be positive")


@dataclass(frozen=True)
class LoopClosureVerificationSample:
    """Single loop-closure candidate with verification statistics."""

    sample_id: str
    query_frame_id: int
    candidate_frame_id: int
    query_timestamp_s: float
    candidate_timestamp_s: float
    match_count: int
    inlier_count: int
    mean_reprojection_error: float
    rotation_error_deg: float | None
    translation_error: float | None
    expected_match: bool

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.match_count < 0:
            raise ValueError("match_count must be non-negative")
        if self.inlier_count < 0:
            raise ValueError("inlier_count must be non-negative")
        if self.inlier_count > self.match_count:
            raise ValueError("inlier_count cannot exceed match_count")
        if self.mean_reprojection_error < 0:
            raise ValueError("mean_reprojection_error must be non-negative")
        if self.rotation_error_deg is not None and self.rotation_error_deg < 0:
            raise ValueError("rotation_error_deg must be non-negative when provided")
        if self.translation_error is not None and self.translation_error < 0:
            raise ValueError("translation_error must be non-negative when provided")


@dataclass(frozen=True)
class LoopClosureSampleResult:
    """Deterministic verdict for a loop-closure candidate."""

    sample_id: str
    query_frame_id: int
    candidate_frame_id: int
    temporal_delta_s: float
    match_count: int
    inlier_ratio: float
    mean_reprojection_error: float
    geometric_score: float
    temporal_score: float
    combined_score: float
    accepted: bool
    expected_match: bool
    classification: str
    rejection_reasons: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LoopClosureValidationReport:
    """Summary of loop-closure verification results for a dataset."""

    dataset_name: str
    total_samples: int
    accepted_count: int
    rejected_count: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    thresholds: LoopClosureVerificationThresholds
    sample_results: tuple[LoopClosureSampleResult, ...] = field(default_factory=tuple)
    report_digest: str = ""

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["thresholds"] = asdict(self.thresholds)
        payload["sample_results"] = [result.asdict() for result in self.sample_results]
        return payload


@dataclass(frozen=True)
class LoopClosureVerificationDataset:
    """Dataset wrapper for loop-closure validation."""

    name: str
    samples: tuple[LoopClosureVerificationSample, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        if not self.samples:
            raise ValueError("dataset must contain at least one sample")

    def evaluate(self, thresholds: LoopClosureVerificationThresholds) -> LoopClosureValidationReport:
        LOGGER.info(
            "Evaluating loop-closure dataset '%s' with %d samples",
            self.name,
            len(self.samples),
        )
        ordered_samples = sorted(
            self.samples,
            key=lambda sample: (sample.sample_id, sample.query_frame_id, sample.candidate_frame_id),
        )
        results = [score_loop_closure_sample(sample, thresholds) for sample in ordered_samples]

        true_positive = sum(1 for result in results if result.classification == "true_positive")
        false_positive = sum(1 for result in results if result.classification == "false_positive")
        true_negative = sum(1 for result in results if result.classification == "true_negative")
        false_negative = sum(1 for result in results if result.classification == "false_negative")
        accepted_count = sum(1 for result in results if result.accepted)
        rejected_count = len(results) - accepted_count

        accuracy = _safe_divide(true_positive + true_negative, len(results))
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)

        report = LoopClosureValidationReport(
            dataset_name=self.name,
            total_samples=len(results),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            sample_results=tuple(results),
            report_digest="",
        )

        digest_payload = report.asdict()
        digest_payload.pop("report_digest", None)
        report_digest = stable_hash(digest_payload)
        return LoopClosureValidationReport(
            dataset_name=report.dataset_name,
            total_samples=report.total_samples,
            accepted_count=report.accepted_count,
            rejected_count=report.rejected_count,
            true_positive=report.true_positive,
            false_positive=report.false_positive,
            true_negative=report.true_negative,
            false_negative=report.false_negative,
            accuracy=report.accuracy,
            precision=report.precision,
            recall=report.recall,
            thresholds=report.thresholds,
            sample_results=report.sample_results,
            report_digest=report_digest,
        )


def score_loop_closure_sample(
    sample: LoopClosureVerificationSample,
    thresholds: LoopClosureVerificationThresholds,
) -> LoopClosureSampleResult:
    temporal_delta = abs(sample.query_timestamp_s - sample.candidate_timestamp_s)
    inlier_ratio = _safe_divide(sample.inlier_count, sample.match_count)

    rejection_reasons: list[str] = []
    if sample.match_count < thresholds.min_match_count:
        rejection_reasons.append("match_count")
    if inlier_ratio < thresholds.min_inlier_ratio:
        rejection_reasons.append("inlier_ratio")
    if sample.mean_reprojection_error > thresholds.max_reprojection_error:
        rejection_reasons.append("reprojection_error")
    if temporal_delta < thresholds.min_temporal_separation_s:
        rejection_reasons.append("temporal_separation")
    if thresholds.max_temporal_separation_s is not None and temporal_delta > thresholds.max_temporal_separation_s:
        rejection_reasons.append("temporal_out_of_range")
    if (
        sample.rotation_error_deg is not None
        and thresholds.max_rotation_error_deg is not None
        and sample.rotation_error_deg > thresholds.max_rotation_error_deg
    ):
        rejection_reasons.append("rotation_error")
    if (
        sample.translation_error is not None
        and thresholds.max_translation_error is not None
        and sample.translation_error > thresholds.max_translation_error
    ):
        rejection_reasons.append("translation_error")

    geometric_score = _compute_geometric_score(sample, thresholds, inlier_ratio)
    temporal_score = _compute_temporal_score(temporal_delta, thresholds)
    combined_score = _weighted_score(geometric_score, temporal_score, thresholds)

    if geometric_score < thresholds.min_geometric_score:
        rejection_reasons.append("geometric_score")
    if temporal_score < thresholds.min_temporal_score:
        rejection_reasons.append("temporal_score")
    if combined_score < thresholds.min_combined_score:
        rejection_reasons.append("combined_score")

    accepted = len(rejection_reasons) == 0
    classification = _classify(accepted, sample.expected_match)

    return LoopClosureSampleResult(
        sample_id=sample.sample_id,
        query_frame_id=sample.query_frame_id,
        candidate_frame_id=sample.candidate_frame_id,
        temporal_delta_s=temporal_delta,
        match_count=sample.match_count,
        inlier_ratio=inlier_ratio,
        mean_reprojection_error=sample.mean_reprojection_error,
        geometric_score=geometric_score,
        temporal_score=temporal_score,
        combined_score=combined_score,
        accepted=accepted,
        expected_match=sample.expected_match,
        classification=classification,
        rejection_reasons=tuple(rejection_reasons),
    )


def _compute_geometric_score(
    sample: LoopClosureVerificationSample,
    thresholds: LoopClosureVerificationThresholds,
    inlier_ratio: float,
) -> float:
    components: list[float] = []
    components.append(min(1.0, inlier_ratio / thresholds.min_inlier_ratio))
    components.append(
        max(0.0, 1.0 - (sample.mean_reprojection_error / thresholds.max_reprojection_error))
    )
    components.append(min(1.0, sample.match_count / thresholds.min_match_count))
    if sample.rotation_error_deg is not None and thresholds.max_rotation_error_deg is not None:
        components.append(
            max(0.0, 1.0 - (sample.rotation_error_deg / thresholds.max_rotation_error_deg))
        )
    if sample.translation_error is not None and thresholds.max_translation_error is not None:
        components.append(
            max(0.0, 1.0 - (sample.translation_error / thresholds.max_translation_error))
        )
    return sum(components) / len(components)


def _compute_temporal_score(
    temporal_delta: float, thresholds: LoopClosureVerificationThresholds
) -> float:
    if temporal_delta < thresholds.min_temporal_separation_s:
        return 0.0
    if thresholds.max_temporal_separation_s is None:
        return 1.0
    if temporal_delta <= thresholds.max_temporal_separation_s:
        return 1.0
    overflow = temporal_delta - thresholds.max_temporal_separation_s
    return max(
        0.0,
        1.0 - (overflow / thresholds.max_temporal_separation_s),
    )


def _weighted_score(
    geometric_score: float, temporal_score: float, thresholds: LoopClosureVerificationThresholds
) -> float:
    total_weight = thresholds.geometric_weight + thresholds.temporal_weight
    return (
        geometric_score * thresholds.geometric_weight
        + temporal_score * thresholds.temporal_weight
    ) / total_weight


def _classify(accepted: bool, expected_match: bool) -> str:
    if accepted and expected_match:
        return "true_positive"
    if accepted and not expected_match:
        return "false_positive"
    if not accepted and expected_match:
        return "false_negative"
    return "true_negative"


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "LoopClosureSampleResult",
    "LoopClosureValidationReport",
    "LoopClosureVerificationDataset",
    "LoopClosureVerificationSample",
    "LoopClosureVerificationThresholds",
    "score_loop_closure_sample",
]
