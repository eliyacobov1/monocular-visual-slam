import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_pipeline import (
    FeaturePipelineConfig,
    adaptive_ransac_threshold,
    build_feature_pipeline,
    matches_to_points,
)


def _make_synthetic_pair(shape: tuple[int, int] = (240, 320)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=shape, dtype=np.uint8)
    shift = np.roll(base, shift=3, axis=1)
    return base, shift


def test_orb_feature_pipeline_matches_and_threshold():
    img_a, img_b = _make_synthetic_pair()
    config = FeaturePipelineConfig(name="orb", nfeatures=500, cross_check=True)
    pipeline = build_feature_pipeline(config)

    kp_a, desc_a = pipeline.detect_and_describe(img_a)
    kp_b, desc_b = pipeline.detect_and_describe(img_b)
    matches = pipeline.match(desc_a, desc_b)
    pts_a, pts_b = matches_to_points(kp_a, kp_b, matches)

    threshold = adaptive_ransac_threshold(
        pts_a,
        pts_b,
        base_threshold=0.01,
        min_threshold=0.005,
        max_threshold=0.03,
    )

    assert isinstance(matches, list)
    assert threshold >= 0.005
    assert threshold <= 0.03
