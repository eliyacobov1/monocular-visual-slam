from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from camera_rig import CameraRig


def test_camera_rig_baseline_validation() -> None:
    fx = 7.0
    fy = 7.0
    cx = 3.0
    cy = 2.0
    baseline = 0.5
    calib = {
        "P2": np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        "P3": np.array(
            [
                [fx, 0.0, cx, -fx * baseline],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    }

    rig = CameraRig.from_kitti_calibration(calib, reference_camera="image_2")
    report = rig.validate()

    assert report.ok
    assert rig.baseline_to("image_3") == pytest.approx(baseline)
    assert report.metrics["baseline_m_image_3"] == pytest.approx(baseline)
