import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from keyframe_manager import KeyframeManager


def _make_keypoints(count: int = 50) -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(count)]


def _make_descriptors(count: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(count, 32), dtype=np.uint8)


def _pose_with_translation(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [tx, ty, tz]
    return pose


def test_should_add_keyframe_requires_motion_or_low_overlap() -> None:
    manager = KeyframeManager(min_translation=0.5, min_rotation_deg=10.0)
    keypoints = _make_keypoints()
    descriptors = _make_descriptors()
    manager.add_keyframe(0, np.eye(4), keypoints, descriptors)

    current_pose = np.eye(4)
    assert manager.should_add_keyframe(current_pose, descriptors) is False


def test_should_add_keyframe_on_translation() -> None:
    manager = KeyframeManager(min_translation=0.1, min_rotation_deg=10.0)
    keypoints = _make_keypoints()
    descriptors = _make_descriptors()
    manager.add_keyframe(0, np.eye(4), keypoints, descriptors)

    current_pose = _pose_with_translation(0.2, 0.0, 0.0)
    assert manager.should_add_keyframe(current_pose, descriptors) is True
