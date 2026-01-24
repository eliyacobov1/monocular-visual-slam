from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kitti_dataset import KittiSequence, parse_kitti_calib_file


def _write_dummy_image(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def test_kitti_odometry_sequence_iteration(tmp_path: Path) -> None:
    seq_root = tmp_path / "sequences" / "00"
    image_dir = seq_root / "image_2"
    image_dir.mkdir(parents=True)
    _write_dummy_image(image_dir / "000000.png")
    _write_dummy_image(image_dir / "000001.png")

    (seq_root / "times.txt").write_text("0.0\n0.1\n")
    (seq_root / "calib.txt").write_text(
        "P2: 7 0 3 0 0 7 2 0 0 0 1 0\n"
    )

    sequence = KittiSequence(tmp_path, "00", camera="image_2")
    frames = list(sequence.iter_frames())

    assert len(frames) == 2
    assert frames[0].timestamp == 0.0
    assert frames[1].timestamp == 0.1

    intrinsics = sequence.camera_intrinsics()
    assert intrinsics is not None
    np.testing.assert_allclose(intrinsics[0, 0], 7.0)
    np.testing.assert_allclose(intrinsics[1, 1], 7.0)
    np.testing.assert_allclose(intrinsics[0, 2], 3.0)
    np.testing.assert_allclose(intrinsics[1, 2], 2.0)


def test_parse_kitti_calibration_file(tmp_path: Path) -> None:
    calib_path = tmp_path / "calib.txt"
    calib_path.write_text(
        "P0: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
    )
    calib = parse_kitti_calib_file(calib_path)

    assert "P0" in calib
    assert calib["P0"].shape == (3, 4)
    assert calib["R0_rect"].shape == (3, 3)
