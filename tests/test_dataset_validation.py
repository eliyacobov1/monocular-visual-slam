from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_validation import validate_kitti, validate_tum


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_validate_kitti_sequence(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequences" / "00"
    image_dir = sequence_path / "image_2"
    _touch(image_dir / "000000.png")
    (sequence_path / "times.txt").write_text("0.0\n", encoding="utf-8")

    calib = "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    (sequence_path / "calib.txt").write_text(calib, encoding="utf-8")

    result = validate_kitti(tmp_path, "00", camera="image_2")

    assert result.ok
    assert result.metadata["num_frames"] == 1
    assert result.metadata["num_timestamps"] == 1


def test_validate_tum_sequence(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    _touch(rgb_dir / "000000.png")
    (tmp_path / "groundtruth.txt").write_text("0 0 0\n", encoding="utf-8")

    result = validate_tum(tmp_path)

    assert result.ok
    assert result.metadata["num_frames"] == 1
