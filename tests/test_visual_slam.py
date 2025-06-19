import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def generate_translation_clip(
    path: Path, num_frames: int = 6, size: tuple[int, int] = (1080, 1920), step=(2, 1)
) -> np.ndarray:
    """Create a video of a noisy image translating by ``step`` each frame.

    Returns the expected translation once frames are resized by the SLAM script.
    """
    h, w = size
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(h, w), dtype=np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (w, h))

    for i in range(num_frames):
        M = np.array([[1, 0, step[0] * i], [0, 1, step[1] * i]], dtype=np.float32)
        frame = cv2.warpAffine(base, M, (w, h))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)
    writer.release()

    scale_x = 1080 / w
    scale_y = 1920 / h
    return np.array([scale_x * step[0], scale_y * step[1]])


def run_slam(video: Path, max_frames: int = 5) -> str:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            "visual_slam_offline_entry_point.py",
            "--video",
            str(video),
            "--max_frames",
            str(max_frames),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
        check=True,
    )
    return result.stdout + result.stderr


def parse_homographies(logs: str) -> list[np.ndarray]:
    lines = logs.splitlines()
    homos: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("DEBUG::cv2_e2e - Homography:"):
            row1 = np.fromstring(lines[i + 1].strip(" []"), sep=" ")
            row2 = np.fromstring(lines[i + 2].strip(" []"), sep=" ")
            row3 = np.fromstring(lines[i + 3].strip(" []"), sep=" ")
            homos.append(np.vstack([row1, row2, row3]))
            i += 4
        else:
            i += 1
    return homos


def test_slam_runs_with_synthetic_clip(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    generate_translation_clip(video_path)
    logs = run_slam(video_path, max_frames=6)
    assert "Adding transform" in logs


def test_homography_consistency(tmp_path):
    video_path = tmp_path / "trans.mp4"
    generate_translation_clip(video_path)
    logs = run_slam(video_path, max_frames=6)
    homos = parse_homographies(logs)
    assert len(homos) >= 3

    for H in homos:
        tx, ty = H[0, 2], H[1, 2]
        rot_deg = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        mag = float(np.hypot(tx, ty))
        assert 0.3 < mag < 5.0
        assert abs(rot_deg) < 5
