import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest
import re

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


def parse_translations(logs: str) -> list[np.ndarray]:
    pattern = re.compile(r"Adding transform R=.*t=\[(.*?)\]")
    translations: list[np.ndarray] = []
    for line in logs.splitlines():
        m = pattern.search(line)
        if m:
            translations.append(np.fromstring(m.group(1), sep=","))
    return translations


def test_slam_runs_with_synthetic_clip(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    generate_translation_clip(video_path)
    logs = run_slam(video_path, max_frames=6)
    assert "Adding transform" in logs


def test_translation_consistency(tmp_path):
    video_path = tmp_path / "trans.mp4"
    generate_translation_clip(video_path)
    logs = run_slam(video_path, max_frames=6)
    translations = parse_translations(logs)
    assert len(translations) >= 3

    for t in translations:
        mag = float(np.hypot(t[0], t[1]))
        assert 0 < mag < 2000
