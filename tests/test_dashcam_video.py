import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest


def run_slam(video: Path, max_frames: int = 30) -> str:
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
            "--log_level",
            "INFO",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        timeout=60,
        check=True,
    )
    return result.stdout + result.stderr

URL = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/solidWhiteRight.mp4"

@pytest.mark.skipif(
    "RUN_DASHCAM_TEST" not in os.environ,
    reason="optional dashcam video test",
)
def test_slam_on_dashcam_video(tmp_path):
    video_path = tmp_path / "solidWhiteRight.mp4"
    urllib.request.urlretrieve(URL, video_path)
    logs = run_slam(video_path, max_frames=30)
    assert "Added loop edge" in logs
    assert "Pose graph optimised" in logs
