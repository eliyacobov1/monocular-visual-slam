import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

# Reuse run_slam helper from test_visual_slam but defined here to avoid package requirements

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

URL = "https://raw.githubusercontent.com/gurugithub/Carnd-Project3-Behavioral-Cloning/master/NVidiaRun2.mp4"

@pytest.mark.skipif(
    "RUN_NVIDIA_VIDEO_TEST" not in os.environ,
    reason="optional real video test",
)
def test_slam_on_nvidia_video(tmp_path):
    video_path = tmp_path / "NVidiaRun2.mp4"
    urllib.request.urlretrieve(URL, video_path)
    logs = run_slam(video_path, max_frames=20)
    assert "Adding transform" in logs

