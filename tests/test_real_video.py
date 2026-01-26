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


def _fetch_video(url: str, filename: str, tmp_path: Path) -> Path:
    cache_dir = os.environ.get("VIDEO_CACHE_DIR")
    if cache_dir:
        cache_path = Path(cache_dir) / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            urllib.request.urlretrieve(url, cache_path)
        return cache_path
    tmp_file = tmp_path / filename
    urllib.request.urlretrieve(url, tmp_file)
    return tmp_file


URL = "https://raw.githubusercontent.com/gurugithub/Carnd-Project3-Behavioral-Cloning/master/NVidiaRun2.mp4"

@pytest.mark.skipif(
    "RUN_NVIDIA_VIDEO_TEST" not in os.environ,
    reason="optional real video test",
)
def test_slam_on_nvidia_video(tmp_path):
    video_path = _fetch_video(URL, "NVidiaRun2.mp4", tmp_path)
    logs = run_slam(video_path, max_frames=20)
    assert "Adding transform" in logs
