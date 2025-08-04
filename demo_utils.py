"""Utilities for demo scripts.

Provides helper to ensure the sample video used in tests is available.
If the expected video file is missing it will be downloaded from the
project's known URL.  This avoids depending on a checked in binary file
while still allowing the demos and visualisations to run out of the box.
"""
from __future__ import annotations

from pathlib import Path
import urllib.request

TEST_VIDEO_URL = (
    "https://raw.githubusercontent.com/gurugithub/Carnd-Project3-Behavioral-Cloning/master/NVidiaRun2.mp4"
)
DEFAULT_VIDEO_PATH = Path("NVidiaRun2.mp4")


def ensure_sample_video(path: str | Path = DEFAULT_VIDEO_PATH) -> Path:
    """Return a path to a local copy of the sample video.

    The test suite downloads this video dynamically.  For the demos we do the
    same if the file is missing so that users can simply run the scripts
    without additional setup.  If the download fails a ``SystemExit`` is
    raised with a helpful message.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        print(f"Downloading sample video to {path}...")
        urllib.request.urlretrieve(TEST_VIDEO_URL, path)
    except Exception as exc:  # pragma: no cover - network failures
        raise SystemExit(f"Unable to download sample video: {exc}")
    return path
