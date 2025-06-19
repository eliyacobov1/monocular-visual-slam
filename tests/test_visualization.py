import os
import sys
import numpy as np
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from slam_path_estimator import VehiclePathLiveAnimator


def test_animator_loop_edge(tmp_path):
    os.environ["MPLBACKEND"] = "Agg"
    anim = VehiclePathLiveAnimator()
    anim.add_transform(np.eye(2), np.array([1.0, 0.0]))
    anim.add_transform(np.eye(2), np.array([1.0, 0.0]))
    anim.add_loop_edge(0, 2)
    time.sleep(0.2)
    save = tmp_path / "plot.png"
    anim.stop(str(save))
    assert (0, 2) in anim.loop_edges
    assert save.exists()
