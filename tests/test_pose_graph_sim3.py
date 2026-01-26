import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pose_graph import PoseGraphSim3D


def test_sim3_pose_graph_reduces_drift():
    pg = PoseGraphSim3D(anchor_weight=10.0)
    drift_scale = 1.2
    steps = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    for step in steps:
        pg.add_pose(np.eye(3), step * drift_scale)

    pg.add_loop(0, len(pg.poses) - 1, np.eye(3), np.zeros(3), 1.0, weight=1.0)
    optimized = pg.optimize()

    orig_end = pg.poses[-1][:3, 3]
    opt_end = optimized[-1][:3, 3]

    assert np.linalg.norm(opt_end) < np.linalg.norm(orig_end)
