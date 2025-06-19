import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from loop_closure import BoWDatabase
from pose_graph import PoseGraph


def test_bow_detects_loop():
    rng = np.random.default_rng(0)
    db = BoWDatabase(vocab_size=2)
    desc1 = rng.integers(0, 256, size=(20, 32)).astype(np.float32)
    desc2 = rng.integers(0, 256, size=(20, 32)).astype(np.float32)
    db.add_frame(0, desc1)
    db.add_frame(1, desc2)
    # descriptors to trigger training quickly
    desc3 = rng.integers(0, 256, size=(20, 32)).astype(np.float32)
    db.add_frame(2, desc3)
    assert db.vocab_trained
    loop_id = db.detect_loop(desc1, threshold=0.5)
    assert loop_id == 0


def test_pose_graph_closure():
    rng = np.random.default_rng(1)
    pg = PoseGraph()

    def rot(th):
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    steps = [
        (rot(0), np.array([1.0, 0.0])),
        (rot(np.pi/2), np.array([0.0, 1.0])),
        (rot(np.pi/2), np.array([-1.0, 0.0])),
        (rot(np.pi/2), np.array([0.0, -1.0])),
    ]
    for R, t in steps:
        noisy_t = t + rng.normal(0, 0.05, size=2)
        pg.add_pose(R, noisy_t)

    pg.add_loop(0, len(pg.poses)-1, np.eye(2), np.zeros(2))
    optimized = pg.optimize()

    orig_end = pg.poses[-1][:2, 2]
    opt_end = optimized[-1][:2, 2]

    assert np.linalg.norm(opt_end) < np.linalg.norm(orig_end)
