import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

@dataclass
class Edge:
    i: int
    j: int
    R: np.ndarray  # 2x2 rotation
    t: np.ndarray  # 2 translation

class PoseGraph:
    def __init__(self) -> None:
        self.poses: List[np.ndarray] = [np.eye(3)]
        self.edges: List[Edge] = []

    def add_pose(self, R: np.ndarray, t: np.ndarray) -> int:
        pose_delta = np.eye(3)
        pose_delta[:2, :2] = R[:2, :2]
        pose_delta[:2, 2] = t[:2]
        new_pose = self.poses[-1] @ pose_delta
        self.poses.append(new_pose)
        if len(self.poses) > 1:
            self.edges.append(Edge(len(self.poses) - 2, len(self.poses) - 1, R, t))
        logger.debug("Added pose %d: t=%s", len(self.poses) - 1, t.tolist())
        return len(self.poses) - 1

    def add_loop(self, i: int, j: int, R: np.ndarray, t: np.ndarray) -> None:
        self.edges.append(Edge(i, j, R, t))
        logger.info("Added loop edge between %d and %d", i, j)

    def optimize(self) -> List[np.ndarray]:
        def pose_vec_to_mats(x: np.ndarray) -> List[np.ndarray]:
            mats = []
            for k in range(len(self.poses)):
                th = x[3*k+2]
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                t = x[3*k:3*k+2]
                mat = np.eye(3)
                mat[:2, :2] = R
                mat[:2, 2] = t
                mats.append(mat)
            return mats

        def residuals(x: np.ndarray) -> np.ndarray:
            mats = pose_vec_to_mats(x)
            res = []
            for e in self.edges:
                Ti = mats[e.i]
                Tj = mats[e.j]
                Tij = np.linalg.inv(Ti) @ Tj
                Rij = Tij[:2, :2]
                tij = Tij[:2, 2]
                r_err = Rij - e.R[:2, :2]
                t_err = tij - e.t[:2]
                res.extend(r_err.ravel())
                res.extend(t_err.ravel())
            return np.array(res)

        x0 = []
        for pose in self.poses:
            x0.extend([pose[0, 2], pose[1, 2], np.arctan2(pose[1,0], pose[0,0])])
        x0 = np.array(x0)
        result = least_squares(residuals, x0, verbose=0)
        logger.info("Pose graph optimisation success: %s", result.success)
        return pose_vec_to_mats(result.x)
