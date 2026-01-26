import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import cv2
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

@dataclass
class Edge:
    i: int
    j: int
    R: np.ndarray  # 2x2 rotation
    t: np.ndarray  # 2 translation
    weight: float = 1.0

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

    def add_loop(self, i: int, j: int, R: np.ndarray, t: np.ndarray, weight: float = 0.5) -> None:
        self.edges.append(Edge(i, j, R, t, weight=weight))
        logger.info("Added loop edge between %d and %d (weight=%.2f)", i, j, weight)

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
                res.extend((r_err * e.weight).ravel())
                res.extend((t_err * e.weight).ravel())
            return np.array(res)

        x0 = []
        for pose in self.poses:
            x0.extend([pose[0, 2], pose[1, 2], np.arctan2(pose[1,0], pose[0,0])])
        x0 = np.array(x0)
        result = least_squares(residuals, x0, verbose=0, loss="soft_l1")
        logger.info("Pose graph optimisation success: %s", result.success)
        return pose_vec_to_mats(result.x)


@dataclass
class Edge3D:
    i: int
    j: int
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3 translation
    weight: float = 1.0


class PoseGraph3D:
    def __init__(self) -> None:
        self.poses: List[np.ndarray] = [np.eye(4)]
        self.edges: List[Edge3D] = []

    def add_pose(self, R: np.ndarray, t: np.ndarray) -> int:
        pose_delta = np.eye(4)
        pose_delta[:3, :3] = R
        pose_delta[:3, 3] = t[:3]
        new_pose = self.poses[-1] @ pose_delta
        self.poses.append(new_pose)
        if len(self.poses) > 1:
            self.edges.append(Edge3D(len(self.poses)-2, len(self.poses)-1, R, t))
        logger.debug("Added pose %d", len(self.poses) - 1)
        return len(self.poses) - 1

    def add_loop(self, i: int, j: int, R: np.ndarray, t: np.ndarray, weight: float = 0.5) -> None:
        self.edges.append(Edge3D(i, j, R, t, weight=weight))
        logger.info("Added loop edge between %d and %d (weight=%.2f)", i, j, weight)

    def optimize(self) -> List[np.ndarray]:
        def pose_vec_to_mats(x: np.ndarray) -> List[np.ndarray]:
            mats = []
            for k in range(len(self.poses)):
                tx, ty, tz, rx, ry, rz = x[6*k:6*k+6]
                R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                mats.append(T)
            return mats

        def residuals(x: np.ndarray) -> np.ndarray:
            mats = pose_vec_to_mats(x)
            res = []
            for e in self.edges:
                Ti = mats[e.i]
                Tj = mats[e.j]
                Tij = np.linalg.inv(Ti) @ Tj
                Rij = Tij[:3, :3]
                tij = Tij[:3, 3]
                r_err = Rij - e.R
                t_err = tij - e.t
                res.extend((r_err * e.weight).ravel())
                res.extend((t_err * e.weight).ravel())
            return np.array(res)

        x0 = []
        for pose in self.poses:
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            x0.extend([pose[0,3], pose[1,3], pose[2,3], *rvec.ravel()])
        x0 = np.array(x0)
        result = least_squares(residuals, x0, verbose=0, loss="soft_l1")
        logger.info("3D pose graph optimisation success: %s", result.success)
        return pose_vec_to_mats(result.x)


@dataclass
class EdgeSim3D:
    i: int
    j: int
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3 translation (map scale)
    s: float  # relative scale
    weight: float = 1.0


class PoseGraphSim3D:
    """Pose graph optimizer with Sim(3) edges for scale-drift correction."""

    def __init__(self, anchor_weight: float = 10.0) -> None:
        self.poses: List[np.ndarray] = [np.eye(4)]
        self.scales: List[float] = [1.0]
        self.edges: List[EdgeSim3D] = []
        self.anchor_weight = anchor_weight

    def add_pose(self, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> int:
        pose_delta = np.eye(4)
        pose_delta[:3, :3] = R
        pose_delta[:3, 3] = t[:3]
        new_pose = self.poses[-1] @ pose_delta
        self.poses.append(new_pose)
        self.scales.append(scale)
        if len(self.poses) > 1:
            self.edges.append(EdgeSim3D(len(self.poses) - 2, len(self.poses) - 1, R, t, scale))
        logger.debug("Added Sim(3) pose %d", len(self.poses) - 1)
        return len(self.poses) - 1

    def add_loop(
        self,
        i: int,
        j: int,
        R: np.ndarray,
        t: np.ndarray,
        s: float,
        weight: float = 0.5,
    ) -> None:
        if s <= 0:
            raise ValueError("Scale must be positive for Sim(3) edges")
        self.edges.append(EdgeSim3D(i, j, R, t, s, weight=weight))
        logger.info(
            "Added Sim(3) loop edge between %d and %d (scale=%.3f weight=%.2f)",
            i,
            j,
            s,
            weight,
        )

    def optimize(self) -> List[np.ndarray]:
        def unpack_pose(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
            tx, ty, tz, rx, ry, rz, log_s = x[7 * k : 7 * k + 7]
            R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
            t = np.array([tx, ty, tz])
            s = float(np.exp(log_s))
            return R, t, s

        def pose_vec_to_mats(x: np.ndarray) -> tuple[List[np.ndarray], List[float]]:
            mats = []
            scales = []
            for k in range(len(self.poses)):
                R, t, s = unpack_pose(x, k)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                mats.append(T)
                scales.append(s)
            return mats, scales

        def residuals(x: np.ndarray) -> np.ndarray:
            mats, scales = pose_vec_to_mats(x)
            res = []
            if self.anchor_weight > 0:
                t0 = x[:3]
                r0 = x[3:6]
                log_s0 = x[6]
                res.extend((self.anchor_weight * t0).ravel())
                res.extend((self.anchor_weight * r0).ravel())
                res.append(self.anchor_weight * log_s0)
            for e in self.edges:
                Ti = mats[e.i]
                Tj = mats[e.j]
                Ri = Ti[:3, :3]
                Rj = Tj[:3, :3]
                ti = Ti[:3, 3]
                tj = Tj[:3, 3]
                si = scales[e.i]
                sj = scales[e.j]
                s_ij = sj / si
                R_ij = Ri.T @ Rj
                t_ij = (1.0 / si) * (Ri.T @ (tj - ti))
                R_err = e.R.T @ R_ij
                rvec_err, _ = cv2.Rodrigues(R_err)
                res.extend((rvec_err.ravel() * e.weight))
                res.extend(((t_ij - e.t) * e.weight).ravel())
                res.append(e.weight * (np.log(s_ij) - np.log(e.s)))
            return np.array(res)

        x0 = []
        for pose, scale in zip(self.poses, self.scales):
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            x0.extend([pose[0, 3], pose[1, 3], pose[2, 3], *rvec.ravel(), np.log(scale)])
        x0 = np.array(x0)
        result = least_squares(residuals, x0, verbose=0, loss="soft_l1")
        mats, scales = pose_vec_to_mats(result.x)
        self.scales = scales
        logger.info("Sim(3) pose graph optimisation success: %s", result.success)
        return mats
