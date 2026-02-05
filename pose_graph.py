"""Pose graph optimizers with modular solver backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np

from graph_optimization import (
    LinearizedResidual,
    PoseGraphProblem,
    PoseGraphSnapshot,
    RobustLossConfig,
    RobustLossType,
    SolverConfig,
    SolverResult,
    get_solver_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    i: int
    j: int
    R: np.ndarray  # 2x2 rotation
    t: np.ndarray  # 2 translation
    weight: float = 1.0


@dataclass
class Edge3D:
    i: int
    j: int
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3 translation
    weight: float = 1.0


@dataclass
class EdgeSim3D:
    i: int
    j: int
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3 translation
    s: float  # relative scale
    weight: float = 1.0


class _BasePoseGraph:
    def __init__(
        self,
        *,
        solver_name: str = "scipy",
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
    ) -> None:
        self._solver_name = solver_name
        self._solver = get_solver_registry().get(solver_name)
        self._solver_config = solver_config or SolverConfig()
        self._loss_config = loss_config or RobustLossConfig()
        self._last_result: SolverResult | None = None
        self._last_snapshot: PoseGraphSnapshot | None = None

    @property
    def last_result(self) -> SolverResult | None:
        return self._last_result

    @property
    def last_snapshot(self) -> PoseGraphSnapshot | None:
        return self._last_snapshot

    def configure_solver(
        self,
        *,
        solver_name: str | None = None,
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
    ) -> None:
        if solver_name is not None:
            self._solver_name = solver_name
            self._solver = get_solver_registry().get(solver_name)
        if solver_config is not None:
            self._solver_config = solver_config
        if loss_config is not None:
            self._loss_config = loss_config

    def _wrap_angle(self, angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _finite_difference_jacobian(
        self,
        residual_fn: callable,
        vec: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        base = residual_fn(vec)
        jac = np.zeros((base.size, vec.size), dtype=float)
        for idx in range(vec.size):
            perturbed = vec.copy()
            perturbed[idx] += epsilon
            diff = residual_fn(perturbed) - base
            jac[:, idx] = diff / epsilon
        return jac


class PoseGraph(_BasePoseGraph):
    def __init__(
        self,
        *,
        solver_name: str = "scipy",
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
    ) -> None:
        super().__init__(solver_name=solver_name, solver_config=solver_config, loss_config=loss_config)
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

    def _pose_to_vec(self, pose: np.ndarray) -> np.ndarray:
        theta = np.arctan2(pose[1, 0], pose[0, 0])
        return np.array([pose[0, 2], pose[1, 2], theta])

    def _vec_to_pose(self, vec: np.ndarray) -> np.ndarray:
        tx, ty, theta = vec
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = [tx, ty]
        return T

    def _edge_residual(self, pose_i: np.ndarray, pose_j: np.ndarray, edge: Edge) -> np.ndarray:
        Tij = np.linalg.inv(pose_i) @ pose_j
        dx, dy = Tij[:2, 2]
        dtheta = np.arctan2(Tij[1, 0], Tij[0, 0])
        meas_theta = np.arctan2(edge.R[1, 0], edge.R[0, 0])
        err = np.array([
            dx - edge.t[0],
            dy - edge.t[1],
            self._wrap_angle(dtheta - meas_theta),
        ])
        return err

    def _pack_variables(self) -> np.ndarray:
        if len(self.poses) <= 1:
            return np.empty(0)
        return np.concatenate([self._pose_to_vec(pose) for pose in self.poses[1:]])

    def _pose_from_vector(self, x: np.ndarray, idx: int) -> np.ndarray:
        if idx == 0:
            return self.poses[0]
        offset = (idx - 1) * 3
        return self._vec_to_pose(x[offset : offset + 3])

    def _linearize_edges(self, x: np.ndarray) -> Iterable:
        for edge in self.edges:
            anchor_i = edge.i == 0
            anchor_j = edge.j == 0
            if edge.i == 0:
                pose_i = self.poses[0]
                vec_i = self._pose_to_vec(pose_i)
            else:
                offset_i = (edge.i - 1) * 3
                vec_i = x[offset_i : offset_i + 3]
                pose_i = self._vec_to_pose(vec_i)
            if edge.j == 0:
                pose_j = self.poses[0]
                vec_j = self._pose_to_vec(pose_j)
            else:
                offset_j = (edge.j - 1) * 3
                vec_j = x[offset_j : offset_j + 3]
                pose_j = self._vec_to_pose(vec_j)

            residual = self._edge_residual(pose_i, pose_j, edge)

            def residual_i(vec: np.ndarray) -> np.ndarray:
                return self._edge_residual(self._vec_to_pose(vec), pose_j, edge)

            def residual_j(vec: np.ndarray) -> np.ndarray:
                return self._edge_residual(pose_i, self._vec_to_pose(vec), edge)

            jac_i = self._finite_difference_jacobian(residual_i, vec_i)
            jac_j = self._finite_difference_jacobian(residual_j, vec_j)
            if anchor_i and not anchor_j:
                zeros = np.zeros_like(jac_j)
                yield (edge.j - 1, edge.j - 1, residual, jac_j, zeros, edge.weight)
            elif anchor_j and not anchor_i:
                zeros = np.zeros_like(jac_i)
                yield (edge.i - 1, edge.i - 1, residual, jac_i, zeros, edge.weight)
            elif not anchor_i and not anchor_j:
                yield (edge.i - 1, edge.j - 1, residual, jac_i, jac_j, edge.weight)

    def optimize(self) -> List[np.ndarray]:
        x0 = self._pack_variables()
        if x0.size == 0:
            return self.poses

        def residuals(x: np.ndarray) -> np.ndarray:
            res = []
            for edge in self.edges:
                pose_i = self._pose_from_vector(x, edge.i)
                pose_j = self._pose_from_vector(x, edge.j)
                res.append(self._edge_residual(pose_i, pose_j, edge) * edge.weight)
            return np.concatenate(res) if res else np.empty(0)

        def linearize(x: np.ndarray) -> Iterable:
            for i, j, residual, jac_i, jac_j, weight in self._linearize_edges(x):
                if i < 0 or j < 0:
                    continue
                yield LinearizedResidual(
                    i=i,
                    j=j,
                    residual=residual,
                    jacobian_i=jac_i,
                    jacobian_j=jac_j,
                    weight=weight,
                )

        edges_payload = [
            {
                "i": edge.i,
                "j": edge.j,
                "R": edge.R[:2, :2].tolist(),
                "t": edge.t[:2].tolist(),
                "weight": edge.weight,
            }
            for edge in self.edges
        ]
        snapshot = PoseGraphSnapshot(
            version=1,
            solver_name=self._solver_name,
            loss_config=self._loss_config,
            solver_config=self._solver_config,
            poses=[pose.tolist() for pose in self.poses],
            edges=edges_payload,
            metadata={"graph_type": "SE2"},
        )
        problem = PoseGraphProblem(
            residual_fn=residuals,
            linearize_fn=linearize,
            parameter_size=x0.size,
            block_size=3,
            snapshot=snapshot,
        )
        x_opt, result = self._solver.solve(problem, x0, self._solver_config, self._loss_config)
        optimized = [self.poses[0]]
        if x_opt.size:
            optimized.extend(
                [self._vec_to_pose(x_opt[i : i + 3]) for i in range(0, x_opt.size, 3)]
            )
        self._last_result = result
        self._last_snapshot = snapshot
        logger.info("Pose graph optimisation success: %s", result.success)
        return optimized


class PoseGraph3D(_BasePoseGraph):
    def __init__(
        self,
        *,
        solver_name: str = "scipy",
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
    ) -> None:
        super().__init__(solver_name=solver_name, solver_config=solver_config, loss_config=loss_config)
        self.poses: List[np.ndarray] = [np.eye(4)]
        self.edges: List[Edge3D] = []

    def add_pose(self, R: np.ndarray, t: np.ndarray) -> int:
        pose_delta = np.eye(4)
        pose_delta[:3, :3] = R
        pose_delta[:3, 3] = t[:3]
        new_pose = self.poses[-1] @ pose_delta
        self.poses.append(new_pose)
        if len(self.poses) > 1:
            self.edges.append(Edge3D(len(self.poses) - 2, len(self.poses) - 1, R, t))
        logger.debug("Added pose %d", len(self.poses) - 1)
        return len(self.poses) - 1

    def add_loop(self, i: int, j: int, R: np.ndarray, t: np.ndarray, weight: float = 0.5) -> None:
        self.edges.append(Edge3D(i, j, R, t, weight=weight))
        logger.info("Added loop edge between %d and %d (weight=%.2f)", i, j, weight)

    def _pose_to_vec(self, pose: np.ndarray) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        return np.hstack([rvec.ravel(), pose[:3, 3]])

    def _vec_to_pose(self, vec: np.ndarray) -> np.ndarray:
        rvec = vec[:3]
        tvec = vec[3:6]
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        return T

    def _edge_residual(self, pose_i: np.ndarray, pose_j: np.ndarray, edge: Edge3D) -> np.ndarray:
        Tij = np.linalg.inv(pose_i) @ pose_j
        Ri = Tij[:3, :3]
        ti = Tij[:3, 3]
        R_err = edge.R.T @ Ri
        rvec_err, _ = cv2.Rodrigues(R_err)
        t_err = ti - edge.t
        return np.hstack([rvec_err.ravel(), t_err.ravel()])

    def _pack_variables(self) -> np.ndarray:
        if len(self.poses) <= 1:
            return np.empty(0)
        return np.concatenate([self._pose_to_vec(pose) for pose in self.poses[1:]])

    def _pose_from_vector(self, x: np.ndarray, idx: int) -> np.ndarray:
        if idx == 0:
            return self.poses[0]
        offset = (idx - 1) * 6
        return self._vec_to_pose(x[offset : offset + 6])

    def _linearize_edges(self, x: np.ndarray) -> Iterable:
        for edge in self.edges:
            anchor_i = edge.i == 0
            anchor_j = edge.j == 0
            if edge.i == 0:
                pose_i = self.poses[0]
                vec_i = self._pose_to_vec(pose_i)
            else:
                offset_i = (edge.i - 1) * 6
                vec_i = x[offset_i : offset_i + 6]
                pose_i = self._vec_to_pose(vec_i)
            if edge.j == 0:
                pose_j = self.poses[0]
                vec_j = self._pose_to_vec(pose_j)
            else:
                offset_j = (edge.j - 1) * 6
                vec_j = x[offset_j : offset_j + 6]
                pose_j = self._vec_to_pose(vec_j)

            residual = self._edge_residual(pose_i, pose_j, edge)

            def residual_i(vec: np.ndarray) -> np.ndarray:
                return self._edge_residual(self._vec_to_pose(vec), pose_j, edge)

            def residual_j(vec: np.ndarray) -> np.ndarray:
                return self._edge_residual(pose_i, self._vec_to_pose(vec), edge)

            jac_i = self._finite_difference_jacobian(residual_i, vec_i)
            jac_j = self._finite_difference_jacobian(residual_j, vec_j)
            if anchor_i and not anchor_j:
                zeros = np.zeros_like(jac_j)
                yield (edge.j - 1, edge.j - 1, residual, jac_j, zeros, edge.weight)
            elif anchor_j and not anchor_i:
                zeros = np.zeros_like(jac_i)
                yield (edge.i - 1, edge.i - 1, residual, jac_i, zeros, edge.weight)
            elif not anchor_i and not anchor_j:
                yield (edge.i - 1, edge.j - 1, residual, jac_i, jac_j, edge.weight)

    def optimize(self) -> List[np.ndarray]:
        x0 = self._pack_variables()
        if x0.size == 0:
            return self.poses

        def residuals(x: np.ndarray) -> np.ndarray:
            res = []
            for edge in self.edges:
                pose_i = self._pose_from_vector(x, edge.i)
                pose_j = self._pose_from_vector(x, edge.j)
                res.append(self._edge_residual(pose_i, pose_j, edge) * edge.weight)
            return np.concatenate(res) if res else np.empty(0)

        edges_payload = [
            {
                "i": edge.i,
                "j": edge.j,
                "R": edge.R.tolist(),
                "t": edge.t.tolist(),
                "weight": edge.weight,
            }
            for edge in self.edges
        ]
        snapshot = PoseGraphSnapshot(
            version=1,
            solver_name=self._solver_name,
            loss_config=self._loss_config,
            solver_config=self._solver_config,
            poses=[pose.tolist() for pose in self.poses],
            edges=edges_payload,
            metadata={"graph_type": "SE3"},
        )
        problem = PoseGraphProblem(
            residual_fn=residuals,
            linearize_fn=lambda x: (
                LinearizedResidual(
                    i=i,
                    j=j,
                    residual=residual,
                    jacobian_i=jac_i,
                    jacobian_j=jac_j,
                    weight=weight,
                )
                for i, j, residual, jac_i, jac_j, weight in self._linearize_edges(x)
                if i >= 0 and j >= 0
            ),
            parameter_size=x0.size,
            block_size=6,
            snapshot=snapshot,
        )
        x_opt, result = self._solver.solve(problem, x0, self._solver_config, self._loss_config)
        optimized = [self.poses[0]]
        if x_opt.size:
            optimized.extend(
                [self._vec_to_pose(x_opt[i : i + 6]) for i in range(0, x_opt.size, 6)]
            )
        self._last_result = result
        self._last_snapshot = snapshot
        logger.info("3D pose graph optimisation success: %s", result.success)
        return optimized


class PoseGraphSim3D(_BasePoseGraph):
    """Pose graph optimizer with Sim(3) edges for scale-drift correction."""

    def __init__(
        self,
        anchor_weight: float = 10.0,
        *,
        solver_name: str = "scipy",
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
    ) -> None:
        super().__init__(solver_name=solver_name, solver_config=solver_config, loss_config=loss_config)
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

    def _pose_to_vec(self, pose: np.ndarray, scale: float) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        return np.hstack([rvec.ravel(), pose[:3, 3], np.log(scale)])

    def _vec_to_pose(self, vec: np.ndarray) -> tuple[np.ndarray, float]:
        rvec = vec[:3]
        tvec = vec[3:6]
        log_s = vec[6]
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        return T, float(np.exp(log_s))

    def _edge_residual(
        self,
        pose_i: np.ndarray,
        scale_i: float,
        pose_j: np.ndarray,
        scale_j: float,
        edge: EdgeSim3D,
    ) -> np.ndarray:
        Ri = pose_i[:3, :3]
        Rj = pose_j[:3, :3]
        ti = pose_i[:3, 3]
        tj = pose_j[:3, 3]
        s_ij = scale_j / scale_i
        R_ij = Ri.T @ Rj
        t_ij = (1.0 / scale_i) * (Ri.T @ (tj - ti))
        R_err = edge.R.T @ R_ij
        rvec_err, _ = cv2.Rodrigues(R_err)
        t_err = t_ij - edge.t
        s_err = np.log(s_ij) - np.log(edge.s)
        residual = np.hstack([rvec_err.ravel(), t_err.ravel(), s_err])
        return residual

    def _pack_variables(self) -> np.ndarray:
        if len(self.poses) <= 1:
            return np.empty(0)
        return np.concatenate(
            [self._pose_to_vec(pose, scale) for pose, scale in zip(self.poses[1:], self.scales[1:])]
        )

    def _pose_from_vector(self, x: np.ndarray, idx: int) -> tuple[np.ndarray, float]:
        if idx == 0:
            return self.poses[0], self.scales[0]
        offset = (idx - 1) * 7
        return self._vec_to_pose(x[offset : offset + 7])

    def _linearize_edges(self, x: np.ndarray) -> Iterable:
        for edge in self.edges:
            anchor_i = edge.i == 0
            anchor_j = edge.j == 0
            if edge.i == 0:
                pose_i = self.poses[0]
                scale_i = self.scales[0]
                vec_i = self._pose_to_vec(pose_i, scale_i)
            else:
                offset_i = (edge.i - 1) * 7
                vec_i = x[offset_i : offset_i + 7]
                pose_i, scale_i = self._vec_to_pose(vec_i)
            if edge.j == 0:
                pose_j = self.poses[0]
                scale_j = self.scales[0]
                vec_j = self._pose_to_vec(pose_j, scale_j)
            else:
                offset_j = (edge.j - 1) * 7
                vec_j = x[offset_j : offset_j + 7]
                pose_j, scale_j = self._vec_to_pose(vec_j)

            residual = self._edge_residual(pose_i, scale_i, pose_j, scale_j, edge)

            def residual_i(vec: np.ndarray) -> np.ndarray:
                pose, scale = self._vec_to_pose(vec)
                return self._edge_residual(pose, scale, pose_j, scale_j, edge)

            def residual_j(vec: np.ndarray) -> np.ndarray:
                pose, scale = self._vec_to_pose(vec)
                return self._edge_residual(pose_i, scale_i, pose, scale, edge)

            jac_i = self._finite_difference_jacobian(residual_i, vec_i)
            jac_j = self._finite_difference_jacobian(residual_j, vec_j)
            if anchor_i and not anchor_j:
                zeros = np.zeros_like(jac_j)
                yield (edge.j - 1, edge.j - 1, residual, jac_j, zeros, edge.weight)
            elif anchor_j and not anchor_i:
                zeros = np.zeros_like(jac_i)
                yield (edge.i - 1, edge.i - 1, residual, jac_i, zeros, edge.weight)
            elif not anchor_i and not anchor_j:
                yield (edge.i - 1, edge.j - 1, residual, jac_i, jac_j, edge.weight)

    def optimize(self) -> List[np.ndarray]:
        x0 = self._pack_variables()
        if x0.size == 0:
            return self.poses

        def residuals(x: np.ndarray) -> np.ndarray:
            res = []
            if self.anchor_weight > 0:
                res.append(self.anchor_weight * self._pose_to_vec(self.poses[0], self.scales[0]))
            for edge in self.edges:
                pose_i, scale_i = self._pose_from_vector(x, edge.i)
                pose_j, scale_j = self._pose_from_vector(x, edge.j)
                res.append(self._edge_residual(pose_i, scale_i, pose_j, scale_j, edge) * edge.weight)
            return np.concatenate(res) if res else np.empty(0)

        edges_payload = [
            {
                "i": edge.i,
                "j": edge.j,
                "R": edge.R.tolist(),
                "t": edge.t.tolist(),
                "s": edge.s,
                "weight": edge.weight,
            }
            for edge in self.edges
        ]
        snapshot = PoseGraphSnapshot(
            version=1,
            solver_name=self._solver_name,
            loss_config=self._loss_config,
            solver_config=self._solver_config,
            poses=[pose.tolist() for pose in self.poses],
            edges=edges_payload,
            metadata={"graph_type": "Sim3"},
        )
        problem = PoseGraphProblem(
            residual_fn=residuals,
            linearize_fn=lambda x: (
                LinearizedResidual(
                    i=i,
                    j=j,
                    residual=residual,
                    jacobian_i=jac_i,
                    jacobian_j=jac_j,
                    weight=weight,
                )
                for i, j, residual, jac_i, jac_j, weight in self._linearize_edges(x)
                if i >= 0 and j >= 0
            ),
            parameter_size=x0.size,
            block_size=7,
            snapshot=snapshot,
        )
        x_opt, result = self._solver.solve(problem, x0, self._solver_config, self._loss_config)
        optimized = [self.poses[0]]
        optimized_scales = [self.scales[0]]
        if x_opt.size:
            for offset in range(0, x_opt.size, 7):
                pose, scale = self._vec_to_pose(x_opt[offset : offset + 7])
                optimized.append(pose)
                optimized_scales.append(scale)
        self._last_result = result
        self._last_snapshot = snapshot
        self.scales = optimized_scales
        logger.info("Sim(3) pose graph optimisation success: %s", result.success)
        return optimized


__all__ = [
    "Edge",
    "Edge3D",
    "EdgeSim3D",
    "PoseGraph",
    "PoseGraph3D",
    "PoseGraphSim3D",
    "RobustLossConfig",
    "RobustLossType",
    "SolverConfig",
]
