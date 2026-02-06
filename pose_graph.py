"""Pose graph optimizers with modular solver backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np

from factor_graph import (
    FactorGraph,
    FactorGraphConfig,
    SE2BetweenFactor,
    SE3BetweenFactor,
    Sim3BetweenFactor,
)
from graph_optimization import (
    PoseGraphProblem,
    PoseGraphSnapshot,
    RobustLossConfig,
    RobustLossType,
    SolverConfig,
    SolverResult,
    get_solver_registry,
)
from optimization_control_plane import OptimizationControlConfig, OptimizationRunReport, OptimizationSupervisor

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
        self._control_config = OptimizationControlConfig()
        self._last_result: SolverResult | None = None
        self._last_snapshot: PoseGraphSnapshot | None = None
        self._last_report: OptimizationRunReport | None = None

    @property
    def last_result(self) -> SolverResult | None:
        return self._last_result

    @property
    def last_snapshot(self) -> PoseGraphSnapshot | None:
        return self._last_snapshot

    @property
    def last_report(self) -> OptimizationRunReport | None:
        return self._last_report

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

    def configure_control_plane(self, *, control_config: OptimizationControlConfig | None = None) -> None:
        if control_config is not None:
            self._control_config = control_config

    def _wrap_angle(self, angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _pose_to_vec(self, pose: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _vec_to_pose(self, vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _build_problem(
        self,
        graph: FactorGraph,
        snapshot: PoseGraphSnapshot,
    ) -> tuple[PoseGraphProblem, np.ndarray, list[int]]:
        problem, x0 = graph.build_problem(snapshot)
        ordered_ids = graph.ordered_variable_ids()
        return problem, x0, ordered_ids

    def _solve_problem(
        self,
        problem: PoseGraphProblem,
        x0: np.ndarray,
        snapshot: PoseGraphSnapshot,
    ) -> tuple[np.ndarray, SolverResult, OptimizationRunReport]:
        supervisor = OptimizationSupervisor(self._control_config)
        x_opt_list, result, report = supervisor.run(
            solver=self._solver,
            problem=problem,
            x0=x0.tolist(),
            solver_config=self._solver_config,
            loss_config=self._loss_config,
            snapshot=snapshot,
            solver_name=self._solver_name,
        )
        x_opt = np.asarray(x_opt_list, dtype=float)
        return x_opt, result, report


class PoseGraph(_BasePoseGraph):
    def __init__(
        self,
        *,
        solver_name: str = "scipy",
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
        numeric_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(solver_name=solver_name, solver_config=solver_config, loss_config=loss_config)
        self.poses: List[np.ndarray] = [np.eye(3)]
        self.edges: List[Edge] = []
        self._numeric_epsilon = numeric_epsilon

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

    def _edge_measurement(self, edge: Edge) -> np.ndarray:
        meas_theta = np.arctan2(edge.R[1, 0], edge.R[0, 0])
        return np.array([edge.t[0], edge.t[1], meas_theta], dtype=float)

    def _build_factor_graph(self) -> FactorGraph:
        config = FactorGraphConfig(state_dim=3, anchor_ids=(0,), numeric_epsilon=self._numeric_epsilon)
        graph = FactorGraph(config)
        for idx, pose in enumerate(self.poses):
            graph.add_variable(idx, self._pose_to_vec(pose))
        for edge in self.edges:
            measurement = self._edge_measurement(edge)
            graph.add_factor(SE2BetweenFactor(edge.i, edge.j, measurement, weight=edge.weight))
        return graph

    def optimize(self) -> List[np.ndarray]:
        graph = self._build_factor_graph()
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
            version=2,
            solver_name=self._solver_name,
            loss_config=self._loss_config,
            solver_config=self._solver_config,
            poses=[pose.tolist() for pose in self.poses],
            edges=edges_payload,
            metadata={
                "graph_type": "SE2",
                "factor_graph": True,
                "numeric_epsilon": self._numeric_epsilon,
            },
        )
        problem, x0, ordered_ids = self._build_problem(graph, snapshot)
        if x0.size == 0:
            return self.poses
        x_opt, result, report = self._solve_problem(problem, x0, snapshot)
        optimized = [self.poses[0]]
        for index, var_id in enumerate(ordered_ids):
            offset = index * 3
            pose_vec = x_opt[offset : offset + 3]
            pose = self._vec_to_pose(pose_vec)
            if var_id < len(self.poses):
                optimized.append(pose)
            else:
                optimized.append(pose)
        self._last_result = result
        self._last_snapshot = snapshot
        self._last_report = report
        logger.info("Pose graph optimisation success: %s", result.success)
        return optimized


class PoseGraph3D(_BasePoseGraph):
    def __init__(
        self,
        *,
        solver_name: str = "scipy",
        solver_config: SolverConfig | None = None,
        loss_config: RobustLossConfig | None = None,
        numeric_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(solver_name=solver_name, solver_config=solver_config, loss_config=loss_config)
        self.poses: List[np.ndarray] = [np.eye(4)]
        self.edges: List[Edge3D] = []
        self._numeric_epsilon = numeric_epsilon

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

    def _build_factor_graph(self) -> FactorGraph:
        config = FactorGraphConfig(state_dim=6, anchor_ids=(0,), numeric_epsilon=self._numeric_epsilon)
        graph = FactorGraph(config)
        for idx, pose in enumerate(self.poses):
            graph.add_variable(idx, self._pose_to_vec(pose))
        for edge in self.edges:
            graph.add_factor(
                SE3BetweenFactor(
                    edge.i,
                    edge.j,
                    measurement_r=edge.R,
                    measurement_t=edge.t,
                    weight=edge.weight,
                    epsilon=self._numeric_epsilon,
                )
            )
        return graph

    def optimize(self) -> List[np.ndarray]:
        graph = self._build_factor_graph()
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
            version=2,
            solver_name=self._solver_name,
            loss_config=self._loss_config,
            solver_config=self._solver_config,
            poses=[pose.tolist() for pose in self.poses],
            edges=edges_payload,
            metadata={
                "graph_type": "SE3",
                "factor_graph": True,
                "numeric_epsilon": self._numeric_epsilon,
            },
        )
        problem, x0, ordered_ids = self._build_problem(graph, snapshot)
        if x0.size == 0:
            return self.poses
        x_opt, result, report = self._solve_problem(problem, x0, snapshot)
        optimized = [self.poses[0]]
        for index, var_id in enumerate(ordered_ids):
            offset = index * 6
            pose_vec = x_opt[offset : offset + 6]
            pose = self._vec_to_pose(pose_vec)
            if var_id < len(self.poses):
                optimized.append(pose)
            else:
                optimized.append(pose)
        self._last_result = result
        self._last_snapshot = snapshot
        self._last_report = report
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
        numeric_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(solver_name=solver_name, solver_config=solver_config, loss_config=loss_config)
        self.poses: List[np.ndarray] = [np.eye(4)]
        self.scales: List[float] = [1.0]
        self.edges: List[EdgeSim3D] = []
        self.anchor_weight = anchor_weight
        self._numeric_epsilon = numeric_epsilon

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

    def _build_factor_graph(self) -> FactorGraph:
        config = FactorGraphConfig(state_dim=7, anchor_ids=(0,), numeric_epsilon=self._numeric_epsilon)
        graph = FactorGraph(config)
        for idx, (pose, scale) in enumerate(zip(self.poses, self.scales)):
            graph.add_variable(idx, self._pose_to_vec(pose, scale))
        for edge in self.edges:
            graph.add_factor(
                Sim3BetweenFactor(
                    edge.i,
                    edge.j,
                    measurement_r=edge.R,
                    measurement_t=edge.t,
                    measurement_s=edge.s,
                    weight=edge.weight,
                    epsilon=self._numeric_epsilon,
                )
            )
        return graph

    def optimize(self) -> List[np.ndarray]:
        graph = self._build_factor_graph()
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
            version=2,
            solver_name=self._solver_name,
            loss_config=self._loss_config,
            solver_config=self._solver_config,
            poses=[pose.tolist() for pose in self.poses],
            edges=edges_payload,
            metadata={
                "graph_type": "Sim3",
                "factor_graph": True,
                "anchor_weight": self.anchor_weight,
                "numeric_epsilon": self._numeric_epsilon,
            },
        )
        problem, x0, ordered_ids = self._build_problem(graph, snapshot)
        if x0.size == 0:
            return self.poses
        x_opt, result, report = self._solve_problem(problem, x0, snapshot)
        optimized = [self.poses[0]]
        optimized_scales = [self.scales[0]]
        for index, var_id in enumerate(ordered_ids):
            offset = index * 7
            pose_vec = x_opt[offset : offset + 7]
            pose, scale = self._vec_to_pose(pose_vec)
            if var_id < len(self.poses):
                optimized.append(pose)
                optimized_scales.append(scale)
            else:
                optimized.append(pose)
                optimized_scales.append(scale)
        self._last_result = result
        self._last_snapshot = snapshot
        self._last_report = report
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
