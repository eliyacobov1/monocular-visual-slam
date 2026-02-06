"""Factor-graph primitives with deterministic ordering and analytic Jacobians."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import cv2
import numpy as np

from graph_optimization import LinearizedResidual, PoseGraphProblem, PoseGraphSnapshot

LOGGER = logging.getLogger(__name__)


class BinaryFactor(Protocol):
    """Protocol for binary factors in a factor graph."""

    i: int
    j: int
    weight: float

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        ...

    def jacobians(self, xi: np.ndarray, xj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class FactorGraphConfig:
    """Configuration for factor graph linearization."""

    state_dim: int
    anchor_ids: tuple[int, ...] = (0,)
    numeric_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.numeric_epsilon <= 0:
            raise ValueError("numeric_epsilon must be positive")


@dataclass
class FactorGraph:
    """Deterministic factor-graph builder with fixed anchors."""

    config: FactorGraphConfig
    variables: dict[int, np.ndarray] = field(default_factory=dict)
    factors: list[BinaryFactor] = field(default_factory=list)

    def add_variable(self, variable_id: int, value: np.ndarray) -> None:
        if variable_id in self.variables:
            raise ValueError(f"Variable {variable_id} already exists")
        value = np.asarray(value, dtype=float)
        if value.size != self.config.state_dim:
            raise ValueError("Variable dimensionality mismatch")
        self.variables[variable_id] = value

    def set_variable(self, variable_id: int, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        if value.size != self.config.state_dim:
            raise ValueError("Variable dimensionality mismatch")
        self.variables[variable_id] = value

    def add_factor(self, factor: BinaryFactor) -> None:
        self.factors.append(factor)

    def ordered_variable_ids(self) -> list[int]:
        return sorted(var_id for var_id in self.variables.keys() if var_id not in self.config.anchor_ids)

    def _pack_state(self) -> np.ndarray:
        ordered_ids = self.ordered_variable_ids()
        if not ordered_ids:
            return np.empty(0)
        return np.concatenate([self.variables[var_id] for var_id in ordered_ids])

    def _unpack_state(self, x: np.ndarray) -> dict[int, np.ndarray]:
        state: dict[int, np.ndarray] = {}
        for anchor_id in self.config.anchor_ids:
            if anchor_id in self.variables:
                state[anchor_id] = self.variables[anchor_id]
        ordered_ids = self.ordered_variable_ids()
        for index, var_id in enumerate(ordered_ids):
            offset = index * self.config.state_dim
            state[var_id] = x[offset : offset + self.config.state_dim]
        return state

    def build_problem(self, snapshot: PoseGraphSnapshot) -> tuple[PoseGraphProblem, np.ndarray]:
        ordered_ids = self.ordered_variable_ids()
        index_map = {var_id: idx for idx, var_id in enumerate(ordered_ids)}
        x0 = self._pack_state()

        def residuals(x: np.ndarray) -> np.ndarray:
            state = self._unpack_state(x)
            residual_blocks: list[np.ndarray] = []
            for factor in self.factors:
                xi = state[factor.i]
                xj = state[factor.j]
                residual = factor.residual(xi, xj)
                residual_blocks.append(np.sqrt(factor.weight) * residual)
            if not residual_blocks:
                return np.empty(0)
            return np.concatenate(residual_blocks)

        def linearize(x: np.ndarray) -> Iterable[LinearizedResidual]:
            state = self._unpack_state(x)
            for factor in self.factors:
                if factor.i in self.config.anchor_ids and factor.j in self.config.anchor_ids:
                    continue
                xi = state[factor.i]
                xj = state[factor.j]
                residual = factor.residual(xi, xj)
                jac_i, jac_j = factor.jacobians(xi, xj)
                if factor.i in self.config.anchor_ids:
                    j_index = index_map.get(factor.j)
                    if j_index is None:
                        continue
                    yield LinearizedResidual(
                        i=j_index,
                        j=None,
                        residual=residual,
                        jacobian_i=jac_j,
                        jacobian_j=None,
                        weight=factor.weight,
                    )
                    continue
                if factor.j in self.config.anchor_ids:
                    i_index = index_map.get(factor.i)
                    if i_index is None:
                        continue
                    yield LinearizedResidual(
                        i=i_index,
                        j=None,
                        residual=residual,
                        jacobian_i=jac_i,
                        jacobian_j=None,
                        weight=factor.weight,
                    )
                    continue
                i_index = index_map.get(factor.i)
                j_index = index_map.get(factor.j)
                if i_index is None or j_index is None:
                    continue
                yield LinearizedResidual(
                    i=i_index,
                    j=j_index,
                    residual=residual,
                    jacobian_i=jac_i,
                    jacobian_j=jac_j,
                    weight=factor.weight,
                )

        problem = PoseGraphProblem(
            residual_fn=residuals,
            linearize_fn=linearize,
            parameter_size=x0.size,
            block_size=self.config.state_dim,
            snapshot=snapshot,
        )
        return problem, x0


@dataclass(frozen=True)
class SE2BetweenFactor:
    """Analytic SE(2) between-factor."""

    i: int
    j: int
    measurement: np.ndarray
    weight: float = 1.0

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        xj = np.asarray(xj, dtype=float)
        dx, dy = _relative_se2(xi, xj)
        dtheta = _wrap_angle(xj[2] - xi[2])
        return np.array([
            dx - self.measurement[0],
            dy - self.measurement[1],
            _wrap_angle(dtheta - self.measurement[2]),
        ])

    def jacobians(self, xi: np.ndarray, xj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xi = np.asarray(xi, dtype=float)
        xj = np.asarray(xj, dtype=float)
        dx = xj[0] - xi[0]
        dy = xj[1] - xi[1]
        theta = xi[2]
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        jac_i = np.array(
            [
                [-cos_t, -sin_t, -sin_t * dx + cos_t * dy],
                [sin_t, -cos_t, -cos_t * dx - sin_t * dy],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        jac_j = np.array(
            [
                [cos_t, sin_t, 0.0],
                [-sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return jac_i, jac_j


@dataclass(frozen=True)
class SE3BetweenFactor:
    """SE(3) between-factor with deterministic numeric Jacobians."""

    i: int
    j: int
    measurement_r: np.ndarray
    measurement_t: np.ndarray
    weight: float = 1.0
    epsilon: float = 1e-6

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        pose_i = _vec_to_se3(xi)
        pose_j = _vec_to_se3(xj)
        Tij = np.linalg.inv(pose_i) @ pose_j
        Ri = Tij[:3, :3]
        ti = Tij[:3, 3]
        r_err = self.measurement_r.T @ Ri
        rvec_err, _ = cv2.Rodrigues(r_err)
        t_err = ti - self.measurement_t
        return np.hstack([rvec_err.ravel(), t_err.ravel()])

    def jacobians(self, xi: np.ndarray, xj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jac_i = _numeric_jacobian(lambda v: self.residual(v, xj), xi, self.epsilon)
        jac_j = _numeric_jacobian(lambda v: self.residual(xi, v), xj, self.epsilon)
        return jac_i, jac_j


@dataclass(frozen=True)
class Sim3BetweenFactor:
    """Sim(3) between-factor with deterministic numeric Jacobians."""

    i: int
    j: int
    measurement_r: np.ndarray
    measurement_t: np.ndarray
    measurement_s: float
    weight: float = 1.0
    epsilon: float = 1e-6

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        pose_i, scale_i = _vec_to_sim3(xi)
        pose_j, scale_j = _vec_to_sim3(xj)
        Ri = pose_i[:3, :3]
        Rj = pose_j[:3, :3]
        ti = pose_i[:3, 3]
        tj = pose_j[:3, 3]
        s_ij = scale_j / scale_i
        R_ij = Ri.T @ Rj
        t_ij = (1.0 / scale_i) * (Ri.T @ (tj - ti))
        r_err = self.measurement_r.T @ R_ij
        rvec_err, _ = cv2.Rodrigues(r_err)
        t_err = t_ij - self.measurement_t
        s_err = np.log(s_ij) - np.log(self.measurement_s)
        return np.hstack([rvec_err.ravel(), t_err.ravel(), s_err])

    def jacobians(self, xi: np.ndarray, xj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jac_i = _numeric_jacobian(lambda v: self.residual(v, xj), xi, self.epsilon)
        jac_j = _numeric_jacobian(lambda v: self.residual(xi, v), xj, self.epsilon)
        return jac_i, jac_j


def _relative_se2(xi: np.ndarray, xj: np.ndarray) -> tuple[float, float]:
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]
    theta = xi[2]
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rel_x = cos_t * dx + sin_t * dy
    rel_y = -sin_t * dx + cos_t * dy
    return rel_x, rel_y


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _numeric_jacobian(func, vec: np.ndarray, epsilon: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    base = func(vec)
    jac = np.zeros((base.size, vec.size), dtype=float)
    for idx in range(vec.size):
        step = np.zeros_like(vec)
        step[idx] = epsilon
        forward = func(vec + step)
        backward = func(vec - step)
        jac[:, idx] = (forward - backward) / (2 * epsilon)
    return jac


def _vec_to_se3(vec: np.ndarray) -> np.ndarray:
    rvec = vec[:3]
    tvec = vec[3:6]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def _vec_to_sim3(vec: np.ndarray) -> tuple[np.ndarray, float]:
    rvec = vec[:3]
    tvec = vec[3:6]
    log_s = vec[6]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T, float(np.exp(log_s))


__all__ = [
    "BinaryFactor",
    "FactorGraph",
    "FactorGraphConfig",
    "SE2BetweenFactor",
    "SE3BetweenFactor",
    "Sim3BetweenFactor",
]
