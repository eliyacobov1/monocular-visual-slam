"""Graph-optimization primitives with pluggable solvers and robust losses."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Iterable, Protocol

import numpy as np
from scipy.optimize import least_squares

LOGGER = logging.getLogger(__name__)


class RobustLossType(str, Enum):
    HUBER = "huber"
    CAUCHY = "cauchy"
    TUKEY = "tukey"


@dataclass(frozen=True)
class RobustLossConfig:
    """Robust loss configuration for solver weighting."""

    loss_type: RobustLossType = RobustLossType.HUBER
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("loss scale must be positive")


@dataclass(frozen=True)
class SolverConfig:
    """Solver configuration with deterministic tolerances."""

    max_iterations: int = 20
    max_nfev: int = 200
    damping: float = 1e-3
    step_scale: float = 1.0
    xtol: float = 1e-10
    ftol: float = 1e-10
    gtol: float = 1e-10

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be positive")
        if self.damping < 0:
            raise ValueError("damping must be non-negative")
        if self.step_scale <= 0:
            raise ValueError("step_scale must be positive")


@dataclass(frozen=True)
class SolverResult:
    """Result summary from a pose-graph optimization."""

    success: bool
    status: int
    cost: float
    residual_norm: float
    iterations: int
    message: str


@dataclass(frozen=True)
class PoseGraphSnapshot:
    """Deterministic snapshot for regression gating and CI artifacts."""

    version: int
    solver_name: str
    loss_config: RobustLossConfig
    solver_config: SolverConfig
    poses: list[list[float]]
    edges: list[dict[str, object]]
    metadata: dict[str, object] = field(default_factory=dict)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PoseGraphProblem:
    """Pose graph optimization problem wrapper."""

    residual_fn: Callable[[np.ndarray], np.ndarray]
    linearize_fn: Callable[[np.ndarray], Iterable["LinearizedResidual"]]
    parameter_size: int
    block_size: int
    snapshot: PoseGraphSnapshot


@dataclass(frozen=True)
class LinearizedResidual:
    """Linearized residual block for sparse Gauss-Newton."""

    i: int
    j: int
    residual: np.ndarray
    jacobian_i: np.ndarray
    jacobian_j: np.ndarray
    weight: float


class PoseGraphSolver(Protocol):
    """Protocol for pose-graph solvers."""

    def solve(
        self,
        problem: PoseGraphProblem,
        x0: np.ndarray,
        solver_config: SolverConfig,
        loss_config: RobustLossConfig,
    ) -> tuple[np.ndarray, SolverResult]:
        ...


class SolverRegistry:
    """Thread-safe registry of pose-graph solvers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._solvers: dict[str, PoseGraphSolver] = {}

    def register(self, name: str, solver: PoseGraphSolver) -> None:
        if not name:
            raise ValueError("solver name must be non-empty")
        with self._lock:
            if name in self._solvers:
                raise ValueError(f"solver '{name}' already registered")
            self._solvers[name] = solver

    def get(self, name: str) -> PoseGraphSolver:
        with self._lock:
            solver = self._solvers.get(name)
        if solver is None:
            raise KeyError(f"solver '{name}' is not registered")
        return solver

    def available(self) -> list[str]:
        with self._lock:
            return sorted(self._solvers.keys())


_SOLVER_REGISTRY = SolverRegistry()


def get_solver_registry() -> SolverRegistry:
    return _SOLVER_REGISTRY


def _robust_loss_rho(z: np.ndarray, loss: RobustLossConfig) -> np.ndarray:
    if loss.loss_type == RobustLossType.HUBER:
        return _huber_rho(z, loss.scale)
    if loss.loss_type == RobustLossType.CAUCHY:
        return _cauchy_rho(z, loss.scale)
    if loss.loss_type == RobustLossType.TUKEY:
        return _tukey_rho(z, loss.scale)
    raise ValueError(f"Unsupported robust loss type {loss.loss_type}")


def _huber_rho(z: np.ndarray, scale: float) -> np.ndarray:
    sqrt_z = np.sqrt(z)
    mask = sqrt_z <= scale
    rho = np.empty((3, z.size), dtype=float)
    rho[0] = np.where(mask, z, 2 * scale * sqrt_z - scale**2)
    rho[1] = np.where(mask, 1.0, scale / np.maximum(sqrt_z, 1e-12))
    rho[2] = np.where(mask, 0.0, -0.5 * scale / np.maximum(z, 1e-12))
    return rho


def _cauchy_rho(z: np.ndarray, scale: float) -> np.ndarray:
    denom = 1.0 + z / (scale**2)
    rho = np.empty((3, z.size), dtype=float)
    rho[0] = (scale**2) * np.log(denom)
    rho[1] = 1.0 / denom
    rho[2] = -1.0 / (scale**2 * denom**2)
    return rho


def _tukey_rho(z: np.ndarray, scale: float) -> np.ndarray:
    sqrt_z = np.sqrt(z)
    mask = sqrt_z <= scale
    rho = np.empty((3, z.size), dtype=float)
    t = 1 - (z / (scale**2))
    rho[0] = np.where(mask, (scale**2 / 6.0) * (1 - t**3), scale**2 / 6.0)
    rho[1] = np.where(mask, t**2, 0.0)
    rho[2] = np.where(mask, -2.0 * t / (scale**2), 0.0)
    return rho


class ScipyLeastSquaresSolver:
    """SciPy-backed least-squares solver with custom robust loss."""

    def solve(
        self,
        problem: PoseGraphProblem,
        x0: np.ndarray,
        solver_config: SolverConfig,
        loss_config: RobustLossConfig,
    ) -> tuple[np.ndarray, SolverResult]:
        def loss_fn(z: np.ndarray) -> np.ndarray:
            return _robust_loss_rho(z, loss_config)

        result = least_squares(
            problem.residual_fn,
            x0,
            loss=loss_fn,
            max_nfev=solver_config.max_nfev,
            xtol=solver_config.xtol,
            ftol=solver_config.ftol,
            gtol=solver_config.gtol,
        )
        cost = float(result.cost)
        residual_norm = float(np.linalg.norm(result.fun))
        summary = SolverResult(
            success=bool(result.success),
            status=int(result.status),
            cost=cost,
            residual_norm=residual_norm,
            iterations=int(result.nfev),
            message=str(result.message),
        )
        return result.x, summary


class BlockSparseNormalEquation:
    """Sparse block normal equation accumulator."""

    def __init__(self, block_size: int, num_blocks: int) -> None:
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._blocks: dict[tuple[int, int], np.ndarray] = {}
        self._rhs: dict[int, np.ndarray] = {}

    def add_block(self, i: int, j: int, block: np.ndarray) -> None:
        key = (i, j)
        if key in self._blocks:
            self._blocks[key] += block
        else:
            self._blocks[key] = block.copy()

    def add_rhs(self, i: int, rhs: np.ndarray) -> None:
        if i in self._rhs:
            self._rhs[i] += rhs
        else:
            self._rhs[i] = rhs.copy()

    def assemble_dense(self) -> tuple[np.ndarray, np.ndarray]:
        size = self._block_size * self._num_blocks
        matrix = np.zeros((size, size), dtype=float)
        rhs = np.zeros(size, dtype=float)
        for (i, j), block in self._blocks.items():
            row = slice(i * self._block_size, (i + 1) * self._block_size)
            col = slice(j * self._block_size, (j + 1) * self._block_size)
            matrix[row, col] += block
        for idx, block_rhs in self._rhs.items():
            row = slice(idx * self._block_size, (idx + 1) * self._block_size)
            rhs[row] += block_rhs
        return matrix, rhs


class GaussNewtonSolver:
    """Sparse Gauss-Newton solver for pose graphs."""

    def solve(
        self,
        problem: PoseGraphProblem,
        x0: np.ndarray,
        solver_config: SolverConfig,
        loss_config: RobustLossConfig,
    ) -> tuple[np.ndarray, SolverResult]:
        x = x0.copy()
        iterations = 0
        for iterations in range(1, solver_config.max_iterations + 1):
            linearized = list(problem.linearize_fn(x))
            if not linearized:
                break
            normal = BlockSparseNormalEquation(problem.block_size, problem.parameter_size // problem.block_size)
            residual_norm = 0.0
            for block in linearized:
                weight = float(block.weight)
                weighted_residual = np.sqrt(weight) * block.residual
                residual_norm += float(weighted_residual @ weighted_residual)
                jac_i = np.sqrt(weight) * block.jacobian_i
                jac_j = np.sqrt(weight) * block.jacobian_j
                normal.add_block(block.i, block.i, jac_i.T @ jac_i)
                normal.add_block(block.j, block.j, jac_j.T @ jac_j)
                normal.add_block(block.i, block.j, jac_i.T @ jac_j)
                normal.add_block(block.j, block.i, jac_j.T @ jac_i)
                normal.add_rhs(block.i, jac_i.T @ weighted_residual)
                normal.add_rhs(block.j, jac_j.T @ weighted_residual)
            matrix, rhs = normal.assemble_dense()
            matrix += np.eye(matrix.shape[0]) * solver_config.damping
            try:
                step = np.linalg.solve(matrix, -rhs)
            except np.linalg.LinAlgError as exc:
                LOGGER.warning("Gauss-Newton failed to solve linear system: %s", exc)
                break
            x += solver_config.step_scale * step
            if np.linalg.norm(step) < solver_config.xtol:
                break
        residual = problem.residual_fn(x)
        summary = SolverResult(
            success=True,
            status=1,
            cost=float(0.5 * np.dot(residual, residual)),
            residual_norm=float(np.linalg.norm(residual)),
            iterations=iterations,
            message="Gauss-Newton completed",
        )
        return x, summary


_SOLVER_REGISTRY.register("scipy", ScipyLeastSquaresSolver())
_SOLVER_REGISTRY.register("gauss_newton", GaussNewtonSolver())


__all__ = [
    "BlockSparseNormalEquation",
    "GaussNewtonSolver",
    "LinearizedResidual",
    "PoseGraphProblem",
    "PoseGraphSnapshot",
    "PoseGraphSolver",
    "RobustLossConfig",
    "RobustLossType",
    "SolverConfig",
    "SolverRegistry",
    "SolverResult",
    "get_solver_registry",
]
