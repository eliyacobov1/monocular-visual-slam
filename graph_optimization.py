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
    linear_solver_max_iter: int = 200
    linear_solver_tol: float = 1e-8
    max_condition_number: float = 1e8
    min_diagonal: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be positive")
        if self.damping < 0:
            raise ValueError("damping must be non-negative")
        if self.step_scale <= 0:
            raise ValueError("step_scale must be positive")
        if self.linear_solver_max_iter <= 0:
            raise ValueError("linear_solver_max_iter must be positive")
        if self.linear_solver_tol <= 0:
            raise ValueError("linear_solver_tol must be positive")
        if self.max_condition_number <= 0:
            raise ValueError("max_condition_number must be positive")
        if self.min_diagonal <= 0:
            raise ValueError("min_diagonal must be positive")


@dataclass(frozen=True)
class IterationDiagnostics:
    """Per-iteration diagnostics for optimization."""

    iteration: int
    residual_norm: float
    step_norm: float
    linear_solver_iterations: int
    linear_solver_residual: float
    damping: float


@dataclass(frozen=True)
class SolverDiagnostics:
    """Aggregated diagnostics for a solver run."""

    iterations: tuple[IterationDiagnostics, ...]
    status: str


@dataclass(frozen=True)
class SolverResult:
    """Result summary from a pose-graph optimization."""

    success: bool
    status: int
    cost: float
    residual_norm: float
    iterations: int
    message: str
    diagnostics: SolverDiagnostics | None = None


@dataclass(frozen=True)
class ConditioningDiagnostics:
    """Diagnostics for normal-equation conditioning."""

    condition_number: float
    min_diagonal: float
    max_diagonal: float
    status: str
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
    j: int | None
    residual: np.ndarray
    jacobian_i: np.ndarray
    jacobian_j: np.ndarray | None
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


def _robust_weight(residual: np.ndarray, loss: RobustLossConfig) -> float:
    if residual.size == 0:
        return 1.0
    z = np.array([float(residual @ residual)])
    rho = _robust_loss_rho(z, loss)
    weight = float(rho[1][0])
    return max(weight, 1e-12)


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
            diagnostics=None,
        )
        return result.x, summary


class BlockSparseMatrix:
    """Deterministic block-sparse matrix with row/column indexing."""

    def __init__(self, block_size: int, num_blocks: int) -> None:
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._blocks: dict[tuple[int, int], np.ndarray] = {}

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def add_block(self, i: int, j: int, block: np.ndarray) -> None:
        if block.shape != (self._block_size, self._block_size):
            raise ValueError("block has incorrect shape")
        key = (i, j)
        if key in self._blocks:
            self._blocks[key] += block
        else:
            self._blocks[key] = block.copy()

    def add_to_diagonal(self, value: float) -> None:
        if value == 0:
            return
        diag = np.eye(self._block_size) * value
        for idx in range(self._num_blocks):
            self.add_block(idx, idx, diag)

    def diagonal_blocks(self) -> dict[int, np.ndarray]:
        diag: dict[int, np.ndarray] = {}
        for (i, j), block in self._blocks.items():
            if i == j:
                diag[i] = block
        return diag

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        size = self._block_size * self._num_blocks
        if vec.size != size:
            raise ValueError("vector size mismatch")
        result = np.zeros(size, dtype=float)
        for i, j in sorted(self._blocks.keys()):
            block = self._blocks[(i, j)]
            row = slice(i * self._block_size, (i + 1) * self._block_size)
            col = slice(j * self._block_size, (j + 1) * self._block_size)
            result[row] += block @ vec[col]
        return result

    def to_dense(self) -> np.ndarray:
        size = self._block_size * self._num_blocks
        dense = np.zeros((size, size), dtype=float)
        for (i, j), block in self._blocks.items():
            row = slice(i * self._block_size, (i + 1) * self._block_size)
            col = slice(j * self._block_size, (j + 1) * self._block_size)
            dense[row, col] += block
        return dense


class BlockSparseNormalEquation:
    """Sparse block normal equation accumulator."""

    def __init__(self, block_size: int, num_blocks: int) -> None:
        self._matrix = BlockSparseMatrix(block_size, num_blocks)
        self._rhs = np.zeros(block_size * num_blocks, dtype=float)

    @property
    def matrix(self) -> BlockSparseMatrix:
        return self._matrix

    @property
    def rhs(self) -> np.ndarray:
        return self._rhs

    def add_block(self, i: int, j: int, block: np.ndarray) -> None:
        self._matrix.add_block(i, j, block)

    def add_rhs(self, i: int, rhs: np.ndarray) -> None:
        block_size = self._matrix.block_size
        if rhs.shape != (block_size,):
            raise ValueError("rhs block has incorrect shape")
        row = slice(i * block_size, (i + 1) * block_size)
        self._rhs[row] += rhs


def compute_conditioning_diagnostics(
    linearized: Iterable[LinearizedResidual],
    *,
    block_size: int,
    num_blocks: int,
    loss_config: RobustLossConfig,
    damping: float,
) -> ConditioningDiagnostics:
    """Compute conditioning diagnostics for a normal equation."""

    normal = BlockSparseNormalEquation(block_size, num_blocks)
    for block in linearized:
        robust_weight = _robust_weight(block.residual, loss_config)
        weight = float(block.weight) * robust_weight
        sqrt_w = np.sqrt(weight)
        jac_i = sqrt_w * block.jacobian_i
        normal.add_block(block.i, block.i, jac_i.T @ jac_i)
        if block.j is None or block.jacobian_j is None:
            continue
        jac_j = sqrt_w * block.jacobian_j
        normal.add_block(block.j, block.j, jac_j.T @ jac_j)
        normal.add_block(block.i, block.j, jac_i.T @ jac_j)
        normal.add_block(block.j, block.i, jac_j.T @ jac_i)
    normal.matrix.add_to_diagonal(damping)
    dense = normal.matrix.to_dense()
    if dense.size == 0:
        return ConditioningDiagnostics(
            condition_number=float("inf"),
            min_diagonal=0.0,
            max_diagonal=0.0,
            status="invalid",
            message="Normal equation is empty",
        )
    diag = np.diag(dense)
    min_diagonal = float(np.min(diag)) if diag.size else 0.0
    max_diagonal = float(np.max(diag)) if diag.size else 0.0
    try:
        condition_number = float(np.linalg.cond(dense))
    except np.linalg.LinAlgError:
        condition_number = float("inf")
    status = "ok"
    message = "Conditioning check completed"
    if not np.isfinite(condition_number):
        status = "invalid"
        message = "Conditioning check failed: non-finite condition number"
    return ConditioningDiagnostics(
        condition_number=condition_number,
        min_diagonal=min_diagonal,
        max_diagonal=max_diagonal,
        status=status,
        message=message,
    )


@dataclass(frozen=True)
class ConjugateGradientResult:
    """Result for conjugate-gradient solves."""

    solution: np.ndarray
    iterations: int
    residual_norm: float
    converged: bool


class BlockDiagonalPreconditioner:
    """Block-diagonal preconditioner for sparse linear systems."""

    def __init__(self, matrix: BlockSparseMatrix, jitter: float = 1e-9) -> None:
        self._block_size = matrix.block_size
        self._num_blocks = matrix.num_blocks
        self._inverse_blocks: dict[int, np.ndarray] = {}
        for idx, block in matrix.diagonal_blocks().items():
            adjusted = block + np.eye(self._block_size) * jitter
            try:
                self._inverse_blocks[idx] = np.linalg.inv(adjusted)
            except np.linalg.LinAlgError:
                self._inverse_blocks[idx] = np.linalg.pinv(adjusted)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        if vec.size != self._block_size * self._num_blocks:
            raise ValueError("vector size mismatch for preconditioner")
        result = np.zeros_like(vec)
        for idx in range(self._num_blocks):
            row = slice(idx * self._block_size, (idx + 1) * self._block_size)
            inv_block = self._inverse_blocks.get(idx)
            if inv_block is None:
                result[row] = vec[row]
            else:
                result[row] = inv_block @ vec[row]
        return result


class ConjugateGradientSolver:
    """Deterministic conjugate-gradient solver for SPD systems."""

    def solve(
        self,
        matrix: BlockSparseMatrix,
        rhs: np.ndarray,
        *,
        max_iter: int,
        tol: float,
        preconditioner: BlockDiagonalPreconditioner | None = None,
    ) -> ConjugateGradientResult:
        size = matrix.block_size * matrix.num_blocks
        if rhs.size != size:
            raise ValueError("rhs size mismatch")
        x = np.zeros(size, dtype=float)
        r = rhs - matrix.matvec(x)
        if preconditioner is None:
            z = r.copy()
        else:
            z = preconditioner.apply(r)
        p = z.copy()
        rz_old = float(r @ z)
        residual_norm = float(np.linalg.norm(r))
        if residual_norm <= tol:
            return ConjugateGradientResult(x, 0, residual_norm, True)
        converged = False
        iterations = 0
        for iterations in range(1, max_iter + 1):
            ap = matrix.matvec(p)
            denom = float(p @ ap)
            if denom == 0:
                break
            alpha = rz_old / denom
            x += alpha * p
            r -= alpha * ap
            residual_norm = float(np.linalg.norm(r))
            if residual_norm <= tol:
                converged = True
                break
            if preconditioner is None:
                z = r.copy()
            else:
                z = preconditioner.apply(r)
            rz_new = float(r @ z)
            if rz_old == 0:
                break
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new
        return ConjugateGradientResult(x, iterations, residual_norm, converged)


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
        diagnostics: list[IterationDiagnostics] = []
        iterations = 0
        status = 1
        message = "Gauss-Newton completed"
        for iterations in range(1, solver_config.max_iterations + 1):
            linearized = list(problem.linearize_fn(x))
            if not linearized:
                LOGGER.info("No linearized residuals available; exiting early")
                break
            num_blocks = problem.parameter_size // problem.block_size
            normal = BlockSparseNormalEquation(problem.block_size, num_blocks)
            residual_norm = 0.0
            for block in linearized:
                robust_weight = _robust_weight(block.residual, loss_config)
                weight = float(block.weight) * robust_weight
                sqrt_w = np.sqrt(weight)
                weighted_residual = sqrt_w * block.residual
                residual_norm += float(weighted_residual @ weighted_residual)
                jac_i = sqrt_w * block.jacobian_i
                normal.add_block(block.i, block.i, jac_i.T @ jac_i)
                normal.add_rhs(block.i, jac_i.T @ weighted_residual)
                if block.j is None or block.jacobian_j is None:
                    continue
                jac_j = sqrt_w * block.jacobian_j
                normal.add_block(block.j, block.j, jac_j.T @ jac_j)
                normal.add_block(block.i, block.j, jac_i.T @ jac_j)
                normal.add_block(block.j, block.i, jac_j.T @ jac_i)
                normal.add_rhs(block.j, jac_j.T @ weighted_residual)
            normal.matrix.add_to_diagonal(solver_config.damping)
            preconditioner = BlockDiagonalPreconditioner(normal.matrix)
            cg = ConjugateGradientSolver()
            cg_result = cg.solve(
                normal.matrix,
                -normal.rhs,
                max_iter=solver_config.linear_solver_max_iter,
                tol=solver_config.linear_solver_tol,
                preconditioner=preconditioner,
            )
            if not cg_result.converged:
                LOGGER.warning(
                    "Conjugate-gradient did not converge: residual=%.6f",
                    cg_result.residual_norm,
                )
            step = cg_result.solution
            step_norm = float(np.linalg.norm(step))
            diagnostics.append(
                IterationDiagnostics(
                    iteration=iterations,
                    residual_norm=float(np.sqrt(residual_norm)),
                    step_norm=step_norm,
                    linear_solver_iterations=cg_result.iterations,
                    linear_solver_residual=cg_result.residual_norm,
                    damping=solver_config.damping,
                )
            )
            x += solver_config.step_scale * step
            if step_norm < solver_config.xtol:
                message = "Converged (step tolerance)"
                break
            if float(np.sqrt(residual_norm)) < solver_config.ftol:
                message = "Converged (residual tolerance)"
                break
            if not cg_result.converged:
                status = 0
                message = "Linear solver failed to converge"
                break
        residual = problem.residual_fn(x)
        summary = SolverResult(
            success=status == 1,
            status=status,
            cost=float(0.5 * np.dot(residual, residual)),
            residual_norm=float(np.linalg.norm(residual)),
            iterations=iterations,
            message=message,
            diagnostics=SolverDiagnostics(iterations=tuple(diagnostics), status=message),
        )
        return x, summary


_SOLVER_REGISTRY.register("scipy", ScipyLeastSquaresSolver())
_SOLVER_REGISTRY.register("gauss_newton", GaussNewtonSolver())


__all__ = [
    "BlockDiagonalPreconditioner",
    "BlockSparseMatrix",
    "BlockSparseNormalEquation",
    "ConditioningDiagnostics",
    "ConjugateGradientResult",
    "ConjugateGradientSolver",
    "GaussNewtonSolver",
    "IterationDiagnostics",
    "LinearizedResidual",
    "PoseGraphProblem",
    "PoseGraphSnapshot",
    "PoseGraphSolver",
    "RobustLossConfig",
    "RobustLossType",
    "SolverConfig",
    "SolverDiagnostics",
    "SolverRegistry",
    "SolverResult",
    "compute_conditioning_diagnostics",
    "get_solver_registry",
]
