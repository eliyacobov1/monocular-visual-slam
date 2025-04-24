"""
freespace_grid.py
=================
Build a 2-D freespace / occupancy grid from a metric depth image.

Assumptions
-----------
* Depth image is a floating-point array in **metres** (shape H×W).
* Camera intrinsics are known (fx, fy, cx, cy).
* Camera extrinsics w.r.t the ground plane are known (4×4 homogeneous T_cam→veh).
  – If the camera is rigidly fixed on a vehicle, extrinsics are usually
    constant and can be read from calibration files.
* +Z forward, +X right, +Y down in camera space; +X right, +Y forward in
  vehicle/world space (common AD/ROS automotive convention).

Grid conventions
----------------
Each grid cell stores one of three states:
    FREE  = 0   (ground within `ground_tol` m of nominal plane)
    OCC   = 100 (obstacle higher than `obstacle_min_z` m above ground)
    UNK   = -1  (no returns / out of FOV)

© 2025 Eli Yacobov.  Licence: MIT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # library style – let the app configure

# ────────────────────────────────────────────────────────────────────────────── #
# Configuration dataclasses
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics for a single monocular camera."""
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Parameters that define the occupancy grid geometry."""
    x_limits_m: Tuple[float, float]          # (min_x, max_x) in vehicle frame
    y_limits_m: Tuple[float, float]          # (min_y, max_y) forward range
    resolution_m: float                      # cell size in metres

    # Height / freespace heuristics
    obstacle_min_z: float = 0.30             # ≥ this height → occupied
    ground_tol: float = 0.20                 # |z| ≤ tol → free

    # Derived sizes (filled at __post_init__)
    size: Tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", (
            int(np.ceil((self.y_limits_m[1] - self.y_limits_m[0]) / self.resolution_m)),
            int(np.ceil((self.x_limits_m[1] - self.x_limits_m[0]) / self.resolution_m)),
        ))


# ────────────────────────────────────────────────────────────────────────────── #
# Core algorithm
# ────────────────────────────────────────────────────────────────────────────── #

FREE: int = 0
OCC:  int = 100
UNK:  int = -1


def depth_to_points(
    depth: np.ndarray,
    intr: CameraIntrinsics,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a pinhole depth map to a 3-D point-cloud in camera space.

    Parameters
    ----------
    depth : (H, W) float32/64
        Depth in metres.
    intr  : CameraIntrinsics
    mask  : optional (H, W) boolean
        True where the depth value should be used (e.g. remove sky/invalid).

    Returns
    -------
    pts_cam : (N, 3) float32
    """
    if mask is None:
        mask = np.isfinite(depth)

    v, u = np.nonzero(mask)          # row = v (y), col = u (x)
    z = depth[v, u].astype(np.float32)
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy   # camera down = positive y
    pts_cam = np.stack((x, y, z), axis=1)
    return pts_cam


def cam_to_vehicle(pts_cam: np.ndarray, T_cv: np.ndarray) -> np.ndarray:
    """
    Transform camera-frame points to vehicle/world frame.

    T_cv : 4×4 float64 homogeneous transform (vehicle ← camera).
    """
    # Homogeneous coords
    pts_h = np.empty((pts_cam.shape[0], 4), dtype=np.float32)
    pts_h[:, :3] = pts_cam
    pts_h[:, 3] = 1.0
    pts_veh = (T_cv @ pts_h.T).T[:, :3]
    return pts_veh


def build_freespace_grid(
    depth: np.ndarray,
    intr: CameraIntrinsics,
    T_cam_to_vehicle: np.ndarray,
    cfg: GridConfig,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a depth image into a freespace / occupancy grid.

    Returns
    -------
    grid : (rows, cols) int8
        UNK (-1), FREE (0) or OCC (100).
        rows index Y (forward), cols index X (right).
    """
    pts_cam = depth_to_points(depth, intr, valid_mask)
    pts_veh = cam_to_vehicle(pts_cam, T_cam_to_vehicle)

    # Filter points that fall inside grid bounds
    x, y, z = pts_veh.T
    in_bounds = (
        (x >= cfg.x_limits_m[0]) & (x < cfg.x_limits_m[1]) &
        (y >= cfg.y_limits_m[0]) & (y < cfg.y_limits_m[1])
    )
    x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]

    # Allocate grid initialised to UNK
    grid = np.full(cfg.size, UNK, dtype=np.int8)

    if x.size == 0:
        logger.warning("No points inside occupancy grid bounds.")
        return grid

    # Grid indices
    col = ((x - cfg.x_limits_m[0]) / cfg.resolution_m).astype(int)
    row = ((y - cfg.y_limits_m[0]) / cfg.resolution_m).astype(int)

    # For each unique (row, col), record the *lowest* z  (closest to ground)
    flat_idx = row * cfg.size[1] + col
    order = np.argsort(flat_idx)            # stable
    flat_idx_sorted = flat_idx[order]
    z_sorted = z[order]

    # Pick first occurrence of each cell (lowest z after sort)
    _, first_idx = np.unique(flat_idx_sorted, return_index=True)
    sel = order[first_idx]
    z_cell = z[sel]
    row_cell = row[sel]
    col_cell = col[sel]

    # Classify
    free_mask = np.abs(z_cell) <= cfg.ground_tol
    occ_mask  = z_cell >= cfg.obstacle_min_z

    grid[row_cell[free_mask], col_cell[free_mask]] = FREE
    grid[row_cell[occ_mask],  col_cell[occ_mask]]  = OCC

    # Optional: fill unknown holes by local expansion (comment out if not needed)
    # -----------------------------------------------------------------
    # from scipy.ndimage import grey_dilation
    # unk_mask = (grid == UNK)
    # grid[unk_mask & grey_dilation(grid == FREE, size=3)] = FREE
    # grid[unk_mask & grey_dilation(grid == OCC,  size=3)] = OCC
    # -----------------------------------------------------------------

    return grid


# ────────────────────────────────────────────────────────────────────────────── #
# Example usage (can be promoted to a pytest later)
# ────────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    H, W = 480, 640
    fake_depth = np.full((H, W), 10.0, dtype=np.float32)  # 10 m everywhere
    fake_depth[H // 2 :, :] = 5.0                        # nearer lower half

    intr = CameraIntrinsics(
        fx=800.0, fy=800.0,
        cx=W / 2.0,
        cy=H / 2.0,
    )

    # Identity extrinsic: camera already aligned with vehicle frame for demo
    T = np.eye(4, dtype=np.float64)

    cfg = GridConfig(
        x_limits_m=(-5.0, 5.0),   # 10 m wide
        y_limits_m=(0.0, 20.0),   # 20 m ahead
        resolution_m=0.25,
    )

    occ_grid = build_freespace_grid(fake_depth, intr, T, cfg)
    logger.info("Grid shape = %s, free=%d, occ=%d, unk=%d",
                occ_grid.shape,
                np.count_nonzero(occ_grid == FREE),
                np.count_nonzero(occ_grid == OCC),
                np.count_nonzero(occ_grid == UNK))