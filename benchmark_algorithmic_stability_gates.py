"""Benchmark algorithmic stability gates for conditioning checks."""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc

import numpy as np

from bundle_adjustment import BundleAdjustmentConfig, Observation, run_bundle_adjustment
from pose_graph import PoseGraph, SolverConfig

LOGGER = logging.getLogger(__name__)


def _project_point(point: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    cam_pose = np.linalg.inv(pose)
    proj = intrinsics @ (cam_pose[:3, :3] @ point + cam_pose[:3, 3])
    return proj[:2] / proj[2]


def _build_bundle_adjustment_case(num_points: int) -> tuple[list[np.ndarray], np.ndarray, list[Observation], np.ndarray]:
    intrinsics = np.eye(3)
    poses = [np.eye(4), np.eye(4)]
    poses[1][:3, 3] = np.array([0.1, 0.0, 0.0])
    rng = np.random.default_rng(42)
    points_3d = rng.uniform(-0.3, 0.3, size=(num_points, 3)) + np.array([0.0, 0.0, 3.0])
    observations: list[Observation] = []
    for idx, point in enumerate(points_3d):
        uv0 = _project_point(point, poses[0], intrinsics)
        uv1 = _project_point(point, poses[1], intrinsics)
        observations.append(Observation(frame_index=0, point_index=idx, uv=uv0))
        observations.append(Observation(frame_index=1, point_index=idx, uv=uv1))
    return poses, points_3d, observations, intrinsics


def _benchmark_bundle_adjustment(num_points: int) -> tuple[float, int]:
    poses, points_3d, observations, intrinsics = _build_bundle_adjustment_case(num_points)
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    start = time.perf_counter()
    _ = run_bundle_adjustment(
        poses=poses,
        points_3d=points_3d,
        observations=observations,
        intrinsics=intrinsics,
        max_nfev=25,
        config=BundleAdjustmentConfig(),
    )
    elapsed = time.perf_counter() - start
    end_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return elapsed, end_mem - start_mem


def _benchmark_pose_graph(num_nodes: int) -> tuple[float, int]:
    rng = np.random.default_rng(7)
    graph = PoseGraph(
        solver_name="gauss_newton",
        solver_config=SolverConfig(max_iterations=10, max_condition_number=1e9, min_diagonal=1e-12),
    )
    for _ in range(1, num_nodes):
        theta = rng.normal(0, 0.05)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = rng.normal(0, 0.1, size=2)
        graph.add_pose(R, t)
    for _ in range(max(num_nodes // 4, 1)):
        i = int(rng.integers(0, num_nodes - 1))
        j = int(rng.integers(i + 1, num_nodes))
        graph.add_loop(i, j, np.eye(2), np.zeros(2), weight=0.8)
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    start = time.perf_counter()
    _ = graph.optimize()
    elapsed = time.perf_counter() - start
    end_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return elapsed, end_mem - start_mem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=120)
    parser.add_argument("--nodes", type=int, default=80)
    args = parser.parse_args()

    ba_elapsed, ba_mem = _benchmark_bundle_adjustment(args.points)
    pg_elapsed, pg_mem = _benchmark_pose_graph(args.nodes)

    print(f"bundle_adjustment_points={args.points} elapsed_s={ba_elapsed:.4f} mem_delta_bytes={ba_mem}")
    print(f"pose_graph_nodes={args.nodes} elapsed_s={pg_elapsed:.4f} mem_delta_bytes={pg_mem}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
