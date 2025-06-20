#!/usr/bin/env python3
"""Compute ATE and RPE between ground truth and estimated poses."""

import argparse
from typing import Iterable, Sequence

import numpy as np


def load_traj(path: str, cols: Sequence[int] = (0, 1)) -> np.ndarray:
    """Load whitespace separated trajectory file.

    Parameters
    ----------
    path:
        File containing a whitespace separated trajectory.
    cols:
        Zero-based column indices storing the ``x`` and ``y`` coordinates.
    """
    data: list[list[float]] = []
    with open(path) as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            if max(cols) >= len(parts):
                continue
            data.append([float(parts[c]) for c in cols])
    return np.array(data, dtype=float)


def align_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return similarity-transform parameters aligning ``src`` to ``dst``."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / len(src)

    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt

    var_src = (src_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(S)) / var_src

    t = mu_dst - scale * (R @ mu_src)
    return R, t, float(scale)


def compute_ate(gt: np.ndarray, est: np.ndarray) -> float:
    """Return root mean square Absolute Trajectory Error."""
    if len(gt) == 0 or len(est) == 0:
        raise ValueError("Trajectories must not be empty")
    min_len = min(len(gt), len(est))
    gt = gt[:min_len]
    est = est[:min_len]
    R, t, s = align_similarity(est, gt)
    est_aligned = (s * (R @ est.T)).T + t
    return float(np.sqrt(np.mean(np.sum((gt - est_aligned) ** 2, axis=1))))


def compute_rpe(gt: np.ndarray, est: np.ndarray, delta: int = 1) -> float:
    """Return root mean square Relative Pose Error."""
    min_len = min(len(gt), len(est)) - delta
    R, t, s = align_similarity(est[: min_len + delta], gt[: min_len + delta])
    est_aligned = (s * (R @ est.T)).T + t
    errors: list[float] = []
    for i in range(min_len):
        gt_rel = gt[i + delta] - gt[i]
        est_rel = est_aligned[i + delta] - est_aligned[i]
        errors.append(np.linalg.norm(gt_rel - est_rel))
    return float(np.sqrt(np.mean(np.square(errors))))


def compute_additional_metrics(
    gt: np.ndarray, est: np.ndarray, delta: int = 1
) -> dict:
    """Return a dictionary with ATE and RPE statistics."""
    if len(gt) == 0 or len(est) == 0:
        raise ValueError("Trajectories must not be empty")

    min_len = min(len(gt), len(est))
    gt = gt[:min_len]
    est = est[:min_len]
    R, t, s = align_similarity(est, gt)
    est_aligned = (s * (R @ est.T)).T + t
    diff = gt - est_aligned
    dists = np.linalg.norm(diff, axis=1)
    ate_rmse = float(np.sqrt(np.mean(dists ** 2)))
    ate_mean = float(np.mean(dists))
    ate_median = float(np.median(dists))

    rpe_errors = []
    for i in range(min_len - delta):
        gt_rel = gt[i + delta] - gt[i]
        est_rel = est_aligned[i + delta] - est_aligned[i]
        rpe_errors.append(np.linalg.norm(gt_rel - est_rel))
    rpe_errors = np.array(rpe_errors)
    rpe_rmse = float(np.sqrt(np.mean(rpe_errors ** 2)))
    rpe_mean = float(np.mean(rpe_errors))
    rpe_median = float(np.median(rpe_errors))

    return {
        "ATE_RMSE": ate_rmse,
        "ATE_MEAN": ate_mean,
        "ATE_MEDIAN": ate_median,
        "RPE_RMSE": rpe_rmse,
        "RPE_MEAN": rpe_mean,
        "RPE_MEDIAN": rpe_median,
    }


def write_report(path: str, lines: Iterable[str]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory ATE/RPE")
    parser.add_argument("--gt", required=True, help="Path to ground truth txt")
    parser.add_argument("--est", required=True, help="Path to estimated poses")
    parser.add_argument(
        "--cols",
        default="0,1",
        help="Comma separated columns for x,y in both files",
    )
    parser.add_argument(
        "--est_cols",
        help="Optional columns for the estimated trajectory if different",
        default=None,
    )
    parser.add_argument(
        "--rpe_delta", type=int, default=1, help="Frame distance for RPE"
    )
    parser.add_argument(
        "--report",
        help="Optional path to save the error metrics",
        default=None,
    )
    args = parser.parse_args()

    col_idx = [int(c) for c in args.cols.split(",")]
    gt = load_traj(args.gt, col_idx)
    est_cols = col_idx if args.est_cols is None else [int(c) for c in args.est_cols.split(",")]
    est = load_traj(args.est, est_cols)

    metrics = compute_additional_metrics(gt, est, delta=args.rpe_delta)

    lines = [f"{k} {v:.4f}" for k, v in metrics.items()]
    for line in lines:
        print(line)
    if args.report:
        write_report(args.report, lines)
