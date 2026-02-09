#!/usr/bin/env python3
"""Unified visualisation interface for the SLAM demo.

This script plays back a video while estimating the camera trajectory.
The left panel shows the current video frame with matched keypoints
between the previous and current image.  The right panel plots the
estimated 2‑D trajectory with the latest camera position highlighted.

Per frame basic diagnostic information is overlayed on the image
including the number of features, matches and the current pose.
"""

from __future__ import annotations

import argparse
import logging
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cam_intrinsics_estimation import make_K, load_K_from_file
from demo_utils import ensure_sample_video, DEFAULT_VIDEO_PATH

LOGGER = logging.getLogger(__name__)


# ------------------------------- helpers -----------------------------------

def rotation_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Return yaw, pitch, roll (Z, Y, X) from a rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[2, 1], R[2, 2])
    else:  # Gimbal lock
        yaw = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees([yaw, pitch, roll])


def apply_axes_limits(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    padding: float = 0.25,
) -> None:
    if xs.size == 0 or ys.size == 0:
        return
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_pad = max((x_max - x_min) * padding, 0.5)
    y_pad = max((y_max - y_min) * padding, 0.5)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


# ------------------------------- main loop ---------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Side‑by‑side SLAM viewer")
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO_PATH),
        help="Path to input video (defaults to sample clip)",
    )
    parser.add_argument(
        "--intrinsics_file", help="Optional intrinsics file with fx fy cx cy"
    )
    parser.add_argument(
        "--step", action="store_true", help="Advance frames on key press"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        if args.video == str(DEFAULT_VIDEO_PATH):
            video_path = ensure_sample_video(video_path)
        else:
            raise SystemExit(f"Could not find video {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit("Video is empty")

    h, w = first_frame.shape[:2]
    if args.intrinsics_file:
        K = load_K_from_file(args.intrinsics_file)
    else:
        K = make_K(w, h)

    orb = cv2.ORB_create(1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    plt.ion()
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax_img = fig.add_subplot(1, 2, 1)
    ax_traj = fig.add_subplot(1, 2, 2)
    ax_traj.set_title("Estimated trajectory")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Z")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, linestyle="--", alpha=0.4)

    poses = [np.eye(4)]
    positions = [np.zeros(3)]
    path_line, = ax_traj.plot([], [], "b-")
    curr_point = ax_traj.scatter([], [], c="r")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)

    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)

            matches = []
            if prev_desc is not None and desc is not None and len(prev_desc) > 0 and len(desc) > 0:
                matches = matcher.match(prev_desc, desc)
                matches = sorted(matches, key=lambda m: m.distance)

            pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            R = np.eye(3)
            t = np.zeros((3, 1))
            inlier_mask = None
            inlier_ratio = 0.0
            if len(matches) >= 8:
                try:
                    E, inlier_mask = cv2.findEssentialMat(
                        pts_prev, pts_curr, K, method=cv2.RANSAC, threshold=1.0
                    )
                    if E is not None:
                        _, R, t, inlier_mask = cv2.recoverPose(
                            E, pts_prev, pts_curr, K, mask=inlier_mask
                        )
                        inlier_mask = inlier_mask.ravel().astype(bool)
                        inlier_ratio = float(inlier_mask.sum()) / len(matches)
                    else:
                        inlier_mask = np.zeros(len(matches), dtype=bool)
                except cv2.error as exc:
                    LOGGER.warning(
                        "Essential matrix estimation failed; keeping identity pose.",
                        exc_info=exc,
                    )
                    inlier_mask = np.zeros(len(matches), dtype=bool)
                    R = np.eye(3)
                    t = np.zeros((3, 1))
            else:
                inlier_mask = np.zeros(len(matches), dtype=bool)

            # Update global pose
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            new_pose = poses[-1] @ T
            poses.append(new_pose)
            pos = new_pose[:3, 3]
            positions.append(pos)

            traj = np.array(positions)
            path_line.set_data(traj[:, 0], traj[:, 2])
            curr_point.remove()
            curr_point = ax_traj.scatter(traj[-1, 0], traj[-1, 2], c="r")
            apply_axes_limits(ax_traj, traj[:, 0], traj[:, 2])

            # Draw matches and inliers on the frame
            display = frame.copy()
            for idx, m in enumerate(matches):
                pt_prev = tuple(map(int, prev_kp[m.queryIdx].pt))
                pt_curr = tuple(map(int, kp[m.trainIdx].pt))
                color = (0, 255, 0) if inlier_mask[idx] else (0, 0, 255)
                cv2.line(display, pt_prev, pt_curr, color, 1)
                cv2.circle(display, pt_curr, 2, color, -1)

            yaw, pitch, roll = rotation_to_euler_xyz(R)
            text = (
                f"Frame: {frame_id}\n"
                f"Features: {len(kp)}\n"
                f"Matches: {len(matches)}\n"
                f"Inlier ratio: {inlier_ratio:.2f}\n"
                f"Pos: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}\n"
                f"Yaw/Pitch/Roll: {yaw:.1f}, {pitch:.1f}, {roll:.1f}"
            )

            ax_img.clear()
            ax_img.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            ax_img.set_title("Frame and matches")
            ax_img.axis("off")
            ax_img.text(
                0.02,
                0.98,
                text,
                color="yellow",
                fontsize=8,
                va="top",
                transform=ax_img.transAxes,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            if args.step:
                plt.waitforbuttonpress()
            else:
                plt.pause(0.001)

            prev_gray, prev_kp, prev_desc = gray, kp, desc
    finally:
        cap.release()
        plt.ioff()

    plt.show()


if __name__ == "__main__":
    main()
