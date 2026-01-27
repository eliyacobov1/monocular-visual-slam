#!/usr/bin/env python3.12

import argparse
import logging
import os
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam_path_estimator import VehiclePathLiveAnimator
from loop_closure import BoWDatabase
from pose_graph import PoseGraph3D, PoseGraphSim3D
from homography import (
    estimate_homography_from_orb,
    estimate_homography_from_orb_with_inliers,
    estimate_pose_from_matches,
)
from keyframe_manager import KeyframeManager
from demo_utils import ensure_sample_video, DEFAULT_VIDEO_PATH
from evaluate_trajectory import compute_additional_metrics
from cam_intrinsics_estimation import make_K, load_K_from_file
from feature_pipeline import (
    FeaturePipelineConfig,
    adaptive_ransac_threshold,
    build_feature_pipeline,
    matches_to_points,
)


def estimate_pose_optical_flow(prev_img: np.ndarray, curr_img: np.ndarray,
                               prev_kp: list[cv2.KeyPoint], K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate relative pose via PnP using tracked keypoints."""
    if len(prev_kp) < 8:
        raise RuntimeError("too few keypoints for optical flow")

    prev_pts = np.float32([k.pt for k in prev_kp]).reshape(-1, 1, 2)
    curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
    st = st.reshape(-1)
    good_prev = prev_pts[st == 1].reshape(-1, 2)
    good_curr = curr_pts[st == 1].reshape(-1, 2)

    if len(good_prev) < 8:
        raise RuntimeError("optical flow tracking failed")

    E, mask = cv2.findEssentialMat(good_prev, good_curr, K, method=cv2.RANSAC, threshold=1.0)
    if E is None:
        raise RuntimeError("findEssentialMat failed")
    _, R, t, _ = cv2.recoverPose(E, good_prev, good_curr, K, mask=mask)
    return R, t.ravel()


def _init_logging(level: str) -> None:
    """Initialise logging with the given level name."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s::cv2_e2e - %(message)s",
    )


# SAMPLE_VIDEO_PATH = "4644521-uhd_2562_1440_30fps.mp4"
# The repository previously relied on a small clip checked into source
# control.  Instead we share the test video used in ``tests/test_real_video``
# so that the demo matches the automated tests and does not depend on large
# binary blobs.  The helper in :mod:`demo_utils` downloads the clip on demand
# if it is missing.
SAMPLE_VIDEO_PATH = str(DEFAULT_VIDEO_PATH)


FeatureDetector_t = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def compute_moments(patch):
    h, w = patch.shape
    M00 = np.sum(patch)

    if M00 == 0:
        return 0, 0, 0

    y_indices, x_indices = np.indices((h, w))

    M10 = np.sum(x_indices * patch)
    M01 = np.sum(y_indices * patch)

    cx = M10 / M00
    cy = M01 / M00

    return M00, cx, cy


def orb_detect(
    img: np.ndarray, detector: FeatureDetector_t, top_significant_kp=1000
) -> np.ndarray:
    keypoints, descriptors = detector(img)
    logging.debug(f"Number of keypoints found: {len(keypoints)}")
    kp_response_sort_indices = np.argsort([-kp.response for kp in keypoints])[
        :top_significant_kp
    ]
    keypoints = [keypoints[i] for i in kp_response_sort_indices]
    descriptors = descriptors[kp_response_sort_indices]
    return keypoints, descriptors


def compute_orientation(img, keypoint, patch_size=31):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    radius = patch_size // 2

    patch = img[
        max(y - radius, 0): y + radius + 1,
        max(x - radius, 0): x + radius + 1,
    ]

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        return 0.0  # not enough space around the point

    # Compute intensity centroid
    _, cx, cy = compute_moments(patch)

    dx = cx - patch_size // 2
    dy = cy - patch_size // 2

    angle = np.arctan2(dy, dx) * 180.0 / np.pi
    return angle


def draw_bbs(img_draw: np.ndarray, results, i):
    bboxes = results.xyxy[i][:, :4].cpu().numpy().astype(int)
    for box in bboxes:
        cv2.rectangle(
            img_draw,
            (box[0], box[1]),
            (box[2], box[3]),
            color=(255, 0, 0),
            thickness=1,
        )


def draw_pred_rect(frames: np.ndarray, results):
    for i, img in enumerate(frames):
        draw_bbs(img, results, i)


def compute_dynamic_mask(
    prev_img: np.ndarray, curr_img: np.ndarray, thresh: int = 25
) -> np.ndarray:
    """Simple absolute difference with a hand-written dilation step."""
    diff = np.abs(prev_img.astype(np.int16) - curr_img.astype(np.int16))
    mask = (diff >= thresh).astype(np.uint8)

    k = 5
    pad = k // 2
    padded = np.pad(mask, pad)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    dilated = windows.max(axis=(2, 3))
    return dilated > 0


def filter_keypoints(keypoints, descriptors, mask):
    if mask is None:
        return keypoints, descriptors
    filtered_kp = []
    filtered_desc = []
    for kp, desc in zip(keypoints, descriptors):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if not mask[y, x]:
            filtered_kp.append(kp)
            filtered_desc.append(desc)
    if len(filtered_desc) == 0:
        return [], np.empty((0, descriptors.shape[1]), dtype=descriptors.dtype)
    return filtered_kp, np.vstack(filtered_desc)


def load_video_frames(path, resize=(1080, 1920), max_frames=50):
    """Yield RGB frames from ``path`` resized for processing.

    The previous implementation returned only grayscale images which meant the
    demo could not easily display the video alongside the trajectory plot.  We
    now provide the colour frames so callers can both run the SLAM pipeline and
    visualise the input at the same time.  Consumers can convert the frames to
    grayscale when required.
    """

    cap = cv2.VideoCapture(path)
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        count += 1
        yield frame

    cap.release()


@dataclass(frozen=True)
class SLAMRunConfig:
    max_frames: int = 10000
    sleep_time: float = 0.1
    pause_time: float = 0.001
    semantic_masking: bool = False
    intrinsics_file: Path | None = None
    save_plot: Path | None = None
    save_poses: Path | None = None
    loop_min_matches: int = 30
    loop_min_inliers: int = 20
    loop_min_inlier_ratio: float = 0.3
    loop_ransac_threshold: float = 0.01
    loop_edge_weight: float = 0.5
    use_sim3_loop_correction: bool = False
    loop_scale_min_translation: float = 1e-3
    feature_type: str = "orb"
    feature_nfeatures: int = 2000
    match_ratio: float = 0.8
    match_cross_check: bool = True
    match_max_matches: int | None = 500
    motion_min_matches: int = 15
    motion_ransac_threshold: float = 0.01
    adaptive_ransac: bool = False
    adaptive_ransac_min: float = 0.005
    adaptive_ransac_max: float = 0.03


@dataclass(frozen=True)
class SLAMInput:
    frames: Iterable[np.ndarray]
    intrinsics: np.ndarray
    kitti_gt_positions: np.ndarray | None = None


@dataclass(frozen=True)
class SLAMResult:
    positions: np.ndarray
    metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class KittiConfig:
    base_dir: Path
    date: str
    drive: str
    camera: str = "image_02"
    report_path: Path | None = None


@dataclass(frozen=True)
class KittiRawSession:
    base_dir: Path
    date: str
    drive: str
    camera: str = "image_02"

    @property
    def date_dir(self) -> Path:
        return self.base_dir / self.date

    @property
    def drive_dir(self) -> Path:
        return self.date_dir / f"{self.date}_drive_{self.drive}_sync"

    @property
    def image_dir(self) -> Path:
        return self.drive_dir / self.camera / "data"

    @property
    def oxts_dir(self) -> Path:
        return self.drive_dir / "oxts" / "data"

    @property
    def calib_cam_to_cam(self) -> Path:
        return self.date_dir / "calib_cam_to_cam.txt"


def _normalize_drive_id(drive: str) -> str:
    drive_str = str(drive)
    return drive_str.zfill(4) if drive_str.isdigit() else drive_str


def load_kitti_image_paths(session: KittiRawSession) -> list[Path]:
    if not session.image_dir.exists():
        raise FileNotFoundError(f"Could not find image directory {session.image_dir}")
    image_paths = sorted(session.image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {session.image_dir}")
    return image_paths


def load_kitti_oxts_positions(session: KittiRawSession) -> np.ndarray:
    if not session.oxts_dir.exists():
        raise FileNotFoundError(f"Could not find oxts directory {session.oxts_dir}")
    oxts_files = sorted(session.oxts_dir.glob("*.txt"))
    if not oxts_files:
        raise FileNotFoundError(f"No oxts files found in {session.oxts_dir}")

    lat0 = lon0 = alt0 = None
    positions: list[np.ndarray] = []
    earth_radius = 6378137.0

    for oxts_file in oxts_files:
        with open(oxts_file) as f:
            line = f.readline().strip()
        if not line:
            continue
        parts = line.split()
        lat, lon, alt = float(parts[0]), float(parts[1]), float(parts[2])
        if lat0 is None:
            lat0, lon0, alt0 = lat, lon, alt
        d_lat = np.radians(lat - lat0)
        d_lon = np.radians(lon - lon0)
        x = d_lon * earth_radius * np.cos(np.radians(lat0))
        y = d_lat * earth_radius
        z = alt - alt0
        positions.append(np.array([x, y, z], dtype=float))

    if not positions:
        raise RuntimeError("No valid OXTS entries found")
    return np.vstack(positions)


def load_kitti_intrinsics(session: KittiRawSession) -> np.ndarray:
    calib_path = session.calib_cam_to_cam
    if not calib_path.exists():
        raise FileNotFoundError(f"Could not find calibration file {calib_path}")
    camera_idx = session.camera.split("_")[-1]
    target_key = f"P_rect_{camera_idx}:"
    with open(calib_path) as f:
        for line in f:
            if line.startswith(target_key):
                values = [float(v) for v in line.split()[1:]]
                P = np.array(values).reshape(3, 4)
                fx, fy = P[0, 0], P[1, 1]
                cx, cy = P[0, 2], P[1, 2]
                return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    raise RuntimeError(f"Camera intrinsics not found in {calib_path}")


def load_kitti_frames(
    image_paths: list[Path],
    resize: tuple[int, int] | None = None,
    max_frames: int | None = None,
):
    count = 0
    for path in image_paths:
        if max_frames is not None and count >= max_frames:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        count += 1
        yield frame


def build_kitti_session(config: KittiConfig) -> KittiRawSession:
    drive_id = _normalize_drive_id(config.drive)
    return KittiRawSession(
        base_dir=config.base_dir,
        date=config.date,
        drive=drive_id,
        camera=config.camera,
    )


def prepare_kitti_input(config: KittiConfig, run_config: SLAMRunConfig) -> SLAMInput:
    """Prepare KITTI frames and metadata for a SLAM run."""
    session = build_kitti_session(config)
    image_paths = load_kitti_image_paths(session)
    gt_positions = load_kitti_oxts_positions(session)
    if run_config.intrinsics_file:
        intrinsics = load_K_from_file(str(run_config.intrinsics_file))
    else:
        intrinsics = load_kitti_intrinsics(session)
    frames = load_kitti_frames(image_paths, resize=None, max_frames=run_config.max_frames)
    return SLAMInput(frames=frames, intrinsics=intrinsics, kitti_gt_positions=gt_positions)


def prepare_video_input(
    video_path: Path,
    run_config: SLAMRunConfig,
    intrinsics_file: Path | None = None,
) -> SLAMInput:
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video file {video_path}")
    frames_iter = load_video_frames(str(video_path), max_frames=run_config.max_frames)
    try:
        first_frame = next(frames_iter)
    except StopIteration as exc:
        raise RuntimeError(f"No frames found in {video_path}") from exc
    frames = chain([first_frame], frames_iter)
    gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    if intrinsics_file:
        intrinsics = load_K_from_file(str(intrinsics_file))
    else:
        intrinsics = make_K(gray.shape[1], gray.shape[0])
    return SLAMInput(frames=frames, intrinsics=intrinsics)


def evaluate_kitti_trajectory(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
) -> dict[str, float]:
    min_len = min(len(gt_positions), len(est_positions))
    gt_xy = gt_positions[:min_len, :2]
    est_xy = est_positions[:min_len, :2]
    return compute_additional_metrics(gt_xy, est_xy)


def estimate_loop_scale(
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    t_measured: np.ndarray,
    min_translation: float,
) -> float:
    """Estimate a loop-closure scale factor using current pose estimates."""
    Tij = np.linalg.inv(pose_i) @ pose_j
    t_est = Tij[:3, 3]
    norm_est = float(np.linalg.norm(t_est))
    norm_meas = float(np.linalg.norm(t_measured))
    if norm_est < min_translation or norm_meas < min_translation:
        return 1.0
    return norm_est / norm_meas


def run_visual_slam(slam_input: SLAMInput, run_config: SLAMRunConfig) -> SLAMResult:
    """Run the SLAM pipeline and optionally compute KITTI metrics."""
    path_estimator = VehiclePathLiveAnimator()
    bow_db = BoWDatabase()
    pose_graph = (
        PoseGraphSim3D() if run_config.use_sim3_loop_correction else PoseGraph3D()
    )
    feature_config = FeaturePipelineConfig(
        name=run_config.feature_type,
        nfeatures=run_config.feature_nfeatures,
        ratio_test=run_config.match_ratio,
        cross_check=run_config.match_cross_check,
        max_matches=run_config.match_max_matches,
    )
    feature_pipeline = build_feature_pipeline(feature_config)
    keyframe_manager = KeyframeManager(matcher=feature_pipeline.match)

    frames_iter = iter(slam_input.frames)
    try:
        first_frame = next(frames_iter)
    except StopIteration as exc:
        raise RuntimeError("No frames available for SLAM") from exc
    prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    frame_id = 0

    prev_keypoints, prev_desc = feature_pipeline.detect_and_describe(prev_frame)
    bow_db.add_frame(frame_id, prev_desc)
    frames_data = {frame_id: (prev_frame, prev_keypoints, prev_desc)}
    pose_graph.add_pose(np.eye(3), np.zeros(3))
    keyframe_manager.add_keyframe(frame_id, pose_graph.poses[-1], prev_keypoints, prev_desc)

    for color_frame in frames_iter:
        frame_id += 1
        curr_img = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
        prev_img = prev_frame
        curr_keypoints, curr_desc = feature_pipeline.detect_and_describe(curr_img)
        mask = (
            compute_dynamic_mask(prev_img, curr_img)
            if run_config.semantic_masking
            else None
        )
        prev_keypoints_filt, prev_desc_filt = filter_keypoints(
            prev_keypoints,
            prev_desc,
            mask,
        )
        curr_keypoints, curr_desc = filter_keypoints(
            curr_keypoints,
            curr_desc,
            mask,
        )
        try:
            R, t = estimate_pose_optical_flow(
                prev_img,
                curr_img,
                prev_keypoints_filt,
                slam_input.intrinsics,
            )
            logging.debug("Pose R=%s t=%s", R.tolist(), t.tolist())
        except Exception as exc:
            logging.warning("Optical flow pose failed: %s", exc)
            try:
                matches = feature_pipeline.match(prev_desc_filt, curr_desc)
                ransac_threshold = run_config.motion_ransac_threshold
                if run_config.adaptive_ransac:
                    pts_prev, pts_curr = matches_to_points(
                        prev_keypoints_filt, curr_keypoints, matches
                    )
                    ransac_threshold = adaptive_ransac_threshold(
                        pts_prev,
                        pts_curr,
                        run_config.motion_ransac_threshold,
                        run_config.adaptive_ransac_min,
                        run_config.adaptive_ransac_max,
                    )
                R, t, inliers, match_count = estimate_pose_from_matches(
                    prev_keypoints_filt,
                    curr_keypoints,
                    matches,
                    slam_input.intrinsics,
                    ransac_threshold=ransac_threshold,
                    min_matches=run_config.motion_min_matches,
                )
                stats = feature_pipeline.match_stats(matches)
                logging.debug(
                    "Feature pose used matches=%d inliers=%d median_dist=%.2f ransac=%.4f",
                    match_count,
                    len(inliers),
                    stats.median_distance,
                    ransac_threshold,
                )
            except Exception as exc2:
                logging.warning("Pose estimation failed: %s", exc2)
                try:
                    _, R, t = estimate_homography_from_orb(
                        prev_keypoints_filt,
                        prev_desc_filt,
                        curr_keypoints,
                        curr_desc,
                        slam_input.intrinsics,
                    )
                    logging.debug("Fallback homography pose used")
                except Exception:
                    prev_frame = curr_img
                    prev_keypoints = curr_keypoints
                    prev_desc = curr_desc
                    continue

        pose_graph.add_pose(R, t)
        path_estimator.add_transform(R, t)
        current_pose = pose_graph.poses[-1]
        if keyframe_manager.should_add_keyframe(current_pose, curr_desc):
            keyframe_manager.add_keyframe(frame_id, current_pose, curr_keypoints, curr_desc)
            ba_result = keyframe_manager.run_local_bundle_adjustment(slam_input.intrinsics)
            if ba_result is not None:
                for frame_index, pose in zip(ba_result.frame_ids, ba_result.poses):
                    if frame_index < len(pose_graph.poses):
                        pose_graph.poses[frame_index] = pose
                path_estimator.set_optimized_poses(list(pose_graph.poses))
                logging.info(
                    "Local bundle adjustment updated %d keyframes",
                    len(ba_result.frame_ids),
                )

        loop_id = bow_db.detect_loop(curr_desc)
        if loop_id is not None and loop_id in frames_data:
            _, loop_kp, loop_desc = frames_data[loop_id]
            try:
                loop_matches = feature_pipeline.match(loop_desc, curr_desc)
                loop_ransac = run_config.loop_ransac_threshold
                if run_config.adaptive_ransac:
                    pts_loop, pts_curr = matches_to_points(
                        loop_kp, curr_keypoints, loop_matches
                    )
                    loop_ransac = adaptive_ransac_threshold(
                        pts_loop,
                        pts_curr,
                        run_config.loop_ransac_threshold,
                        run_config.adaptive_ransac_min,
                        run_config.adaptive_ransac_max,
                    )
                R_loop, t_loop, inliers, match_count = estimate_pose_from_matches(
                    loop_kp,
                    curr_keypoints,
                    loop_matches,
                    slam_input.intrinsics,
                    ransac_threshold=loop_ransac,
                    min_matches=run_config.loop_min_matches,
                )
                inlier_count = len(inliers)
                inlier_ratio = inlier_count / max(match_count, 1)
                if inlier_count < run_config.loop_min_inliers or inlier_ratio < run_config.loop_min_inlier_ratio:
                    logging.info(
                        "Loop candidate rejected: frame=%d matches=%d inliers=%d ratio=%.2f",
                        loop_id,
                        match_count,
                        inlier_count,
                        inlier_ratio,
                    )
                    R_loop = t_loop = None
            except Exception as exc:
                logging.warning("Loop closure transform failed: %s", exc)
                try:
                    _, R_loop, t_loop, inliers, match_count = estimate_homography_from_orb_with_inliers(
                        loop_kp,
                        loop_desc,
                        curr_keypoints,
                        curr_desc,
                        slam_input.intrinsics,
                        min_matches=run_config.loop_min_matches,
                    )
                    inlier_count = len(inliers)
                    inlier_ratio = inlier_count / max(match_count, 1)
                    if inlier_count < run_config.loop_min_inliers or inlier_ratio < run_config.loop_min_inlier_ratio:
                        logging.info(
                            "Loop candidate rejected (homography): frame=%d matches=%d inliers=%d ratio=%.2f",
                            loop_id,
                            match_count,
                            inlier_count,
                            inlier_ratio,
                        )
                        R_loop = t_loop = None
                    else:
                        logging.debug("Fallback homography for loop used")
                except Exception as exc2:
                    logging.warning("Fallback loop closure failed: %s", exc2)
                    R_loop = t_loop = None
            if R_loop is not None:
                if run_config.use_sim3_loop_correction:
                    loop_scale = estimate_loop_scale(
                        pose_graph.poses[loop_id],
                        pose_graph.poses[frame_id],
                        t_loop,
                        run_config.loop_scale_min_translation,
                    )
                    pose_graph.add_loop(
                        loop_id,
                        frame_id,
                        R_loop,
                        t_loop,
                        loop_scale,
                        weight=run_config.loop_edge_weight,
                    )
                    logging.info(
                        "Loop scale estimate: frame=%d scale=%.3f",
                        loop_id,
                        loop_scale,
                    )
                else:
                    pose_graph.add_loop(
                        loop_id,
                        frame_id,
                        R_loop,
                        t_loop,
                        weight=run_config.loop_edge_weight,
                    )
                path_estimator.add_loop_edge(loop_id, frame_id)
                optimized = pose_graph.optimize()
                path_estimator.set_optimized_poses(optimized)
                orig = np.array([p[:2, 2] for p in pose_graph.poses])
                opt = np.array([p[:2, 2] for p in optimized])
                rmse = np.sqrt(np.mean((orig - opt) ** 2))
                logging.info("Pose graph optimised (RMSE=%.4f)", rmse)

        bow_db.add_frame(frame_id, curr_desc)
        frames_data[frame_id] = (curr_img, curr_keypoints, curr_desc)
        prev_frame = curr_img
        prev_keypoints = curr_keypoints
        prev_desc = curr_desc
        time.sleep(run_config.sleep_time)
        if run_config.pause_time > 0:
            plt.pause(run_config.pause_time)
    path_estimator.stop(run_config.save_plot)

    positions = np.array(path_estimator.positions)
    if run_config.save_poses:
        np.savetxt(run_config.save_poses, positions, fmt="%.6f")

    metrics = None
    if slam_input.kitti_gt_positions is not None:
        metrics = evaluate_kitti_trajectory(
            slam_input.kitti_gt_positions,
            positions,
        )
        for key, value in metrics.items():
            logging.info("KITTI comparison: %s %.4f", key, value)
    return SLAMResult(positions=positions, metrics=metrics)


def run_kitti_test(
    kitti_config: KittiConfig,
    run_config: SLAMRunConfig,
) -> SLAMResult:
    """Convenience helper to run and report a KITTI evaluation."""
    slam_input = prepare_kitti_input(kitti_config, run_config)
    result = run_visual_slam(slam_input, run_config)
    if result.metrics and kitti_config.report_path:
        with open(kitti_config.report_path, "w") as f:
            for key, value in result.metrics.items():
                f.write(f"{key} {value:.4f}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visual SLAM demo")
    parser.set_defaults(match_cross_check=True, adaptive_ransac=False)
    parser.add_argument(
        "--video",
        default=os.environ.get("SAMPLE_VIDEO_PATH", SAMPLE_VIDEO_PATH),
        help="Path to the input video",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--kitti_base_dir",
        help="Base directory containing KITTI raw data (e.g., /data/kitti_raw)",
        default=None,
    )
    parser.add_argument(
        "--kitti_date",
        help="KITTI date folder (e.g., 2011_09_26)",
        default=None,
    )
    parser.add_argument(
        "--kitti_drive",
        help="KITTI drive id (e.g., 0001)",
        default=None,
    )
    parser.add_argument(
        "--kitti_camera",
        default="image_02",
        help="KITTI camera folder to use (default: image_02)",
    )
    parser.add_argument(
        "--kitti_report",
        help="Optional path to save KITTI trajectory comparison metrics",
        default=None,
    )
    parser.add_argument(
        "--log_level",
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--sleep_time",
        type=float,
        default=0.1,
        help="Delay between frames to simulate slow processing",
    )
    parser.add_argument(
        "--pause_time",
        type=float,
        default=0.001,
        help="Matplotlib pause duration",
    )
    parser.add_argument(
        "--semantic_masking",
        action="store_true",
        help="Mask dynamic regions before feature detection",
    )
    parser.add_argument(
        "--intrinsics_file",
        help="Optional path to camera intrinsics (fx fy cx cy)",
        default=None,
    )
    parser.add_argument(
        "--save_plot",
        help="Path to save the final trajectory plot",
        default=None,
    )
    parser.add_argument(
        "--save_poses",
        help="Optional path to save estimated 2D poses (frame_index x y)",
        default=None,
    )
    parser.add_argument(
        "--feature_type",
        default="orb",
        help="Feature pipeline to use (default: orb)",
    )
    parser.add_argument(
        "--feature_nfeatures",
        type=int,
        default=2000,
        help="Number of features for the detector (default: 2000)",
    )
    parser.add_argument(
        "--match_ratio",
        type=float,
        default=0.8,
        help="Lowe ratio threshold when cross-check is disabled",
    )
    parser.add_argument(
        "--match_cross_check",
        dest="match_cross_check",
        action="store_true",
        help="Enable cross-check matching (default: True)",
    )
    parser.add_argument(
        "--no_match_cross_check",
        dest="match_cross_check",
        action="store_false",
        help="Disable cross-check matching (use ratio test instead)",
    )
    parser.add_argument(
        "--match_max_matches",
        type=int,
        default=500,
        help="Maximum number of matches to keep (default: 500)",
    )
    parser.add_argument(
        "--motion_min_matches",
        type=int,
        default=15,
        help="Minimum matches required for motion estimation",
    )
    parser.add_argument(
        "--motion_ransac_threshold",
        type=float,
        default=0.01,
        help="Base RANSAC threshold for motion estimation",
    )
    parser.add_argument(
        "--adaptive_ransac",
        action="store_true",
        help="Enable adaptive RANSAC thresholding",
    )
    parser.add_argument(
        "--adaptive_ransac_min",
        type=float,
        default=0.005,
        help="Minimum adaptive RANSAC threshold",
    )
    parser.add_argument(
        "--adaptive_ransac_max",
        type=float,
        default=0.03,
        help="Maximum adaptive RANSAC threshold",
    )
    parser.add_argument(
        "--loop_min_matches",
        type=int,
        default=30,
        help="Minimum ORB matches required before loop verification",
    )
    parser.add_argument(
        "--loop_min_inliers",
        type=int,
        default=20,
        help="Minimum inliers required to accept a loop candidate",
    )
    parser.add_argument(
        "--loop_min_inlier_ratio",
        type=float,
        default=0.3,
        help="Minimum inlier ratio required to accept a loop candidate",
    )
    parser.add_argument(
        "--loop_ransac_threshold",
        type=float,
        default=0.01,
        help="RANSAC threshold for loop verification (essential matrix)",
    )
    parser.add_argument(
        "--loop_edge_weight",
        type=float,
        default=0.5,
        help="Pose-graph weight for accepted loop edges",
    )
    parser.add_argument(
        "--use_sim3_loop_correction",
        action="store_true",
        help="Enable Sim(3) pose-graph optimization for scale-drift correction",
    )
    parser.add_argument(
        "--loop_scale_min_translation",
        type=float,
        default=1e-3,
        help="Minimum translation norm for estimating loop scale",
    )
    args = parser.parse_args()

    _init_logging(args.log_level)

    run_config = SLAMRunConfig(
        max_frames=args.max_frames,
        sleep_time=args.sleep_time,
        pause_time=args.pause_time,
        semantic_masking=args.semantic_masking,
        intrinsics_file=Path(args.intrinsics_file) if args.intrinsics_file else None,
        save_plot=Path(args.save_plot) if args.save_plot else None,
        save_poses=Path(args.save_poses) if args.save_poses else None,
        loop_min_matches=args.loop_min_matches,
        loop_min_inliers=args.loop_min_inliers,
        loop_min_inlier_ratio=args.loop_min_inlier_ratio,
        loop_ransac_threshold=args.loop_ransac_threshold,
        loop_edge_weight=args.loop_edge_weight,
        use_sim3_loop_correction=args.use_sim3_loop_correction,
        loop_scale_min_translation=args.loop_scale_min_translation,
        feature_type=args.feature_type,
        feature_nfeatures=args.feature_nfeatures,
        match_ratio=args.match_ratio,
        match_cross_check=args.match_cross_check,
        match_max_matches=args.match_max_matches,
        motion_min_matches=args.motion_min_matches,
        motion_ransac_threshold=args.motion_ransac_threshold,
        adaptive_ransac=args.adaptive_ransac,
        adaptive_ransac_min=args.adaptive_ransac_min,
        adaptive_ransac_max=args.adaptive_ransac_max,
    )

    if args.kitti_base_dir:
        if not args.kitti_date or not args.kitti_drive:
            raise SystemExit("KITTI mode requires --kitti_date and --kitti_drive")
        kitti_config = KittiConfig(
            base_dir=Path(args.kitti_base_dir),
            date=args.kitti_date,
            drive=args.kitti_drive,
            camera=args.kitti_camera,
            report_path=Path(args.kitti_report) if args.kitti_report else None,
        )
        run_kitti_test(kitti_config, run_config)
        return

    # Ensure the input video exists. When the user does not provide a video
    # path we fall back to the sample clip used in tests and download it on
    # demand. This provides a better out-of-the-box experience while still
    # allowing custom videos to be supplied.
    video_path = Path(args.video)
    if not video_path.exists():
        if args.video == SAMPLE_VIDEO_PATH:
            video_path = ensure_sample_video(video_path)
        else:
            raise SystemExit(f"Could not find video file {video_path}")
    slam_input = prepare_video_input(video_path, run_config, run_config.intrinsics_file)
    run_visual_slam(slam_input, run_config)

    # plt.axis('off')
    # plt.gca().set_position([0, 0, 1, 1])  # Fill the entire figure
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    # plt.imshow(frame_with_keypoints)
    # plt.show()


if __name__ == "__main__":
    main()
