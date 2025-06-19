#!/usr/bin/env python3.12

import cv2
import time
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import argparse
import os

from slam_path_estimator import VehiclePathLiveAnimator
from loop_closure import BoWDatabase
from pose_graph import PoseGraph
from homography import estimate_homography_from_orb
from cam_intrinsics_estimation import make_K
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s::cv2_e2e - %(message)s')

# SAMPLE_VIDEO_PATH = "4644521-uhd_2562_1440_30fps.mp4"
SAMPLE_VIDEO_PATH = "sharp_curve.mp4"

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

def orb_detect(img: np.ndarray, detector: FeatureDetector_t, top_significant_kp=1000) -> np.ndarray:
    keypoints, descriptors = detector(img)
    logging.debug(f"Number of keypoints found: {len(keypoints)}")
    kp_response_sort_indices = np.argsort([-kp.response for kp in keypoints])[:top_significant_kp]
    keypoints = [keypoints[i] for i in kp_response_sort_indices]
    descriptors = descriptors[kp_response_sort_indices]
    return keypoints, descriptors

def compute_orientation(img, keypoint, patch_size=31):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    radius = patch_size // 2

    patch = img[max(y - radius, 0): y + radius + 1,
                max(x - radius, 0): x + radius + 1]

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
      cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)

def draw_pred_rect(frames: np.ndarray, results):
  for i, img in enumerate(frames):
    draw_bbs(img, results, i)

def compute_dynamic_mask(prev_img: np.ndarray, curr_img: np.ndarray, thresh: int = 25) -> np.ndarray:
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
    cap = cv2.VideoCapture(path)
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        count += 1
        yield grayscale_frame
        # frames.append(frame)
    # cap.release()

    # frames_np = np.stack(frames, axis=0)  # Shape: (T, H, W, C)
    # # frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # (T, C, H, W)
    # # frames_tensor = frames_np / 255.0  # Normalize to [0,1]
    # return frames_np


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visual SLAM demo")
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
        "--semantic_masking",
        action="store_true",
        help="Mask dynamic regions before feature detection",
    )
    parser.add_argument(
        "--save_plot",
        help="Path to save the final trajectory plot",
        default=None,
    )
    args = parser.parse_args()

    path_estimator = VehiclePathLiveAnimator()
    bow_db = BoWDatabase()
    pose_graph = PoseGraph()
    frames = load_video_frames(args.video, max_frames=args.max_frames)
    prev_frame = next(frames)
    frame_id = 0

    # Process first frame
    cv2_orb_detector: FeatureDetector_t = lambda img: cv2.ORB_create().detectAndCompute(img, None)
    prev_keypoints, prev_desc = orb_detect(prev_frame, cv2_orb_detector)
    bow_db.add_frame(frame_id, prev_desc)
    frames_data = {frame_id: (prev_frame, prev_keypoints, prev_desc)}
    pose_graph.add_pose(np.eye(2), np.zeros(2))  # initial pose

    for frame in frames:
        frame_id += 1
        curr_img, prev_img = frame, prev_frame
        cv2_orb_detector: FeatureDetector_t = lambda img: cv2.ORB_create().detectAndCompute(img, None)
        cv2_sift_detector: FeatureDetector_t = lambda img: cv2.SIFT_create().detectAndCompute(img, None)
        # custom_orb_detector: FeatureDetector_t = lambda img: my_custom_orb(img)
        prev_keypoints, prev_desc = orb_detect(prev_img, cv2_orb_detector)
        curr_keypoints, curr_desc = orb_detect(curr_img, cv2_orb_detector)
        mask = compute_dynamic_mask(prev_img, curr_img) if args.semantic_masking else None
        prev_keypoints, prev_desc = filter_keypoints(prev_keypoints, prev_desc, mask)
        curr_keypoints, curr_desc = filter_keypoints(curr_keypoints, curr_desc, mask)
        # frame_with_keypoints = cv2.drawKeypoints(
        #     gray_additional_frame, keypoints, None,
        #     flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        # )
        K = make_K(curr_img.shape[1], curr_img.shape[0])  # estimate intrinsics
        try:
            H, R, t = estimate_homography_from_orb(prev_keypoints, prev_desc,
                                                  curr_keypoints, curr_desc, K)
            logging.debug("Homography:\n%s", H)
        except Exception as exc:
            logging.warning("Homography estimation failed: %s", exc)
            prev_frame = curr_img
            continue
        

        
        # H, R, t = estimate_homography_from_sift(prev_keypoints, prev_desc, curr_keypoints, curr_desc)
        pose_graph.add_pose(R, t)
        path_estimator.add_transform(R, t)

        # Loop closure detection
        loop_id = bow_db.detect_loop(curr_desc)
        if loop_id is not None and loop_id in frames_data:
            loop_img, loop_kp, loop_desc = frames_data[loop_id]
            try:
                _, R_loop, t_loop = estimate_homography_from_orb(loop_kp, loop_desc, curr_keypoints, curr_desc, K)
                pose_graph.add_loop(loop_id, frame_id, R_loop, t_loop)
                path_estimator.add_loop_edge(loop_id, frame_id)
                optimized = pose_graph.optimize()
                path_estimator.set_optimized_poses(optimized)
                orig = np.array([p[:2,2] for p in pose_graph.poses])
                opt = np.array([p[:2,2] for p in optimized])
                rmse = np.sqrt(np.mean((orig - opt)**2))
                logging.info("Pose graph optimised (RMSE=%.4f)", rmse)
            except Exception as exc:
                logging.warning("Loop closure transform failed: %s", exc)

        bow_db.add_frame(frame_id, curr_desc)
        frames_data[frame_id] = (curr_img, curr_keypoints, curr_desc)
        prev_frame = curr_img
        time.sleep(0.1)  # simulate results arriving slowly
        plt.pause(0.001)  # <-- THIS IS CRITICAL!!!
    path_estimator.stop(args.save_plot)

    # plt.axis('off')
    # plt.gca().set_position([0, 0, 1, 1])  # Fill the entire figure
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    # plt.imshow(frame_with_keypoints)
    # plt.show()


if __name__ == "__main__":
    main()
