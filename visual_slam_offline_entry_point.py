#!/usr/bin/env python3.12

import cv2
import time
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

import feature_detection.fast as fast
from slam_path_estimator import VehiclePathLiveAnimator
from homography import estimate_homography_from_orb, estimate_pose_from_orb
from cam_intrinsics_estimation import make_K
from bev import generate_bev_image, generate_bev_remap, get_extrinsics, compute_ground_to_image_homography
import logging

logging.basicConfig(level=logging.INFO, format='INFO::cv2_e2e - %(message)s')

SAMPLE_VIDEO_PATH = "4644521-uhd_2562_1440_30fps.mp4"

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


if __name__ == "__main__":
    path_estimator = VehiclePathLiveAnimator()
    frames = load_video_frames(SAMPLE_VIDEO_PATH, max_frames=10000)
    prev_frame = next(frames)

    for frame in frames:
        curr_img, prev_img = frame, prev_frame
        cv2_orb_detector: FeatureDetector_t = lambda img: cv2.ORB_create().detectAndCompute(img, None)
        cv2_sift_detector: FeatureDetector_t = lambda img: cv2.SIFT_create().detectAndCompute(img, None)
        # custom_orb_detector: FeatureDetector_t = lambda img: fast.orb_detect_and_compute(img)
        prev_keypoints, prev_desc = orb_detect(prev_img, cv2_orb_detector)
        curr_keypoints, curr_desc = orb_detect(curr_img, cv2_orb_detector)
        # frame_with_keypoints = cv2.drawKeypoints(
        #     gray_additional_frame, keypoints, None,
        #     flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        # )
        K = make_K(curr_img.shape[1], curr_img.shape[0]) # currently estimating K by assuming FOV
        R, t = estimate_pose_from_orb(prev_keypoints, prev_desc, curr_keypoints, curr_desc, K)
        
        # TODO: this is for testing BEV only
        # pitch_deg = 5
        # camera_height = 1.6
        # R_extrinsic, t_extrinsic = get_extrinsics(pitch_deg, camera_height)
        # H = compute_ground_to_image_homography(K, pitch_deg, camera_height)
        # resolution = 0.05  # meters per pixel
        # bev_size_m = (10, 20)  # width x height in meters
        # map_x, map_y = generate_bev_remap(H, bev_size_m, resolution)
        # bev_image = generate_bev_image(curr_img, map_x, map_y)
        # plt.imsave("bev_image.png", bev_image, cmap='gray')
        # import sys
        # sys.exit(0)
        # TODO: this is for testing BEV only
        
        # H, R, t = estimate_homography_from_sift(prev_keypoints, prev_desc, curr_keypoints, curr_desc)
        path_estimator.add_transform(R, t)
        prev_frame = curr_img
        time.sleep(0.1)  # simulate results arriving slowly
        plt.pause(0.001)  # <-- THIS IS CRITICAL!!!
    path_estimator.stop()

    # plt.axis('off')
    # plt.gca().set_position([0, 0, 1, 1])  # Fill the entire figure
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    # plt.imshow(frame_with_keypoints)
    # plt.show()
