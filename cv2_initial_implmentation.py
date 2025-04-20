#!/usr/bin/env python3.12

import cv2
import numpy as np
import matplotlib.pyplot as plt

import feature_detection.fast as fast

SAMPLE_VIDEO_PATH = "4599004-hd_1080_1920_30fps.mp4"

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

def orb_cv2(img: np.ndarray, top_significant_kp=10) -> np.ndarray:
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:top_significant_kp]
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    return img_with_keypoints

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

def orb_impl(img: np.ndarray, top_significant_kp=10):
    ...

def draw_bbs(img_draw: np.ndarray, results, i):
  bboxes = results.xyxy[i][:, :4].cpu().numpy().astype(int)
  for box in bboxes:
      cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)

def draw_pred_rect(frames: np.ndarray, results):
  for i, img in enumerate(frames):
    draw_bbs(img, results, i)
    
def load_video_frames(path, resize=(1080, 1920), max_frames=50):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()

    frames_np = np.stack(frames, axis=0)  # Shape: (T, H, W, C)
    # frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # (T, C, H, W)
    # frames_tensor = frames_np / 255.0  # Normalize to [0,1]
    return frames_np


if __name__ == "__main__":
    print("Loading video frames...")
    single_frame = load_video_frames(SAMPLE_VIDEO_PATH, max_frames=1)[0]
    gray_single_frame = cv2.cvtColor(single_frame, cv2.COLOR_BGR2GRAY)
    frame_with_keypoints = orb_cv2(gray_single_frame)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])  # Fill the entire figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    plt.imshow(frame_with_keypoints)
    plt.show()
    