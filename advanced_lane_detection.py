
"""advanced_lane_detection.py
Full laneâ€‘detection pipeline based on a preâ€‘trained YOLOP model.
Features
--------
* Loads YOLOP from PyTorchâ€‘Hub (hustvl/yolop).
* Secondâ€‘order polynomial fit for left/right lanes.
* Î±â€‘Î²â€‘Î³ Kalman filtering to smooth polynomial coefficients.
* Inverseâ€‘perspective mapping (IPM) to obtain a BEV lane mask.
* Writes a stacked video (original frame over BEV mask).

Usage
-----
pip install torch torchvision torchaudio opencv-python --extra-index-url \
    https://download.pytorch.org/whl/cu118
python advanced_lane_detection.py        # downloads a sample clip

To process your own dashâ€‘cam:
python advanced_lane_detection.py --input my_clip.mp4 --output out.mp4

Author: ChatGPT (OpenAI) â€“ AprilÂ 2025
"""

import argparse
from pathlib import Path
import urllib.request
import numpy as np
import cv2
import torch
from torchvision import transforms

# --------------------------------------------------------------------- #
# 1.  Î±â€‘Î²â€‘Î³ Kalman filters
# --------------------------------------------------------------------- #
class ABGKalman:
    """Î±â€‘Î²â€‘Î³ tracker for a constantâ€‘acceleration model."""
    def __init__(self, alpha=0.4, beta=0.2, gamma=0.1):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.x = self.v = self.a = None

    def update(self, z, dt=1.0):
        # Initialise on first call
        if self.x is None:
            self.x, self.v, self.a = z, 0.0, 0.0
            return z
        # Prediction
        x_pred = self.x + self.v * dt + 0.5 * self.a * dt * dt
        v_pred = self.v + self.a * dt
        r = z - x_pred                         # residual
        # Update
        self.x = x_pred + self.alpha * r
        self.v = v_pred + self.beta * r / dt
        self.a = self.a + self.gamma * r / (0.5 * dt * dt)
        return self.x

class PolyKalman:
    """Kalmanâ€‘smooth each coefficient of a polynomial independently."""
    def __init__(self, order=2, alpha=0.3, beta=0.1, gamma=0.05):
        self.filters = [ABGKalman(alpha, beta, gamma) for _ in range(order + 1)]

    def update(self, coeffs):
        return np.array([flt.update(c) for flt, c in zip(self.filters, coeffs)])

# --------------------------------------------------------------------- #
# 2.  Perspective geometry helpers
# --------------------------------------------------------------------- #
def compute_ipm_homography(K, pitch_deg, cam_height_m, roi_y_px):
    """Compute homography that maps road pixels (y â‰¥ roi_y_px) to BEV."""
    pitch = np.deg2rad(pitch_deg)
    R_pitch = np.array([[1, 0, 0],
                        [0,  np.cos(pitch), -np.sin(pitch)],
                        [0,  np.sin(pitch),  np.cos(pitch)]])
    R = R_pitch
    normal = R[1, :]              # ground plane normal in camera coordinates
    d = cam_height_m
    t = np.array([0, cam_height_m, 0])
    H = R - np.outer(t, normal) / d
    H = K @ H @ np.linalg.inv(K)
    T = np.eye(3); T[1, 2] = -roi_y_px
    return H @ T

# --------------------------------------------------------------------- #
# 3.  YOLOP model utilities
# --------------------------------------------------------------------- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True).to(device).eval()

resize   = transforms.Resize((640, 640))
toTensor = transforms.ToTensor()
norm     = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))

def preprocess_bgr(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil     = transforms.functional.to_pil_image(img_rgb)
    x = norm(toTensor(resize(pil))).unsqueeze(0)
    return x.to(device)

def get_lane_mask(frame_bgr):
    with torch.inference_mode():
        _det, _da, ll = model(preprocess_bgr(frame_bgr))
    mask = ll.argmax(dim=1).squeeze().byte().cpu().numpy()
    return cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

# --------------------------------------------------------------------- #
# 4.  Polynomial fitting & drawing helpers
# --------------------------------------------------------------------- #
def fit_polynomial(mask, side='left', order=2, min_pixels=300):
    """Fit x = a yÂ² + b y + c (returns coeffs highestâ€‘order first)."""
    h, w = mask.shape
    mid  = w // 2
    if side == 'left':
        xs, ys = np.where(mask[:, :mid] == 1)
    else:
        xs, ys = np.where(mask[:, mid:] == 1)
        xs += mid
    if len(xs) < min_pixels:
        return None
    coeffs = np.polyfit(ys, xs, order)
    return coeffs

def render_polynomial(frame_bgr, coeffs, colour, thickness=5, y_samples=100):
    if coeffs is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    ys = np.linspace(0, h-1, y_samples)
    xs = np.polyval(coeffs, ys)
    pts = np.vstack([xs, ys]).T.astype(np.int32)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
    for i in range(len(pts) - 1):
        cv2.line(frame_bgr, tuple(pts[i]), tuple(pts[i+1]), colour, thickness)
    return frame_bgr

# --------------------------------------------------------------------- #
# 5.  Main processing function
# --------------------------------------------------------------------- #
def run_video(input_path, output_path,
              K=np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0,    0,   1]], dtype=np.float32),
              pitch_deg=3.0,
              cam_height_m=1.5,
              roi_ratio=0.6):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {input_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H*2))

    roi_y   = int(H * roi_ratio)
    H_ipm   = compute_ipm_homography(K, pitch_deg, cam_height_m, roi_y)

    kalman_L = PolyKalman()
    kalman_R = PolyKalman()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mask = get_lane_mask(frame)
        left_coeffs  = fit_polynomial(mask, 'left')
        right_coeffs = fit_polynomial(mask, 'right')

        if left_coeffs is not None:
            left_coeffs  = kalman_L.update(left_coeffs)
        if right_coeffs is not None:
            right_coeffs = kalman_R.update(right_coeffs)

        vis = frame.copy()
        vis = render_polynomial(vis, left_coeffs,  (0, 0, 255))
        vis = render_polynomial(vis, right_coeffs, (0, 255, 0))

        bev = cv2.warpPerspective(mask.astype(np.uint8)*255, H_ipm, (W, H))
        bev_col = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        stacked = np.vstack([vis, bev_col])
        out.write(stacked)

    cap.release(); out.release()
    print(f"âœ“ Done. Saved to {output_path}")

# --------------------------------------------------------------------- #
# 6.  CLI entryâ€‘point
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="YOLOPâ€‘based lane detection")
    parser.add_argument('--input',  type=str, default='dashcam_sample.mp4',
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='lanes_output.mp4',
                        help='Path for output video')
    args = parser.parse_args()

    # Autoâ€‘download a tiny sample clip if the input path is missing
    if not Path(args.input).exists():
        url = ('https://github.com/bunditmongkon/watchroad/releases/download/'
               'v0.1/drive_sample.mp4')
        print('Downloading sample clip ...')
        urllib.request.urlretrieve(url, args.input)

    run_video(args.input, args.output)

if __name__ == '__main__':
    main()