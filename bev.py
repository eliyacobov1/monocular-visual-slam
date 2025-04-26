import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_camera_matrix(fx, fy, cx, cy):
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

def get_extrinsics(pitch_deg, camera_height):
    pitch_rad = np.deg2rad(pitch_deg)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ])
    t = np.array([[0], [0], [-camera_height]])
    return R, t

def compute_ground_to_image_homography(K, pitch_deg, camera_height):
    pitch = np.deg2rad(pitch_deg)

    # R : world → camera  (positive pitch = camera tips down)
    R = np.array([[1, 0,               0],
                  [0, np.cos(pitch),  -np.sin(pitch)],
                  [0, np.sin(pitch),   np.cos(pitch)]])

    # Camera is h metres above the ground, i.e. –h along WORLD‑Y
    t = np.array([[0], [-camera_height], [0]])

    # Keep X (left–right) and Z (forward) columns of R
    H_world2img = K @ np.hstack((R[:, [0, 2]], t))
    return H_world2img

def compute_homography(K, R, t):
    # Ground plane assumption: Z = 0, so we use only R[:, :2]
    H_world_to_img = K @ np.hstack((R[:, :2], t))
    return H_world_to_img

def generate_bev_remap(H, bev_size_m, resolution):
    bev_w_m, bev_l_m = bev_size_m        # width  (X), length (Z)
    bev_w_px = int(bev_w_m / resolution)
    bev_l_px = int(bev_l_m / resolution)

    x_vals = np.linspace(-bev_w_m/2, bev_w_m/2, bev_w_px)   # left → right
    z_vals = np.linspace(0, bev_l_m,   bev_l_px)            # near → far
    X, Z = np.meshgrid(x_vals, z_vals)

    world_pts = np.stack([X.ravel(), Z.ravel(), np.ones_like(X).ravel()], 0)
    img_pts   = H @ world_pts
    img_pts  /= img_pts[2]                                  # normalise

    map_x = img_pts[0].reshape(bev_l_px, bev_w_px).astype(np.float32)
    map_y = img_pts[1].reshape(bev_l_px, bev_w_px).astype(np.float32)
    return map_x, map_y

def generate_bev_image(image, map_x, map_y):
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def show_results(original, bev):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Bird's Eye View (BEV)")
    plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Camera parameters (example: 1280x720)
    fx, fy = 800, 800
    cx, cy = 640, 360
    pitch_deg = 5
    camera_height = 1.6
    resolution = 0.05  # meters per pixel
    bev_size_m = (10, 20)  # width x height in meters

    # Load image
    image = cv2.imread("road_sample.jpg")  # Replace with your image path
    assert image is not None, "Image not found!"

    K = get_camera_matrix(fx, fy, cx, cy)
    R, t = get_extrinsics(pitch_deg, camera_height)
    H = compute_homography(K, R, t)
    map_x, map_y = generate_bev_remap(H, bev_size_m, resolution)
    bev_image = generate_bev_image(image, map_x, map_y)

    show_results(image, bev_image)

if __name__ == "__main__":
    main()