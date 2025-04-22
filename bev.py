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

def compute_homography(K, R, t):
    # Ground plane assumption: Z = 0, so we use only R[:, :2]
    H_world_to_img = K @ np.hstack((R[:, :2], t))
    return H_world_to_img

def generate_bev_remap(H_world_to_img, bev_size_m, resolution):
    bev_w_m, bev_h_m = bev_size_m
    bev_w_px = int(bev_w_m / resolution)
    bev_h_px = int(bev_h_m / resolution)

    # Grid in world coordinates
    x_vals = np.linspace(-bev_w_m/2, bev_w_m/2, bev_w_px)
    y_vals = np.linspace(0, bev_h_m, bev_h_px)
    X, Y = np.meshgrid(x_vals, y_vals)

    world_coords = np.stack([X.ravel(), Y.ravel(), np.ones_like(X).ravel()], axis=0)
    img_coords = H_world_to_img @ world_coords
    img_coords /= img_coords[2]

    map_x = img_coords[0].reshape(bev_h_px, bev_w_px).astype(np.float32)
    map_y = img_coords[1].reshape(bev_h_px, bev_w_px).astype(np.float32)
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