import cv2
import numpy as np
import time

BRESENHAM_CIRCLE_16_PIXELS = np.array([
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)
])

# Indexes of North, East, South, West
NEIGHBOR_IDXS = np.array([0, 4, 8, 12])

def generate_point_pairs(patch_size, num_pairs, seed=42):
    np.random.seed(seed)
    coords = np.random.randint(-patch_size//2, patch_size//2, (num_pairs * 2, 2))
    return coords[:num_pairs], coords[num_pairs:]

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

def compute_orientation(patch):
    cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
    _, mx, my  = compute_moments(patch)
    dx = mx - cx
    dy = my - cy
    angle = np.arctan2(dy, dx)
    return angle

def rotate_points(points, angle):
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
    return np.dot(points, rot_matrix.T).astype(np.int32)

def compute_brief_descriptors(img, keypoints, angles, patch_size=31, num_pairs=256):
    keypoints = [kp for kp in keypoints if 
                 patch_size//2 <= kp.pt[0] < img.shape[1] - patch_size//2 and 
                 patch_size//2 <= kp.pt[1] < img.shape[0] - patch_size//2]

    keypoints_coords = np.array([[int(kp.pt[0]), int(kp.pt[1])] for kp in keypoints])
    p1, p2 = generate_point_pairs(patch_size, num_pairs)

    descriptors = []
    for (x, y), angle in zip(keypoints_coords, angles):
        patch = img[y - patch_size//2 : y + patch_size//2 + 1,
                    x - patch_size//2 : x + patch_size//2 + 1]

        rp1 = rotate_points(p1, angle)
        rp2 = rotate_points(p2, angle)

        desc = []
        for (dx1, dy1), (dx2, dy2) in zip(rp1, rp2):
            x1, y1 = dx1 + patch_size//2, dy1 + patch_size//2
            x2, y2 = dx2 + patch_size//2, dy2 + patch_size//2

            # Safe indexing with clipping
            if 0 <= x1 < patch.shape[1] and 0 <= y1 < patch.shape[0] and \
               0 <= x2 < patch.shape[1] and 0 <= y2 < patch.shape[0]:
                val1 = patch[y1, x1]
                val2 = patch[y2, x2]
                desc.append(1 if val1 < val2 else 0)
            else:
                desc.append(0)  # Padding for out-of-bounds

        descriptors.append(np.packbits(desc))  # 256 bits → 32 bytes

    return keypoints, np.array(descriptors, dtype=np.uint8)

def is_corner(img: np.ndarray, x, y, threshold, contiguous=12):
    center_intensity = img[y, x]

    circle_vals = img[
        y + BRESENHAM_CIRCLE_16_PIXELS[NEIGHBOR_IDXS, 1],
        x + BRESENHAM_CIRCLE_16_PIXELS[NEIGHBOR_IDXS, 0]
    ]

    brighter = (circle_vals > center_intensity + threshold).sum()
    darker = (circle_vals < center_intensity - threshold).sum()
    
    if max(brighter, darker) < 3:  # require at least 3 of 4 to be brighter or darker
        return False

    circle_vals_full = img[
        y + BRESENHAM_CIRCLE_16_PIXELS[:, 1],
        x + BRESENHAM_CIRCLE_16_PIXELS[:, 0]
    ]
    bright = circle_vals_full > center_intensity + threshold
    dark = circle_vals_full < center_intensity - threshold

    arr_extended = np.concatenate([bright, bright[:contiguous-1]])
    bright_contig = np.convolve(arr_extended, np.ones(contiguous, dtype=int), mode='valid').max() >= contiguous

    arr_extended = np.concatenate([dark, dark[:contiguous-1]])
    dark_contig = np.convolve(arr_extended, np.ones(contiguous, dtype=int), mode='valid').max() >= contiguous

    return bright_contig or dark_contig

def fast_detect(img, threshold=20, contiguous=12):
    h, w = img.shape
    keypoints = []
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            if is_corner(img, x, y, threshold, contiguous):
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=7)
                keypoints.append(kp)
    return keypoints

def orb_detect_and_compute(img, patch_size=31, num_pairs=256):
    img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = fast_detect(img_gray)

    angles = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = img_gray[y - patch_size//2:y + patch_size//2 + 1,
                         x - patch_size//2:x + patch_size//2 + 1]
        angle = compute_orientation(patch)
        angles.append(angle)

    keypoints, descriptors = compute_brief_descriptors(img_gray, keypoints, angles,
                                                       patch_size=patch_size,
                                                       num_pairs=num_pairs)
    return keypoints, descriptors


if __name__ == "__main__":
    img = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found.")

    keypoints, descriptors = orb_detect_and_compute(img)

    img_out = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    cv2.imshow("ORB Keypoints", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
