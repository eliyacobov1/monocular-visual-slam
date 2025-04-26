import numpy as np
import matplotlib.pyplot as plt

def compute_line_from_points(p1, p2):
    """Returns the homogeneous line equation (a, b, c) for line through p1 and p2"""
    x1, y1 = p1
    x2, y2 = p2
    return np.cross([x1, y1, 1], [x2, y2, 1])

def compute_vanishing_point_from_lines(lines):
    """
    Estimate vanishing point as intersection of lines using SVD
    lines: list of homogeneous lines (a, b, c)
    """
    A = np.array(lines)
    _, _, vh = np.linalg.svd(A)
    vp_hom = vh[-1]
    return vp_hom[:2] / vp_hom[2]

def estimate_focal_length_from_vps(vp1, vp2, img_center):
    """
    Estimate focal length from two orthogonal vanishing points
    vp1, vp2: (x, y)
    img_center: (cx, cy)
    """
    x0, y0 = img_center
    dx1, dy1 = vp1[0] - x0, vp1[1] - y0
    dx2, dy2 = vp2[0] - x0, vp2[1] - y0
    dot = dx1 * dx2 + dy1 * dy2
    f_squared = -dot
    if f_squared <= 0:
        return None
    return np.sqrt(f_squared)

def estimate_intrinsics_from_matched_points(matched_pairs, image_shape, label1='VP1', label2='VP2'):
    """
    Estimate camera intrinsics from matched keypoints using vanishing points
    matched_pairs: list of ((x1, y1), (x2, y2)) from consecutive frames
    image_shape: (height, width)
    """
    h, w = image_shape
    cx, cy = w / 2, h / 2

    # Assume first N lines are one direction, rest are another (you can cluster by angle in real use)
    N = len(matched_pairs) // 2
    lines_vp1 = [compute_line_from_points(p1, p2) for p1, p2 in matched_pairs[:N]]
    lines_vp2 = [compute_line_from_points(p1, p2) for p1, p2 in matched_pairs[N:]]

    vp1 = compute_vanishing_point_from_lines(lines_vp1)
    vp2 = compute_vanishing_point_from_lines(lines_vp2)
    f = estimate_focal_length_from_vps(vp1, vp2, (cx, cy))

    if f is None:
        raise ValueError("Failed to estimate focal length from vanishing points.")

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])

    # Optional plot
    plt.figure(figsize=(8, 6))
    for (x1, y1), (x2, y2) in matched_pairs:
        plt.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5)
    plt.scatter([vp1[0]], [vp1[1]], c='r', label=label1)
    plt.scatter([vp2[0]], [vp2[1]], c='g', label=label2)
    plt.scatter([cx], [cy], c='b', label='Principal Point')
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.title('Vanishing Points and Lines')
    plt.legend()
    plt.grid(True)
    plt.show()

    return K

def make_K(width, height, fov_x_deg=90):
    f = width / (2 * np.tan(np.deg2rad(fov_x_deg) / 2))
    return np.array([[f, 0, width/2],
                     [0, f, height/2],
                     [0, 0,        1 ]], dtype=np.float64)

# ======= EXAMPLE USAGE =======
if __name__ == "__main__":
    # Simulated matched keypoints (should come from actual feature tracking)
    matched_pairs = [
        ((100, 500), (300, 300)), ((110, 510), (310, 310)),  # Direction 1
        ((500, 100), (500, 300)), ((510, 110), (510, 310))   # Direction 2 (orthogonal)
    ]

    image_shape = (600, 800)  # height, width
    K = estimate_intrinsics_from_matched_points(matched_pairs, image_shape)
    print("Estimated Camera Intrinsic Matrix:\n", K)
