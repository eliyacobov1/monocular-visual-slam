"""
Pureâ€‘NumPy lane detection pipeline (no OpenCV).
"""
import numpy as np

def gaussian_kernel(size=5, sigma=1.0):
    """Generate a (size x size) Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def convolve(img, kernel):
    """2â€‘D convolution (zeroâ€‘padding) of img with kernel."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            out[i, j] = np.sum(region * kernel)
    return out


def sobel_edges(gray):
    """Return gradient magnitude & angle maps (Sobels from scratch)."""
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = convolve(gray, kx)
    gy = convolve(gray, ky)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    return mag, ang


def threshold(img, low, high):
    """Simple thresholding into a binary edge map."""
    return np.logical_and(img >= low, img <= high).astype(np.uint8)


def region_of_interest(edges, ratio=0.6):
    """Zero every row above ratio*height (simple trapezoidal ROI)."""
    mask = np.zeros_like(edges)
    h, w = edges.shape
    poly = np.array([
        [int(0.1 * w), h - 1],
        [int(0.9 * w), h - 1],
        [int(0.55 * w), int(ratio * h)],
        [int(0.45 * w), int(ratio * h)]
    ])
    # Rayâ€‘casting PIP test for each pixel (vectorized)
    # Build bounding box to reduce compute
    min_x, min_y = poly[:, 0].min(), poly[:, 1].min()
    max_x, max_y = poly[:, 0].max(), poly[:, 1].max()
    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten(); yv = yv.flatten()
    # Rayâ€‘casting: count intersections of horizontal ray
    inside = np.zeros_like(xv, dtype=bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cond = ((y1 > yv) != (y2 > yv)) & \
               (xv < (x2 - x1) * (yv - y1) / (y2 - y1 + 1e-6) + x1)
        inside = inside ^ cond
    mask[yv[inside], xv[inside]] = 1
    return edges * mask


def hough_lines(edge_img, rho_res=1, theta_res=np.deg2rad(1), thresh=150):
    """Return list of (rho, theta) pairs over threshold."""
    h, w = edge_img.shape
    diag_len = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    thetas = np.arange(-np.pi / 2, np.pi / 2, theta_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    ys, xs = np.nonzero(edge_img)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for t_idx in range(len(thetas)):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
            accumulator[rho, t_idx] += 1

    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] >= thresh:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                lines.append((rho, theta))
    return lines


def separate_lanes(lines):
    """Split raw Hough lines into left / right using slope sign."""
    left, right = [], []
    for rho, theta in lines:
        if np.sin(theta) == 0:
            continue  # skip perfectly vertical
        slope = -np.cos(theta) / np.sin(theta)
        intercept = rho / np.sin(theta)
        if slope < -0.5:  # left
            left.append((slope, intercept))
        elif slope > 0.5:  # right
            right.append((slope, intercept))
    def average(group):
        if not group:
            return None
        m = np.mean([g[0] for g in group])
        b = np.mean([g[1] for g in group])
        return m, b
    return average(left), average(right)


def render_lanes(image, lanes, y_top_ratio=0.6, thickness=8):
    """Draw averaged lane lines onto an RGB image (inâ€‘place)."""
    out = image.copy()
    h, w = out.shape[:2]
    y1, y2 = h - 1, int(h * y_top_ratio)
    for lane in lanes:
        if lane is None:
            continue
        m, b = lane
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        # Simple DDA line drawing
        length = max(abs(x2 - x1), abs(y2 - y1))
        xs = np.linspace(x1, x2, length).astype(int)
        ys = np.linspace(y1, y2, length).astype(int)
        for dx in range(-thickness // 2, thickness // 2 + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                idx = (ys + dy, xs + dx)
                valid = (idx[0] >= 0) & (idx[0] < h) & (idx[1] >= 0) & (idx[1] < w)
                out[idx[0][valid], idx[1][valid]] = [255, 0, 0]  # red
    return out


def detect_lanes(rgb_image):
    """Full pipeline from RGB -> laneâ€‘overlay RGB."""
    # 1. Preâ€‘process
    gray = rgb_image.mean(axis=2).astype(np.float32)
    blur = convolve(gray, gaussian_kernel(5, 1.0))
    mag, ang = sobel_edges(blur)
    edges = threshold(mag, 50, mag.max())  # simple global threshold
    roi_edges = region_of_interest(edges, ratio=0.6)
    # 2. Hough voting
    lines = hough_lines(roi_edges, rho_res=1, theta_res=np.deg2rad(1), thresh=120)
    # 3. Average into two lane lines
    left_lane, right_lane = separate_lanes(lines)
    # 4. Overlay
    overlay = render_lanes(rgb_image, [left_lane, right_lane])
    return overlay

