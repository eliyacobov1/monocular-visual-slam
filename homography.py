import numpy as np

def hamming_distance(desc1, desc2):
    return np.count_nonzero(desc1 != desc2)

def match_descriptors(descs1, descs2, threshold=30):
    matches = []
    for i, d1 in enumerate(descs1):
        best_dist = float('inf')
        best_j = -1
        for j, d2 in enumerate(descs2):
            dist = hamming_distance(d1, d2)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_dist < threshold:
            matches.append((i, best_j))
    return matches

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts - mean)
    scale = np.sqrt(2) / std
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def estimate_homography(src_pts, dst_pts):
    src_pts_norm, T_src = normalize_points(src_pts)
    dst_pts_norm, T_dst = normalize_points(dst_pts)

    A = []
    for (x1, y1), (x2, y2) in zip(src_pts_norm, dst_pts_norm):
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[-1, -1]

def decompose_homography(H, K=np.eye(3)):
    # Assume pinhole camera model with identity intrinsics if not given
    # Simplified decomposition assuming planar scene
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = H[:, 2]

    # Normalize
    norm = np.linalg.norm(np.linalg.inv(K) @ H1)
    r1 = np.linalg.inv(K) @ H1 / norm
    r2 = np.linalg.inv(K) @ H2 / norm
    t = np.linalg.inv(K) @ H3 / norm

    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)

    # Ensure R is a valid rotation matrix using SVD
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    return R, t

def estimate_homography_from_orb(kp1, desc1, kp2, desc2):
    matches = match_descriptors(desc1, desc2)

    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography")

    src_pts = np.array([kp1[i].pt for i, _ in matches])
    dst_pts = np.array([kp2[j].pt for _, j in matches])

    H = estimate_homography(src_pts, dst_pts)
    R, t = decompose_homography(H)
    return H, R, t
