import numpy as np
import cv2


def hamming_distance(desc1, desc2):
    return np.count_nonzero(desc1 != desc2)


def match_orb_descriptors(desc1, desc2, ratio=0.8):
    """Mutual best match with Lowe ratio test and Hamming distance."""
    best12 = []
    for i, d1 in enumerate(desc1):
        # XOR all descriptors at once -> 512-bit strings -> popcount
        hamming = popcount_uint8(np.bitwise_xor(d1, desc2)).sum(axis=1)
        j_best, j_second = np.argsort(hamming)[:2]
        if hamming[j_best] < ratio * hamming[j_second]:
            best12.append((i, j_best, hamming[j_best]))

    # Symmetry test
    matches = []
    for i, j, dist_ij in best12:
        hamming = popcount_uint8(np.bitwise_xor(desc2[j], desc1)).sum(axis=1)
        i_best = np.argmin(hamming)
        if i_best == i:
            matches.append((i, j))
    return matches


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts - mean)
    scale = np.sqrt(2) / std
    T = np.array(
        [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
    )
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


def estimate_homography_from_sift(kp1, desc1, kp2, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        best_dist = float("inf")
        best_j = -1
        for j, d2 in enumerate(desc2):
            dist = np.linalg.norm(d1 - d2)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_dist < 0.75 * np.linalg.norm(desc2[best_j]):
            matches.append((i, best_j))

    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography")

    src_pts = np.array([kp1[i].pt for i, _ in matches])
    dst_pts = np.array([kp2[j].pt for _, j in matches])

    H = estimate_homography(src_pts, dst_pts)
    R, t = decompose_homography(H)
    return H, R, t


def popcount_uint8(x: np.ndarray) -> np.ndarray:
    # table-based popcount on bytes, ~10× faster than np.count_nonzero
    _tbl = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    return _tbl[x]


def hamming_distance_bitpop(d1: np.ndarray, d2: np.ndarray) -> int:
    return popcount_uint8(np.bitwise_xor(d1, d2)).sum()


# ---------------------  normalisation (Hartley 1997)  ---------------------- #


def normalise_points(pts):
    c = pts.mean(axis=0)
    diffs = pts - c
    rms = np.sqrt((diffs**2).sum(axis=1).mean())
    s = np.sqrt(2) / rms
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ pts_h.T).T[:, :2], T


# ---------------------  DLT homography, reused inside RANSAC --------------- #


def dlt_homography(src, dst):
    src_n, T_src = normalise_points(src)
    dst_n, T_dst = normalise_points(dst)

    A = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ Hn @ T_src
    return H / H[2, 2]


# ---------------------  RANSAC wrapper  ------------------------------------ #


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    th: float = 3.0,
    max_iter: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a homography with a simple RANSAC loop.

    The implementation mirrors the classical algorithm: random 4‑point minimal
    samples with symmetrical reprojection error checking.  The best model is
    refined using all inliers.

    Parameters
    ----------
    src, dst : (N,2) arrays
        Matched coordinates ``src[i] -> dst[i]`` in pixels.
    th : float, optional
        Inlier reprojection threshold in pixels.
    max_iter : int, optional
        Maximum number of RANSAC iterations.
    rng : np.random.Generator, optional
        Random number generator used for sampling.  If ``None`` (default), a
        fresh generator is created.

    Returns
    -------
    H : (3,3) ndarray
        Estimated homography mapping ``src`` to ``dst``.
    inliers : (M,) ndarray
        Indices of inlier correspondences.
    """

    n = len(src)
    if n < 4:
        raise ValueError("At least four correspondences are required")

    best_H = None
    best_inliers: np.ndarray = np.array([], dtype=int)

    src_h = np.hstack([src, np.ones((n, 1))])
    dst_h = np.hstack([dst, np.ones((n, 1))])

    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_iter):
        idx = rng.choice(n, 4, replace=False)
        H = dlt_homography(src[idx], dst[idx])

        proj_f = (H @ src_h.T).T
        proj_f = proj_f[:, :2] / proj_f[:, 2, None]
        proj_b = (np.linalg.inv(H) @ dst_h.T).T
        proj_b = proj_b[:, :2] / proj_b[:, 2, None]
        err = np.linalg.norm(proj_f - dst, axis=1) + np.linalg.norm(
            proj_b - src, axis=1
        )
        inliers = np.flatnonzero(err < th)

        if inliers.size > best_inliers.size:
            best_H = H
            best_inliers = inliers
            if inliers.size > 0.8 * n:
                break

    if best_H is None or best_inliers.size < 4:
        raise RuntimeError("RANSAC failed — too few inliers")

    refined_H = dlt_homography(src[best_inliers], dst[best_inliers])
    return refined_H, best_inliers


# ---------------------  pipeline entry point ------------------------------ #


def estimate_homography_from_orb(kp1, desc1, kp2, desc2, K=np.eye(3)):
    matches = match_orb_descriptors(desc1, desc2)
    if len(matches) < 15:
        raise RuntimeError("Too few matches after ratio+sym test")

    src = np.asarray([kp1[i].pt for i, _ in matches])
    dst = np.asarray([kp2[j].pt for _, j in matches])

    H, inliers = ransac_homography(src, dst)

    # *If* your scene is truly planar and K is correct, you may still
    # decompose H.  Otherwise prefer the essential matrix path (see below).
    R, t = decompose_homography(H, K)
    return H, R, t  # , inliers


def estimate_pose_from_orb(kp1, des1, kp2, des2, K):
    matches = match_orb_descriptors(des1, des2)
    if len(matches) < 15:
        raise RuntimeError("too few matches")

    pts1 = np.float32([kp1[i].pt for i, _ in matches])
    pts2 = np.float32([kp2[j].pt for _, j in matches])

    # (1) Essential matrix with RANSAC (five‑point inside OpenCV)
    E, inliers = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0
    )

    # (2) Recover the only physically valid pose
    _, R, t, _ = cv2.recoverPose(
        E, pts1[inliers.ravel() == 1], pts2[inliers.ravel() == 1], K
    )

    return R, t.reshape(-1)


# ---------------------  (optional) essential matrix variant  --------------- #
"""
If the road is NOT perfectly planar, replace the call above with:

    # 1. Eight-point fundamental with the same RANSAC loop
    F, inliers = ransac_fundamental(src, dst)   # implement similarly
    # 2. E = K.T @ F @ K
    E = K.T @ F @ K
    # 3. Decompose E (5-point or SVD, keep the solution that yields positive depth)
    R, t = decompose_essential(E, src[inliers], dst[inliers], K)

This gives a physically consistent rotation/translation pair and removes the
“planar trap” that destabilises your VO in mixed-depth scenes.
"""
