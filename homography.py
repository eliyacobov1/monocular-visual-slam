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


# ---------------------  Essential matrix utilities  ------------------------ #


def eight_point_E(src: np.ndarray, dst: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute the essential matrix using the eight-point algorithm."""
    n = len(src)
    if n < 8:
        raise ValueError("Eight correspondences required")

    Kinv = np.linalg.inv(K)
    src_h = np.hstack([src, np.ones((n, 1))])
    dst_h = np.hstack([dst, np.ones((n, 1))])
    x1 = (Kinv @ src_h.T).T
    x2 = (Kinv @ dst_h.T).T

    A = []
    for p1, p2 in zip(x1, x2):
        x, y, _ = p1 / p1[2]
        u, v, _ = p2 / p2[2]
        A.append([u * x, u * y, u, v * x, v * y, v, x, y, 1])
    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt

    return K.T @ F @ K


def decompose_essential(E: np.ndarray, src: np.ndarray, dst: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the correct R,t pair from the essential matrix."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    candidates = [
        (U @ W @ Vt, U[:, 2]),
        (U @ W @ Vt, -U[:, 2]),
        (U @ W.T @ Vt, U[:, 2]),
        (U @ W.T @ Vt, -U[:, 2]),
    ]

    def triangulate(P1, P2, p1, p2):
        A = np.vstack([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    src_h = np.hstack([src, np.ones((len(src), 1))])
    dst_h = np.hstack([dst, np.ones((len(dst), 1))])

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best = None
    best_count = -1
    for R, t in candidates:
        P2 = K @ np.hstack([R, t.reshape(3, 1)])
        count = 0
        for x1, x2 in zip(src_h, dst_h):
            X = triangulate(P1, P2, x1, x2)
            if X[2] > 0 and (R @ X + t)[2] > 0:
                count += 1
        if count > best_count:
            best = (R, t)
            best_count = count

    if best is None:
        raise RuntimeError("Essential matrix decomposition failed")
    return best


def ransac_essential(
    src: np.ndarray,
    dst: np.ndarray,
    K: np.ndarray,
    th: float = 0.01,
    max_iter: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an essential matrix with a basic RANSAC loop."""
    n = len(src)
    if n < 8:
        raise ValueError("At least eight correspondences are required")

    if rng is None:
        rng = np.random.default_rng()

    best_E = None
    best_inliers = np.array([], dtype=int)

    src_h = np.hstack([src, np.ones((n, 1))])
    dst_h = np.hstack([dst, np.ones((n, 1))])

    for _ in range(max_iter):
        idx = rng.choice(n, 8, replace=False)
        E = eight_point_E(src[idx], dst[idx], K)

        Ex1 = (E @ src_h.T).T
        Etx2 = (E.T @ dst_h.T).T
        err = np.abs(np.sum(dst_h * (E @ src_h.T).T, axis=1))
        denom = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
        err = err ** 2 / denom
        inliers = np.flatnonzero(err < th ** 2)

        if inliers.size > best_inliers.size:
            best_E = E
            best_inliers = inliers
            if inliers.size > 0.8 * n:
                break

    if best_E is None or best_inliers.size < 8:
        raise RuntimeError("RANSAC essential matrix failed")

    refined_E = eight_point_E(src[best_inliers], dst[best_inliers], K)
    return refined_E, best_inliers


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


def estimate_homography_from_orb_with_inliers(
    kp1,
    desc1,
    kp2,
    desc2,
    K=np.eye(3),
    min_matches: int = 15,
):
    matches = match_orb_descriptors(desc1, desc2)
    if len(matches) < min_matches:
        raise RuntimeError("Too few matches after ratio+sym test")

    src = np.asarray([kp1[i].pt for i, _ in matches])
    dst = np.asarray([kp2[j].pt for _, j in matches])

    H, inliers = ransac_homography(src, dst)
    R, t = decompose_homography(H, K)
    return H, R, t, inliers, len(matches)


def estimate_pose_from_orb(kp1, des1, kp2, des2, K):
    """Estimate camera pose using our minimal essential matrix implementation."""
    matches = match_orb_descriptors(des1, des2)
    if len(matches) < 15:
        raise RuntimeError("too few matches")

    pts1 = np.float32([kp1[i].pt for i, _ in matches])
    pts2 = np.float32([kp2[j].pt for _, j in matches])

    E, inliers = ransac_essential(pts1, pts2, K)
    R, t = decompose_essential(E, pts1[inliers], pts2[inliers], K)
    return R, t


def estimate_pose_from_orb_with_inliers(
    kp1,
    des1,
    kp2,
    des2,
    K,
    ransac_threshold: float = 0.01,
    min_matches: int = 15,
):
    """Estimate camera pose and return R,t plus inlier diagnostics."""
    matches = match_orb_descriptors(des1, des2)
    if len(matches) < min_matches:
        raise RuntimeError("too few matches")

    pts1 = np.float32([kp1[i].pt for i, _ in matches])
    pts2 = np.float32([kp2[j].pt for _, j in matches])

    E, inliers = ransac_essential(pts1, pts2, K, th=ransac_threshold)
    R, t = decompose_essential(E, pts1[inliers], pts2[inliers], K)
    return R, t, inliers, len(matches)


def estimate_pose_from_matches(
    kp1,
    kp2,
    matches,
    K,
    ransac_threshold: float = 0.01,
    min_matches: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Estimate camera pose from provided matches and return inliers."""
    if len(matches) < min_matches:
        raise RuntimeError("too few matches")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, inliers = ransac_essential(pts1, pts2, K, th=ransac_threshold)
    R, t = decompose_essential(E, pts1[inliers], pts2[inliers], K)
    return R, t, inliers, len(matches)


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
