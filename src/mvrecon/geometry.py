"""Two-view geometry from scratch (NumPy only).

Conventions used everywhere in the project:

* A camera has intrinsics ``K`` (3x3) and a world-to-camera pose ``(R, t)``:
  ``x_cam = R @ X_world + t``. Its centre in the world is ``C = -R.T @ t``.
* Pixel points are (N, 2) arrays; "normalised" points are ``K^-1 [u, v, 1]`` (N, 2).
* The essential matrix of a pair maps view 1 to view 2: ``x2n^T E x1n = 0`` with
  ``E = [t]_x R`` where (R, t) is the pose of camera 2 relative to camera 1.

What is implemented here (and unit-tested against synthetic scenes):

1. Hartley normalisation + the normalised 8-point algorithm for F (rank 2 enforced by SVD);
2. the calibrated 8-point algorithm for E, projected onto the essential manifold
   (singular values (1, 1, 0));
3. RANSAC around it with the Sampson (first-order geometric) error as the residual;
4. the four-fold (R, t) decomposition of E and the cheirality (points in front) test;
5. linear (DLT) triangulation, reprojection errors and triangulation angles;
6. Umeyama's closed-form similarity alignment, used to compare against ground truth.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- basics
def to_h(x: np.ndarray) -> np.ndarray:
    """(N, d) -> (N, d+1) homogeneous coordinates."""
    return np.hstack([x, np.ones((len(x), 1), dtype=x.dtype)])


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: skew(a) @ b == np.cross(a, b)."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def normalize_points(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels -> normalised image coordinates (remove the intrinsics)."""
    xn = to_h(x.astype(float)) @ np.linalg.inv(K).T
    return xn[:, :2] / xn[:, 2:]


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """World points (N, 3) -> pixels (N, 2)."""
    xc = X @ R.T + t
    xp = xc @ K.T
    return xp[:, :2] / xp[:, 2:]


def relative_pose(R1, t1, R2, t2):
    """Pose of camera 2 relative to camera 1 (both world-to-camera)."""
    R = R2 @ R1.T
    return R, t2 - R @ t1


# ----------------------------------------------------------------------------- F and E
def hartley_transform(x: np.ndarray) -> np.ndarray:
    """Similarity T that moves the points' centroid to 0 and their mean distance to sqrt(2).

    Without it the 8-point design matrix mixes entries of size ~1 and ~1e6 (pixel products), and
    the SVD solution is dominated by rounding error.
    """
    c = x.mean(axis=0)
    d = np.sqrt(((x - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def _eight_point_raw(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Least-squares solution of x2^T M x1 = 0 for a 3x3 M (|M| = 1), no constraint enforced."""
    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    A = np.stack([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, np.ones_like(u1)], axis=1)
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 3)


def eight_point_fundamental(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Normalised 8-point algorithm (Hartley 1997). x1, x2: (N >= 8, 2) pixel matches."""
    T1, T2 = hartley_transform(x1), hartley_transform(x2)
    p1 = (to_h(x1) @ T1.T)[:, :2]
    p2 = (to_h(x2) @ T2.T)[:, :2]
    F = _eight_point_raw(p1, p2)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0.0]) @ Vt  # a fundamental matrix has rank 2
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def eight_point_essential(x1n: np.ndarray, x2n: np.ndarray) -> np.ndarray:
    """Calibrated 8-point: E from normalised points, projected to singular values (1, 1, 0)."""
    T1, T2 = hartley_transform(x1n), hartley_transform(x2n)
    p1 = (to_h(x1n) @ T1.T)[:, :2]
    p2 = (to_h(x2n) @ T2.T)[:, :2]
    E = T2.T @ _eight_point_raw(p1, p2) @ T1
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def fundamental_from_essential(E, K1, K2):
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order approximation of the squared reprojection error of a match under F (px^2)."""
    h1, h2 = to_h(x1), to_h(x2)
    Fx1 = h1 @ F.T
    Ftx2 = h2 @ F
    num = np.sum(h2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def ransac_essential(
    x1: np.ndarray,
    x2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    thresh_px: float = 1.5,
    iters: int = 2000,
    confidence: float = 0.999,
    seed: int = 0,
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC over 8-point minimal samples; residual = Sampson distance in pixels.

    The iteration count adapts to the best inlier ratio found so far:
    ``N = log(1 - confidence) / log(1 - w^8)``. The final E is re-estimated from all inliers.
    Returns (E, inlier_mask).
    """
    n = len(x1)
    if n < 8:
        raise ValueError("need at least 8 matches")
    rng = np.random.default_rng(seed)
    x1n, x2n = normalize_points(x1, K1), normalize_points(x2, K2)
    best_mask = np.zeros(n, bool)
    thr2 = thresh_px**2
    needed, i = iters, 0
    while i < min(iters, needed):
        idx = rng.choice(n, 8, replace=False)
        E = eight_point_essential(x1n[idx], x2n[idx])
        d = sampson_distance(fundamental_from_essential(E, K1, K2), x1, x2)
        mask = d < thr2
        if mask.sum() > best_mask.sum():
            best_mask = mask
            w = mask.mean()
            p_good = w**8  # chance that one sample is all-inlier
            if p_good > 1e-9:
                needed = int(np.ceil(np.log(1 - confidence) / np.log1p(-min(p_good, 1 - 1e-12))))
        i += 1
    # The linear 8-point estimate minimises an algebraic error, not a geometric one; on narrow
    # fields of view it fits noticeably worse than the true E. Polish it on the inliers by
    # minimising the Sampson error over (R, t), then re-score (twice: inliers can change).
    E = eight_point_essential(x1n[best_mask], x2n[best_mask])
    for _ in range(2 if refine else 0):
        E = refine_essential(E, x1[best_mask], x2[best_mask], K1, K2)
        mask = sampson_distance(fundamental_from_essential(E, K1, K2), x1, x2) < thr2
        if mask.sum() < 8:
            break
        best_mask = mask
    return E, best_mask


def refine_essential(E: np.ndarray, x1: np.ndarray, x2: np.ndarray, K1, K2) -> np.ndarray:
    """Non-linear refinement of E = [t]x R: minimise the Sampson distances (pixels).

    Parameters: a rotation vector (3) and the translation direction (3, re-normalised), which
    keeps E on the essential manifold at every step (5 true degrees of freedom).
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    R, t, _ = recover_pose(E, normalize_points(x1, K1), normalize_points(x2, K2))

    def residuals(p):
        Rm = Rotation.from_rotvec(p[:3]).as_matrix()
        tt = p[3:] / np.linalg.norm(p[3:])
        F = fundamental_from_essential(skew(tt) @ Rm, K1, K2)
        return np.sqrt(sampson_distance(F, x1, x2))

    p0 = np.concatenate([Rotation.from_matrix(R).as_rotvec(), t])
    p = least_squares(residuals, p0, loss="huber", f_scale=1.0, max_nfev=50).x
    return skew(p[3:] / np.linalg.norm(p[3:])) @ Rotation.from_rotvec(p[:3]).as_matrix()


# ----------------------------------------------------------------------------- pose from E
def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four (R, t) candidates of E = [t]x R (t up to scale and sign)."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation, vectorised over N points. Returns (N, 3) world points.

    Each view gives two rows of A X = 0: u * P[2] - P[0] and v * P[2] - P[1]. The solution is
    the right singular vector of A with the smallest singular value.
    """
    A = np.stack(
        [
            x1[:, 0:1] * P1[2] - P1[0],
            x1[:, 1:2] * P1[2] - P1[1],
            x2[:, 0:1] * P2[2] - P2[0],
            x2[:, 1:2] * P2[2] - P2[1],
        ],
        axis=1,
    )  # (N, 4, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1]
    return X[:, :3] / X[:, 3:4]


def triangulate_multi(Ps: list[np.ndarray], xs: list[np.ndarray]) -> np.ndarray:
    """DLT triangulation of ONE point seen in several views (used for long tracks)."""
    rows = []
    for P, x in zip(Ps, xs, strict=True):
        rows += [x[0] * P[2] - P[0], x[1] * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def depths(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X @ R.T + t)[:, 2]


def recover_pose(
    E: np.ndarray, x1n: np.ndarray, x2n: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the (R, t) candidate with most points in front of both cameras (cheirality).

    Works in normalised coordinates (K = I). Returns (R, t with |t| = 1, in-front mask).
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in decompose_essential(E):
        X = triangulate(P1, np.hstack([R, t.reshape(3, 1)]), x1n, x2n)
        front = (X[:, 2] > 0) & (depths(R, t, X) > 0)
        if best is None or front.sum() > best[2].sum():
            best = (R, t / np.linalg.norm(t), front)
    return best


def triangulation_angles(C1: np.ndarray, C2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the two viewing rays of each point: small = depth is uncertain."""
    a, b = X - C1, X - C2
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def reprojection_error(K, R, t, X, x) -> np.ndarray:
    return np.linalg.norm(project(K, R, t, X) - x, axis=1)


# ----------------------------------------------------------------------------- comparisons
def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix (geodesic distance to identity), degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    return rotation_angle_deg(R_est @ R_gt.T)


def direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle between two translation directions (scale is unobservable in two views)."""
    c = t_est @ t_gt / (np.linalg.norm(t_est) * np.linalg.norm(t_gt) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity dst ~ s * R @ src + t (Umeyama 1991). Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (a**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s
