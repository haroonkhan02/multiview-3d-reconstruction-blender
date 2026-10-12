"""Bundle adjustment from scratch with scipy.optimize.least_squares.

Unknowns: each free camera's rotation vector (3) and translation (3), and every 3D point (3).
Residuals: for every observation (camera c sees point p at pixel x), ``project(c, p) - x``.
Intrinsics are known and fixed (calibrated SfM).

The Jacobian is very sparse: an observation's two residuals depend only on its own camera's
6 parameters and its own point's 3. Passing that sparsity pattern lets SciPy's trust-region
solver estimate the Jacobian with few function evaluations and solve the normal equations with
LSMR instead of a dense matrix. A Huber loss limits the pull of the remaining outliers.

Gauge freedom: a reconstruction can be moved, rotated and scaled without changing any
residual. Holding the first camera fixed removes rotation and translation; scale stays free,
which the trust-region method tolerates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class BAResult:
    R: np.ndarray  # (n_cams, 3, 3)
    t: np.ndarray  # (n_cams, 3)
    X: np.ndarray  # (n_pts, 3)
    rmse_before: float
    rmse_after: float
    nfev: int


def project_points(rvec, t, K, X):
    """Vectorised projection: per-observation rotation vectors, translations, intrinsics."""
    xc = Rotation.from_rotvec(rvec).apply(X) + t
    z = xc[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = xc[:, :2] / z
    return np.stack([K[:, 0, 0] * uv[:, 0] + K[:, 0, 2], K[:, 1, 1] * uv[:, 1] + K[:, 1, 2]], 1)


def sparsity(n_cams_free, n_pts, cam_idx_free, pt_idx):
    """Jacobian sparsity: observation k touches camera cam_idx_free[k] (-1 = fixed) and point."""
    m = len(pt_idx) * 2
    A = lil_matrix((m, n_cams_free * 6 + n_pts * 3), dtype=np.uint8)
    rows = np.arange(len(pt_idx))
    free = cam_idx_free >= 0
    for s in range(6):
        A[2 * rows[free], cam_idx_free[free] * 6 + s] = 1
        A[2 * rows[free] + 1, cam_idx_free[free] * 6 + s] = 1
    for s in range(3):
        A[2 * rows, n_cams_free * 6 + pt_idx * 3 + s] = 1
        A[2 * rows + 1, n_cams_free * 6 + pt_idx * 3 + s] = 1
    return A


def bundle_adjust(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    X: np.ndarray,
    cam_idx: np.ndarray,
    pt_idx: np.ndarray,
    obs: np.ndarray,
    fixed: tuple[int, ...] = (0,),
    huber_px: float = 1.0,
    max_nfev: int = 100,
) -> BAResult:
    """Refine poses and points.

    R (C, 3, 3), t (C, 3), K (C, 3, 3) per camera; X (P, 3); observations: camera index,
    point index and measured pixel (O, 2). Cameras listed in ``fixed`` are not optimised.
    """
    C, P = len(R), len(X)
    free_cams = [c for c in range(C) if c not in fixed]
    remap = -np.ones(C, int)
    remap[free_cams] = np.arange(len(free_cams))
    cam_free = remap[cam_idx]
    rv_all = Rotation.from_matrix(R).as_rotvec()
    t_all = t.copy()
    K_obs = K[cam_idx]

    def unpack(x):
        rv, tt = rv_all.copy(), t_all.copy()
        cams = x[: len(free_cams) * 6].reshape(-1, 6)
        rv[free_cams], tt[free_cams] = cams[:, :3], cams[:, 3:]
        return rv, tt, x[len(free_cams) * 6:].reshape(P, 3)

    def residuals(x):
        rv, tt, pts = unpack(x)
        return (project_points(rv[cam_idx], tt[cam_idx], K_obs, pts[pt_idx]) - obs).ravel()

    x0 = np.concatenate([np.hstack([rv_all[free_cams], t_all[free_cams]]).ravel(), X.ravel()])
    r0 = residuals(x0)
    res = least_squares(
        residuals, x0, jac_sparsity=sparsity(len(free_cams), P, cam_free, pt_idx),
        method="trf", loss="huber", f_scale=huber_px, x_scale="jac", tr_solver="lsmr",
        max_nfev=max_nfev, verbose=0,
    )
    rv, tt, pts = unpack(res.x)
    return BAResult(
        R=Rotation.from_rotvec(rv).as_matrix(), t=tt, X=pts,
        rmse_before=float(np.sqrt(np.mean(r0.reshape(-1, 2) ** 2) * 2)),
        rmse_after=float(np.sqrt(np.mean(res.fun.reshape(-1, 2) ** 2) * 2)),
        nfev=int(res.nfev),
    )
