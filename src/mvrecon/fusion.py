"""Depth-map fusion and meshing.

Fusion (the step that turns noisy per-view depth maps into one clean point cloud):

* drop pixels whose ZNCC confidence is below ``min_conf``;
* **geometric consistency**: project each remaining pixel into every source view, read that
  view's depth there, back-project it and re-project it into the reference view. The pixel is
  consistent with that view if it comes back within ``max_px`` pixels and its depth agrees to
  ``max_rel_depth``. Keep pixels consistent with at least ``min_views`` views. A wrong depth
  is very unlikely to be confirmed by independent views, so this removes most outliers;
* back-project the survivors to world space, colour them from the reference image and merge
  all views with a voxel grid.

Meshing: Open3D normals (oriented towards the camera that produced each point) and screened
Poisson surface reconstruction, then removal of the low-density triangles that Poisson invents
far from any point (it always produces a closed surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionConfig:
    min_conf: float = 0.5
    max_px: float = 1.0
    max_rel_depth: float = 0.01
    min_views: int = 2


def backproject(depth: np.ndarray, K, R, t, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) of the pixels in mask, and their (N, 2) pixel coordinates."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    if mask is None:
        mask = np.isfinite(depth)
    uv = np.stack([u[mask], v[mask]], 1).astype(float)
    d = depth[mask]
    xc = (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T) * d[:, None]
    return (xc - t) @ R, uv


def _project(K, R, t, X):
    xc = X @ R.T + t
    z = xc[:, 2]
    uv = (xc @ K.T)[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    return uv, z


def consistency_counts(r, depths, confs, Ks, Rs, ts, sources, cfg: FusionConfig):
    """Per valid reference pixel: how many source views confirm its depth."""
    D = depths[r]
    valid = (confs[r] > cfg.min_conf) & np.isfinite(D) & (D > 0)
    X, uv = backproject(D, Ks[r], Rs[r], ts[r], valid)
    d_ref = D[valid]
    count = np.zeros(len(X), int)
    for s in sources:
        Ds = depths[s]
        H, W = Ds.shape
        uv_s, z_s = _project(Ks[s], Rs[s], ts[s], X)
        ui, vi = np.round(uv_s[:, 0]).astype(int), np.round(uv_s[:, 1]).astype(int)
        inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z_s > 0)
        ok = np.zeros(len(X), bool)
        idx = np.flatnonzero(inside)
        ds = Ds[vi[idx], ui[idx]]
        good = (confs[s][vi[idx], ui[idx]] > cfg.min_conf) & (ds > 0)
        idx, ds = idx[good], ds[good]
        # Back-project the source pixel with the source's depth, then re-project into the ref.
        xc = np.c_[uv_s[idx], np.ones(len(idx))] @ np.linalg.inv(Ks[s]).T * ds[:, None]
        Xs = (xc - ts[s]) @ Rs[s]
        uv_r, z_r = _project(Ks[r], Rs[r], ts[r], Xs)
        px = np.linalg.norm(uv_r - uv[idx], axis=1)
        rel = np.abs(z_r - d_ref[idx]) / d_ref[idx]
        ok[idx] = (px < cfg.max_px) & (rel < cfg.max_rel_depth)
        count += ok
    return valid, X, count


def fuse(images, depths, confs, Ks, Rs, ts, sources, cfg: FusionConfig):
    """All views -> (points (N, 3), colours (N, 3) uint8, camera centre per point (N, 3))."""
    pts, cols, cams = [], [], []
    stats = {}
    for r in depths:
        valid, X, count = consistency_counts(r, depths, confs, Ks, Rs, ts, sources[r], cfg)
        keep = count >= cfg.min_views
        rgb = images[r][..., ::-1][valid][keep]
        pts.append(X[keep])
        cols.append(rgb)
        cams.append(np.repeat((-Rs[r].T @ ts[r])[None], keep.sum(), 0))
        stats[int(r)] = {"confident": int(valid.sum()), "consistent": int(keep.sum()),
                         "pixels": int(valid.size)}
    return np.concatenate(pts), np.concatenate(cols).astype(np.uint8), np.concatenate(cams), stats


def to_o3d(points, colors=None):
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return pc


def voxel_merge(points, colors, cams, voxel: float):
    """Average points, colours and viewing camera per voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    def mean(a):
        out = np.zeros((len(cnt), a.shape[1]))
        np.add.at(out, inv, a)
        return out / cnt[:, None]

    return mean(points), mean(colors.astype(float)).astype(np.uint8), mean(cams)


def poisson_mesh(points, colors, cams, voxel: float, depth: int = 10, trim_quantile: float = 0.04):
    """Screened Poisson surface + density trimming. Returns an Open3D TriangleMesh."""
    import open3d as o3d

    pc = to_o3d(points, colors)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    n = np.asarray(pc.normals)
    flip = np.sum(n * (cams - points), axis=1) < 0  # normals must face the camera that saw them
    n[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(n)
    # n_threads=1: the multi-threaded Open3D 0.20 build aborted ("Failed to close loop") on
    # the ETH3D pipes cloud on macOS; single-threaded is slower but stable.
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth, n_threads=1)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_quantile))
    # Poisson also extrapolates surface far from the data: drop vertices > 3 voxels from a point.
    from scipy.spatial import cKDTree

    v = np.asarray(mesh.vertices)
    far = cKDTree(points).query(v, k=1, workers=-1)[0] > 3 * voxel
    mesh.remove_vertices_by_mask(far)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh
