"""Comparing reconstructions with ground truth: camera poses and dense point clouds.

Poses. A reconstruction is only defined up to a similarity transform (position, orientation and
scale of the world are arbitrary), so the estimated camera centres are first aligned to the
ground-truth centres with Umeyama's closed-form least squares. Then, per registered image:

* rotation error: angle of R_est_aligned * R_gt^T (degrees);
* position error: distance between aligned and true camera centres (metres on ETH3D).

Point clouds (a simplified version of the ETH3D benchmark protocol, see the README):

* accuracy@tau: fraction of reconstructed points within tau of the laser scan;
* completeness@tau: fraction of laser-scan points within tau of the reconstruction;
* F1@tau: their harmonic mean.

The official ETH3D evaluator also ignores reconstructed points in space the scanner never saw;
this simplified version only crops to the scan's bounding box, so accuracy here is a lower bound.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mvrecon import geometry as g


def align_and_compare(
    R_est: dict[int, np.ndarray],
    t_est: dict[int, np.ndarray],
    R_gt: list[np.ndarray],
    t_gt: list[np.ndarray],
) -> dict:
    ids = sorted(R_est)
    C_est = np.array([g.camera_center(R_est[i], t_est[i]) for i in ids])
    C_gt = np.array([g.camera_center(R_gt[i], t_gt[i]) for i in ids])
    s, Rs, ts = g.umeyama(C_est, C_gt, with_scale=True)
    # Least squares lets one badly registered camera tilt the whole alignment (on relief a
    # single 54 cm outlier added ~0.5 deg to EVERY camera's rotation error). Refit on the
    # cameras within 3x the median error; all cameras are still scored.
    used = np.ones(len(ids), bool)
    for _ in range(3):
        pos = np.linalg.norm(s * C_est @ Rs.T + ts - C_gt, axis=1)
        keep = pos <= max(3 * np.median(pos), 1e-9)
        if keep.sum() < 3 or (keep == used).all():
            break
        used = keep
        s, Rs, ts = g.umeyama(C_est[used], C_gt[used], with_scale=True)
    C_al = s * C_est @ Rs.T + ts
    pos = np.linalg.norm(C_al - C_gt, axis=1)
    # world_est -> world_gt is X' = s Rs X + ts, so camera rotations become R_est Rs^T.
    rot = np.array([g.rotation_error_deg(R_est[i] @ Rs.T, R_gt[i]) for i in ids])
    extent = float(np.linalg.norm(C_gt.max(0) - C_gt.min(0)))
    return {
        "registered": len(ids),
        "alignment_inliers": int(used.sum()),
        "rot_err_median_deg": float(np.median(rot)),
        "rot_err_max_deg": float(rot.max()),
        "pos_err_median_m": float(np.median(pos)),
        "pos_err_max_m": float(pos.max()),
        "pos_err_median_rel": float(np.median(pos) / max(extent, 1e-9)),
        "camera_extent_m": extent,
        "sim3": {"s": float(s), "R": Rs.tolist(), "t": ts.tolist()},
        "per_image": {int(i): {"rot_deg": float(r), "pos_m": float(p)}
                      for i, r, p in zip(ids, rot, pos, strict=True)},
    }


def apply_sim3(X: np.ndarray, sim3: dict) -> np.ndarray:
    return sim3["s"] * X @ np.asarray(sim3["R"]).T + np.asarray(sim3["t"])


def cloud_metrics(
    rec: np.ndarray,
    gt: np.ndarray,
    taus=(0.02, 0.05, 0.10, 0.20),
    crop_margin: float = 0.1,
) -> dict:
    """Accuracy / completeness / F1 of a reconstructed cloud against a GT scan (same frame)."""
    lo, hi = gt.min(0) - crop_margin, gt.max(0) + crop_margin
    inside = np.all((rec >= lo) & (rec <= hi), axis=1)
    rec_c = rec[inside]
    out = {"points": int(len(rec)), "points_in_scan_box": int(len(rec_c)), "gt_points": len(gt)}
    if len(rec_c) == 0:
        return out | {f"tau_{t}": {"accuracy": 0.0, "completeness": 0.0, "f1": 0.0} for t in taus}
    d_acc = cKDTree(gt).query(rec_c, k=1, workers=-1)[0]
    d_comp = cKDTree(rec_c).query(gt, k=1, workers=-1)[0]
    out["median_dist_to_gt_m"] = float(np.median(d_acc))
    for tau in taus:
        a, c = float((d_acc < tau).mean()), float((d_comp < tau).mean())
        out[f"tau_{tau:g}"] = {"accuracy": a, "completeness": c,
                               "f1": 2 * a * c / (a + c) if a + c > 0 else 0.0}
    return out


def load_scan(scene, voxel: float = 0.01) -> np.ndarray:
    """All laser scans of a Scene in the image frame, voxel-downsampled (Open3D)."""
    import open3d as o3d

    pts = []
    for path, M in zip(scene.scan_paths, scene.scan_transforms, strict=True):
        pc = o3d.io.read_point_cloud(str(path)).voxel_down_sample(voxel)
        P = np.asarray(pc.points)
        pts.append(P @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(pts)
