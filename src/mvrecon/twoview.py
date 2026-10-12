"""Two-view Structure-from-Motion benchmark: my from-scratch solver vs OpenCV's.

For image pairs of the ETH3D scenes (chosen from the ground-truth co-visibility, so they really
overlap) this runs, on the same SIFT matches:

* ``ours``         RANSAC + normalised 8-point E + Sampson refinement + cheirality (geometry.py)
* ``ours_linear``  the same without the non-linear refinement (ablation)
* ``ours_20k``     ``ours`` with the RANSAC iteration cap raised from 2,000 to 20,000
* ``opencv``       cv2.findEssentialMat (Nister 5-point inside RANSAC) + cv2.recoverPose

and compares the relative pose with the ground truth: rotation error, translation-direction
error, and the pose AUC at 5/10/20 degrees (area under the cumulative curve of
max(rotation error, direction error), the metric used by SuperGlue/LoFTR papers).

    mv3d-twoview --scenes pipes meadow terrace relief --pairs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mvrecon import geometry as g
from mvrecon.datasets import load_scene
from mvrecon.features import extract_scene, match

METHODS = ("ours", "ours_linear", "ours_20k", "opencv")


def choose_pairs(covis: np.ndarray, n_pairs: int, min_shared: int = 50) -> list[tuple[int, int]]:
    """Pairs spread over the co-visibility range (easy -> hard), deterministic."""
    iu = np.triu_indices(len(covis), 1)
    cand = [(covis[i, j], i, j) for i, j in zip(*iu, strict=True) if covis[i, j] >= min_shared]
    cand.sort(reverse=True)
    if len(cand) <= n_pairs:
        return [(i, j) for _, i, j in cand]
    pick = np.linspace(0, len(cand) - 1, n_pairs).round().astype(int)
    return [(cand[k][1], cand[k][2]) for k in pick]


def solve(method: str, x1, x2, K1, K2, thresh_px: float, seed: int = 0):
    """Return (R, t, inlier mask) of camera 2 relative to camera 1."""
    if method.startswith("ours"):
        E, mask = g.ransac_essential(x1, x2, K1, K2, thresh_px=thresh_px, seed=seed,
                                     iters=20000 if method == "ours_20k" else 2000,
                                     refine=method != "ours_linear")
        R, t, _ = g.recover_pose(E, g.normalize_points(x1[mask], K1),
                                 g.normalize_points(x2[mask], K2))
        return R, t, mask
    if method == "opencv":
        # Normalised coordinates let the two cameras have different K; the threshold is
        # converted from pixels with the mean focal length.
        x1n, x2n = g.normalize_points(x1, K1), g.normalize_points(x2, K2)
        f = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4
        cv2.setRNGSeed(seed)
        E, m = cv2.findEssentialMat(x1n, x2n, np.eye(3), cv2.RANSAC, 0.999, thresh_px / f)
        if E is None:
            raise RuntimeError("findEssentialMat failed")
        E = E[:3]
        _, R, t, m2 = cv2.recoverPose(E, x1n, x2n, np.eye(3), mask=m.copy())
        return R, t.ravel(), m.ravel().astype(bool)
    raise ValueError(method)


def pose_auc(errors: np.ndarray, thresholds=(5, 10, 20)) -> dict[str, float]:
    """Area under the recall-vs-error curve up to each threshold, normalised to [0, 1]."""
    errors = np.sort(np.asarray(errors, float))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    out = {}
    for th in thresholds:
        last = np.searchsorted(errors, th)
        e = np.r_[errors[:last], th]
        r = np.r_[recall[:last], recall[last - 1]]
        out[f"auc@{th}"] = float(np.trapezoid(r, x=e) / th)
    return out


def run_pair(scene, feats, i, j, thresh_px, ratio, seed=0):
    K1, K2 = scene.K[i], scene.K[j]
    m = match(feats[i], feats[j], ratio=ratio)
    x1, x2 = feats[i].xy[m[:, 0]].astype(float), feats[j].xy[m[:, 1]].astype(float)
    R_gt, t_gt = g.relative_pose(scene.R_gt[i], scene.t_gt[i], scene.R_gt[j], scene.t_gt[j])
    F_gt = g.fundamental_from_essential(g.skew(t_gt) @ R_gt, K1, K2)
    gt_ok = g.sampson_distance(F_gt, x1, x2) < thresh_px**2  # matches consistent with GT
    C1 = g.camera_center(scene.R_gt[i], scene.t_gt[i])
    C2 = g.camera_center(scene.R_gt[j], scene.t_gt[j])
    row = {"scene": scene.name, "i": int(i), "j": int(j), "seed": seed, "matches": len(m),
           "gt_inlier_ratio": float(gt_ok.mean()) if len(m) else 0.0,
           "baseline_m": float(np.linalg.norm(C1 - C2)),
           "view_angle_deg": g.rotation_angle_deg(R_gt)}
    for method in METHODS:
        t0 = time.perf_counter()
        try:
            R, t, mask = solve(method, x1, x2, K1, K2, thresh_px, seed)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            row[method] = {"failed": True, "err_R": 180.0, "err_t": 180.0, "err": 180.0}
            continue
        dt = time.perf_counter() - t0
        eR, et = g.rotation_error_deg(R, R_gt), g.direction_error_deg(t, t_gt)
        # Triangulate the inliers with the estimated pose to report the 2-view geometry quality.
        P1 = g.projection_matrix(K1, np.eye(3), np.zeros(3))
        P2 = g.projection_matrix(K2, R, t)
        X = g.triangulate(P1, P2, x1[mask], x2[mask])
        rep = np.r_[g.reprojection_error(K1, np.eye(3), np.zeros(3), X, x1[mask]),
                    g.reprojection_error(K2, R, t, X, x2[mask])]
        row[method] = {
            "err_R": eR, "err_t": et, "err": max(eR, et), "inliers": int(mask.sum()),
            "inlier_precision": float(gt_ok[mask].mean()) if mask.any() else 0.0,
            "reproj_median_px": float(np.median(rep)) if len(rep) else None,
            "time_ms": dt * 1000,
        }
    return row, (x1, x2)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for method in METHODS:
        err = np.array([r[method]["err"] for r in rows])
        eR = np.array([r[method]["err_R"] for r in rows])
        et = np.array([r[method]["err_t"] for r in rows])
        tm = [r[method]["time_ms"] for r in rows if "time_ms" in r[method]]
        prec = [r[method]["inlier_precision"] for r in rows if "inlier_precision" in r[method]]
        out[method] = {
            "runs": len(rows),
            **pose_auc(err),
            "median_err_R_deg": float(np.median(eR)),
            "median_err_t_deg": float(np.median(et)),
            "success_5deg": float((err < 5).mean()),
            "median_inlier_precision": float(np.median(prec)) if prec else None,
            "median_time_ms": float(np.median(tm)) if tm else None,
        }
    return out


def epipolar_figure(scene, i, j, x1, x2, path: Path, n_lines: int = 14, seed: int = 0):
    """Left: points in image i. Right: their epipolar lines in image j (estimated F, ours)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E, mask = g.ransac_essential(x1, x2, scene.K[i], scene.K[j], thresh_px=1.5)
    F = g.fundamental_from_essential(E, scene.K[i], scene.K[j])
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(mask), min(n_lines, mask.sum()), replace=False)
    im1, im2 = scene.image(i)[:, :, ::-1], scene.image(j)[:, :, ::-1]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(idx)))
    ax[0].imshow(im1)
    ax[1].imshow(im2)
    w = im2.shape[1]
    for c, k in zip(colors, idx, strict=True):
        a, b, cc = F @ np.r_[x1[k], 1]
        xs = np.array([0, w - 1])
        ax[1].plot(xs, -(a * xs + cc) / b, color=c, lw=1.2)
        ax[0].plot(*x1[k], "o", color=c, ms=6, mec="k")
        ax[1].plot(*x2[k], "o", color=c, ms=6, mec="k")
    ax[1].set_ylim(im2.shape[0], 0)
    for a_, name in zip(ax, (scene.names[i], scene.names[j]), strict=True):
        a_.set_title(name)
        a_.axis("off")
    fig.suptitle(f"ETH3D {scene.name}: {mask.sum()} RANSAC inliers of {len(x1)} SIFT matches; "
                 "each point on the left must lie on the line of the same colour on the right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def error_cdf_figure(rows: list[dict], path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = {"ours": "ours: 8-point + Sampson refinement", "ours_linear": "ours: 8-point only",
              "ours_20k": "ours, 20k RANSAC iterations", "opencv": "OpenCV: 5-point"}
    for m in METHODS:
        e = np.sort([r[m]["err"] for r in rows])
        ax.step(e, (np.arange(len(e)) + 1) / len(e), where="post", label=labels[m])
    ax.set_xscale("log")
    ax.set_xlim(0.05, 30)
    ax.set_xlabel("pose error = max(rotation, translation direction) [deg]")
    ax.set_ylabel("fraction of pairs")
    ax.set_title(f"Relative pose error on {len(rows)} ETH3D pairs")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=["pipes", "meadow", "terrace", "relief"])
    ap.add_argument("--pairs", type=int, default=12, help="pairs per scene")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--thresh-px", type=float, default=1.5)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("reports/twoview"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.scenes:
        scene = load_scene(f"eth3d/{name}", max_side=args.max_side)
        feats = extract_scene(scene, cache_dir="outputs/cache")
        pairs = choose_pairs(scene.covis, args.pairs)
        for k, (i, j) in enumerate(pairs):
            for seed in range(args.seeds):  # RANSAC is random: every pair runs with 3 seeds
                row, (x1, x2) = run_pair(scene, feats, i, j, args.thresh_px, args.ratio, seed)
                rows.append(row)
            print(f"{name} {scene.names[i]}-{scene.names[j]}: {row['matches']} matches "
                  f"({row['gt_inlier_ratio']:.0%} GT-consistent), "
                  + ", ".join(f"{m} {row[m]['err']:.2f}deg" for m in METHODS))
            if name == args.scenes[-1] and k == 0:  # an easy, well-textured pair
                epipolar_figure(scene, i, j, x1, x2, args.out / "epipolar_lines.jpg")
    summary = {"all": summarise(rows)}
    for name in args.scenes:
        summary[name] = summarise([r for r in rows if r["scene"] == name])
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "results.json").write_text(json.dumps(
        {"config": cfg, "summary": summary, "pairs": rows}, indent=1))
    error_cdf_figure(rows, args.out / "pose_error_cdf.png")
    for m in METHODS:
        s = summary["all"][m]
        print(f"{m:12s} AUC@5/10/20 {s['auc@5']:.3f}/{s['auc@10']:.3f}/{s['auc@20']:.3f}  "
              f"median R {s['median_err_R_deg']:.3f} t {s['median_err_t_deg']:.3f} deg")


if __name__ == "__main__":
    main()
