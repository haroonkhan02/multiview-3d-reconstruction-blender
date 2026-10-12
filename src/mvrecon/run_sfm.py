"""Run my incremental SfM and COLMAP on the same scenes and compare both with ground truth.

    mv3d-sfm --scenes eth3d/pipes eth3d/meadow eth3d/terrace eth3d/relief middlebury/templeRing

Writes per scene ``models/sfm/<scene>_<engine>.npz`` (poses + sparse points, used by the MVS
stage) and ``reports/sfm/results.json`` + ``sfm_<scene>.png`` (cameras and points, top view).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mvrecon import geometry as g
from mvrecon.datasets import load_scene
from mvrecon.evaluate import align_and_compare
from mvrecon.features import extract_scene
from mvrecon.sfm import IncrementalSfM, Reconstruction, SfMConfig, match_all_pairs


def save_reconstruction(rec: Reconstruction, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(sorted(rec.R))
    np.savez(path, ids=ids, R=np.array([rec.R[i] for i in ids]),
             t=np.array([rec.t[i] for i in ids]), X=rec.X, colors=rec.colors)


def load_reconstruction(path: Path) -> Reconstruction:
    z = np.load(path)
    rec = Reconstruction(X=z["X"], colors=z["colors"])
    for i, R, t in zip(z["ids"], z["R"], z["t"], strict=True):
        rec.R[int(i)], rec.t[int(i)] = R, t
    return rec


def write_ply(path: Path, X: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Minimal binary-free ASCII PLY writer (no Open3D needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["ply", "format ascii 1.0", f"element vertex {len(X)}",
            "property float x", "property float y", "property float z"]
    if colors is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n")
        for k, p in enumerate(X):
            line = f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}"
            if colors is not None:
                line += " {} {} {}".format(*colors[k])
            f.write(line + "\n")


def cached_pair_matches(scene, feats, cfg: SfMConfig) -> dict:
    """Verified pairwise matches, cached per scene + matching settings (the slow part)."""
    key = f"{scene.name}_{scene.size[0]}_r{cfg.ratio}_px{cfg.ransac_px}_it{cfg.ransac_iters}"
    path = Path("outputs/cache") / f"pairs_{key}.npz"
    if path.exists():
        z = np.load(path)
        return {tuple(int(v) for v in k.split("_")): z[k] for k in z.files}
    pairs = match_all_pairs(feats, scene.K, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{i}_{j}": m for (i, j), m in pairs.items()})
    return pairs


def plot_scene(scene, recs: dict[str, tuple[Reconstruction, dict]], path: Path) -> None:
    """Top view (GT frame) of the aligned cameras of every engine plus my sparse points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mvrecon.evaluate import apply_sim3

    C_gt = np.array([g.camera_center(R, t) for R, t in zip(scene.R_gt, scene.t_gt, strict=True)])
    # Plot in the plane spanned by the two largest principal axes of the GT camera centres.
    ref = next(iter(recs.values()))
    pts = apply_sim3(ref[0].X, ref[1]["sim3"])
    mu = C_gt.mean(0)
    _, _, Vt = np.linalg.svd(np.vstack([C_gt, pts]) - mu, full_matrices=False)
    proj = lambda P: (P - mu) @ Vt[:2].T  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lim = np.percentile(np.abs(proj(pts)), 98, axis=0) * 1.15
    p2 = proj(pts)
    ax.scatter(p2[:, 0], p2[:, 1], s=0.4, c=ref[0].colors / 255.0, alpha=0.6, rasterized=True)
    ax.scatter(*proj(C_gt).T, marker="o", s=70, facecolors="none", edgecolors="k",
               label="ground-truth cameras")
    marks = {"ours": ("x", "tab:red"), "colmap": ("+", "tab:blue")}
    for name, (rec, m) in recs.items():
        C = np.array([g.camera_center(rec.R[i], rec.t[i]) for i in sorted(rec.R)])
        ax.scatter(*proj(apply_sim3(C, m["sim3"])).T, marker=marks[name][0], s=60,
                   color=marks[name][1],
                   label=f"{name}: {m['registered']}/{len(scene)} cams, "
                         f"median {m['pos_err_median_m'] * 100:.1f} cm")
    ax.set_xlim(-lim[0], lim[0])
    ax.set_ylim(-lim[1], lim[1])
    ax.set_aspect("equal")
    ax.set_title(f"{scene.name}: cameras and my sparse points (aligned to GT)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["eth3d/pipes", "eth3d/meadow", "eth3d/terrace", "eth3d/relief",
                             "middlebury/templeRing"])
    ap.add_argument("--engines", nargs="+", default=["ours", "colmap"])
    ap.add_argument("--out", type=Path, default=Path("reports/sfm"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    res_path = args.out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for spec in args.scenes:
        scene = load_scene(spec)
        recs = {}
        for engine in args.engines:
            t0 = time.perf_counter()
            if engine == "ours":
                feats = extract_scene(scene, cache_dir="outputs/cache")
                cfg = SfMConfig()
                pairs = cached_pair_matches(scene, feats, cfg)
                rec = IncrementalSfM(feats, scene.K, cfg).run(pairs, verbose=False)
            else:
                from mvrecon.colmap_sfm import run_colmap

                rec = run_colmap(scene, Path("outputs/colmap") / scene.name)
            rec.log["wall_seconds"] = time.perf_counter() - t0
            save_reconstruction(rec, Path("models/sfm") / f"{scene.name}_{engine}.npz")
            write_ply(Path("models/sfm") / f"{scene.name}_{engine}_sparse.ply", rec.X, rec.colors)
            m = align_and_compare(rec.R, rec.t, scene.R_gt, scene.t_gt)
            recs[engine] = (rec, m)
            results.setdefault(scene.name, {})[engine] = {"log": rec.log, "poses": m}
            print(f"{scene.name:12s} {engine:6s}: {m['registered']}/{len(scene)} registered, "
                  f"{rec.log['points']} pts, rot {m['rot_err_median_deg']:.3f} deg, "
                  f"pos {m['pos_err_median_m'] * 100:.2f} cm (max {m['pos_err_max_m'] * 100:.1f}), "
                  f"reproj {rec.log['mean_reproj_px']:.2f} px, {rec.log['wall_seconds']:.0f} s")
            res_path.write_text(json.dumps(results, indent=1, default=float))
        plot_scene(scene, recs, args.out / f"sfm_{scene.name}.png")


if __name__ == "__main__":
    main()
