"""Dense reconstruction: plane-sweep MVS -> consistency fusion -> Poisson mesh -> evaluation.

    mv3d-dense --scene eth3d/pipes --poses gt       # dense stage alone (ground-truth cameras)
    mv3d-dense --scene eth3d/pipes --poses ours     # full pipeline on my SfM cameras
    mv3d-dense --scene middlebury/templeRing --poses colmap

With ``--poses ours|colmap`` the SfM result (models/sfm/<scene>_<engine>.npz) is first moved
into the ground-truth frame with the similarity found by ``mv3d-sfm`` (only when ground truth
exists), so distances are in metres and the laser scan can be used directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mvrecon.datasets import load_scene
from mvrecon.evaluate import apply_sim3, cloud_metrics, load_scan
from mvrecon.fusion import FusionConfig, fuse, poisson_mesh, to_o3d, voxel_merge
from mvrecon.mvs import MVSConfig, depth_range, device, plane_sweep, select_sources
from mvrecon.run_sfm import load_reconstruction


def get_cameras(scene, poses: str):
    """(R, t dicts, sparse points) in the GT frame when possible."""
    sim3 = None
    if poses != "gt":
        rec = load_reconstruction(Path("models/sfm") / f"{scene.name}_{poses}.npz")
        res = Path("reports/sfm/results.json")
        if scene.R_gt is not None and res.exists():
            sim3 = json.loads(res.read_text())[scene.name][poses]["poses"]["sim3"]
        R, t, X = dict(rec.R), dict(rec.t), rec.X
        if sim3 is not None:  # X' = s Rs X + ts  =>  R' = R Rs^T, t' = s t - R' ts
            Rs, ts, s = np.asarray(sim3["R"]), np.asarray(sim3["t"]), sim3["s"]
            for i in R:
                R[i] = R[i] @ Rs.T
                t[i] = s * t[i] - R[i] @ ts
            X = apply_sim3(X, sim3)
        return R, t, X
    R = dict(enumerate(scene.R_gt))
    t = dict(enumerate(scene.t_gt))
    X = scene.gt_points
    if X is None:  # Middlebury: GT poses but no GT sparse points -> use COLMAP's, aligned
        _, _, X = get_cameras(scene, "colmap")
    return R, t, X


def depth_gallery(scene, views, imgs, depths, confs, keep_masks, path: Path, cfg: FusionConfig):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(views), 4, figsize=(15, 2.9 * len(views)))
    for row, r in zip(np.atleast_2d(axes), views, strict=True):
        d = depths[r]
        lo, hi = np.percentile(d, [2, 98])
        filt = np.where(keep_masks[r], d, np.nan)
        row[0].imshow(imgs[r][..., ::-1])
        row[1].imshow(d, cmap="turbo", vmin=lo, vmax=hi)
        row[2].imshow(confs[r], cmap="gray", vmin=-0.2, vmax=1)
        row[3].imshow(filt, cmap="turbo", vmin=lo, vmax=hi)
        row[0].set_ylabel(scene.names[r], fontsize=8)
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    titles = ["reference image", "raw plane-sweep depth (winner-takes-all)",
              "ZNCC confidence (white = sure)",
              f"after fusion filter (conf > {cfg.min_conf}, >= {cfg.min_views} views agree)"]
    for a, tt in zip(np.atleast_2d(axes)[0], titles, strict=True):
        a.set_title(tt, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def error_figure(points, dist, path: Path, title: str, tau: float = 0.05):
    """Top view of the fused cloud coloured by distance to the laser scan."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mu = points.mean(0)
    _, _, Vt = np.linalg.svd(points[:: max(1, len(points) // 20000)] - mu, full_matrices=False)
    p2 = (points - mu) @ Vt[:2].T
    order = np.random.default_rng(0).permutation(len(dist))  # no colour drawn on top of another
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(p2[order, 0], p2[order, 1], c=np.minimum(dist[order], 2 * tau), s=0.3,
                    cmap="RdYlGn_r", vmin=0, vmax=2 * tau, rasterized=True)
    fig.colorbar(sc, ax=ax, label=f"distance to laser scan [m] (clipped at {2 * tau:g})")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--poses", choices=["gt", "ours", "colmap"], default="gt")
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--n-depths", type=int, default=192)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--voxel-px", type=float, default=1.0,
                    help="voxel size as a multiple of the median pixel footprint")
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("reports/dense"))
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached depth maps (outputs/dense) and only redo fusion + meshing")
    ap.add_argument("--tag", default="", help="suffix for result keys (fusion ablations)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.scene, max_side=args.max_side)
    tag = f"{scene.name}_{args.poses}"
    cache = Path("outputs/dense") / f"{tag}_{args.max_side}_{args.n_depths}.npz"
    tag += args.tag
    mcfg = MVSConfig(n_depths=args.n_depths)
    fcfg = FusionConfig(min_conf=args.min_conf, min_views=args.min_views)
    R, t, Xs = get_cameras(scene, args.poses)
    views = sorted(R)
    imgs = {i: scene.image(i) for i in views}
    grays = {i: cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 for i in views}
    depths, confs, sources = {}, {}, {}
    dev = device()
    t0 = time.perf_counter()
    if args.reuse and cache.exists():
        z = np.load(cache, allow_pickle=True)
        depths, confs, sources = z["depths"].item(), z["confs"].item(), z["sources"].item()
    for r in tqdm([] if depths else views, desc=f"plane sweep {tag}"):
        src = select_sources(r, scene.K, R, t, Xs, scene.size, mcfg)
        if not src:
            continue
        dmin, dmax = depth_range(scene.K[r], R[r], t[r], Xs, scene.size)
        depths[r], confs[r] = plane_sweep(
            grays[r], [grays[s] for s in src], scene.K[r], R[r], t[r],
            [scene.K[s] for s in src], [R[s] for s in src], [t[s] for s in src],
            dmin, dmax, mcfg, dev)
        sources[r] = src
    t_mvs = time.perf_counter() - t0
    if not args.reuse or not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, depths=np.array(depths, dtype=object), confs=np.array(confs, dtype=object),
                 sources=np.array(sources, dtype=object))
    pts, cols, cams, stats = fuse(imgs, depths, confs, scene.K, R, t, sources, fcfg)
    # Voxel = median pixel footprint (depth / focal) x voxel_px: unit-free across datasets.
    med_depth = np.median([np.median(d) for d in depths.values()])
    voxel = float(args.voxel_px * med_depth / scene.K[0][0, 0])
    pts, cols, cams = voxel_merge(pts, cols, cams, voxel)
    t_fuse = time.perf_counter() - t0 - t_mvs
    import open3d as o3d

    mdir = Path("models/dense")
    mdir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(mdir / f"{tag}_fused.ply"), to_o3d(pts, cols))
    mesh = poisson_mesh(pts, cols, cams, voxel, depth=args.poisson_depth)
    o3d.io.write_triangle_mesh(str(mdir / f"{tag}_mesh.ply"), mesh)
    t_mesh = time.perf_counter() - t0 - t_mvs - t_fuse
    result = {
        "scene": scene.name, "poses": args.poses, "views": len(depths),
        "image_size": list(scene.size), "device": str(dev),
        "mvs": vars(mcfg), "fusion": vars(fcfg), "voxel_m": voxel,
        "fused_points": int(len(pts)), "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "kept_fraction": float(sum(s["consistent"] for s in stats.values())
                               / sum(s["pixels"] for s in stats.values())),
        "seconds": {"plane_sweep": t_mvs, "fusion": t_fuse, "poisson": t_mesh,
                    "per_view_sweep": t_mvs / max(len(depths), 1)},
    }
    figures = not args.tag  # ablation runs only add numbers
    gallery_views = views[:: max(1, len(views) // 3)][:3] if figures else []
    keep_masks = {}
    from mvrecon.fusion import consistency_counts

    for r in gallery_views:
        valid, _, count = consistency_counts(r, depths, confs, scene.K, R, t, sources[r], fcfg)
        m = np.zeros_like(valid)
        m[valid] = count >= fcfg.min_views
        keep_masks[r] = m
    if figures:
        depth_gallery(scene, gallery_views, imgs, depths, confs, keep_masks,
                      args.out / f"depth_{tag}.jpg", fcfg)
    if scene.scan_paths:
        from scipy.spatial import cKDTree

        gt = load_scan(scene, voxel=0.01)
        result["cloud"] = cloud_metrics(pts, gt)
        mesh_pts = np.asarray(mesh.sample_points_uniformly(max(len(pts), 100000)).points)
        result["mesh"] = cloud_metrics(mesh_pts, gt)
        dist = cKDTree(gt).query(pts, workers=-1)[0]
        c = result["cloud"]["tau_0.05"]
        if figures:
            error_figure(pts, dist, args.out / f"error_{tag}.png",
                         f"{scene.name}, {args.poses} cameras: fused cloud coloured by error "
                         f"(acc@5cm {c['accuracy']:.2f}, comp@5cm {c['completeness']:.2f})")
    res_path = args.out / "results.json"
    allres = json.loads(res_path.read_text()) if res_path.exists() else {}
    if args.reuse and tag in allres:  # keep the timing of the run that did the sweep
        result["seconds"]["plane_sweep"] = allres[tag]["seconds"]["plane_sweep"]
        result["seconds"]["per_view_sweep"] = allres[tag]["seconds"]["per_view_sweep"]
    allres[tag] = result
    res_path.write_text(json.dumps(allres, indent=1, default=float))
    summary = {k: result[k] for k in ("fused_points", "mesh_triangles", "kept_fraction")}
    if "cloud" in result:
        summary |= {f"{w}@{tau}": round(result[w][f"tau_{tau}"]["f1"], 3)
                    for w in ("cloud", "mesh") for tau in ("0.02", "0.05", "0.1")}
    print(tag, summary, {k: round(v, 1) for k, v in result["seconds"].items()})


if __name__ == "__main__":
    main()
