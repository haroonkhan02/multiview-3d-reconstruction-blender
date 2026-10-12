"""COLMAP (via pycolmap) incremental SfM on the same resized images, as the reference system.

When the scene's intrinsics are known (ETH3D, Middlebury) COLMAP gets them as a fixed PINHOLE
camera, so both systems solve the same calibrated problem. For your own photos COLMAP starts
from a guessed focal length and refines it (SIMPLE_RADIAL model, one shared camera).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from mvrecon.sfm import Reconstruction


def write_resized(scene, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(scene.paths):
        dst = out_dir / p.name
        if not dst.exists():
            cv2.imwrite(str(dst), scene.image(i), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_dir


def run_colmap(scene, work: Path, known_intrinsics: bool = True, seed: int = 0) -> Reconstruction:
    import pycolmap

    img_dir = write_resized(scene, work / "images")
    db = work / "database.db"
    if db.exists():
        db.unlink()
    shutil.rmtree(work / "sparse", ignore_errors=True)
    reader = pycolmap.ImageReaderOptions()
    same_K = all(np.allclose(K, scene.K[0]) for K in scene.K)
    if known_intrinsics and same_K:
        K = scene.K[0]
        reader.camera_model = "PINHOLE"
        reader.camera_params = ",".join(f"{v:.6f}" for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    else:
        reader.camera_model = "SIMPLE_RADIAL"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max(scene.size)
    t0 = time.perf_counter()
    pycolmap.extract_features(db, img_dir, image_names=scene.names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extraction, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    t_match = time.perf_counter() - t0
    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = seed
    if known_intrinsics and same_K:
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
    (work / "sparse").mkdir(parents=True, exist_ok=True)
    recs = pycolmap.incremental_mapping(db, img_dir, work / "sparse", opts)
    total = time.perf_counter() - t0
    if not recs:
        raise RuntimeError("COLMAP registered no images")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    best.write(work / "sparse")
    return to_reconstruction(best, scene, {"match_seconds": t_match, "total_seconds": total,
                                           "models": len(recs)})


def to_reconstruction(rc, scene, log: dict) -> Reconstruction:
    name_to_idx = {n: i for i, n in enumerate(scene.names)}
    rec = Reconstruction()
    for img in rc.images.values():
        if not img.has_pose:
            continue
        T = img.cam_from_world()
        M = T.matrix()
        idx = name_to_idx[img.name]
        rec.R[idx], rec.t[idx] = M[:, :3].copy(), M[:, 3].copy()
    pts = list(rc.points3D.values())
    rec.X = np.array([p.xyz for p in pts])
    rec.colors = np.array([p.color for p in pts], np.uint8)
    track_len = [p.track.length() for p in pts]
    cam = next(iter(rc.cameras.values()))
    rec.log = {"registered": len(rec.R), "images": len(scene), "points": len(pts),
               "mean_track_length": float(np.mean(track_len)),
               "mean_reproj_px": float(rc.compute_mean_reprojection_error()),
               "camera": {"model": str(cam.model), "params": [float(v) for v in cam.params]},
               **log}
    return rec
