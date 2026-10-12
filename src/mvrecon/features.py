"""SIFT keypoints and descriptor matching (Lowe ratio test + mutual nearest neighbour)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    xy: np.ndarray  # (N, 2) float32 pixel coordinates
    desc: np.ndarray  # (N, 128) float32, RootSIFT-normalised
    colors: np.ndarray | None = None  # (N, 3) uint8 RGB at each keypoint


def root_sift(desc: np.ndarray) -> np.ndarray:
    """RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise then sqrt.

    Euclidean distance between RootSIFT vectors equals the Hellinger kernel on the original
    histograms, which is a better similarity for histograms and gives more correct matches.
    """
    desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc).astype(np.float32)


def extract(image_bgr: np.ndarray, max_features: int = 10000, contrast: float = 0.01) -> Features:
    """SIFT with a lower contrast threshold than OpenCV's 0.04 default.

    At 1600 px the default kept only ~2-3k keypoints on the ETH3D images and some wide-baseline
    pairs had < 10 correct matches; 0.01 roughly doubles the correct matches on those pairs.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    xy = np.array([k.pt for k in kps], np.float32)
    colors = None
    if image_bgr.ndim == 3:
        ij = np.round(xy).astype(int)
        ij[:, 0] = ij[:, 0].clip(0, image_bgr.shape[1] - 1)
        ij[:, 1] = ij[:, 1].clip(0, image_bgr.shape[0] - 1)
        colors = image_bgr[ij[:, 1], ij[:, 0], ::-1].copy()
    return Features(xy, root_sift(desc), colors)


def match(f1: Features, f2: Features, ratio: float = 0.8, mutual: bool = True) -> np.ndarray:
    """(M, 2) index pairs. Ratio test: best distance < ratio * second best (ambiguity filter)."""
    if len(f1.xy) < 2 or len(f2.xy) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(f1.desc, f2.desc, k=2)
    fwd = {m.queryIdx: m.trainIdx for m, n in knn if m.distance < ratio * n.distance}
    if mutual:
        back = bf.match(f2.desc, f1.desc)
        best_back = {m.queryIdx: m.trainIdx for m in back}
        fwd = {q: t for q, t in fwd.items() if best_back.get(t) == q}
    return np.array(sorted(fwd.items()), dtype=int).reshape(-1, 2)


def extract_scene(scene, max_features: int = 10000, cache_dir=None) -> list[Features]:
    """SIFT for every image of a Scene, cached as one .npz per scene and image size."""
    from pathlib import Path

    from tqdm import tqdm

    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{scene.name}_{scene.size[0]}_{max_features}.npz"
        if cache.exists():
            z = np.load(cache)
            return [Features(z[f"xy{i}"], z[f"d{i}"], z[f"c{i}"]) for i in range(len(scene))]
    feats = [extract(scene.image(i), max_features)
             for i in tqdm(range(len(scene)), desc=f"SIFT {scene.name}", leave=False)]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, f in enumerate(feats):
            arrays |= {f"xy{i}": f.xy, f"d{i}": f.desc, f"c{i}": f.colors}
        np.savez(cache, **arrays)
    return feats
