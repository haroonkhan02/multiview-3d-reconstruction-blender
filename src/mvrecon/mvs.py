"""Plane-sweep multi-view stereo in PyTorch (runs on the Apple GPU through MPS).

For every reference view:

1. **Neighbours**: pick the source views that see many of the same sparse SfM points from a
   useful angle (5-40 degrees between viewing rays; too small = no depth signal, too large =
   the patches look different).
2. **Depth hypotheses**: ``n_depths`` planes parallel to the reference image, uniform in
   *inverse* depth between the 2nd and 98th percentile of the sparse points' depths (widened by
   20 %). Uniform in 1/d means equal steps in pixel disparity, so near and far are treated alike.
3. **Warp**: for each plane, back-project every reference pixel to that depth, project it into
   each source view and sample the source image there (bilinear ``grid_sample``).
4. **Photo-consistency**: zero-mean normalised cross-correlation (ZNCC) between the reference
   patch and the warped source patch (``window`` x ``window``), computed with box filters. ZNCC
   ignores gain and offset changes between photos. Pixels that warp outside a source score -1.
5. **Aggregate** over sources by averaging the best ``top_k`` scores per pixel and depth, so a
   view in which the point is occluded does not veto it.
6. **Winner-takes-all** over depth, with a parabola through the best score and its neighbours
   for sub-plane precision. The best score is kept as a confidence.

Known limitation: fronto-parallel windows assume the surface faces the camera, so slanted
surfaces (floors, walls seen at grazing angles) get noisier depth. PatchMatch-style methods
(COLMAP, ACMM) also estimate a normal per pixel to remove that bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mvrecon import geometry as g


@dataclass
class MVSConfig:
    n_depths: int = 192
    n_sources: int = 5
    top_k: int = 2
    window: int = 7
    depth_chunk: int = 16
    min_angle: float = 5.0
    max_angle: float = 40.0


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def visible_points(K, R, t, X, size) -> np.ndarray:
    """Boolean mask of sparse points that project inside the image with positive depth."""
    xc = X @ R.T + t
    z = xc[:, 2]
    uv = (xc @ K.T)[:, :2] / np.maximum(z[:, None], 1e-9)
    w, h = size
    return (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


def select_sources(r, Ks, Rs, ts, X, size, cfg: MVSConfig) -> list[int]:
    vis = {i: visible_points(Ks[i], Rs[i], ts[i], X, size) for i in Rs}
    Cr = g.camera_center(Rs[r], ts[r])
    scores = {}
    for s in Rs:
        if s == r:
            continue
        common = vis[r] & vis[s]
        if common.sum() < 10:
            continue
        ang = g.triangulation_angles(Cr, g.camera_center(Rs[s], ts[s]), X[common])
        good = (ang >= cfg.min_angle) & (ang <= cfg.max_angle)
        scores[s] = good.sum() + 0.1 * common.sum()
    return [s for s, _ in sorted(scores.items(), key=lambda kv: -kv[1])[: cfg.n_sources]]


def depth_range(K, R, t, X, size) -> tuple[float, float]:
    vis = visible_points(K, R, t, X, size)
    z = (X[vis] @ R.T + t)[:, 2]
    if len(z) < 10:
        raise ValueError("too few sparse points to set a depth range")
    lo, hi = np.percentile(z, [2, 98])
    return lo * 0.8, hi * 1.2


def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def zncc(ref: torch.Tensor, src: torch.Tensor, k: int) -> torch.Tensor:
    """ZNCC of every k x k window. ref (1,1,H,W) broadcast against src (D,1,H,W) -> (D,H,W)."""
    mu_r, mu_s = _box(ref, k), _box(src, k)
    var_r = _box(ref * ref, k) - mu_r**2
    var_s = _box(src * src, k) - mu_s**2
    cov = _box(ref * src, k) - mu_r * mu_s
    return (cov / torch.sqrt(var_r.clamp_min(1e-6) * var_s.clamp_min(1e-6)))[:, 0]


def plane_sweep(
    ref_img: np.ndarray,
    src_imgs: list[np.ndarray],
    K_ref: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    K_src: list[np.ndarray],
    R_src: list[np.ndarray],
    t_src: list[np.ndarray],
    dmin: float,
    dmax: float,
    cfg: MVSConfig,
    dev: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth map (H, W) and ZNCC confidence (H, W) for one reference view (grey float images)."""
    dev = dev or device()
    H, W = ref_img.shape
    to_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    ref = to_t(ref_img)[None, None]
    srcs = [to_t(s)[None, None] for s in src_imgs]
    inv = torch.linspace(1.0 / dmin, 1.0 / dmax, cfg.n_depths, device=dev)
    depths = 1.0 / inv  # near -> far, uniform in inverse depth
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                          torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    rays = torch.stack([u, v, torch.ones_like(u)], 0).reshape(3, -1)
    rays = to_t(np.linalg.inv(K_ref)) @ rays  # (3, HW): depth-1 points in the ref camera
    # Precompute per source: pixel(d) = A * d + b in homogeneous source pixels.
    proj = []
    for Ks, Rs, ts in zip(K_src, R_src, t_src, strict=True):
        R_rel, t_rel = g.relative_pose(R_ref, t_ref, Rs, ts)
        proj.append((to_t(Ks @ R_rel) @ rays, to_t(Ks @ t_rel)[:, None]))
    k = min(cfg.top_k, len(srcs))
    agg = torch.empty((cfg.n_depths, H, W), device=dev)
    # Depth chunks outside, sources inside: only one chunk of per-source scores is in memory.
    for d0 in range(0, cfg.n_depths, cfg.depth_chunk):
        d = depths[d0 : d0 + cfg.depth_chunk]
        chunk = []
        for src, (A, b) in zip(srcs, proj, strict=True):
            Hs, Ws = src.shape[-2:]
            p = A[None] * d[:, None, None] + b[None]  # (D, 3, HW)
            z = p[:, 2].clamp_min(1e-6)
            gx = p[:, 0] / z / (Ws - 1) * 2 - 1
            gy = p[:, 1] / z / (Hs - 1) * 2 - 1
            grid = torch.stack([gx, gy], -1).reshape(len(d), H, W, 2)
            warped = F.grid_sample(src.expand(len(d), -1, -1, -1), grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=True)
            sc = zncc(ref, warped, cfg.window)
            inside = (gx.abs() <= 1) & (gy.abs() <= 1) & (p[:, 2] > 0)
            chunk.append(torch.where(inside.reshape(len(d), H, W), sc, -1.0))
        agg[d0 : d0 + len(d)] = torch.stack(chunk).topk(k, dim=0).values.mean(0)
    best, idx = agg.max(0)
    # Sub-plane refinement: vertex of the parabola through (idx-1, idx, idx+1), in inverse depth.
    i0 = (idx - 1).clamp(0, cfg.n_depths - 1)
    i2 = (idx + 1).clamp(0, cfg.n_depths - 1)
    c0 = agg.gather(0, i0[None])[0]
    c2 = agg.gather(0, i2[None])[0]
    den = c0 - 2 * best + c2
    off = torch.where(den.abs() > 1e-6, 0.5 * (c0 - c2) / den, torch.zeros_like(den)).clamp(-1, 1)
    interior = (idx > 0) & (idx < cfg.n_depths - 1)
    off = torch.where(interior, off, torch.zeros_like(off))
    step = inv[1] - inv[0]
    inv_best = inv[idx] + off * step
    depth = 1.0 / inv_best
    return depth.cpu().numpy(), best.cpu().numpy()
