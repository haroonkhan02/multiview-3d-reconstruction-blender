"""Incremental Structure-from-Motion written from scratch (calibrated cameras).

1. SIFT in every image; match every pair; keep the RANSAC inliers of my essential-matrix solver.
2. Chain pairwise matches into **tracks** with union-find (one track = one physical point);
   drop tracks that contain two different keypoints of the same image (inconsistent).
3. **Initialise** from the pair with the most inliers among those whose median triangulation
   angle is >= ``init_min_angle`` (a wide enough baseline for well-conditioned depth).
4. **Register** the next image with PnP + RANSAC on its 2D-3D correspondences (OpenCV's
   solver; the 2D-3D matching and track bookkeeping are mine), refined with Levenberg-Marquardt.
5. **Triangulate** every track that now has >= 2 registered observations; keep points with a
   small reprojection error and a triangulation angle >= ``min_tri_angle``.
6. **Bundle adjust** all cameras and points (bundle.py), drop observations with a large error,
   and repeat 4-6 until no more images can be registered.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from tqdm import tqdm

from mvrecon import geometry as g
from mvrecon.bundle import bundle_adjust
from mvrecon.features import Features, match


@dataclass
class SfMConfig:
    ratio: float = 0.8
    ransac_px: float = 1.5
    ransac_iters: int = 20000
    min_pair_inliers: int = 30
    init_min_angle: float = 5.0
    min_tri_angle: float = 2.0
    max_reproj_px: float = 4.0
    pnp_min_inliers: int = 30
    huber_px: float = 1.0
    ba_nfev: int = 40
    seed: int = 0


@dataclass
class Reconstruction:
    R: dict[int, np.ndarray] = field(default_factory=dict)
    t: dict[int, np.ndarray] = field(default_factory=dict)
    X: np.ndarray | None = None  # (P, 3)
    colors: np.ndarray | None = None  # (P, 3) uint8
    track_of_point: np.ndarray | None = None
    observations: list | None = None  # per point: list of (img, keypoint)
    log: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- matching
def match_all_pairs(feats: list[Features], K: list[np.ndarray], cfg: SfMConfig):
    """Geometrically verified matches for every image pair: {(i, j): (M, 2) keypoint indices}."""
    out = {}
    pairs = list(itertools.combinations(range(len(feats)), 2))
    for i, j in tqdm(pairs, desc="match + verify", leave=False):
        m = match(feats[i], feats[j], ratio=cfg.ratio)
        if len(m) < cfg.min_pair_inliers:
            continue
        x1, x2 = feats[i].xy[m[:, 0]].astype(float), feats[j].xy[m[:, 1]].astype(float)
        _, mask = g.ransac_essential(x1, x2, K[i], K[j], cfg.ransac_px, cfg.ransac_iters,
                                     seed=cfg.seed)
        if mask.sum() >= cfg.min_pair_inliers:
            out[(i, j)] = m[mask]
    return out


def build_tracks(n_kps: list[int], pair_matches: dict) -> list[list[tuple[int, int]]]:
    """Union-find over (image, keypoint) nodes; returns consistent tracks of length >= 2."""
    offset = np.cumsum([0] + n_kps)
    parent = np.arange(offset[-1])

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    for (i, j), m in pair_matches.items():
        for a, b in zip(offset[i] + m[:, 0], offset[j] + m[:, 1], strict=True):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list[int]] = {}
    used = np.unique(np.concatenate([np.r_[offset[i] + m[:, 0], offset[j] + m[:, 1]]
                                     for (i, j), m in pair_matches.items()]))
    for node in used:
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        imgs = np.searchsorted(offset, nodes, side="right") - 1
        if len(nodes) < 2 or len(set(imgs.tolist())) != len(nodes):
            continue  # one image would see the same point twice: an inconsistent track
        tracks.append([(int(im), int(n - offset[im])) for im, n in zip(imgs, nodes, strict=True)])
    return tracks


# ----------------------------------------------------------------------------- the pipeline
class IncrementalSfM:
    def __init__(self, feats: list[Features], K: list[np.ndarray], cfg: SfMConfig | None = None):
        self.feats, self.K, self.cfg = feats, K, cfg or SfMConfig()
        self.n = len(feats)

    def run(self, pair_matches: dict | None = None, verbose: bool = True) -> Reconstruction:
        t0 = time.perf_counter()
        cfg = self.cfg
        if pair_matches is None:
            pair_matches = match_all_pairs(self.feats, self.K, cfg)
        t_match = time.perf_counter() - t0
        self.tracks = build_tracks([len(f.xy) for f in self.feats], pair_matches)
        # For each image: keypoint -> track id.
        self.kp_track = [dict() for _ in range(self.n)]
        for tid, tr in enumerate(self.tracks):
            for im, kp in tr:
                self.kp_track[im][kp] = tid
        self.X = np.full((len(self.tracks), 3), np.nan)
        self.R, self.t = {}, {}
        i, j = self._initialise(pair_matches)
        self._triangulate_new()
        self._bundle()
        failed: set[int] = set()
        while True:
            nxt = self._next_image(failed)
            if nxt is None:
                break
            if self._register(nxt):
                # New points may now make earlier failures registrable: retry them.
                failed.clear()
                self._triangulate_new()
                self._bundle()
                if verbose:
                    print(f"  registered {nxt:2d}: {len(self.R)}/{self.n} images, "
                          f"{np.isfinite(self.X[:, 0]).sum()} points")
            else:
                failed.add(nxt)
        self._bundle(final=True)
        rec = self._export()
        rec.log.update({"init_pair": [int(i), int(j)], "n_tracks": len(self.tracks),
                        "verified_pairs": len(pair_matches), "match_seconds": t_match,
                        "total_seconds": time.perf_counter() - t0})
        return rec

    # -- steps
    def _initialise(self, pair_matches):
        cands = sorted(pair_matches.items(), key=lambda kv: -len(kv[1]))
        best = None
        for (i, j), m in cands[:20]:
            x1 = self.feats[i].xy[m[:, 0]].astype(float)
            x2 = self.feats[j].xy[m[:, 1]].astype(float)
            E, mask = g.ransac_essential(x1, x2, self.K[i], self.K[j], self.cfg.ransac_px,
                                         self.cfg.ransac_iters, seed=self.cfg.seed)
            R, t, _ = g.recover_pose(E, g.normalize_points(x1[mask], self.K[i]),
                                     g.normalize_points(x2[mask], self.K[j]))
            P1 = g.projection_matrix(self.K[i], np.eye(3), np.zeros(3))
            X = g.triangulate(P1, g.projection_matrix(self.K[j], R, t), x1[mask], x2[mask])
            ang = np.median(g.triangulation_angles(np.zeros(3), g.camera_center(R, t), X))
            if ang >= self.cfg.init_min_angle:
                best = (i, j, R, t)
                break
            if best is None:
                best = (i, j, R, t)
        i, j, R, t = best
        self.R[i], self.t[i] = np.eye(3), np.zeros(3)
        self.R[j], self.t[j] = R, t
        return i, j

    def _next_image(self, failed):
        best, best_n = None, 0
        for im in range(self.n):
            if im in self.R or im in failed:
                continue
            n = sum(np.isfinite(self.X[tid, 0]) for tid in self.kp_track[im].values())
            if n > best_n:
                best, best_n = im, n
        return best if best_n >= self.cfg.pnp_min_inliers else None

    def _register(self, im) -> bool:
        kps = [(kp, tid) for kp, tid in self.kp_track[im].items() if np.isfinite(self.X[tid, 0])]
        obj = np.array([self.X[tid] for _, tid in kps])
        img = np.array([self.feats[im].xy[kp] for kp, _ in kps], dtype=float)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, self.K[im], None, reprojectionError=self.cfg.max_reproj_px,
            iterationsCount=2000, confidence=0.9999, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < self.cfg.pnp_min_inliers:
            return False
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K[im], None, rvec, tvec)
        self.R[im], self.t[im] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        return True

    def _triangulate_new(self):
        """Triangulate every untriangulated track with >= 2 registered observations."""
        cfg = self.cfg
        for tid, tr in enumerate(self.tracks):
            if np.isfinite(self.X[tid, 0]):
                continue
            obs = [(im, kp) for im, kp in tr if im in self.R]
            if len(obs) < 2:
                continue
            Ps = [g.projection_matrix(self.K[im], self.R[im], self.t[im]) for im, _ in obs]
            xs = [self.feats[im].xy[kp].astype(float) for im, kp in obs]
            X = g.triangulate_multi(Ps, xs)
            errs = [g.reprojection_error(self.K[im], self.R[im], self.t[im], X[None], x[None])[0]
                    for (im, _), x in zip(obs, xs, strict=True)]
            deps = [g.depths(self.R[im], self.t[im], X[None])[0] for im, _ in obs]
            Cs = np.array([g.camera_center(self.R[im], self.t[im]) for im, _ in obs])
            ang = max(g.triangulation_angles(Cs[a], Cs[b], X[None])[0]
                      for a, b in itertools.combinations(range(len(Cs)), 2))
            if max(errs) < cfg.max_reproj_px and min(deps) > 0 and ang >= cfg.min_tri_angle:
                self.X[tid] = X

    def _observations(self):
        cams, pts, uv = [], [], []
        for tid, tr in enumerate(self.tracks):
            if not np.isfinite(self.X[tid, 0]):
                continue
            for im, kp in tr:
                if im in self.R:
                    cams.append(im)
                    pts.append(tid)
                    uv.append(self.feats[im].xy[kp])
        return np.array(cams), np.array(pts), np.array(uv, dtype=float)

    def _bundle(self, final: bool = False):
        cams, pts, uv = self._observations()
        reg = sorted(self.R)
        cmap = {im: k for k, im in enumerate(reg)}
        live = np.unique(pts)
        pmap = -np.ones(len(self.tracks), int)
        pmap[live] = np.arange(len(live))
        res = bundle_adjust(
            np.array([self.R[im] for im in reg]), np.array([self.t[im] for im in reg]),
            np.array([self.K[im] for im in reg]), self.X[live],
            np.array([cmap[c] for c in cams]), pmap[pts], uv,
            fixed=(0,), huber_px=self.cfg.huber_px,
            max_nfev=self.cfg.ba_nfev * (3 if final else 1),
        )
        for k, im in enumerate(reg):
            self.R[im], self.t[im] = res.R[k], res.t[k]
        self.X[live] = res.X
        # Points that still reproject badly in any registered view (or end up behind a camera)
        # are removed; their tracks may be re-triangulated later from better poses.
        Rs = np.array([self.R[c] for c in cams])
        ts = np.array([self.t[c] for c in cams])
        Ks = np.array([self.K[c] for c in cams])
        xc = np.einsum("nij,nj->ni", Rs, self.X[pts]) + ts
        depth = xc[:, 2]
        pix = np.einsum("nij,nj->ni", Ks, xc)
        err = np.linalg.norm(pix[:, :2] / pix[:, 2:] - uv, axis=1)
        bad = np.unique(pts[(err > self.cfg.max_reproj_px) | (depth <= 0)])
        self.X[bad] = np.nan
        self.last_ba = {"rmse_before": res.rmse_before, "rmse_after": res.rmse_after,
                        "removed_points": int(len(bad)), "nfev": res.nfev}

    def _export(self) -> Reconstruction:
        live = np.flatnonzero(np.isfinite(self.X[:, 0]))
        colors, obs_lists, errs = [], [], []
        for tid in live:
            obs = [(im, kp) for im, kp in self.tracks[tid] if im in self.R]
            obs_lists.append(obs)
            cs = [self.feats[im].colors[kp] for im, kp in obs if self.feats[im].colors is not None]
            colors.append(np.mean(cs, 0) if cs else np.array([200, 200, 200]))
            errs += [g.reprojection_error(self.K[im], self.R[im], self.t[im], self.X[tid][None],
                                          self.feats[im].xy[kp][None].astype(float))[0]
                     for im, kp in obs]
        rec = Reconstruction(R=dict(self.R), t=dict(self.t), X=self.X[live],
                             colors=np.array(colors, np.uint8), track_of_point=live,
                             observations=obs_lists)
        rec.log = {"registered": len(self.R), "images": self.n, "points": int(len(live)),
                   "mean_track_length": float(np.mean([len(o) for o in obs_lists])),
                   "mean_reproj_px": float(np.mean(errs)),
                   "median_reproj_px": float(np.median(errs)),
                   "last_ba": self.last_ba}
        return rec
