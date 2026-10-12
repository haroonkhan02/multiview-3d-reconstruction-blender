"""Plane-sweep MVS, fusion and point-cloud metrics on a synthetic textured plane."""

import cv2
import numpy as np
import pytest

from mvrecon.evaluate import cloud_metrics
from mvrecon.fusion import FusionConfig, backproject, consistency_counts, fuse, voxel_merge

K = np.array([[200.0, 0, 79.5], [0, 200.0, 59.5], [0, 0, 1]])
SIZE = (160, 120)
PLANE_Z = 2.0


def render_plane(R, t, texture, extent=2.0):
    """Image of the textured plane z = PLANE_Z (world frame) seen by camera (R, t)."""
    W, H = SIZE
    v, u = np.mgrid[0:H, 0:W].astype(float)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T  # camera frame
    d = rays @ R  # world-frame directions (R^T applied to each ray)
    C = -R.T @ t
    s = (PLANE_Z - C[2]) / d[..., 2]
    P = C + s[..., None] * d
    n = texture.shape[0]
    mx = ((P[..., 0] + extent) / (2 * extent) * (n - 1)).astype(np.float32)
    my = ((P[..., 1] + extent) / (2 * extent) * (n - 1)).astype(np.float32)
    return cv2.remap(texture, mx, my, cv2.INTER_LINEAR)


@pytest.fixture(scope="module")
def plane_views():
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((512, 512)).astype(np.float32), (0, 0), 2)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    cams = [(np.eye(3), np.array([-x, 0.0, 0.0])) for x in (0.0, 0.25, -0.25, 0.4)]
    return [render_plane(R, t, tex) for R, t in cams], cams


def test_plane_sweep_recovers_plane_depth(plane_views):
    torch = pytest.importorskip("torch")
    from mvrecon.mvs import MVSConfig, plane_sweep

    imgs, cams = plane_views
    cfg = MVSConfig(n_depths=64, n_sources=3, depth_chunk=16)
    depth, conf = plane_sweep(imgs[0], imgs[1:], K, *cams[0], [K] * 3,
                              [c[0] for c in cams[1:]], [c[1] for c in cams[1:]],
                              1.0, 4.0, cfg, torch.device("cpu"))
    inner = (slice(15, -15), slice(25, -25))
    assert np.median(np.abs(depth[inner] - PLANE_Z)) < 0.02
    assert np.mean(np.abs(depth[inner] - PLANE_Z) < 0.05) > 0.95
    assert np.median(conf[inner]) > 0.9


def test_zncc_is_invariant_to_gain_and_offset():
    torch = pytest.importorskip("torch")
    from mvrecon.mvs import zncc

    x = torch.rand(1, 1, 32, 32)
    s = zncc(x, (3 * x + 0.5).expand(2, 1, 32, 32), 7)
    assert torch.allclose(s[:, 5:-5, 5:-5], torch.ones(2, 22, 22), atol=1e-3)


def test_fusion_keeps_consistent_and_drops_inconsistent_depths():
    cams = {i: (np.eye(3), np.array([-x, 0.0, 0.0])) for i, x in enumerate((0.0, 0.2, -0.2))}
    Ks = {i: K for i in cams}
    Rs = {i: c[0] for i, c in cams.items()}
    ts = {i: c[1] for i, c in cams.items()}
    depths = {i: np.full(SIZE[::-1], PLANE_Z) for i in cams}
    confs = {i: np.ones(SIZE[::-1]) for i in cams}
    depths[0][:, :40] = 3.0  # a wrong (but confident) region in view 0
    cfg = FusionConfig(min_views=2)
    valid, X, count = consistency_counts(0, depths, confs, Ks, Rs, ts, [1, 2], cfg)
    cols = np.mgrid[0:SIZE[1], 0:SIZE[0]][1][valid]
    assert (count[cols < 40] < 2).all()
    assert (count[(cols > 60) & (cols < 100)] == 2).mean() > 0.95
    images = {i: np.zeros((*SIZE[::-1], 3), np.uint8) for i in cams}
    pts, colors, cam_c, stats = fuse(images, depths, confs, Ks, Rs, ts,
                                     {0: [1, 2], 1: [0, 2], 2: [0, 1]}, cfg)
    assert np.allclose(pts[:, 2], PLANE_Z, atol=1e-6)
    assert stats[0]["consistent"] < stats[0]["confident"]  # the wrong region is dropped


def test_backproject_inverts_projection():
    D = np.full((4, 5), 2.5)
    X, uv = backproject(D, K, np.eye(3), np.zeros(3))
    proj = X @ K.T
    assert np.allclose(proj[:, :2] / proj[:, 2:], uv)
    assert np.allclose(X[:, 2], 2.5)


def test_voxel_merge_averages_within_voxels():
    pts = np.array([[0.01, 0, 0], [0.03, 0, 0], [0.51, 0, 0]])
    cols = np.array([[0, 0, 0], [100, 100, 100], [50, 50, 50]], np.uint8)
    p, c, _ = voxel_merge(pts, cols, pts, voxel=0.1)
    assert len(p) == 2
    assert np.allclose(sorted(p[:, 0]), [0.02, 0.51])


def test_cloud_metrics_accuracy_and_completeness():
    gt = np.stack(np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50), [0.0]), -1)
    gt = gt.reshape(-1, 3)
    half = gt[gt[:, 0] < 0.5] + [0, 0, 0.01]  # half the surface, 1 cm off
    outliers = np.array([[0.5, 0.5, 0.08]] * 10)  # inside the scan's box, 8 cm off
    far = np.array([[0.5, 0.5, 5.0]])  # outside the box: cropped, not counted
    m = cloud_metrics(np.vstack([half, outliers, far]), gt, taus=(0.02,))
    assert m["points_in_scan_box"] == len(half) + 10
    a, c = m["tau_0.02"]["accuracy"], m["tau_0.02"]["completeness"]
    assert np.isclose(a, len(half) / (len(half) + 10))
    assert 0.48 < c < 0.55
