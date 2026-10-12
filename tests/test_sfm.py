"""Bundle adjustment, track building and the full incremental SfM on a synthetic scene."""

import numpy as np

from mvrecon.bundle import bundle_adjust
from mvrecon.evaluate import align_and_compare
from mvrecon.features import Features
from mvrecon.sfm import IncrementalSfM, SfMConfig, build_tracks
from tests.conftest import make_scene


def synthetic_features(obs, n_points, seed=0):
    """One unique random descriptor per 3D point, shared by all views (perfect matches)."""
    rng = np.random.default_rng(seed)
    desc = rng.random((n_points, 128)).astype(np.float32)
    out = []
    for k, x in enumerate(obs):
        order = np.random.default_rng(seed + k + 1).permutation(n_points)  # shuffle keypoints
        out.append(Features(x[order].astype(np.float32), desc[order],
                            np.full((n_points, 3), 128, np.uint8)))
    return out


def test_bundle_adjustment_reduces_reprojection_error():
    K, cams, X, obs = make_scene(n_views=4, n_points=120, noise_px=0.3, seed=2)
    rng = np.random.default_rng(0)
    R = np.array([c[0] for c in cams])
    t = np.array([c[1] for c in cams])
    t[1:] += rng.normal(0, 0.02, (3, 3))  # camera 0 is the fixed gauge camera: leave it exact
    Xn = X + rng.normal(0, 0.02, X.shape)
    cam_idx = np.repeat(np.arange(4), len(X))
    pt_idx = np.tile(np.arange(len(X)), 4)
    uv = np.concatenate(obs)
    # Huber with a far-off start needs ~70 evaluations (only ~4 with a plain squared loss).
    res = bundle_adjust(R, t, np.repeat(K[None], 4, 0), Xn, cam_idx, pt_idx, uv, max_nfev=200)
    assert res.rmse_after < res.rmse_before / 5
    assert res.rmse_after < 0.6  # close to the 0.3 px noise level
    assert np.allclose(res.R[0], R[0]) and np.allclose(res.t[0], t[0])  # gauge camera fixed


def test_build_tracks_chains_pairs_and_rejects_conflicts():
    pairs = {(0, 1): np.array([[0, 5], [1, 6]]), (1, 2): np.array([[5, 9], [6, 7]]),
             (0, 2): np.array([[1, 8]])}  # keypoint 1 of image 0 -> image 2 kps 7 and 8: conflict
    tracks = build_tracks([3, 10, 10], pairs)
    as_sets = [set(t) for t in tracks]
    assert {(0, 0), (1, 5), (2, 9)} in as_sets
    assert all((0, 1) not in s for s in as_sets)  # conflicting track dropped


def test_incremental_sfm_recovers_all_cameras():
    K, cams, X, obs = make_scene(n_views=6, n_points=250, noise_px=0.3, seed=4)
    feats = synthetic_features(obs, len(X))
    cfg = SfMConfig(ransac_iters=500, init_min_angle=3.0, min_tri_angle=1.0, pnp_min_inliers=20)
    rec = IncrementalSfM(feats, [K] * 6, cfg).run(verbose=False)
    assert rec.log["registered"] == 6
    m = align_and_compare(rec.R, rec.t, [c[0] for c in cams], [c[1] for c in cams])
    assert m["rot_err_median_deg"] < 0.1
    assert m["pos_err_median_m"] < 0.01  # scene is ~4 m across
    assert rec.log["mean_reproj_px"] < 0.6
    assert rec.log["points"] > 200


def test_align_and_compare_is_zero_for_a_similarity_copy():
    K, cams, X, obs = make_scene(n_views=5)
    from scipy.spatial.transform import Rotation

    s = 0.37

    Q = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    ts = np.array([0.5, -1.0, 2.0])
    R_est, t_est = {}, {}
    for i, (R, t) in enumerate(cams):  # X' = s Q X + ts  =>  R' = R Q^T, t' = s t - R' ts
        R_est[i] = R @ Q.T
        t_est[i] = s * t - R_est[i] @ ts
    m = align_and_compare(R_est, t_est, [c[0] for c in cams], [c[1] for c in cams])
    assert m["rot_err_max_deg"] < 1e-4
    assert m["pos_err_max_m"] < 1e-9
