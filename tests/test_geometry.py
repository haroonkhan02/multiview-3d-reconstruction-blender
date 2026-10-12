import numpy as np
import pytest

from mvrecon import geometry as g
from tests.conftest import random_rotation


def test_hartley_transform_centres_and_scales():
    x = np.random.default_rng(0).uniform(0, 4000, (50, 2))
    p = (g.to_h(x) @ g.hartley_transform(x).T)[:, :2]
    assert np.allclose(p.mean(0), 0, atol=1e-9)
    assert np.isclose(np.linalg.norm(p, axis=1).mean(), np.sqrt(2))


def test_eight_point_fundamental_satisfies_epipolar_constraint(scene):
    K, cams, X, obs = scene
    F = g.eight_point_fundamental(obs[0], obs[2])
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    res = np.abs(np.sum(g.to_h(obs[2]) * (g.to_h(obs[0]) @ F.T), axis=1))
    assert res.max() < 1e-6
    assert g.sampson_distance(F, obs[0], obs[2]).max() < 1e-12


def test_essential_has_two_equal_singular_values(scene):
    K, cams, X, obs = scene
    E = g.eight_point_essential(g.normalize_points(obs[0], K), g.normalize_points(obs[1], K))
    s = np.linalg.svd(E, compute_uv=False)
    assert np.allclose(s, [1, 1, 0], atol=1e-9)


def test_recover_pose_matches_ground_truth(scene):
    K, cams, X, obs = scene
    (R1, t1), (R2, t2) = cams[0], cams[3]
    R_gt, t_gt = g.relative_pose(R1, t1, R2, t2)
    x1n, x2n = g.normalize_points(obs[0], K), g.normalize_points(obs[3], K)
    E = g.eight_point_essential(x1n, x2n)
    R, t, front = g.recover_pose(E, x1n, x2n)
    assert g.rotation_error_deg(R, R_gt) < 1e-6
    assert g.direction_error_deg(t, t_gt) < 1e-3  # arccos near 0 limits precision
    assert front.all()


def test_ransac_rejects_outliers(noisy_scene):
    K, cams, X, obs = noisy_scene
    x1, x2 = obs[0].copy(), obs[4].copy()
    rng = np.random.default_rng(3)
    bad = rng.choice(len(x1), 90, replace=False)  # 30 % gross outliers
    x2[bad] = rng.uniform(0, 640, (90, 2))
    E, mask = g.ransac_essential(x1, x2, K, K, thresh_px=2.0, seed=0)
    assert mask[bad].mean() < 0.05
    good = np.setdiff1d(np.arange(len(x1)), bad)
    assert mask[good].mean() > 0.9
    R_gt, t_gt = g.relative_pose(*cams[0], *cams[4])
    R, t, _ = g.recover_pose(E, g.normalize_points(x1[mask], K), g.normalize_points(x2[mask], K))
    assert g.rotation_error_deg(R, R_gt) < 0.5
    assert g.direction_error_deg(t, t_gt) < 2.0


def test_decompose_essential_returns_four_valid_rotations():
    R = random_rotation(1)
    t = np.array([1.0, 0.2, -0.3])
    for Rc, tc in g.decompose_essential(g.skew(t) @ R):
        assert np.allclose(Rc @ Rc.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(Rc), 1)
        assert np.isclose(np.linalg.norm(tc), 1)


def test_triangulation_is_exact_without_noise(scene):
    K, cams, X, obs = scene
    P1, P2 = (g.projection_matrix(K, *cams[i]) for i in (0, 5))
    assert np.allclose(g.triangulate(P1, P2, obs[0], obs[5]), X, atol=1e-8)
    Ps = [g.projection_matrix(K, *c) for c in cams]
    Xm = g.triangulate_multi(Ps, [o[7] for o in obs])
    assert np.allclose(Xm, X[7], atol=1e-8)
    assert g.reprojection_error(K, *cams[2], X, obs[2]).max() < 1e-9


def test_triangulation_angle_grows_with_baseline(scene):
    K, cams, X, obs = scene
    C = [g.camera_center(*c) for c in cams]
    a_small = g.triangulation_angles(C[0], C[1], X).mean()
    a_big = g.triangulation_angles(C[0], C[5], X).mean()
    assert 0 < a_small < a_big


@pytest.mark.parametrize("with_scale", [True, False])
def test_umeyama_recovers_similarity(with_scale):
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    R = random_rotation(2)
    s = 2.5 if with_scale else 1.0
    dst = s * src @ R.T + [1, -2, 3]
    s_e, R_e, t_e = g.umeyama(src, dst, with_scale)
    assert np.isclose(s_e, s)
    assert np.allclose(R_e, R)
    assert np.allclose(t_e, [1, -2, 3])


def test_rotation_and_direction_errors():
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    assert np.isclose(g.rotation_error_deg(R, np.eye(3)), 10)
    assert np.isclose(g.direction_error_deg(np.array([1, 0, 0]), np.array([0, 2, 0])), 90)


def test_small_rotation_errors_are_not_rounded_to_zero():
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("x", 0.01, degrees=True).as_matrix()
    assert np.isclose(g.rotation_angle_deg(R), 0.01, rtol=1e-6)
    assert np.isclose(g.rotation_angle_deg(R * (1 + 1e-6)), 0.01, rtol=1e-3)  # trace > 3


def test_qvec_is_normalised_before_conversion():
    from mvrecon.datasets import qvec_to_rotmat

    R = qvec_to_rotmat(np.array([0.9, 0.1, -0.2, 0.3]))  # not unit length
    assert np.allclose(R @ R.T, np.eye(3))
