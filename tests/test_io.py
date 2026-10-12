"""Dataset readers, pair selection, the pose-AUC metric and the render helpers."""

import numpy as np
import pytest

from mvrecon.datasets import qvec_to_rotmat, read_colmap_text, read_mlp, scale_K, scaled_size
from mvrecon.render import view_frame
from mvrecon.twoview import choose_pairs, pose_auc


def test_qvec_identity_and_90deg():
    assert np.allclose(qvec_to_rotmat(np.array([1.0, 0, 0, 0])), np.eye(3))
    c = np.cos(np.pi / 4)
    Rz = qvec_to_rotmat(np.array([c, 0, 0, c]))
    assert np.allclose(Rz @ [1, 0, 0], [0, 1, 0])


def test_scale_K_keeps_pixel_centres():
    K = np.array([[1000.0, 0, 1999.5], [0, 1000.0, 999.5], [0, 0, 1]])
    (w, h), s = scaled_size(4000, 2000, 1000)
    assert (w, h) == (1000, 500) and s == 0.25
    Ks = scale_K(K, s)
    assert np.allclose(Ks[:2, 2], [499.5, 249.5])  # centre of the image stays the centre
    assert np.isclose(Ks[0, 0], 250)


def test_read_colmap_text_and_mlp(tmp_path):
    (tmp_path / "cameras.txt").write_text("# c\n0 PINHOLE 100 80 90 91 50 40\n")
    (tmp_path / "images.txt").write_text(
        "# i\n1 1 0 0 0 0.1 0.2 0.3 0 dslr_images/a.JPG\n10 20 5 11 21 -1\n"
        "2 1 0 0 0 0 0 0 0 dslr_images/b.JPG\n12 22 5\n")
    (tmp_path / "points3D.txt").write_text("# p\n5 1.0 2.0 3.0 255 0 0 0.1 1 0 2 0\n")
    cams, images, xyz = read_colmap_text(tmp_path)
    assert cams["0"][2][0, 0] == 90 and cams["0"][:2] == (100, 80)
    assert images[0]["name"] == "dslr_images/a.JPG"
    assert np.allclose(images[0]["t"], [0.1, 0.2, 0.3])
    assert images[0]["point_ids"] == {5} and images[1]["point_ids"] == {5}
    assert np.allclose(xyz, [[1, 2, 3]])
    mlp = tmp_path / "scan_alignment.mlp"
    mlp.write_text('<MeshLabProject><MeshGroup><MLMesh label="s" filename="scan1.ply">'
                   "<MLMatrix44>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1 </MLMatrix44></MLMesh>"
                   "</MeshGroup></MeshLabProject>")
    (name, M), = read_mlp(mlp)
    assert name == "scan1.ply" and np.allclose(M[:3, 3], [1, 2, 3])


def test_read_colmap_rejects_distorted_models(tmp_path):
    (tmp_path / "cameras.txt").write_text("0 SIMPLE_RADIAL 100 80 90 50 40 0.1\n")
    with pytest.raises(ValueError):
        read_colmap_text(tmp_path)


def test_choose_pairs_spreads_over_covisibility():
    covis = np.zeros((6, 6), int)
    for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]):
        covis[i, j] = covis[j, i] = 100 * (k + 1)
    pairs = choose_pairs(covis, 3, min_shared=50)
    assert pairs[0] == (0, 5) and pairs[-1] == (0, 1) and len(pairs) == 3


def test_pose_auc():
    assert np.isclose(pose_auc([0.0] * 10)["auc@5"], 1.0)
    assert pose_auc([100.0] * 10)["auc@5"] == 0.0
    # Uniform errors on [0, 10]: recall(e) = e / 10, AUC@10 = 0.5.
    assert abs(pose_auc(np.linspace(0, 10, 1001))["auc@10"] - 0.5) < 0.01


def test_view_frame_up_and_front():
    # Camera at (0, -5, 0) looking along +y with image y pointing down (-z world).
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    t = -R @ np.array([0, -5.0, 0])
    up, front = view_frame({0: R}, {0: t}, np.zeros(3))
    assert np.allclose(up, [0, 0, 1])
    assert np.allclose(front, [0, -1, 0])
