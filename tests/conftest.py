"""Synthetic multi-view scenes with known ground truth (no dataset, GPU or Blender needed)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def look_at(C: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera (R, t) for a camera at C looking at target (y axis roughly down)."""
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, -1.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def make_scene(n_views=6, n_points=300, radius=4.0, seed=0, noise_px=0.0):
    rng = np.random.default_rng(seed)
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    X = rng.uniform(-1, 1, (n_points, 3))
    cams = []
    for i in range(n_views):
        a = np.radians(-30 + 60 * i / max(n_views - 1, 1))
        C = np.array([radius * np.sin(a), 0.3 * np.cos(3 * a), -radius * np.cos(a)])
        cams.append(look_at(C, np.zeros(3)))
    obs = []
    for R, t in cams:
        xc = X @ R.T + t
        x = xc[:, :2] / xc[:, 2:] * 800 + [320, 240]
        obs.append(x + rng.normal(0, noise_px, x.shape))
    return K, cams, X, obs


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def noisy_scene():
    return make_scene(noise_px=0.5, seed=1)


def random_rotation(seed=0):
    return Rotation.random(random_state=seed).as_matrix()
