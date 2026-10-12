"""Readers for the calibrated multi-view datasets and for plain photo folders.

Every reader returns a :class:`Scene`: image paths plus, when known, per-image intrinsics and
ground-truth world-to-camera poses. Images are processed at a reduced size (``max_side``);
the intrinsics are scaled to match, so all downstream code works in the reduced pixel grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Scene:
    name: str
    paths: list[Path]
    K: list[np.ndarray]  # per image, already scaled to the processing size
    size: tuple[int, int]  # (width, height) at the processing size
    R_gt: list[np.ndarray] | None = None
    t_gt: list[np.ndarray] | None = None
    gt_points: np.ndarray | None = None  # (N, 3) GT sparse points (ETH3D) for depth ranges
    scan_paths: list[Path] = field(default_factory=list)
    scan_transforms: list[np.ndarray] = field(default_factory=list)
    covis: np.ndarray | None = None  # (n, n) number of shared GT points

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.paths]

    def image(self, i: int, gray: bool = False) -> np.ndarray:
        return load_image(self.paths[i], self.size, gray)


def load_image(path: Path, size: tuple[int, int], gray: bool = False) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def scaled_size(w: int, h: int, max_side: int) -> tuple[tuple[int, int], float]:
    s = min(1.0, max_side / max(w, h))
    return (round(w * s), round(h * s)), s


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    """Scale intrinsics for an image resized by s (pixel centres stay consistent)."""
    K = K.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    # Normalise first: files store ~7 decimals, and an unnormalised quaternion gives a
    # scaled (non-orthonormal) matrix whose trace can exceed 3.
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_colmap_text(calib_dir: Path):
    """Parse COLMAP cameras.txt / images.txt / points3D.txt (PINHOLE cameras only)."""
    cams = {}
    for line in (calib_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cid, model, w, h, *p = line.split()
        if model != "PINHOLE":
            raise ValueError(f"expected undistorted PINHOLE cameras, got {model}")
        fx, fy, cx, cy = map(float, p)
        cams[cid] = (int(w), int(h), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
    images = []
    lines = [ln for ln in (calib_dir / "images.txt").read_text().splitlines()
             if not ln.startswith("#")]
    for head, pts in zip(lines[0::2], lines[1::2], strict=False):
        f = head.split()
        q, t = np.array(f[1:5], float), np.array(f[5:8], float)
        obs = np.array(pts.split(), dtype=float).reshape(-1, 3) if pts.strip() else np.zeros((0, 3))
        ids = obs[:, 2].astype(np.int64)
        images.append({"name": f[9], "camera": f[8], "R": qvec_to_rotmat(q), "t": t,
                       "point_ids": set(ids[ids >= 0].tolist())})
    xyz = []
    for line in (calib_dir / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        xyz.append([float(v) for v in line.split()[1:4]])
    return cams, images, np.asarray(xyz)


def read_mlp(path: Path) -> list[tuple[str, np.ndarray]]:
    """MeshLab project: (ply filename, 4x4 transform) for each laser scan."""
    import xml.etree.ElementTree as ET

    out = []
    for mesh in ET.parse(path).getroot().iter("MLMesh"):
        M = np.array(mesh.find("MLMatrix44").text.split(), float).reshape(4, 4)
        out.append((mesh.get("filename"), M))
    return out


def load_eth3d(root: Path, scene: str, max_side: int = 1600) -> Scene:
    base = Path(root) / "eth3d" / scene
    cams, images, xyz = read_colmap_text(base / "dslr_calibration_undistorted")
    images.sort(key=lambda d: d["name"])
    w, h, _ = cams[images[0]["camera"]]
    size, s = scaled_size(w, h, max_side)
    n = len(images)
    covis = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            covis[i, j] = covis[j, i] = len(images[i]["point_ids"] & images[j]["point_ids"])
    scans = read_mlp(base / "dslr_scan_eval" / "scan_alignment.mlp")
    return Scene(
        name=scene,
        paths=[base / "images" / d["name"] for d in images],
        K=[scale_K(cams[d["camera"]][2], s) for d in images],
        size=size,
        R_gt=[d["R"] for d in images],
        t_gt=[d["t"] for d in images],
        gt_points=xyz,
        scan_paths=[base / "dslr_scan_eval" / f for f, _ in scans],
        scan_transforms=[M for _, M in scans],
        covis=covis,
    )


def load_middlebury(root: Path, name: str = "templeRing", max_side: int = 640) -> Scene:
    """Middlebury multi-view: ``<name>/<prefix>_par.txt`` lists K (9), R (9), t (3) per image."""
    base = Path(root) / "middlebury" / name
    par = next(base.glob("*_par.txt"))
    rows = [ln.split() for ln in par.read_text().splitlines()[1:] if ln.strip()]
    first = cv2.imread(str(base / rows[0][0]))
    size, s = scaled_size(first.shape[1], first.shape[0], max_side)
    return Scene(
        name=name,
        paths=[base / r[0] for r in rows],
        K=[scale_K(np.array(r[1:10], float).reshape(3, 3), s) for r in rows],
        size=size,
        R_gt=[np.array(r[10:19], float).reshape(3, 3) for r in rows],
        t_gt=[np.array(r[19:22], float) for r in rows],
    )


def load_folder(folder: Path, max_side: int = 1600, focal_factor: float = 1.2) -> Scene:
    """A folder of your own photos: no ground truth, intrinsics guessed (f = 1.2 * max side).

    The guess is only a starting value; COLMAP refines it during bundle adjustment.
    """
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix in exts)
    if len(paths) < 3:
        raise ValueError(f"need at least 3 images in {folder}")
    first = cv2.imread(str(paths[0]))
    size, _ = scaled_size(first.shape[1], first.shape[0], max_side)
    f = focal_factor * max(size)
    K = np.array([[f, 0, (size[0] - 1) / 2], [0, f, (size[1] - 1) / 2], [0, 0, 1.0]])
    return Scene(name=Path(folder).name, paths=paths, K=[K] * len(paths), size=size)


def load_scene(spec: str, root: Path = Path("data"), max_side: int | None = None) -> Scene:
    """``eth3d/<scene>``, ``middlebury/<name>`` or a path to a folder of photos."""
    kind, _, name = spec.partition("/")
    if kind == "eth3d":
        return load_eth3d(root, name, max_side or 1600)
    if kind == "middlebury":
        return load_middlebury(root, name, max_side or 640)
    return load_folder(Path(spec), max_side or 1600)
