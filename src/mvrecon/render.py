"""Render a reconstructed mesh in Blender (headless) and encode MP4 + GIF with ffmpeg.

    mv3d-render --scene middlebury/templeRing --poses colmap --sweep 360
    mv3d-render --scene eth3d/relief --poses gt --sweep 50

The up and front directions come from the cameras that took the photos: "up" is minus the
average camera y axis (image rows point down), "front" points from the mesh centre to the
average camera centre. Blender is found via $BLENDER, the macOS app bundle or the PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

BLENDER_CANDIDATES = ("/Applications/Blender.app/Contents/MacOS/Blender", "blender")


def find_blender() -> str:
    for c in ([os.environ["BLENDER"]] if "BLENDER" in os.environ else []) + list(
        BLENDER_CANDIDATES
    ):
        path = shutil.which(c) or (c if Path(c).exists() else None)
        if path:
            return path
    raise FileNotFoundError("Blender not found: install it or set $BLENDER")


def view_frame(R: dict, t: dict, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(up, front) from world-to-camera rotations: camera y axis in world = R[1]."""
    up = -np.mean([R[i][1] for i in R], axis=0)
    C = np.mean([-R[i].T @ t[i] for i in R], axis=0)
    front = C - center
    return up / np.linalg.norm(up), front / np.linalg.norm(front)


def encode(frames: Path, mp4: Path, gif: Path | None, fps: int = 24, gif_width: int = 640):
    pattern = str(frames / "frame_%04d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    if gif is not None:
        vf = (f"fps=12,scale={gif_width}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        pattern, "-vf", vf, str(gif)], check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--poses", default="gt")
    ap.add_argument("--mesh", type=Path, default=None, help="default: models/dense/<tag>_mesh.ply")
    ap.add_argument("--sweep", type=float, default=360)
    ap.add_argument("--elevation", type=float, default=20)
    ap.add_argument("--distance", type=float, default=2.4)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--res", type=int, nargs=2, default=[960, 720])
    ap.add_argument("--gif", type=Path, default=None, help="also write a GIF here")
    args = ap.parse_args(argv)

    import open3d as o3d

    from mvrecon.datasets import load_scene
    from mvrecon.dense import get_cameras

    scene = load_scene(args.scene)
    tag = f"{scene.name}_{args.poses}"
    mesh_path = args.mesh or Path("models/dense") / f"{tag}_mesh.ply"
    V = np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices)
    center = np.median(V, axis=0)
    radius = float(np.percentile(np.linalg.norm(V - center, axis=1), 90))
    R, t, _ = get_cameras(scene, args.poses)
    up, front = view_frame(R, t, center)
    out = Path("outputs/render") / tag
    frames = out / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    script = Path(__file__).with_name("blender_turntable.py")
    cmd = [find_blender(), "--background", "--python", str(script), "--",
           "--mesh", str(mesh_path.resolve()), "--out", str(frames.resolve()),
           "--up", *map(str, up), "--front", *map(str, front), "--center", *map(str, center),
           "--radius", str(radius), "--sweep", str(args.sweep), "--elevation",
           str(args.elevation), "--distance", str(args.distance), "--frames", str(args.frames),
           "--res", *map(str, args.res)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    encode(frames, out / f"{tag}_turntable.mp4", args.gif)
    print(f"rendered {args.frames} frames -> {out / f'{tag}_turntable.mp4'}"
          + (f" and {args.gif}" if args.gif else ""))


if __name__ == "__main__":
    main()
