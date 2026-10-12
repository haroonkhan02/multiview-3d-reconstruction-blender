"""Download the public multi-view datasets used by the project.

* **ETH3D high-resolution multi-view (training scenes)**: undistorted DSLR images
  (about 6200 x 4140 JPEG) with ground-truth intrinsics and poses in COLMAP text format, plus the
  laser-scan point cloud used for evaluation. CC BY-NC-SA 4.0, no account needed.
  https://www.eth3d.net/datasets
* **Middlebury multi-view "templeRing"** (Seitz et al. 2006): 47 calibrated 640 x 480 views
  of a plaster temple on a turntable. Used for the object-centred demo and the Blender render.
  https://vision.middlebury.edu/mview/data/

Usage::

    mv3d-download eth3d --scenes pipes meadow terrace relief
    mv3d-download temple
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

ETH3D_URL = "https://www.eth3d.net/data/{scene}_{kind}.7z"
ETH3D_KINDS = ("dslr_undistorted", "dslr_scan_eval")
TEMPLE_URL = "https://vision.middlebury.edu/mview/data/data/templeRing.zip"
DEFAULT_SCENES = ("pipes", "meadow", "terrace", "relief")


def fetch(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream `url` to `dest` with a progress bar; resumes via a .part file and skips if present."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if done and resp.status != 206:  # server ignored the range: start again
            done = 0
            part.unlink()
        total = int(resp.headers.get("Content-Length", 0)) + done
        with open(part, "ab") as f, tqdm(total=total, initial=done, unit="B", unit_scale=True,
                                         desc=dest.name) as bar:
            while block := resp.read(chunk):
                f.write(block)
                bar.update(len(block))
    part.rename(dest)
    return dest


def extract_7z(archive: Path, out_dir: Path) -> None:
    import py7zr

    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(out_dir)


def download_eth3d(root: Path, scenes: list[str], keep_archives: bool = False) -> None:
    for scene in scenes:
        scene_dir = root / "eth3d" / scene
        for kind in ETH3D_KINDS:
            marker = scene_dir / f".{kind}.done"
            if marker.exists():
                print(f"{scene}/{kind}: already extracted")
                continue
            archive = fetch(ETH3D_URL.format(scene=scene, kind=kind),
                            root / "eth3d" / "_archives" / f"{scene}_{kind}.7z")
            # Archives contain a top-level "<scene>/" folder.
            extract_7z(archive, root / "eth3d")
            marker.touch()
            if not keep_archives:
                archive.unlink()
            print(f"{scene}/{kind}: extracted to {scene_dir}")
    shutil.rmtree(root / "eth3d" / "_archives", ignore_errors=True)


def download_temple(root: Path) -> None:
    out = root / "middlebury"
    if (out / "templeRing" / "templeR_par.txt").exists():
        print("templeRing: already extracted")
        return
    archive = fetch(TEMPLE_URL, out / "templeRing.zip")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    archive.unlink()
    print(f"templeRing: extracted to {out / 'templeRing'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", choices=["eth3d", "temple", "all"])
    ap.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args(argv)
    if args.dataset in ("eth3d", "all"):
        download_eth3d(args.root, args.scenes, args.keep_archives)
    if args.dataset in ("temple", "all"):
        download_temple(args.root)


if __name__ == "__main__":
    main()
