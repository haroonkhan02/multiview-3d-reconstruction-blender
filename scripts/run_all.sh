#!/bin/sh
# Reproduce every number and figure in the README (Apple M5: about 1.5 h in total).
set -e
mv3d-download eth3d --scenes pipes meadow terrace relief      # ~2.1 GB
mv3d-download temple                                          # 11 MB

# 1. Two-view solvers vs ground truth (48 pairs x 3 seeds)
mv3d-twoview

# 2. My incremental SfM vs COLMAP on all five scenes
mv3d-sfm

# 3. Dense stage alone (ground-truth cameras), then the full pipeline on SfM cameras
for s in pipes meadow terrace relief; do mv3d-dense --scene eth3d/$s --poses gt; done
for s in pipes terrace relief; do mv3d-dense --scene eth3d/$s --poses ours; done
mv3d-dense --scene middlebury/templeRing --poses colmap --max-side 640

# 4. Fusion-filter ablation on cached depth maps (no new plane sweeps)
for s in pipes relief; do
  for v in 1 2 3; do
    for c in 0.3 0.5 0.7; do
      mv3d-dense --scene eth3d/$s --poses gt --reuse --min-views $v --min-conf $c \
        --tag "_v${v}_c${c}"
    done
  done
done

# 5. Blender turntables
mv3d-render --scene middlebury/templeRing --poses colmap --sweep 360 --gif assets/temple_turntable.gif
mv3d-render --scene eth3d/relief --poses ours --sweep 50 --gif assets/relief_swing.gif
