#!/bin/sh
# Reproduce every number and figure in the README (Apple M5: about 1.5 h in total).
set -e
mv3d-download eth3d --scenes pipes meadow terrace relief      # ~1.5 GB download, ~2.1 GB extracted
mv3d-download temple                                          # 11 MB

# 1. Two-view solvers vs ground truth (48 pairs x 3 seeds)
mv3d-twoview

# 2. My incremental SfM vs COLMAP on all five scenes
mv3d-sfm
mv3d-sfm --eval-only   # re-score with the robust alignment (same numbers as a fresh run)
# Meadow with relaxed thresholds (chosen after looking at meadow: reported separately)
mv3d-sfm --scenes eth3d/meadow --engines ours --pnp-min-inliers 20 --min-tri-angle 1.5 --suffix _relaxed

# 3. Dense stage alone (ground-truth cameras), then the full pipeline on SfM cameras
for s in pipes meadow terrace relief; do mv3d-dense --scene eth3d/$s --poses gt; done
for s in pipes terrace relief; do mv3d-dense --scene eth3d/$s --poses ours; done
mv3d-dense --scene middlebury/templeRing --poses ours --max-side 640

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
mv3d-render --scene middlebury/templeRing --poses ours --sweep 360 --frames 120 --min-brightness 0.15 \
  --elevation 20 --distance 3.4 --gif assets/temple_turntable.gif
mv3d-render --scene eth3d/relief --poses ours --sweep 50 --frames 96 --elevation 15 --distance 2.0 \
  --gif assets/relief_swing.gif
mv3d-plots
