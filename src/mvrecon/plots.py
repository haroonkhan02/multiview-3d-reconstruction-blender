"""Summary figures and markdown tables from the results JSON files.

    mv3d-plots      # reads reports/{twoview,sfm,dense}/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ETH3D = ["pipes", "meadow", "terrace", "relief"]
TAUS = ["0.02", "0.05", "0.1", "0.2"]


def sfm_figure(res: dict, path: Path) -> None:
    scenes = [s for s in ETH3D + ["templeRing"] if s in res]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(scenes))
    for k, (eng, color) in enumerate([("ours", "tab:red"), ("colmap", "tab:blue")]):
        reg = [res[s][eng]["poses"]["registered"] / res[s][eng]["log"]["images"] for s in scenes]
        pos = [res[s][eng]["poses"]["pos_err_median_rel"] * 100 for s in scenes]
        ax[0].bar(x + (k - 0.5) * 0.38, reg, 0.38, color=color, label=eng)
        ax[1].bar(x + (k - 0.5) * 0.38, pos, 0.38, color=color, label=eng)
    ax[0].set_ylabel("cameras registered (fraction)")
    ax[1].set_ylabel("median camera position error\n[% of camera-path extent]")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xticks(x, scenes)
        a.grid(alpha=0.3, axis="y")
        a.legend()
    fig.suptitle("My incremental SfM vs COLMAP (same images, same known intrinsics)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dense_figure(res: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    taus = [float(t) * 100 for t in TAUS]
    for s, color in zip(ETH3D, plt.cm.tab10.colors, strict=False):
        for poses, ls in (("gt", "-"), ("ours", "--")):
            r = res.get(f"{s}_{poses}")
            if r is None or "cloud" not in r:
                continue
            for a, key in zip(ax, ("accuracy", "completeness", "f1"), strict=True):
                a.plot(taus, [r["cloud"][f"tau_{t}"][key] for t in TAUS], ls, marker="o",
                       color=color, label=f"{s} ({poses} cameras)")
    for a, title in zip(ax, ("accuracy", "completeness", "F1"), strict=True):
        a.set_xscale("log")
        a.set_xticks(taus, [f"{t:g}" for t in taus])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel("distance threshold tau [cm]")
        a.set_title(title)
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
    ax[2].legend(fontsize=7)
    fig.suptitle("Fused plane-sweep cloud vs ETH3D laser scan "
                 "(solid: ground-truth cameras, dashed: my SfM cameras)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ablation_figure(res: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = ["| Scene | min views | min ZNCC | accuracy@5cm | completeness@5cm | F1@5cm |",
            "|---|---|---|---|---|---|"]
    for a, s in zip(ax, ("pipes", "relief"), strict=True):
        for v, marker in zip((1, 2, 3), ("o", "s", "^"), strict=True):
            pts = []
            for c in ("0.3", "0.5", "0.7"):
                r = res.get(f"{s}_gt_v{v}_c{c}")
                if r is None:
                    continue
                m = r["cloud"]["tau_0.05"]
                pts.append((m["completeness"], m["accuracy"]))
                a.annotate(c, (m["completeness"], m["accuracy"]), fontsize=7,
                           xytext=(4, 2), textcoords="offset points")
                rows.append(f"| {s} | {v} | {c} | {m['accuracy']:.3f} | {m['completeness']:.3f} "
                            f"| {m['f1']:.3f} |")
            if pts:
                p = np.array(pts)
                a.plot(p[:, 0], p[:, 1], marker=marker, label=f">= {v} agreeing views")
        a.set_xlabel("completeness @ 5 cm")
        a.set_ylabel("accuracy @ 5 cm")
        a.set_title(f"{s}: fusion filter trade-off (labels = min ZNCC)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return rows


def main() -> None:
    root = Path("reports")
    tables = []
    if (root / "sfm/results.json").exists():
        res = json.loads((root / "sfm/results.json").read_text())
        sfm_figure(res, root / "sfm/sfm_summary.png")
        tables += ["## SfM",
                   "| Scene | Engine | Registered | Points | Track length | Reproj. [px] "
                   "| Rot. err. median [deg] | Pos. err. median / max [cm] | Time [s] |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for s, engines in res.items():
            for e, r in engines.items():
                lg, p = r["log"], r.get("poses", {})
                tables.append(
                    f"| {s} | {e} | {lg['registered']}/{lg['images']} | {lg['points']} | "
                    f"{lg['mean_track_length']:.2f} | {lg['mean_reproj_px']:.2f} | "
                    f"{p.get('rot_err_median_deg', float('nan')):.3f} | "
                    f"{p.get('pos_err_median_m', float('nan')) * 100:.2f} / "
                    f"{p.get('pos_err_max_m', float('nan')) * 100:.1f} | "
                    f"{lg['wall_seconds']:.0f} |")
    if (root / "dense/results.json").exists():
        res = json.loads((root / "dense/results.json").read_text())
        dense_figure(res, root / "dense/dense_summary.png")
        tables += ["", "## Dense",
                   "| Run | Points | Kept | Acc@2/5/10 cm | Comp@2/5/10 cm | F1@2/5/10 cm "
                   "| Mesh F1@5 cm | Sweep s/view |", "|---|---|---|---|---|---|---|---|"]
        for k, r in res.items():
            if "cloud" not in r or "_v" in k:
                continue
            acc, comp, f1 = ("/".join(f"{r['cloud'][f'tau_{t}'][key]:.2f}" for t in TAUS[:3])
                             for key in ("accuracy", "completeness", "f1"))
            tables.append(f"| {k} | {r['fused_points']:,} | {r['kept_fraction']:.2f} | "
                          f"{acc} | {comp} | {f1} | "
                          f"{r['mesh']['tau_0.05']['f1']:.2f} | "
                          f"{r['seconds']['per_view_sweep']:.1f} |")
        tables += ["", "## Fusion ablation (ground-truth cameras, cached depth maps)"]
        tables += ablation_figure(res, root / "dense/fusion_ablation.png")
    (root / "tables.md").write_text("\n".join(tables) + "\n")
    print("\n".join(tables))


if __name__ == "__main__":
    main()
