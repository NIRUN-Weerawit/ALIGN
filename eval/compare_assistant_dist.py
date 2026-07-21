"""Side-by-side comparison of assistant head output distributions across runs.

Loads the .npz arrays for each run and produces a comparison figure:
  - Per-step mean abs std (bar chart) — collapse check
  - Per-step abs mean (bar chart) — bias check
  - Per-dim std at k=0 (grouped bar chart) — per-axis collapse
  - k0 sample histogram (overlay) — shape comparison

Output: reports/assistant_dist_comparison.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch  # not used; kept for compat with the conda env's matplotlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--runs", nargs="+", default=["run_3", "run_7", "run_9", "run_10"])
    p.add_argument("--out", default="reports/assistant_dist_comparison.png")
    args = p.parse_args()

    arrays = {}
    for r in args.runs:
        path = Path(args.reports_dir) / f"assistant_dist_{r}_arrays.npz"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {r}")
            continue
        d = np.load(path)
        arrays[r] = d["preds"]   # (N, K, 6)

    if not arrays:
        raise SystemExit("no arrays found")

    runs = list(arrays.keys())
    K = next(iter(arrays.values())).shape[1]
    dim_names = ["x", "y", "z", "ax", "ay", "az"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ── (0,0) Per-step mean abs std
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(K)
    width = 0.8 / len(runs)
    for i, r in enumerate(runs):
        sds = np.array([
            [arrays[r][:, k, d].std() for d in range(6)]
            for k in range(K)
        ]).mean(axis=1)
        ax.bar(x + i * width - 0.4 + width/2, sds, width, color=colors[i], label=r)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in range(K)])
    ax.set_ylabel("mean std across 6 dims")
    ax.set_title("Per-step mean abs std\n(higher = more spread; lower = more collapsed)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── (0,1) Per-step abs mean
    ax = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(runs):
        ms = np.array([
            [abs(arrays[r][:, k, d].mean()) for d in range(6)]
            for k in range(K)
        ]).mean(axis=1)
        ax.bar(x + i * width - 0.4 + width/2, ms, width, color=colors[i], label=r)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in range(K)])
    ax.set_ylabel("mean |mean| across 6 dims")
    ax.set_title("Per-step mean abs mean\n(bias — should be ~0)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── (0,2) Per-dim std at k=0
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(6)
    for i, r in enumerate(runs):
        sds = [arrays[r][:, 0, d].std() for d in range(6)]
        ax.bar(x + i * width - 0.4 + width/2, sds, width, color=colors[i], label=r)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.set_ylabel("std")
    ax.set_title("Per-dim std at k=0 (the step used at inference)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.001, color="red", linestyle="--", alpha=0.5, label="collapse threshold")

    # ── (1,0..2) k=0 sample histogram overlay, one panel per dim group
    # Position dims (x, y, z)
    ax = fig.add_subplot(gs[1, 0])
    for i, r in enumerate(runs):
        flat = arrays[r][:, 0, 0:3].flatten()
        ax.hist(flat, bins=50, histtype="step", color=colors[i], label=r, alpha=0.8)
    ax.set_title("k=0 position (x, y, z) histogram")
    ax.set_xlabel("output value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Rotation dims (ax, ay, az)
    ax = fig.add_subplot(gs[1, 1])
    for i, r in enumerate(runs):
        flat = arrays[r][:, 0, 3:6].flatten()
        ax.hist(flat, bins=50, histtype="step", color=colors[i], label=r, alpha=0.8)
    ax.set_title("k=0 rotation (ax, ay, az) histogram")
    ax.set_xlabel("output value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (1,2) Cumulative std across all (K, 6) dims sorted
    ax = fig.add_subplot(gs[1, 2])
    for i, r in enumerate(runs):
        all_stds = []
        for k in range(K):
            for d in range(6):
                all_stds.append(arrays[r][:, k, d].std())
        all_stds = sorted(all_stds)
        ax.plot(all_stds, color=colors[i], label=r, linewidth=1.5)
    ax.axhline(0.001, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("dim rank (sorted by std)")
    ax.set_ylabel("std")
    ax.set_title("Sorted std across all 30 dims\n(should be smooth; cliff = collapsed dims)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (2, 0..2) Magnitude growth across steps
    ax = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(runs):
        mags = []
        for k in range(K):
            mag = np.linalg.norm(arrays[r][:, k, :], axis=-1).mean()
            mags.append(mag)
        ax.plot(range(K), mags, marker="o", color=colors[i], label=r)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"k={k}" for k in range(K)])
    ax.set_ylabel("mean L2 norm of 6D delta")
    ax.set_title("Output magnitude per step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (2,1) Per-dim max range (k4 only)
    ax = fig.add_subplot(gs[2, 1])
    x = np.arange(6)
    for i, r in enumerate(runs):
        ranges = [arrays[r][:, -1, d].max() - arrays[r][:, -1, d].min() for d in range(6)]
        ax.bar(x + i * width - 0.4 + width/2, ranges, width, color=colors[i], label=r)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.set_ylabel("max - min")
    ax.set_title("Per-dim range at k=4 (final step)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── (2,2) k=0 prediction mean comparison (bar chart)
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(6)
    for i, r in enumerate(runs):
        ms = [arrays[r][:, 0, d].mean() for d in range(6)]
        ax.bar(x + i * width - 0.4 + width/2, ms, width, color=colors[i], label=r)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.set_ylabel("mean prediction")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Per-dim mean at k=0 (bias visible if non-zero)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Assistant head output distribution — cross-run comparison\n"
        f"runs: {', '.join(runs)}    (data: h5_data/libero_spatial.h5, N=320)",
        fontsize=12,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
