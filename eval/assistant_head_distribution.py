#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN: analyze the output distribution of a trained assistant head.

For a given assistant head checkpoint, runs the head on a validation set
and reports per-output-dimension statistics (mean, std, percentiles,
near-zero fraction, kurtosis), per-dim histograms, and correlation
matrices. Useful for diagnosing mode collapse, dimensional collapse,
or training pathologies without running full sim evals.

Output:
  - <out_prefix>_stats.json    per-dim statistics + raw arrays
  - <out_prefix>_dist.png      30-panel histogram grid + 2 correlation plots

Usage:
    PYTHONNOUSERSITE=1 python eval/assistant_head_distribution.py \\
        --heads-checkpoint checkpoints/heads/.../assistant_best.pt \\
        --encoder-checkpoint checkpoints/pretrain/.../best.pt \\
        --data h5_data/libero_spatial.h5 \\
        --out-prefix reports/assistant_dist_run7 \\
        --n-batches 50 \\
        --batch-size 32

The 30 output dims are organized as a 5×6 grid (chunk_size × action_dim).
A "healthy" head shows:
  - Each dim's std > 0.01 (not collapsed)
  - Each dim's mean near 0 (symmetric, not biased)
  - Each dim's distribution roughly bell-shaped or at least multimodal
  - Cross-dim correlations < 0.7 (not redundant)

A "broken" head shows:
  - One or more dims with std < 0.001 (mode collapse on that axis)
  - Strong correlation (>0.9) between dims that should be independent
  - Very heavy tails or extreme outliers
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from data.align_dataset import ALIGNDataset, head_collate
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────
# Checkpoint loading (same logic as eval_assistant_head.py)
# ──────────────────────────────────────────────────────────────────

def load_assistant_head(
    heads_checkpoint: str,
    encoder_checkpoint: str,
    device: torch.device,
):
    enc_ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    enc_cfg = enc_ckpt.get("config", {})

    heads_ckpt = torch.load(heads_checkpoint, map_location=device, weights_only=False)
    heads_cfg = heads_ckpt.get("config", {})
    chunk_size = heads_cfg.get("chunk_size", 5)

    model = ALIGNModel(
        embed_dim=256,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=enc_cfg.get("mixer_dim", 512),
        num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
    ).to(device)

    if "trainable_state_dict" in enc_ckpt:
        model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    model.freeze_backbone()
    model.freeze_all_encoders()
    model.eval()

    if "trainable_state_dict" in heads_ckpt:
        head_state = heads_ckpt["trainable_state_dict"]
        current_state = model.state_dict()
        compatible = {
            k: v for k, v in head_state.items()
            if k in current_state and v.shape == current_state[k].shape
        }
        if compatible:
            model.load_state_dict(compatible, strict=False)

    print(f"  Loaded heads (epoch={heads_ckpt.get('epoch', '?')}, "
          f"loss={heads_ckpt.get('loss', float('nan')):.4f})")
    return model, chunk_size


# ──────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────

def percentile_stats(x: np.ndarray) -> Dict[str, float]:
    """Per-dim summary statistics."""
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "abs_mean": float(np.abs(x).mean()),
        "frac_near_zero": float((np.abs(x) < 1e-3).mean()),
        "kurtosis": float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-12) - 3),
    }


def health_check(stats: Dict[str, Dict[str, float]], chunk_size: int) -> List[str]:
    """Per-dim health flags. Returns a list of warning strings."""
    warnings = []
    for k in range(chunk_size):
        for d in range(6):
            key = f"k{k}_d{d}"
            s = stats[key]
            if s["std"] < 0.001:
                warnings.append(f"  ⚠ k{k} d{d} std={s['std']:.6f}  → DIMENSION COLLAPSE")
            if abs(s["mean"]) > 0.1:
                warnings.append(f"  ⚠ k{k} d{d} mean={s['mean']:.4f}  → BIASED")
            if s["frac_near_zero"] > 0.5:
                warnings.append(f"  ⚠ k{k} d{d} frac_near_zero={s['frac_near_zero']:.2f}  → MOSTLY ZERO")
            if abs(s["kurtosis"]) > 10:
                warnings.append(f"  ⚠ k{k} d{d} kurtosis={s['kurtosis']:.1f}  → HEAVY-TAILED")
    return warnings


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Analyze the output distribution of a trained assistant head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--heads-checkpoint", required=True)
    p.add_argument("--encoder-checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out-prefix", required=True,
                   help="Output prefix (writes <prefix>_stats.json and <prefix>_dist.png)")
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the PNG plot (only emit JSON + console)")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Assistant Head Distribution Analysis ===")
    print(f"  Heads:    {args.heads_checkpoint}")
    print(f"  Encoder:  {args.encoder_checkpoint}")
    print(f"  Data:     {args.data}")
    print(f"  Device:   {device}")

    # ── Load model
    print("\nLoading...")
    model, chunk_size = load_assistant_head(
        args.heads_checkpoint, args.encoder_checkpoint, device,
    )
    print(f"  chunk_size={chunk_size}")

    # ── Load validation data
    ds = ALIGNDataset(args.data, mode="head", traj_window=5)
    n_total = len(ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    _, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    print(f"  Dataset: {n_train} train, {n_val} val")

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size, vision_window_size=chunk_size),
    )

    # ── Run model, collect outputs
    all_preds: List[np.ndarray] = []   # each: (B, K, 6)
    all_targets: List[np.ndarray] = []
    n_samples = 0

    print(f"\nRunning {args.n_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break

            frames = torch.from_numpy(batch["frames"]).to(device)
            texts = batch["texts"]
            current_action = torch.from_numpy(batch["current_action"]).float().to(device)
            # v2: prefer one-step state (B, 7); fall back to legacy
            # "trajectory" (B, K, 6). encode_mixed handles both.
            traj_view = torch.from_numpy(
                batch.get("robot_state", batch["trajectory"])
            ).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                mixed = model.encode_mixed(frames, traj_view, texts)
                z_v = mixed["z_v"].float()
                z_s = mixed["z_s"].float()
                z_sext = mixed["z_sext"].float()
                action_pred = model.assistant_head(z_v, z_s, z_sext)

            all_preds.append(action_pred.float().cpu().numpy())
            all_targets.append(current_action.float().cpu().numpy())
            n_samples += action_pred.shape[0]

    preds = np.concatenate(all_preds, axis=0)        # (N, 6)
    targets = np.concatenate(all_targets, axis=0)    # (N, 6)
    print(f"  Collected {n_samples} predictions of shape {preds.shape}")

    # ── Per-dim statistics
    print(f"\n{'='*70}")
    print(f"Per-dim statistics (N={n_samples}, K={chunk_size}, action_dim=6)")
    print(f"{'='*70}\n")
    print(f"  {'Dim':<8}{'Mean':<10}{'Std':<10}{'Min':<10}{'P50':<10}"
          f"{'P95':<10}{'Max':<10}{'absμ':<10}{'kurt':<8}{'≈0':<8}")
    print(f"  {'-'*8}{'-'*9}{'-'*9}{'-'*9}{'-'*9}{'-'*9}{'-'*9}{'-'*9}{'-'*7}{'-'*7}")

    stats: Dict[str, Dict[str, float]] = {}
    for k in range(chunk_size):
        for d in range(6):
            x = preds[:, k, d]
            s = percentile_stats(x)
            stats[f"k{k}_d{d}"] = s
            tag = f"k{k}_d{d}"
            print(f"  {tag:<8}"
                  f"{s['mean']:<10.4f}{s['std']:<10.4f}"
                  f"{s['min']:<10.4f}{s['p50']:<10.4f}"
                  f"{s['p95']:<10.4f}{s['max']:<10.4f}"
                  f"{s['abs_mean']:<10.4f}{s['kurtosis']:<8.1f}"
                  f"{s['frac_near_zero']:<8.2f}")

    # Target distribution for comparison
    target_stats: Dict[str, Dict[str, float]] = {}
    for k in range(chunk_size):
        for d in range(6):
            x = targets[:, k, d]
            target_stats[f"k{k}_d{d}"] = percentile_stats(x)

    # ── Per-dim correlations
    # Flatten (N, K, 6) → (N, K*6), then compute correlation matrix
    preds_flat = preds.reshape(n_samples, -1)              # (N, K*6)
    targets_flat = targets.reshape(n_samples, -1)
    pred_corr = np.corrcoef(preds_flat.T)                  # (K*6, K*6)
    target_corr = np.corrcoef(targets_flat.T)

    # Cross-step correlation: do outputs at step k=0 correlate with step k=4?
    # We look at the diagonal blocks of pred_corr.
    cross_step_corr = {}
    for k1 in range(chunk_size):
        for k2 in range(chunk_size):
            if k1 < k2:
                block = pred_corr[k1*6:(k1+1)*6, k2*6:(k2+1)*6]
                cross_step_corr[f"k{k1}_vs_k{k2}"] = float(block[np.triu_indices(6, k=1)].mean())

    # ── Health warnings
    warnings = health_check(stats, chunk_size)
    print(f"\n{'='*70}")
    print(f"Health check")
    print(f"{'='*70}")
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(w)
    else:
        print(f"  ✓ All {chunk_size * 6} dimensions look healthy")

    # ── JSON output
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(out_prefix) + "_stats.json")
    payload = {
        "checkpoint": args.heads_checkpoint,
        "n_samples": int(n_samples),
        "chunk_size": chunk_size,
        "preds_shape": list(preds.shape),
        "preds_per_dim": stats,
        "targets_per_dim": target_stats,
        "cross_step_correlation_means": cross_step_corr,
        "warnings": warnings,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    # Also save raw arrays in a separate .npz for re-plotting
    npz_path = Path(str(out_prefix) + "_arrays.npz")
    np.savez_compressed(npz_path, preds=preds, targets=targets)
    print(f"\n  Wrote: {json_path}")
    print(f"  Wrote: {npz_path}")

    # ── Plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n  matplotlib not available, skipping PNG plot")
            return

        png_path = Path(str(out_prefix) + "_dist.png")
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 6, height_ratios=[3, 3, 3, 3, 1.2][:4], hspace=0.4, wspace=0.3)

        # 5×6 grid of histograms: row = step k, col = action dim
        dim_names = ["x", "y", "z", "ax", "ay", "az"]
        for k in range(min(chunk_size, 3)):    # up to 3 step rows
            for d in range(6):
                ax = fig.add_subplot(gs[k, d])
                ax.hist(preds[:, k, d], bins=40, color="steelblue", alpha=0.7,
                        label="pred")
                ax.hist(targets[:, k, d], bins=40, color="darkorange", alpha=0.5,
                        label="target")
                ax.set_title(f"k={k} dim={dim_names[d]}", fontsize=8)
                ax.tick_params(labelsize=6)
                if k == 0 and d == 0:
                    ax.legend(fontsize=6, loc="upper right")
                ax.set_yticks([])

        # Step-0 within-step dim correlation heatmap
        ax_h1 = fig.add_subplot(gs[3, 0:2])
        im1 = ax_h1.imshow(pred_corr[:6, :6], cmap="RdBu_r", vmin=-1, vmax=1)
        ax_h1.set_title("pred: k=0 dim correlations", fontsize=9)
        ax_h1.set_xticks(range(6))
        ax_h1.set_yticks(range(6))
        ax_h1.set_xticklabels(dim_names, fontsize=7)
        ax_h1.set_yticklabels(dim_names, fontsize=7)
        plt.colorbar(im1, ax=ax_h1, fraction=0.046)

        # Cross-step correlation heatmap (steps × steps)
        ax_h2 = fig.add_subplot(gs[3, 2:4])
        cross_step_matrix = np.full((chunk_size, chunk_size), np.nan)
        for k1 in range(chunk_size):
            for k2 in range(chunk_size):
                if k1 == k2:
                    cross_step_matrix[k1, k2] = 1.0
                elif k1 < k2:
                    cross_step_matrix[k1, k2] = cross_step_corr[f"k{k1}_vs_k{k2}"]
                    cross_step_matrix[k2, k1] = cross_step_corr[f"k{k1}_vs_k{k2}"]
        im2 = ax_h2.imshow(cross_step_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_h2.set_title("pred: cross-step correlation (off-diag mean)", fontsize=9)
        ax_h2.set_xticks(range(chunk_size))
        ax_h2.set_yticks(range(chunk_size))
        plt.colorbar(im2, ax=ax_h2, fraction=0.046)

        # Cumulative std across dims (collapse check)
        ax_h3 = fig.add_subplot(gs[3, 4:6])
        stds = np.array([stats[f"k0_d{d}"]["std"] for d in range(6)])
        ax_h3.bar(dim_names, stds, color="steelblue")
        ax_h3.axhline(0.01, color="red", linestyle="--", label="collapse threshold")
        ax_h3.set_title("k=0 per-dim std (collapse check)", fontsize=9)
        ax_h3.set_ylabel("std", fontsize=8)
        ax_h3.tick_params(labelsize=7)
        ax_h3.legend(fontsize=7)

        fig.suptitle(
            f"Assistant head output distribution — {Path(args.heads_checkpoint).name}\n"
            f"N={n_samples}, K={chunk_size}, blue=pred, orange=target",
            fontsize=11,
        )
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote: {png_path}")


if __name__ == "__main__":
    main()
