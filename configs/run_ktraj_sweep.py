#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Programmatic K_traj sweep runner.

Runs multiple pretrain_from_stream calls with different K_traj values,
logs all to the same W&B project, and produces a comparison plot at the end.

Usage:
    python configs/run_ktraj_sweep.py \\
        --data-dir /path/to/libero_10 \\
        --ks 5 10 15 20 25 30 40 50 \\
        --epochs 10 \\
        --output-dir ./sweep_ktraj

After running, look at wandb.ai/<entity>/align-ktraj-sweep for comparison,
or check the local analysis plot at ./sweep_ktraj/comparison.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_ktraj_sweep(
    repo_id: str,
    data_dir: str,
    k_values: List[int],
    output_dir: str,
    epochs: int,
    fps: int,
    batch_size: int,
    lr: float,
    max_steps_per_epoch: int,
    wandb_project: str,
):
    """Run pretrain_from_stream for each K_traj value."""
    import numpy as np
    from training.pretrain_streaming import pretrain_from_stream

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"K_traj = {k}  (window: {k/fps:.2f}s @ {fps}fps)")
        print(f"{'='*60}")

        run_output = out / f"k{k:02d}"
        run_output.mkdir(parents=True, exist_ok=True)

        best_ckpt = pretrain_from_stream(
            repo_ids=[repo_id],
            output_dir=str(run_output),
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_steps_per_epoch=max_steps_per_epoch,
            traj_window=k,
            fps=fps,
            wandb_project=wandb_project,
            wandb_run=f"k{k:02d}",
            enable_wandb=True,
            num_workers=0,
        )

        # Read final loss from log
        log_path = run_output / "streaming_training_log.jsonl"
        if log_path.exists():
            lines = [json.loads(l) for l in log_path.read_text().splitlines() if l]
            if lines:
                final = lines[-1]
                results.append({
                    "k_traj": k,
                    "window_seconds": k / fps,
                    "final_loss": final["loss"],
                    "final_cos_vt": final["cos_vt"],
                    "final_cos_vl": final["cos_vl"],
                    "final_cos_tl": final["cos_tl"],
                })
                print(f"\n  K={k}: final loss={final['loss']:.4f}  "
                      f"cos_vt={final['cos_vt']:.3f}  "
                      f"cos_vl={final['cos_vl']:.3f}  "
                      f"cos_tl={final['cos_tl']:.3f}")

    # Save results
    with open(out / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("K_TRAJ SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  {'K':>4s}  {'Window':>8s}  {'Loss':>8s}  {'cos_vt':>8s}  {'cos_vl':>8s}  {'cos_tl':>8s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['k_traj']:4d}  {r['window_seconds']:.2f}s   "
              f"{r['final_loss']:.4f}  "
              f"{r['final_cos_vt']:.4f}  {r['final_cos_vl']:.4f}  {r['final_cos_tl']:.4f}")

    # Find best
    if results:
        best = min(results, key=lambda r: r["final_loss"])
        print(f"\n  Best K_traj by loss: K={best['k_traj']}  "
              f"(window={best['window_seconds']:.2f}s, loss={best['final_loss']:.4f})")
    return results


def plot_results(results_path: str):
    """Generate a comparison plot of K_traj sweep results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting. pip install matplotlib")
        return

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print("No results to plot.")
        return

    ks = [r["k_traj"] for r in results]
    losses = [r["final_loss"] for r in results]
    cos_vt = [r["final_cos_vt"] for r in results]
    cos_vl = [r["final_cos_vl"] for r in results]
    cos_tl = [r["final_cos_tl"] for r in results]
    windows = [r["window_seconds"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(ks, losses, "o-", linewidth=2, markersize=8, color="tab:red")
    ax1.set_xlabel("K_traj (trajectory window size)")
    ax1.set_ylabel("Final contrastive loss")
    ax1.set_title("K_traj vs Loss (lower is better)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ks)

    # Cosine similarity plot
    ax2.plot(ks, cos_vt, "o-", label="cos(v,t)", linewidth=2, markersize=8)
    ax2.plot(ks, cos_vl, "s-", label="cos(v,l)", linewidth=2, markersize=8)
    ax2.plot(ks, cos_tl, "^-", label="cos(t,l)", linewidth=2, markersize=8)
    ax2.set_xlabel("K_traj (trajectory window size)")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("K_traj vs Final Cosine Alignments (higher is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ks)

    fig.suptitle(
        f"K_traj ablation study — {len(results)} K values tested",
        fontsize=14, fontweight="bold",
    )

    out_path = Path(results_path).parent / "comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="K_traj ablation sweep")
    parser.add_argument("--data-dir", required=True, help="Local LIBERO data directory")
    parser.add_argument("--repo-id", default="nvidia/LIBERO_LeRobot_v3")
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[5, 10, 15, 20, 25, 30, 40, 50],
                        help="K_traj values to test (default: 5,10,15,20,25,30,40,50)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps-per-epoch", type=int, default=500)
    parser.add_argument("--output-dir", default="./sweep_ktraj")
    parser.add_argument("--wandb-project", default="align-ktraj-sweep")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plot from existing sweep_results.json")
    args = parser.parse_args()

    if args.plot:
        plot_results(Path(args.output_dir) / "sweep_results.json")
        return

    run_ktraj_sweep(
        repo_id=args.repo_id,
        data_dir=args.data_dir,
        k_values=args.ks,
        output_dir=args.output_dir,
        epochs=args.epochs,
        fps=args.fps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps_per_epoch=args.max_steps_per_epoch,
        wandb_project=args.wandb_project,
    )

    # Auto-plot
    if (Path(args.output_dir) / "sweep_results.json").exists():
        try:
            plot_results(str((Path(args.output_dir) / "sweep_results.json")))
        except Exception as e:
            print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()