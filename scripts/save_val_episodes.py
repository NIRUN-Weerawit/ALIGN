#!/usr/bin/env python3
"""Save validation episode keys from a training run to a text file.

Run this after training to extract which episodes were held out for
validation. The output file can be passed to eval_libero_v4_trajectory.py
via --val-episodes to ensure evaluation only uses held-out episodes.

Usage:
    python scripts/save_val_episodes.py \\
        --checkpoint checkpoints/v4/libero_spatial/run_15/intention_best.pt \\
        --data data/libero_spatial.h5 \\
        --output val_episodes.txt

    # Then use in eval:
    python eval/eval_libero_v4_trajectory.py \\
        --data data/libero_spatial.h5 \\
        --checkpoint checkpoints/v4/libero_spatial/run_15/intention_best_fixed.pt \\
        --cameras image wrist_image \\
        --val-episodes val_episodes.txt
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.align_dataset import ALIGNDataset
from torch.utils.data import random_split


def main():
    parser = argparse.ArgumentParser(
        description="Save validation episode keys from a training run."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (to read config)")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output", default="val_episodes.txt", help="Output text file")
    parser.add_argument("--val-split", type=float, default=None,
                        help="Override val split fraction (default: from checkpoint config or 0.1)")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"],
                        help="Camera names (must match training config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (default: from checkpoint config or 42)")
    args = parser.parse_args()

    # Load checkpoint config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    val_split = args.val_split or cfg.get("val_split", 0.1)
    seed = args.seed or cfg.get("seed", 42)
    chunk_size = cfg.get("chunk_size", 10)
    history_size = cfg.get("history_size", chunk_size)
    segment_max_mult = cfg.get("segment_max_mult", 5)
    traj_window = max(history_size * segment_max_mult, chunk_size)

    print(f"Building dataset from {args.data}...")
    ds = ALIGNDataset(
        args.data, mode="head",
        traj_window=traj_window, cameras=args.cameras,
    )
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    # Use the same random split as training
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)

    # Get the original indices that fell into the val split
    val_indices = val_ds.indices  # list of ints

    # Map indices back to episode keys
    # Each index corresponds to a specific (ep_idx, start, count) in ds._index
    ep_keys_seen = set()
    for idx in val_indices:
        ep_idx, start, count = ds._index[idx]
        ep_key = ds._episode_keys[ep_idx]
        ep_keys_seen.add(ep_key)

    ep_keys_sorted = sorted(ep_keys_seen)
    print(f"Total samples: {n_total}")
    print(f"Val split: {n_val} samples ({val_split*100:.0f}%)")
    print(f"Unique val episodes: {len(ep_keys_sorted)}")

    with open(args.output, "w") as f:
        for k in ep_keys_sorted:
            f.write(k + "\n")

    print(f"Saved to {args.output}")
    print(f"\nTo use: python eval/eval_libero_v4_trajectory.py \\")
    print(f"    --data {args.data} \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --val-episodes {args.output}")


if __name__ == "__main__":
    main()
