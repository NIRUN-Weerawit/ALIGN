#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train v3 ALIGNIntentionModel on synthetic Gym data (CartPole or Pendulum).

This is a SMALL evaluation script — used to verify the model works on
easy tasks before training on real LIBERO data.

────────────────────────────────────────────────────────────────────────
TRAINING CONTRACT — train_gym_synthetic.py
────────────────────────────────────────────────────────────────────────
INPUT  (per sample, B = batch):
  - frames_window:      (B, K, 64, 64, 3) uint8     — K past frames
  - robot_state_window: (B, K, 7)                — K past states (padded)
  - actions_window:     (B, K, 6)                — K past actions (padded)

OUTPUT (per sample):
  - actions_pred: (B, K, 6) — predicted K future actions

TARGET:
  - actions_window: (B, K, 6) — ground truth

LOSS:
  - F.mse_loss(actions_pred, actions_window)

METRICS (per epoch):
  - train/loss, val/loss, val/action_mean
────────────────────────────────────────────────────────────────────────

Usage:
    python training/train_gym_synthetic.py --env CartPole-v1 --n-train 200 \\
        --head-type mamba --epochs 30 --chunk-size 10
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Disable cuDNN band-aid
torch.backends.cudnn.enabled = False

from data.gym_synthetic_dataset import GymSyntheticDataset
from models.align_intention import ALIGNIntentionModel


# ================================================================
# Custom collate
# ================================================================

def gym_collate(batch):
    """Stack a list of items into a batch.

    Each item: {frames_window, robot_state_window, actions_window}
    """
    return {
        'frames_window': np.stack([item['frames_window'] for item in batch]),
        'robot_state_window': np.stack([item['robot_state_window'] for item in batch]).astype(np.float32),
        'actions_window': np.stack([item['actions_window'] for item in batch]).astype(np.float32),
    }


# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALIGN Intention training on synthetic Gym data (ablation test)"
    )
    # Data
    parser.add_argument("--env", default="CartPole-v1",
                        choices=["CartPole-v1", "Pendulum-v1"],
                        help="Gym environment to use")
    parser.add_argument("--n-train", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--n-val", type=int, default=50,
                        help="Number of val episodes")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="K — past steps / future actions")
    parser.add_argument("--image-size", type=int, default=64)
    # Model
    parser.add_argument("--head-type", choices=["transformer", "mamba", "hybrid"],
                        default="mamba", help="Which head to use")
    parser.add_argument("--vision-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--mamba-output-dim", type=int, default=512)
    # Training
    parser.add_argument("--output-dir", default="checkpoints/gym_synthetic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    return parser.parse_args()


# ================================================================
# Train / val loops
# ================================================================

def train_one_epoch(model, loader, optimizer, device, max_steps=0):
    model.train()
    losses = []
    n_steps = min(max_steps, len(loader)) if max_steps else len(loader)
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        frames = torch.from_numpy(batch['frames_window']).to(device)
        state = torch.from_numpy(batch['robot_state_window']).to(device).float()
        target = torch.from_numpy(batch['actions_window']).to(device).float()

        out = model(frames, state)
        h_current = out['h_seq'][:, -1]
        actions_pred = model.predict_actions(
            out['z_v_pooled_seq'], out['z_s_seq'], h_current,
        )
        loss = F.mse_loss(actions_pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        frames = torch.from_numpy(batch['frames_window']).to(device)
        state = torch.from_numpy(batch['robot_state_window']).to(device).float()
        target = torch.from_numpy(batch['actions_window']).to(device).float()

        out = model(frames, state)
        h_current = out['h_seq'][:, -1]
        actions_pred = model.predict_actions(
            out['z_v_pooled_seq'], out['z_s_seq'], h_current,
        )
        loss = F.mse_loss(actions_pred, target)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  Device: {device}")
    print(f"  Env:    {args.env}")
    print(f"  Head:   {args.head_type}")
    print(f"  K:      {args.chunk_size}")

    # Output dir
    out_dir = Path(args.output_dir) / f"{args.env}_{args.head_type}_K{args.chunk_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}")

    # Data
    print(f"\n  Building dataset ({args.n_train} train, {args.n_val} val)...")
    train_ds = GymSyntheticDataset(
        env_name=args.env, n_samples=args.n_train, K=args.chunk_size,
        image_size=args.image_size, seed=args.seed,
    )
    val_ds = GymSyntheticDataset(
        env_name=args.env, n_samples=args.n_val, K=args.chunk_size,
        image_size=args.image_size, seed=args.seed + 1000,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=gym_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, collate_fn=gym_collate, num_workers=0,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print(f"\n  Building model (head_type={args.head_type})...")
    model = ALIGNIntentionModel(
        vision_dim=args.vision_dim,
        state_dim=args.state_dim,
        mamba_output_dim=args.mamba_output_dim,
        action_dim=6,
        chunk_size=args.chunk_size,
        num_cameras=1,
        head_type=args.head_type,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # Save config
    config = vars(args).copy()
    config['n_params'] = n_params
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Log
    log_path = out_dir / "train_log.jsonl"
    log_fp = open(log_path, "a")

    # Training
    print(f"\n  Training for {args.epochs} epochs...")
    best_val = float("inf")
    t_start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            max_steps=args.max_steps_per_epoch,
        )
        val_loss = validate(model, val_loader, device)
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)")

        log_fp.write(json.dumps({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "head_type": args.head_type,
            }, out_dir / "best.pt")

    log_fp.close()
    total_time = time.time() - t_start_all
    print(f"\n  Done. Best val_loss={best_val:.4f}  Total: {total_time:.0f}s")
    print(f"  Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()