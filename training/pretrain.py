#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN pretraining script.

3-way contrastive pretraining: InfoNCE on (vision, trajectory, text) triples.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from training.contrastive_loss import ContrastiveLoss3Way
from data.align_dataset import ALIGNDataset, pretrain_collate


# ================================================================
# Config
# ================================================================

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_TEMPERATURE = 0.07
DEFAULT_EMBED_DIM = 256
DEFAULT_TRAJ_WINDOW = 10
DEFAULT_FRAMES_PER_EP = 8
DEFAULT_EPISODES_PER_BATCH = 8


def main():
    parser = argparse.ArgumentParser(description="ALIGN Contrastive Pretraining")
    parser.add_argument("--data", required=True, help="Path to align.h5 dataset")
    parser.add_argument("--output-dir", default="./checkpoints/pretrain", help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--traj-window", type=int, default=DEFAULT_TRAJ_WINDOW)
    parser.add_argument("--frames-per-ep", type=int, default=DEFAULT_FRAMES_PER_EP)
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--val-every", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--no-text", action="store_true", help="Disable text modality")
    parser.add_argument("--resume", help="Resume from checkpoint")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== ALIGN Contrastive Pretraining ===")
    print(f"  Dataset:  {args.data}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.lr}")
    print(f"  Text:     {'disabled' if args.no_text else 'enabled'}")

    # ── Dataset ──
    full_ds = ALIGNDataset(
        args.data,
        mode="pretrain",
        frames_per_ep=args.frames_per_ep,
        traj_window=args.traj_window,
    )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    print(f"  {n_train} train samples, {n_val} val samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: pretrain_collate(b, traj_window=args.traj_window),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda b: pretrain_collate(b, traj_window=args.traj_window),
        num_workers=2,
    )

    # ── Model ──
    model = ALIGNModel(
        embed_dim=args.embed_dim,
        use_text=not args.no_text,
        device=str(device),
    ).to(device)

    # Only train projection heads + trajectory encoder
    for p in model.vision_encoder.backbone.parameters():
        p.requires_grad = False
    if model.text_encoder is not None:
        for p in model.text_encoder.model.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_params:,}")

    # ── Loss ──
    criterion = ContrastiveLoss3Way(temperature=args.temperature)

    # ── Optimizer ──
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # ── Resume ──
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    best_val_loss = float("inf")
    log_path = output_dir / "training_log.jsonl"
    log_fp = open(log_path, "a")

    for epoch in range(start_epoch, args.epochs):
        # ── Train ──
        model.train()
        train_losses = []
        train_vt = []
        train_vl = []
        train_tl = []

        for i, batch in enumerate(train_loader):
            frames = torch.from_numpy(batch["frames"]).to(device)
            trajs = torch.from_numpy(batch["trajectories"]).float().to(device)
            texts = batch["texts"]

            z_v = model.encode_vision(frames)
            z_t = model.encode_trajectory(trajs)
            z_text = model.encode_text(texts)

            stats = criterion(z_v, z_t, z_text)
            loss = stats["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()

            train_losses.append(loss.item())
            train_vt.append(stats["avg_cos_vt"].item())
            train_vl.append(stats["avg_cos_vl"].item())
            train_tl.append(stats["avg_cos_tl"].item())

        avg_train_loss = float(np.mean(train_losses))
        print(f"  Epoch {epoch + 1:3d}  train loss: {avg_train_loss:.4f}  "
              f"cos_vt: {float(np.mean(train_vt)):.3f}  "
              f"cos_vl: {float(np.mean(train_vl)):.3f}  "
              f"cos_tl: {float(np.mean(train_tl)):.3f}")

        # ── Val ──
        if (epoch + 1) % args.val_every == 0:
            model.eval()
            val_losses = []
            val_vt = []
            val_vl = []
            val_tl = []
            with torch.no_grad():
                for batch in val_loader:
                    frames = torch.from_numpy(batch["frames"]).to(device)
                    trajs = torch.from_numpy(batch["trajectories"]).float().to(device)
                    texts = batch["texts"]
                    z_v = model.encode_vision(frames)
                    z_t = model.encode_trajectory(trajs)
                    z_text = model.encode_text(texts)
                    stats = criterion(z_v, z_t, z_text)
                    val_losses.append(stats["loss"].item())
                    val_vt.append(stats["avg_cos_vt"].item())
                    val_vl.append(stats["avg_cos_vl"].item())
                    val_tl.append(stats["avg_cos_tl"].item())

            avg_val_loss = float(np.mean(val_losses))
            print(f"  Epoch {epoch + 1:3d}  val loss:   {avg_val_loss:.4f}  "
                  f"cos_vt: {float(np.mean(val_vt)):.3f}  "
                  f"cos_vl: {float(np.mean(val_vl)):.3f}  "
                  f"cos_tl: {float(np.mean(val_tl)):.3f}")

            # Log
            log_fp.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_cos_vt": float(np.mean(train_vt)),
                "train_cos_vl": float(np.mean(train_vl)),
                "train_cos_tl": float(np.mean(train_tl)),
                "val_cos_vt": float(np.mean(val_vt)),
                "val_cos_vl": float(np.mean(val_vl)),
                "val_cos_tl": float(np.mean(val_tl)),
                "timestamp": datetime.now().isoformat(),
            }) + "\n")
            log_fp.flush()

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                }, output_dir / "best.pt")
                print(f"  -> best checkpoint saved (val_loss: {avg_val_loss:.4f})")

        # ── Periodic checkpoint ──
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_dir / f"epoch_{epoch + 1:04d}.pt")

    log_fp.close()
    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Logs saved to {log_path}")


if __name__ == "__main__":
    main()
