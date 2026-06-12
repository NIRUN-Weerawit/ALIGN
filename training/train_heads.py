#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN head training script — single joint stage.

Phase 2: Trains Decision + Assistant heads with joint loss on frozen
encoder+mixer embeddings.

Usage:
    python training/train_heads.py --data ./align.h5 \\
        --pretrained ./checkpoints/pretrain/best.pt \\
        --output-dir ./checkpoints/heads \\
        --epochs-heads 30
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from training.wandb_utils import init_wandb
from data.align_dataset import ALIGNDataset, head_collate


def train_heads_hdf5(
    data_path: str,
    pretrained_checkpoint: str,
    output_dir: str,
    epochs_heads: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    chunk_size: int = 5,
    val_split: float = 0.1,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    wandb_project: str = "align-heads",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 0,
    traj_window: int = 20,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    use_bf16: bool = True,
):
    """Train heads from HDF5 data — single joint loss, all encoders frozen."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ALIGN Head Training (HDF5) ===")
    print(f"  Data:       {data_path}")
    print(f"  Pretrained: {pretrained_checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs_heads} (joint BCE + MSE)")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-heads-hdf5",
            "data": str(data_path),
            "pretrained_checkpoint": pretrained_checkpoint,
            "epochs_heads": epochs_heads,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "chunk_size": chunk_size,
            "device": str(device),
            "use_bf16": use_bf16,
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:         {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Dataset ──
    full_ds = ALIGNDataset(data_path, mode="head", traj_window=traj_window)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    print(f"  {n_train} train, {n_val} val samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size),
        num_workers=num_workers,
    )

    # -- Model ──
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
    ).to(device)

    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    model.freeze_backbone()
    model.freeze_all_encoders()  # explicit: no gradient leaks to encoders or mixer
    print(f"  Loaded pretrained backbone from {pretrained_checkpoint}")
    print(f"  All encoders + mixer frozen — only heads train")

    # -- Single joint head optimizer ──
    optimizer = optim.AdamW(
        model.get_head_params(),
        lr=lr,
        weight_decay=weight_decay,
    )

    log_path = output_dir / "head_log.jsonl"
    log_fp = open(log_path, "a")
    best_loss = float("inf")
    _step_start = time.time()

    for epoch in range(epochs_heads):
        model.train()
        losses, alphas, deltas = [], [], []

        for step, batch in enumerate(train_loader):
            if step >= max_steps_per_epoch:
                break

            frames = torch.from_numpy(batch["frames"]).to(device)
            poses = torch.from_numpy(batch["noisy_pose"]).float().to(device)
            texts = batch["texts"]
            alpha_t = torch.from_numpy(batch["alpha_target"]).float().to(device)
            delta_t = torch.from_numpy(batch["delta_target"]).float().to(device)

            # Encode through frozen encoders + mixer
            with torch.no_grad():
                traj_view = torch.from_numpy(batch["trajectory"]).float().to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frames, traj_view, texts)
                z_v = mixed["z_v"].float()
                z_t = mixed["z_t"].float()
                z_text = mixed["z_text"].float()

            # Joint loss: BCE(α) + 0.5 × MSE(Δ)
            alpha_pred = model.decision_head(z_v, z_t, z_text)
            delta_pred = model.assistant_head(z_v, z_t, z_text, poses)

            loss = (F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_t) +
                    0.5 * F.mse_loss(delta_pred, delta_t))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_head_params(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            alphas.append(alpha_pred.detach().mean().item())
            deltas.append(delta_pred.detach().abs().mean().item())

            if (step + 1) % 100 == 0:
                _now = time.time()
                print(f"  Epoch {epoch + 1}, step {step + 1}/{max_steps_per_epoch}  "
                      f"loss: {loss.item():.4f}  α: {alpha_pred.mean().item():.3f}  "
                      f"Δ: {delta_pred.abs().mean().item():.4f}", flush=True)

        avg_loss = float(np.mean(losses))
        avg_alpha = float(np.mean(alphas))
        avg_delta = float(np.mean(deltas))

        print(f"  Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
              f"α: {avg_alpha:.3f}  Δ: {avg_delta:.4f}")

        # W&B logging
        wandb_trainer.log({
            "head/loss": avg_loss,
            "head/alpha_mean": avg_alpha,
            "head/delta_mean": avg_delta,
            "head/epoch": epoch + 1,
        }, step=epoch + 1)

        log_fp.write(json.dumps({
            "stage": "joint", "epoch": epoch + 1,
            "loss": avg_loss, "alpha_mean": avg_alpha, "delta_mean": avg_delta,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_heads_checkpoint(
                str(output_dir / "best.pt"), epoch, avg_loss,
                optimizer.state_dict(), {"chunk_size": chunk_size})
            wandb_trainer.save(str(output_dir / "best.pt"))
            print(f"  -> best.pt (loss: {avg_loss:.4f})")

    log_fp.close()
    print(f"\n  Head training complete. Best loss: {best_loss:.4f}")
    print(f"  Logs: {log_path}")
    wandb_trainer.finish()
    return str(output_dir / "best.pt")


def main():
    parser = argparse.ArgumentParser(description="ALIGN Head Training (HDF5, single joint stage)")
    parser.add_argument("--data", required=True, help="Path to align.h5 dataset")
    parser.add_argument("--pretrained", required=True, help="Path to Phase 1 pretrained checkpoint")
    parser.add_argument("--output-dir", default="./checkpoints/heads")
    parser.add_argument("--epochs-heads", type=int, default=30,
                        help="Total joint head training epochs (default 30)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--mixer-dim", type=int, default=512)
    parser.add_argument("--num-mixer-blocks", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 autocast (on by default)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable BF16 autocast")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-heads")
    parser.add_argument("--wandb-run", default=None)

    args = parser.parse_args()

    train_heads_hdf5(
        data_path=args.data,
        pretrained_checkpoint=args.pretrained,
        output_dir=args.output_dir,
        epochs_heads=args.epochs_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        chunk_size=args.chunk_size,
        val_split=args.val_split,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        enable_wandb=args.wandb,
        num_workers=args.num_workers,
        traj_window=args.traj_window,
        mixer_dim=args.mixer_dim,
        num_mixer_blocks=args.num_mixer_blocks,
        use_bf16=args.bf16,
    )


if __name__ == "__main__":
    main()