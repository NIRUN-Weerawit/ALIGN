#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN head training script.

Stages:
  1. Decision head (BCE loss)
  2. Assistant head (MSE loss)
  3. Joint fine-tuning

All stages use frozen vision/text/trajectory backbones from pretrained checkpoint.
"""

import argparse
import json
import sys
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
from training.wandb_utils import init_wandb, WandBTrainer
from data.align_dataset import ALIGNDataset, head_collate


# ================================================================
# Config
# ================================================================

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS_DECISION = 10
DEFAULT_EPOCHS_ASSISTANT = 20
DEFAULT_EPOCHS_JOINT = 20
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_CHUNK_SIZE = 5


def train_epoch(
    model: ALIGNModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    stage: str,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    losses = []
    alphas = []
    deltas = []

    for batch in loader:
        frames = torch.from_numpy(batch["frames"]).to(device)
        traj = torch.from_numpy(batch["trajectory"]).float().to(device)
        noisy = torch.from_numpy(batch["noisy_pose"]).float().to(device)
        texts = batch["texts"]
        dists = torch.from_numpy(batch["distances"]).float().to(device)
        alpha_t = torch.from_numpy(batch["alpha_target"]).float().to(device)
        delta_t = torch.from_numpy(batch["delta_target"]).float().to(device)

        with torch.no_grad():
            z_v = model.encode_vision(frames)
            z_t = model.encode_trajectory(traj)
            z_text = model.encode_text(texts)

        if stage == "decision":
            alpha_pred = model.decision_head(z_v, z_t, z_text, dists)
            loss = F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_t)
        elif stage == "assistant":
            delta_pred = model.assistant_head(z_v, z_t, z_text, noisy)
            loss = F.mse_loss(delta_pred, delta_t)
        elif stage == "joint":
            alpha_pred = model.decision_head(z_v, z_t, z_text, dists)
            delta_pred = model.assistant_head(z_v, z_t, z_text, noisy)
            loss = F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_t) + 0.5 * F.mse_loss(delta_pred, delta_t)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        if stage in ("decision", "joint"):
            alphas.append(alpha_pred.detach().mean().item())
        if stage in ("assistant", "joint"):
            deltas.append(delta_pred.detach().abs().mean().item())

        # Progress every 50 batches during training
        if (len(losses)) % 50 == 0:
            print(f"  [{stage}] batch {len(losses)}/{len(loader) if hasattr(loader, '__len__') else '?'}  "
                  f"loss: {loss.item():.4f}", flush=True)

    return {
        "loss": float(np.mean(losses)),
        "alpha_mean": float(np.mean(alphas)) if alphas else 0.0,
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
    }


@torch.no_grad()
def validate(model: ALIGNModel, loader: DataLoader, stage: str, device: torch.device) -> dict:
    """Validate model."""
    model.eval()
    losses = []
    alphas = []
    deltas = []

    for batch in loader:
        frames = torch.from_numpy(batch["frames"]).to(device)
        traj = torch.from_numpy(batch["trajectory"]).float().to(device)
        noisy = torch.from_numpy(batch["noisy_pose"]).float().to(device)
        texts = batch["texts"]
        dists = torch.from_numpy(batch["distances"]).float().to(device)
        alpha_t = torch.from_numpy(batch["alpha_target"]).float().to(device)
        delta_t = torch.from_numpy(batch["delta_target"]).float().to(device)

        z_v = model.encode_vision(frames)
        z_t = model.encode_trajectory(traj)
        z_text = model.encode_text(texts)

        if stage == "decision":
            alpha_pred = model.decision_head(z_v, z_t, z_text, dists)
            loss = F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_t)
        elif stage == "assistant":
            delta_pred = model.assistant_head(z_v, z_t, z_text, noisy)
            loss = F.mse_loss(delta_pred, delta_t)
        elif stage == "joint":
            alpha_pred = model.decision_head(z_v, z_t, z_text, dists)
            delta_pred = model.assistant_head(z_v, z_t, z_text, noisy)
            loss = F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_t) + 0.5 * F.mse_loss(delta_pred, delta_t)

        losses.append(loss.item())
        if stage in ("decision", "joint"):
            alphas.append(alpha_pred.mean().item())
        if stage in ("assistant", "joint"):
            deltas.append(delta_pred.abs().mean().item())

    return {
        "loss": float(np.mean(losses)),
        "alpha_mean": float(np.mean(alphas)) if alphas else 0.0,
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="ALIGN Head Training")
    parser.add_argument("--data", required=True, help="Path to align.h5 dataset")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained checkpoint (.pt)")
    parser.add_argument("--output-dir", default="./checkpoints/heads", help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs-decision", type=int, default=DEFAULT_EPOCHS_DECISION)
    parser.add_argument("--epochs-assistant", type=int, default=DEFAULT_EPOCHS_ASSISTANT)
    parser.add_argument("--epochs-joint", type=int, default=DEFAULT_EPOCHS_JOINT)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="align-heads", help="W&B project name")
    parser.add_argument("--wandb-run", default=None, help="W&B run name")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"=== ALIGN Head Training ===")
    print(f"  Data:       {args.data}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Output:     {args.output_dir}")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "model": "align-heads",
            "data": str(args.data),
            "pretrained": str(args.pretrained),
            "epochs_decision": args.epochs_decision,
            "epochs_assistant": args.epochs_assistant,
            "epochs_joint": args.epochs_joint,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "chunk_size": args.chunk_size,
            "val_split": args.val_split,
            "device": str(device),
        },
    ) if args.wandb else init_wandb(project=args.wandb_project, name=args.wandb_run, config={})
    print(f"  W&B:        {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Dataset ──
    full_ds = ALIGNDataset(args.data, mode="head", traj_window=10)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    print(f"  {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: head_collate(b, chunk_size=args.chunk_size),
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda b: head_collate(b, chunk_size=args.chunk_size),
        num_workers=2,
    )

    # -- Model ──
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=args.chunk_size,
        use_text=True,
        device=str(device),
    ).to(device)

    ckpt = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.freeze_backbone()
    print(f"  Loaded pretrained backbone from {args.pretrained}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "head_training_log.jsonl"
    log_fp = open(log_path, "a")

    def log_entry(stage: str, epoch: int, train_stats: dict, val_stats: dict):
        entry = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_alpha_mean": train_stats["alpha_mean"],
            "val_alpha_mean": val_stats["alpha_mean"],
            "train_delta_mean": train_stats["delta_mean"],
            "val_delta_mean": val_stats["delta_mean"],
            "timestamp": datetime.now().isoformat(),
        }
        log_fp.write(json.dumps(entry) + "\n")
        log_fp.flush()

    # -- Stage 1: Decision Head ──
    print(f"\n  --- Stage 1: Decision Head ({args.epochs_decision} epochs) ---")
    opt_decision = optim.AdamW(
        model.decision_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs_decision):
        train_stats = train_epoch(model, train_loader, opt_decision, "decision", device)
        val_stats = validate(model, val_loader, "decision", device)
        print(f"  Epoch {epoch + 1:3d}  train: {train_stats['loss']:.4f}  val: {val_stats['loss']:.4f}  "
              f"α: {val_stats['alpha_mean']:.3f}")
        log_entry("decision", epoch + 1, train_stats, val_stats)

        # W&B logging
        wandb_trainer.log({
            "decision/train_loss": train_stats["loss"],
            "decision/val_loss": val_stats["loss"],
            "decision/train_alpha_mean": train_stats["alpha_mean"],
            "decision/val_alpha_mean": val_stats["alpha_mean"],
            "decision/epoch": epoch + 1,
            "best_val_loss": best_val_loss,
        }, step=epoch + 1)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), output_dir / "decision_best.pt")
            wandb_trainer.save(str(output_dir / "decision_best.pt"))
    torch.save(model.state_dict(), output_dir / "decision_last.pt")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # -- Stage 2: Assistant Head ──
    print(f"\n  --- Stage 2: Assistant Head ({args.epochs_assistant} epochs) ---")
    opt_assistant = optim.AdamW(
        model.assistant_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs_assistant):
        train_stats = train_epoch(model, train_loader, opt_assistant, "assistant", device)
        val_stats = validate(model, val_loader, "assistant", device)
        print(f"  Epoch {epoch + 1:3d}  train: {train_stats['loss']:.4f}  val: {val_stats['loss']:.4f}  "
              f"Δ: {val_stats['delta_mean']:.4f}")
        log_entry("assistant", epoch + 1, train_stats, val_stats)

        # W&B logging
        wandb_trainer.log({
            "assistant/train_loss": train_stats["loss"],
            "assistant/val_loss": val_stats["loss"],
            "assistant/train_delta_mean": train_stats["delta_mean"],
            "assistant/val_delta_mean": val_stats["delta_mean"],
            "assistant/epoch": epoch + 1,
            "best_val_loss": best_val_loss,
        }, step=epoch + 1)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), output_dir / "assistant_best.pt")
            wandb_trainer.save(str(output_dir / "assistant_best.pt"))
    torch.save(model.state_dict(), output_dir / "assistant_last.pt")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # -- Stage 3: Joint Fine-tuning ──
    print(f"\n  --- Stage 3: Joint Fine-Tuning ({args.epochs_joint} epochs) ---")
    opt_joint = optim.AdamW(
        [p for p in model.decision_head.parameters()] + [p for p in model.assistant_head.parameters()],
        lr=args.lr * 0.5,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs_joint):
        train_stats = train_epoch(model, train_loader, opt_joint, "joint", device)
        val_stats = validate(model, val_loader, "joint", device)
        print(f"  Epoch {epoch + 1:3d}  train: {train_stats['loss']:.4f}  val: {val_stats['loss']:.4f}  "
              f"α: {val_stats['alpha_mean']:.3f}  Δ: {val_stats['delta_mean']:.4f}")
        log_entry("joint", epoch + 1, train_stats, val_stats)

        # W&B logging
        wandb_trainer.log({
            "joint/train_loss": train_stats["loss"],
            "joint/val_loss": val_stats["loss"],
            "joint/train_alpha_mean": train_stats["alpha_mean"],
            "joint/val_alpha_mean": val_stats["alpha_mean"],
            "joint/train_delta_mean": train_stats["delta_mean"],
            "joint/val_delta_mean": val_stats["delta_mean"],
            "joint/epoch": epoch + 1,
            "best_val_loss": best_val_loss,
        }, step=epoch + 1)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), output_dir / "joint_best.pt")
            wandb_trainer.save(str(output_dir / "joint_best.pt"))
    torch.save(model.state_dict(), output_dir / "joint_last.pt")
    print(f"  Best val loss: {best_val_loss:.4f}")

    log_fp.close()
    print(f"\n  Training complete.")
    print(f"  Logs: {log_path}")
    print(f"  Best checkpoint: {output_dir}/joint_best.pt")
    wandb_trainer.finish()


if __name__ == "__main__":
    main()
