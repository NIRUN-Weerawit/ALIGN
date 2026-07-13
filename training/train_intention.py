#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN Intention Head training — Mamba-based.

Trains the ALIGNIntentionModel (state-conditioned attention pool +
Mamba recurrence + IntentionTransformerHead).

Usage:
    python3 training/train_intention.py --data data/libero_spatial.h5 \\
        --output-dir checkpoints/intention \\
        --cameras wrist_image --chunk-size 10 --epochs 20

────────────────────────────────────────────────────────────────────────
TRAINING CONTRACT — train_intention.py (v3: Intention Head)
────────────────────────────────────────────────────────────────────────
INPUT  (per sample, B = batch; K = --chunk-size):
  - frames_window:     (B, K, H, W, 3) uint8  — K past frames (new v3)
  - robot_state_window:(B, K, 7)             — K past robot states (new v3)
  - cameras:           (B, K, V, H, W, 3) optional — multi-cam variant

OUTPUT (per sample, B = batch):
  - z_v_pooled_seq:    (B, K, vision_dim)    — pooled visual per step
  - z_t_seq:           (B, K, state_dim)     — state embedding per step
  - h_seq:             (B, K, mamba_dim)     — Mamba hidden state per step
  - actions_pred:      (B, K, 6)             — K future OSC actions from
                                                IntentionTransformerHead

TARGET:
  - actions_window:    (B, K, 6)             — K past ground-truth
                                                human actions
  - (or, with --loss-mode delta:) delta_target (B, K, 6) — pose-relative
    goals (clean_pose[t+k+1] − clean_pose[t])

LOSS:
  - F.mse_loss(actions_pred, actions_window)    (--loss-mode action)
  - F.mse_loss(actions_pred, delta_target)      (--loss-mode delta)

METRICS (per epoch, logged to wandb + JSONL):
  - train/loss        (float, action²)   MSE between actions_pred & target
  - train/action_mean (float, |action|)  Mean |actions_pred| — magnitude
                                          sanity check (helps diagnose
                                          mode collapse)
  - val/loss          (float, action²)   Val-set MSE
  - val/action_mean   (float, |action|)  Val-set mean |actions_pred|

BEST CHECKPOINT:
  - Lowest val/loss across epochs, saved as intention_best.pt
────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.align_dataset import ALIGNDataset, MultiALIGNDataset, head_collate
from models.align_intention import ALIGNIntentionModel
from training.wandb_utils import init_wandb


# ================================================================
# Data
# ================================================================

def build_datasets(args):
    """Build train and val datasets from one or more HDF5 files."""
    if len(args.data) == 1:
        ds = ALIGNDataset(
            args.data[0], mode="head",
            traj_window=args.chunk_size, cameras=args.cameras,
        )
    else:
        ds = MultiALIGNDataset(
            args.data, mode="head",
            traj_window=args.chunk_size, cameras=args.cameras,
        )
    n_total = len(ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  {n_train} train, {n_val} val samples")
    return ds, train_ds, val_ds


def build_loaders(train_ds, val_ds, args):
    """Build train and val dataloaders."""
    collate_fn = lambda b: head_collate(
        b, chunk_size=args.chunk_size, vision_window_size=args.chunk_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ================================================================
# Model
# ================================================================

def build_model(args, num_cameras, device):
    """Build ALIGNIntentionModel. Optionally warm-start from a pretrained
    encoder+mixer checkpoint for the vision/state encoders.
    """
    model = ALIGNIntentionModel(
        vision_dim=args.vision_dim,
        state_dim=args.state_dim,
        mamba_output_dim=args.mamba_output_dim,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        num_cameras=num_cameras,
        use_patch_tokens=args.use_patch_tokens,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        head_d_model=args.head_d_model,
        head_nhead=args.head_nhead,
        head_num_layers=args.head_num_layers,
        head_dim_ff=args.head_dim_ff,
    )
    if args.pretrained:
        # Load only the encoder parts (vision_encoder.backbone, state_encoder)
        # by name-matching compatible weights. Shape mismatches are skipped
        # so we can warm-start from an older / differently-shaped checkpoint.
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        sd = ckpt.get("trainable_state_dict",
                      ckpt.get("model_state_dict", ckpt))
        own = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(own, strict=False)
        print(f"  Loaded {loaded} params from {args.pretrained} "
              f"({skipped} skipped due to shape/key mismatch)")
    model = model.to(device)
    return model


# ================================================================
# Training / Validation
# ================================================================

def train_one_epoch(model, loader, optimizer, device, args, max_steps=0):
    """Train for one epoch. Returns (avg_loss, avg_action_mean)."""
    model.train()
    losses, actions_pred_list = [], []
    n_steps = min(max_steps, len(loader)) if max_steps else len(loader)
    pbar = tqdm(
        range(n_steps), total=n_steps,
        desc=f"  [train]", unit="step", leave=False,
    )
    step_iter = iter(loader)
    for _ in pbar:
        try:
            batch = next(step_iter)
        except StopIteration:
            break

        frames = torch.from_numpy(batch["frames_window"]).to(device)  # (B, K, H, W, 3) or (B, K, V, H, W, 3)
        state = torch.from_numpy(batch["robot_state_window"]).float().to(device)  # (B, K, 7)
        if args.loss_mode == "action":
            target = torch.from_numpy(batch["actions_window"]).float().to(device)  # (B, K, 6)
        else:  # delta
            target = torch.from_numpy(batch["delta_target"]).float().to(device)  # (B, K, 6)

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.bf16 and device.type == "cuda"):
            out = model(frames, state)
            # out: z_v_pooled_seq (B, K, pool_out_dim), z_t_seq (B, K, state_dim),
            #      h_seq (B, K, mamba_output_dim)
            h_current = out["h_seq"][:, -1]  # (B, mamba_output_dim) — latest
            actions_pred = model.predict_actions(
                out["z_v_pooled_seq"], out["z_t_seq"], h_current,
            )  # (B, K, 6)
            loss = F.mse_loss(actions_pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()

        losses.append(loss.item())
        actions_pred_list.append(actions_pred.detach().abs().mean().item())

        pbar.set_postfix(
            mse=f"{loss.item():.5f}",
            a_mean=f"{actions_pred.detach().abs().mean().item():.4f}",
        )

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    avg_action = float(np.mean(actions_pred_list)) if actions_pred_list else 0.0
    return avg_loss, avg_action


@torch.no_grad()
def validate(model, loader, device, args):
    """Validate on the val set. Returns (avg_loss, avg_action_mean)."""
    model.eval()
    losses, actions_pred_list = [], []
    pbar = tqdm(loader, desc="  [val]  ", unit="batch", leave=False)
    for batch in pbar:
        frames = torch.from_numpy(batch["frames_window"]).to(device)
        state = torch.from_numpy(batch["robot_state_window"]).float().to(device)
        if args.loss_mode == "action":
            target = torch.from_numpy(batch["actions_window"]).float().to(device)
        else:
            target = torch.from_numpy(batch["delta_target"]).float().to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.bf16 and device.type == "cuda"):
            out = model(frames, state)
            h_current = out["h_seq"][:, -1]
            actions_pred = model.predict_actions(
                out["z_v_pooled_seq"], out["z_t_seq"], h_current,
            )
            loss = F.mse_loss(actions_pred, target)

        losses.append(loss.item())
        actions_pred_list.append(actions_pred.detach().abs().mean().item())
        pbar.set_postfix(mse=f"{loss.item():.5f}")

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    avg_action = float(np.mean(actions_pred_list)) if actions_pred_list else 0.0
    return avg_loss, avg_action


# ================================================================
# Main
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALIGN Intention Head training (v3: Mamba-based)"
    )
    # Data
    parser.add_argument("--data", nargs="+", required=True,
                        help="Path(s) to HDF5 data file(s).")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"],
                        help="Camera names to load (e.g. 'wrist_image image').")
    parser.add_argument("--num-cameras", type=int, default=1,
                        help="Number of cameras. Auto-derived from --cameras "
                             "if not specified.")
    parser.add_argument("--val-split", type=float, default=0.1)
    # Model
    parser.add_argument("--pretrained", default=None,
                        help="Path to pretrained encoder checkpoint "
                             "(optional — vision/state encoders only).")
    parser.add_argument("--vision-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--mamba-output-dim", type=int, default=512)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--use-patch-tokens", action="store_true", default=True,
                        help="Use DINOv2 patch tokens (default on).")
    parser.add_argument("--no-patch-tokens", dest="use_patch_tokens",
                        action="store_false")
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=10)
    # IntentionTransformerHead params
    parser.add_argument("--head-d-model", type=int, default=384)
    parser.add_argument("--head-nhead", type=int, default=4)
    parser.add_argument("--head-num-layers", type=int, default=2)
    parser.add_argument("--head-dim-ff", type=int, default=1024)
    # Training
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss-mode", choices=["action", "delta"],
                        default="action",
                        help="'action' = predict human's K past actions; "
                             "'delta' = predict pose-relative goals.")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 autocast (default on).")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable BF16 autocast.")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Cap steps per epoch (0 = use full loader).")
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="align-intention")
    parser.add_argument("--wandb-run", default=None)
    # Other
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"\n=== ALIGN Intention (Mamba) Training ===")
    print(f"  Device:     {device}")
    print(f"  Chunk (K):  {args.chunk_size}")
    print(f"  Cameras:    {args.cameras}")
    print(f"  Loss mode:  {args.loss_mode}")
    print(f"  Pretrained: {args.pretrained or '(none — training from scratch)'}")

    # Output dir
    out_dir = Path(args.output_dir)
    if len(args.data) == 1:
        ds_name = Path(args.data[0]).stem
    else:
        ds_name = "+".join(Path(p).stem for p in args.data)
    out_dir = out_dir / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {out_dir}")

    # Wandb
    wandb_trainer = init_wandb(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args),
    )
    print(f"  W&B:        {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # Data
    print("\n  Loading data...")
    full_ds, train_ds, val_ds = build_datasets(args)
    train_loader, val_loader = build_loaders(train_ds, val_ds, args)

    # Auto-derive num_cameras from the actual data shape
    sample = train_ds[0]
    frames_shape = sample["frames_window"].shape
    if frames_shape.ndim == 5:
        # (K, V, H, W, 3) — multi-cam
        num_cameras = frames_shape[1]
    else:
        num_cameras = 1
    print(f"  Cameras detected: {num_cameras}")

    # Model
    print("\n  Building model...")
    model = build_model(args, num_cameras, device)

    # Freeze vision and state encoders; train intention encoder + head
    model.freeze_encoders()
    for p in model.intention_encoder.parameters():
        p.requires_grad = True
    for p in model.intention_head.parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params:   {n_trainable:,}")
    print(f"  Total model params: {n_total:,}")
    print(f"  Vision + state encoders frozen; training IntentionEncoder + Head")

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )

    # Save config snapshot
    config_snapshot = vars(args).copy()
    config_snapshot["num_cameras"] = num_cameras
    config_snapshot["n_trainable_params"] = n_trainable
    config_snapshot["n_total_params"] = n_total
    config_snapshot["model_class"] = "ALIGNIntentionModel"
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Log file
    log_path = out_dir / "intention_log.jsonl"
    log_fp = open(log_path, "w")

    # Training loop
    print(f"\n  Training for {args.epochs} epochs...")
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        train_loss, train_action = train_one_epoch(
            model, train_loader, optimizer, device, args,
            max_steps=args.max_steps_per_epoch,
        )
        val_loss, val_action = validate(model, val_loader, device, args)
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train: loss={train_loss:.5f} a_mean={train_action:.4f}  |  "
              f"val: loss={val_loss:.5f} a_mean={val_action:.4f}  "
              f"({elapsed:.0f}s)")

        wandb_trainer.log({
            "train/loss": train_loss,
            "train/action_mean": train_action,
            "val/loss": val_loss,
            "val/action_mean": val_action,
            "epoch": epoch,
        }, step=epoch)

        log_fp.write(json.dumps({
            "stage": "intention",
            "epoch": epoch,
            "train/loss": train_loss,
            "train/action_mean": train_action,
            "val/loss": val_loss,
            "val/action_mean": val_action,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": config_snapshot,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_action_mean": val_action,
                "phase": "intention_head",
            }
            torch.save(ckpt, out_dir / "intention_best.pt")
            print(f"    ↳ new best (val_loss={val_loss:.5f}), "
                  f"saved to intention_best.pt")

    log_fp.close()
    wandb_trainer.finish()
    print(f"\n  Done. Best val_loss={best_val_loss:.5f}")
    print(f"  Best checkpoint: {out_dir / 'intention_best.pt'}")
    print(f"  Logs:            {log_path}")


if __name__ == "__main__":
    main()
