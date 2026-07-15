#!/usr/bin/env python3
"""ALIGN Intention Head training — Mamba-based.

Trains the ALIGNIntentionModel end-to-end (one training pass, no separate
pretraining needed for v3).

What's trained (default):
  - Vision projection (Linear 768→256)        — adapts generic DINOv2 features
  - State encoder (MLP 7→256)                 — no pretrained weights available
  - Per-camera state-conditioned pool         — small, learnable
  - Mamba history encoder (optional)          — controlled by --use-history
  - Head (transformer or mamba)               — always trainable

What's frozen (default):
  - DINOv2 backbone                           — ImageNet-pretrained, kept frozen

There is no flag to train DINOv2: it's always frozen.

Usage:
  # Default training (single-stage, no pretraining needed)
  python3 training/train_intention.py --data data/libero_spatial.h5 \\
      --output-dir checkpoints/intention \\
      --cameras wrist_image --chunk-size 10 --epochs 200

  # Ablation: try all 6 head+history combinations
  for head in transformer mamba hybrid; do
    for hist in --use-history --no-history; do
      name="${head}_$(echo $hist | tr -d --)"
      python3 training/train_intention.py \\
        --data data/libero_spatial.h5 \\
        --output-dir checkpoints/$name \\
        --cameras wrist_image --chunk-size 10 \\
        --head-type $head $hist --epochs 100
    done
  done

────────────────────────────────────────────────────────────────────────
TRAINING CONTRACT — train_intention.py (v3: Intention Head)
────────────────────────────────────────────────────────────────────────

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
# Disable cuDNN — needed when DINOv2 (with torch.no_grad) and Mamba's CUDA
# kernel interact badly. Without this, you get CUDNN_STATUS_NOT_INITIALIZED
# on the first conv2d inside DINOv2. See setup.sh for the same fix.
torch.backends.cudnn.enabled = False
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

# ================================================================
# V3 defaults — these are now CLI-configurable.
# Kept here only as documentation of the default values.
# The training script reads them from `args` (see argparse below).
# ================================================================
# VISION_DIM = 256          (--vision-dim)
# STATE_DIM = 256           (--state-dim)
# MAMBA_OUTPUT_DIM = 512    (--mamba-output-dim)
# MAMBA_D_STATE = 16        (--mamba-d-state)
# MAMBA_D_CONV = 4          (--mamba-d-conv)
# MAMBA_EXPAND = 2          (--mamba-expand)
# ACTION_DIM = 6            (--action-dim)
# USE_PATCH_TOKENS = True   (--no-patch-tokens to disable)


def build_model(args, num_cameras, device):
    """Build ALIGNIntentionModel. All dimensions come from `args`."""
    model = ALIGNIntentionModel(
        vision_dim=args.vision_dim,
        state_dim=args.state_dim,
        mamba_output_dim=args.mamba_output_dim if args.use_history else 0,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        num_cameras=num_cameras,
        use_patch_tokens=args.use_patch_tokens,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        head_type=args.head_type,
        head_d_model=args.head_d_model,
        head_nhead=args.head_nhead,
        head_num_layers=args.head_num_layers,
        head_dim_ff=args.head_dim_ff,
        use_text=args.use_text,
        text_dim=args.text_dim,
    )
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
        target = torch.from_numpy(batch["actions_window"]).float().to(device)  # (B, K, 7)

        # Optional text encoding (only if --use-text was set)
        z_text = None
        if args.use_text:
            # Build text list: use --task-text for all items, or pull from batch if present
            B = frames.shape[0]
            if "texts" in batch and batch["texts"]:
                texts = batch["texts"]
            elif args.task_text:
                texts = [args.task_text] * B
            else:
                texts = ["default task"] * B
            z_text = model.text_encoder(texts)  # (B, text_dim)

        # Forward (BF16 always on for speed; disabled automatically on CPU)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            out = model(frames, state)
            h_current = out["h_seq"][:, -1]  # (B, mamba_output_dim) — latest
            # predict_actions returns actions (direct regression) or cond (flow head)
            actions_pred = model.predict_actions(
                out["z_v_pooled_seq"], out["z_t_seq"], h_current, z_text=z_text,
            )  # (B, K, 7) or (B, K, cond_dim)
            # Loss depends on head type
            if args.head_type == "flow":
                # Flow-matching: cond → velocity field loss
                loss = model.intention_head.loss(target, actions_pred)
            else:
                # Direct regression: MSE on actions
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
    """Validate on the val set. Returns (avg_loss, avg_action_mean, per_dim_metrics).

    per_dim_metrics: dict like {"pos_mse": 0.04, "rot_mse": 0.05, "grip_mse": 0.02}
    """
    model.eval()
    losses, actions_pred_list = [], []
    per_dim_squared = np.zeros(6, dtype=np.float64)
    per_dim_abs = np.zeros(6, dtype=np.float64)
    n_samples = 0
    pbar = tqdm(loader, desc="  [val]  ", unit="batch", leave=False)
    for batch in pbar:
        frames = torch.from_numpy(batch["frames_window"]).to(device)
        state = torch.from_numpy(batch["robot_state_window"]).float().to(device)
        target = torch.from_numpy(batch["actions_window"]).float().to(device)

        # Optional text encoding (only if --use-text was set)
        z_text = None
        if args.use_text:
            B = frames.shape[0]
            if "texts" in batch and batch["texts"]:
                texts = batch["texts"]
            elif args.task_text:
                texts = [args.task_text] * B
            else:
                texts = ["default task"] * B
            z_text = model.text_encoder(texts)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            out = model(frames, state)
            h_current = out["h_seq"][:, -1]
            if args.head_type == "flow":
                # For flow head, sample actions via ODE integration
                actions_pred = model.sample_actions(
                    out["z_v_pooled_seq"], out["z_t_seq"], h_current, z_text=z_text,
                )
                # For loss reporting, also compute the flow-matching loss
                cond = model.intention_head(
                    out["z_v_pooled_seq"], out["z_t_seq"], h_current, z_text=z_text,
                )
                loss = model.intention_head.loss(target, cond)
            else:
                actions_pred = model.predict_actions(
                    out["z_v_pooled_seq"], out["z_t_seq"], h_current, z_text=z_text,
                )
                loss = F.mse_loss(actions_pred, target)

        # Per-dim error accumulation (across batch and time)
        diff = (actions_pred - target).detach().float().cpu().numpy()
        B, T, D = diff.shape
        per_dim_squared += (diff ** 2).sum(axis=(0, 1))  # (D,)
        per_dim_abs += np.abs(diff).sum(axis=(0, 1))      # (D,)
        n_samples += B * T

        losses.append(loss.item())
        actions_pred_list.append(actions_pred.detach().abs().mean().item())
        pbar.set_postfix(mse=f"{loss.item():.5f}")

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    avg_action = float(np.mean(actions_pred_list)) if actions_pred_list else 0.0

    # Per-dim metrics (only meaningful if action_dim=6)
    per_dim_metrics = {}
    if n_samples > 0:
        # Per-dim MSE and MAE
        per_dim_mse = per_dim_squared / n_samples
        per_dim_mae = per_dim_abs / n_samples
        per_dim_metrics = {
            "pos_mse": float((per_dim_mse[0] + per_dim_mse[1] + per_dim_mse[2]) / 3),
            "rot_mse": float((per_dim_mse[3] + per_dim_mse[4] + per_dim_mse[5]) / 3),
            "grip_mse": float(per_dim_mse[5]) if len(per_dim_mse) > 5 else 0.0,
            "pos_mae": float((per_dim_mae[0] + per_dim_mae[1] + per_dim_mae[2]) / 3),
            "rot_mae": float((per_dim_mae[3] + per_dim_mae[4] + per_dim_mae[5]) / 3),
            # Per-axis (for fine-grained debugging)
            "px_mse": float(per_dim_mse[0]),
            "py_mse": float(per_dim_mse[1]),
            "pz_mse": float(per_dim_mse[2]),
            "rx_mse": float(per_dim_mse[3]),
            "ry_mse": float(per_dim_mse[4]),
            "rz_mse": float(per_dim_mse[5]),
        }

    return avg_loss, avg_action, per_dim_metrics


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
    # NOTE: --num-cameras removed; auto-derived from --cameras.
    parser.add_argument("--val-split", type=float, default=0.1)
    # Model
    # NOTE: --pretrained removed; warm-starting is rare and complicates the CLI.
    # NOTE: --vision-dim, --state-dim, --mamba-output-dim hardcoded to v3 defaults.
    # NOTE: --mamba-d-state, --mamba-d-conv, --mamba-expand are Mamba internals; hardcoded.
    # NOTE: --use-patch-tokens / --no-patch-tokens removed; always on for v3.
    # NOTE: DINOv2 is always frozen (no flag). State encoder is always
    # trainable (no flag) since we don't have pretrained weights for it.
    # NOTE: --action-dim hardcoded to 6 (OSC pose deltas).
    parser.add_argument("--chunk-size", type=int, default=10)
    # Head selection
    parser.add_argument("--head-type", choices=["transformer", "mamba", "hybrid", "flow"],
                        default="mamba",
                        help="Which head architecture: transformer, mamba, hybrid, or flow")
    parser.add_argument("--use-history", action="store_true", default=True,
                        help="Include Mamba history component (h) in head input.")
    parser.add_argument("--no-history", dest="use_history", action="store_false",
                        help="Disable Mamba history component.")
    # V3 architecture dimensions (configurable, no longer hardcoded)
    parser.add_argument("--vision-dim", type=int, default=256,
                        help="Per-patch vision dim after projection (default 256).")
    parser.add_argument("--state-dim", type=int, default=256,
                        help="Robot state encoder output dim (default 256).")
    parser.add_argument("--mamba-output-dim", type=int, default=512,
                        help="Mamba output dim (history state h). Default 512.")
    parser.add_argument("--mamba-d-state", type=int, default=16,
                        help="Mamba inner state dim (default 16).")
    parser.add_argument("--mamba-d-conv", type=int, default=4,
                        help="Mamba conv kernel size (default 4).")
    parser.add_argument("--mamba-expand", type=int, default=2,
                        help="Mamba block expansion factor (default 2).")
    parser.add_argument("--action-dim", type=int, default=6,
                        help="Action output dim (default 6 for OSC).")
    parser.add_argument("--no-patch-tokens", dest="use_patch_tokens",
                        action="store_false", default=True,
                        help="Use CLS token instead of patch tokens from DINOv2.")
    parser.set_defaults(use_patch_tokens=True)
    # Text modality (optional)
    parser.add_argument("--use-text", action="store_true", default=True,
                        help="Enable text encoder + text-conditioned head.")
    parser.add_argument("--text-dim", type=int, default=256,
                        help="Text encoder output dim (default 256).")
    parser.add_argument("--task-text", type=str, default=None,
                        help="Task description for text conditioning (default: auto from dataset).")
    # IntentionTransformerHead params
    parser.add_argument("--head-d-model", type=int, default=384,
                        help="IntentionTransformerHead model dimension (default: 384)")
    parser.add_argument("--head-nhead", type=int, default=4,
                        help="IntentionTransformerHead number of head (default: 4)")
    parser.add_argument("--head-num-layers", type=int, default=2,
                        help="IntentionTransformerHead number of layer (default: 2)")
    parser.add_argument("--head-dim-ff", type=int, default=1024,
                        help="IntentionTransformerHead fead-forward dimension (default: 1024)")
    # Training
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None,
                        help="Custom run folder name (default: run_N auto-incremented).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    # NOTE: --loss-mode removed; always 'action' for v3.
    # NOTE: --bf16 / --no-bf16 removed; BF16 is always on for speed.
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Cap steps per epoch (0 = use full loader).")
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="align-intention")
    parser.add_argument("--wandb-run", default=None)
    # Other
    parser.add_argument("--seed", type=int, default=42)
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

    # Output dir
    out_dir = Path(args.output_dir)
    if len(args.data) == 1:
        ds_name = Path(args.data[0]).stem
    else:
        ds_name = "+".join(Path(p).stem for p in args.data)
    out_dir = out_dir / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        # Custom name provided
        out_dir = out_dir / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Auto-create a new run_N subfolder to avoid overwriting previous runs.
        # e.g. checkpoints/v3/libero_object/run_1, run_2, ...
        existing_runs = sorted(
            int(p.name.split("_")[1])
            for p in out_dir.glob("run_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        next_run = (existing_runs[-1] + 1) if existing_runs else 1
        out_dir = out_dir / f"run_{next_run}"
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

    # Determine num_cameras from --cameras argument (auto-derived)
    num_cameras = len(args.cameras)
    print(f"  Cameras: {num_cameras} (from --cameras {args.cameras})")

    # Model
    print("\n  Building model...")
    model = build_model(args, num_cameras, device)

    # Configure trainable parameters
    # - DINOv2 backbone: always frozen (ImageNet-pretrained)
    # - Vision projection: always trainable (small, adapts generic features)
    # - State encoder: always trainable (no pretrained weights available)
    # - Intention encoder + head: always trainable
    # Freeze the DINOv2 backbone (the only frozen component)
    for p in model.vision_encoder.backbone.parameters():
        p.requires_grad = False
    print("  DINOv2 backbone: frozen (ImageNet-pretrained)")
    # Enable training for everything else
    if model.intention_encoder is not None:
        for p in model.intention_encoder.parameters():
            p.requires_grad = True
    for p in model.intention_head.parameters():
        p.requires_grad = True
    # If text encoder exists, train the projection (frozen CLIP under it)
    if model.text_encoder is not None:
        for p in model.text_encoder.projection.parameters():
            p.requires_grad = True
        print("  Text encoder: CLIP frozen, projection trainable")
    # State encoder and vision projection are trainable by default (no action needed)
    print("  Trainable: vision projection + state encoder + intention encoder + head"
          + (" + text projection" if model.text_encoder is not None else ""))
    # Collect trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params:   {n_trainable:,}")
    print(f"  Total model params: {n_total:,}")
    # Update wandb config with model-derived info
    if wandb_trainer.enabled:
        wandb_trainer.run.config.update({
            "num_cameras": num_cameras,
            "n_params": n_total,
            "n_trainable_params": n_trainable,
        })
        # Explicitly log v3 architecture dimensions (in case args rename)
        wandb_trainer.run.config.update({
            "v3/vision_dim": args.vision_dim,
            "v3/state_dim": args.state_dim,
            "v3/mamba_output_dim": args.mamba_output_dim,
            "v3/mamba_d_state": args.mamba_d_state,
            "v3/mamba_d_conv": args.mamba_d_conv,
            "v3/mamba_expand": args.mamba_expand,
            "v3/action_dim": args.action_dim,
            "v3/use_patch_tokens": args.use_patch_tokens,
            "v3/use_history": args.use_history,
            "v3/use_text": args.use_text,
            "v3/text_dim": args.text_dim,
        })
        # Watch gradients (for monitoring)
        wandb_trainer.watch(model, log="gradients", log_freq=200, log_graph=False)
    if model.intention_encoder is not None:
        print("  Training: vision projection + state encoder + intention encoder + head")
        print("  (DINOv2 backbone frozen)")
    else:
        print("  Training: vision projection + state encoder + head (no Mamba history)")
        print("  (DINOv2 backbone frozen)")

    # Optimizer — single LR group (only trainable params receive gradients)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"  Optimizer: 1 LR group (lr={args.lr:.2e}, {n_trainable:,} trainable params)")

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
        val_loss, val_action, val_per_dim = validate(model, val_loader, device, args)
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train: loss={train_loss:.5f} a_mean={train_action:.4f}  |  "
              f"val: loss={val_loss:.5f} a_mean={val_action:.4f}  "
              f"pos_mse={val_per_dim.get('pos_mse', 0):.4f} "
              f"rot_mse={val_per_dim.get('rot_mse', 0):.4f}  "
              f"({elapsed:.0f}s)")

        # Build log dict
        log_dict = {
            "train/loss": train_loss,
            "train/action_mean": train_action,
            "val/loss": val_loss,
            "val/action_mean": val_action,
            "epoch": epoch,
        }
        # Add per-dim metrics (only if action_dim == 6)
        for k, v in val_per_dim.items():
            log_dict[f"val/{k}"] = v
        wandb_trainer.log(log_dict, step=epoch)

        log_record = {
            "stage": "intention",
            "epoch": epoch,
            "train/loss": train_loss,
            "train/action_mean": train_action,
            "val/loss": val_loss,
            "val/action_mean": val_action,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        # Add per-dim to JSONL log
        for k, v in val_per_dim.items():
            log_record[f"val/{k}"] = v
        log_fp.write(json.dumps(log_record) + "\n")
        log_fp.flush()

        # Save best (based on val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": config_snapshot,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_action_mean": val_action,
                "val_per_dim": val_per_dim,
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
