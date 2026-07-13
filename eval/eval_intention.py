#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN Intention Head eval — Mamba-based.

Loads a trained ALIGNIntentionModel and evaluates on a held-out split.

Usage:
    python3 eval/eval_intention.py --data data/libero_spatial.h5 \\
        --checkpoint checkpoints/intention/.../intention_best.pt \\
        --cameras wrist_image --chunk-size 10
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Disable cuDNN band-aid
import torch
torch.backends.cudnn.enabled = False

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.align_dataset import ALIGNDataset, MultiALIGNDataset, head_collate
from models.align_intention import ALIGNIntentionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="ALIGN Intention Head eval (v3: Mamba-based)"
    )
    parser.add_argument("--data", nargs="+", required=True,
                        help="Path(s) to HDF5 data file(s)")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to intention_best.pt")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--vision-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--mamba-output-dim", type=int, default=512)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--num-cameras", type=int, default=0,
                        help="0 = auto-detect from data")
    parser.add_argument("--loss-mode", choices=["action", "delta"], default="action")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-batches", type=int, default=0,
                        help="If >0, limit eval to N batches (for quick smoke)")
    parser.add_argument("--out-json", default=None,
                        help="Optional path to write metrics as JSON")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, args):
    """Run the model on the val split and report metrics."""
    model.eval()
    all_losses = []
    all_per_dim_mse = []
    all_per_dim_mae = []
    all_targets = []
    all_preds = []
    for batch_idx, batch in enumerate(loader):
        if args.n_batches > 0 and batch_idx >= args.n_batches:
            break
        frames = batch["frames_window"].to(device)
        state = batch["robot_state_window"].to(device).float()
        if args.loss_mode == "action":
            target = batch["actions_window"].to(device).float()
        else:
            target = batch["delta_target"].to(device).float()

        out = model(frames, state)
        h_current = out["h_seq"][:, -1]
        actions_pred = model.predict_actions(
            out["z_v_pooled_seq"], out["z_t_seq"], h_current,
        )
        loss = F.mse_loss(actions_pred, target)
        all_losses.append(loss.item())

        # Per-dim metrics
        diff = (actions_pred - target).detach().cpu().numpy()  # (B, K, 6)
        per_dim_mse = (diff ** 2).mean(axis=(0, 1))  # (6,)
        per_dim_mae = np.abs(diff).mean(axis=(0, 1))  # (6,)
        all_per_dim_mse.append(per_dim_mse)
        all_per_dim_mae.append(per_dim_mae)

        all_targets.append(target.cpu().numpy())
        all_preds.append(actions_pred.detach().cpu().numpy())

    if not all_losses:
        return None

    avg_loss = float(np.mean(all_losses))
    avg_per_dim_mse = np.mean(all_per_dim_mse, axis=0)  # (6,)
    avg_per_dim_mae = np.mean(all_per_dim_mae, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    # Relative error (per-dim)
    target_std = targets.std(axis=(0, 1)) + 1e-6
    rel_err = avg_per_dim_mse ** 0.5 / target_std

    return {
        "mse": avg_loss,
        "per_dim_mse": avg_per_dim_mse.tolist(),
        "per_dim_mae": avg_per_dim_mae.tolist(),
        "per_dim_rel_error": rel_err.tolist(),
        "n_samples": int(targets.shape[0]),
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  Device: {device}")

    # Load checkpoint first to get config
    print(f"  Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    # Override args with ckpt config
    if "chunk_size" in cfg:
        args.chunk_size = cfg["chunk_size"]
    if "vision_dim" in cfg:
        args.vision_dim = cfg["vision_dim"]
    if "state_dim" in cfg:
        args.state_dim = cfg["state_dim"]
    if "mamba_output_dim" in cfg:
        args.mamba_output_dim = cfg["mamba_output_dim"]
    if "action_dim" in cfg:
        args.action_dim = cfg["action_dim"]
    if "num_cameras" in cfg:
        args.num_cameras = cfg["num_cameras"]
    if "loss_mode" in cfg:
        args.loss_mode = cfg["loss_mode"]
    print(f"  Config: chunk_size={args.chunk_size}, vision_dim={args.vision_dim}, "
          f"state_dim={args.state_dim}, mamba_dim={args.mamba_output_dim}, "
          f"num_cameras={args.num_cameras}, loss_mode={args.loss_mode}")

    # Build dataset
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
    print(f"  Dataset: {n_total} total, {n_val} val samples")

    # Build loader
    collate_fn = lambda b: head_collate(
        b, chunk_size=args.chunk_size, vision_window_size=args.chunk_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn, num_workers=0,
    )

    # Auto-detect num_cameras if not specified
    if args.num_cameras == 0:
        sample = val_ds[0]
        frames_shape = sample["frames_window"].shape
        if frames_shape.ndim == 5:
            num_cameras = frames_shape[2]
        else:
            num_cameras = 1
        args.num_cameras = num_cameras
        print(f"  Auto-detected num_cameras={num_cameras}")

    # Build model
    print(f"\n  Building model...")
    model = ALIGNIntentionModel(
        vision_dim=args.vision_dim,
        state_dim=args.state_dim,
        mamba_output_dim=args.mamba_output_dim,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        num_cameras=args.num_cameras,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Load weights
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"  Loaded weights from {args.checkpoint}")

    # Evaluate
    print(f"\n  Evaluating on val split ({n_val} samples)...")
    t_start = time.time()
    metrics = evaluate(model, val_loader, device, args)
    elapsed = time.time() - t_start
    if metrics is None:
        print("  No samples evaluated!")
        return

    dim_names = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
    print(f"\n=== Intention Head Eval Results ({elapsed:.1f}s) ===")
    print(f"  N samples:     {metrics['n_samples']}")
    print(f"  MSE:           {metrics['mse']:.4f}")
    print(f"\n  Per-dimension:")
    print(f"    {'dim':<10} {'MSE':>10} {'MAE':>10} {'rel_err':>10}")
    for i, name in enumerate(dim_names):
        print(f"    {name:<10} {metrics['per_dim_mse'][i]:>10.6f} "
              f"{metrics['per_dim_mae'][i]:>10.6f} "
              f"{metrics['per_dim_rel_error'][i]:>10.3f}")

    # Save metrics
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()
