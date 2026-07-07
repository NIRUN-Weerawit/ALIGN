#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PoseDeltaToAction — learned inverse-dynamics connector.

Maps EEF pose deltas (m, rad) → OSC actions (normalized), replacing
fixed per-axis scales. Captures non-linear Jacobian coupling and
configuration-dependent dynamics that fixed scales miss.

Training data: every LIBERO demo step already has
  (pose_delta = clean_pose[t+1] - noisy_pose[t], expert_action[t])
No new data collection needed — just mine from HDF5.

Benefits:
  - Learns coupling (rotation affects position and vice versa)
  - Fine-tune only this ~30K param layer for a new robot/environment
  - Sim-to-real: retrain just this connector on the real robot's dynamics
  - No manual scale measurement per suite

Usage:
    PYTHONNOUSERSITE=1 python training/train_pose_to_action.py \\
        --data h5_data/libero_spatial.h5 \\
        --output-dir checkpoints/pose_to_action
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# Model
# ================================================================

class PoseDeltaToAction(nn.Module):
    """Maps EEF pose delta (m, rad) → OSC action (normalized).

    Input:  (B, 6) — [dx, dy, dz, dax, day, daz] in meters and radians
    Output: (B, 6) — [ax, ay, az, arx, ary, arz] in OSC action space

    The mapping is the inverse dynamics of the OSC controller. For a
    simple controller with a fixed Jacobian, this reduces to per-axis
    scaling. For real controllers with configuration-dependent Jacobians,
    the learned model captures the non-linear coupling.
    """

    def __init__(self, pose_dim: int = 6, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.pose_dim = pose_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, pose_delta: torch.Tensor) -> torch.Tensor:
        return self.net(pose_delta)


# ================================================================
# Dataset
# ================================================================

class PoseActionDataset(Dataset):
    """Mine (pose_delta, expert_action) pairs from HDF5.

    For each episode, for each timestep t:
      pose_delta = poses[t+1] - poses[t]   (the actual EEF movement)
      action = actions[t]                   (the OSC command that caused it)

    The model learns the mapping: "given I want to move by this delta,
    what OSC action do I send?" — i.e. the inverse dynamics of the
    OSC controller.

    Uses 'poses' (or 'noisy_poses') for poses, 'actions' for expert actions.
    """

    def __init__(self, h5_paths: List[str], max_samples: int = 0):
        self.samples = []  # list of (pose_delta, action) tuples
        for h5_path in h5_paths:
            self._load_h5(h5_path)
        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  PoseActionDataset: {len(self.samples)} samples from {len(h5_paths)} file(s)")

    def _load_h5(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            ep_keys = sorted([k for k in f.keys() if k.startswith("ep_")])
            for ep_key in ep_keys:
                ep = f[ep_key]
                # Poses — prefer 'poses' (clean), fallback 'noisy_poses'
                if "poses" in ep:
                    poses = ep["poses"][:]
                elif "noisy_poses" in ep:
                    poses = ep["noisy_poses"][:]
                else:
                    continue
                # Actions
                if "actions" not in ep:
                    continue
                actions = ep["actions"][:]

                N = min(len(poses), len(actions))
                if N < 2:
                    continue

                # For each timestep: pose_delta = poses[t+1] - poses[t]
                # This is the actual EEF movement caused by action[t]
                for t in range(N - 1):
                    pose_delta = (poses[t + 1, :6] - poses[t, :6]).astype(np.float32)
                    action = actions[t, :6].astype(np.float32)
                    self.samples.append((pose_delta, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pd, a = self.samples[idx]
        return {"pose_delta": pd, "action": a}


# ================================================================
# Training
# ================================================================

def train_pose_to_action(
    data_paths: List[str],
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    hidden_dim: int = 128,
    max_samples: int = 0,
    seed: int = 0,
    use_bf16: bool = True,
    device: Optional[str] = None,
) -> str:
    """Train PoseDeltaToAction from HDF5 demo data.

    Returns:
        Path to the best checkpoint.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Run directory ----
    ds_name = "+".join(Path(p).stem for p in data_paths)
    base_dir = Path(output_dir) / ds_name
    existing = sorted(base_dir.glob("run_*")) if base_dir.exists() else []
    next_run = max([int(d.name.split("_")[-1]) for d in existing
                    if d.name.split("_")[-1].isdigit()] + [0]) + 1
    out_dir = base_dir / f"run_{next_run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PoseDeltaToAction Training ===")
    print(f"  Run:           {out_dir}")
    print(f"  Data:          {data_paths}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {epochs}")
    print(f"  LR:            {lr}")
    print(f"  Hidden dim:    {hidden_dim}")

    # ---- Dataset ----
    full_ds = PoseActionDataset(data_paths, max_samples=max_samples)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    indices = list(range(n_total))
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    print(f"  {n_train} train, {n_val} val samples")

    train_loader = DataLoader(
        full_ds, batch_size=batch_size, drop_last=True,
        sampler=train_indices, num_workers=0,
    )
    val_loader = DataLoader(
        full_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        sampler=val_indices, num_workers=0,
    )

    # ---- Model ----
    model = PoseDeltaToAction(pose_dim=6, action_dim=6, hidden_dim=hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params:        {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Config ----
    config = {
        "model": "pose-delta-to-action",
        "data": [str(p) for p in data_paths],
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "hidden_dim": hidden_dim,
        "max_samples": max_samples,
        "seed": seed,
        "use_bf16": use_bf16,
        "device": str(device),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- Training loop ----
    best_val_mse = float("inf")
    best_ckpt = out_dir / "pose_to_action_best.pt"
    last_ckpt = out_dir / "pose_to_action_last.pt"
    log_path = out_dir / "log.jsonl"
    log_fp = open(log_path, "w")

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda"
        else torch.amp.autocast("cuda", enabled=False)
    )

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        pbar = tqdm(train_loader, desc=f"ep {epoch}/{epochs}")
        for batch in pbar:
            pose_delta = batch["pose_delta"].to(device)
            action = batch["action"].to(device)

            with autocast_ctx:
                pred = model(pose_delta)
                loss = F.mse_loss(pred, action)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.detach())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        train_mse = epoch_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_abs_errors = []
        val_rel_errors = []
        with torch.no_grad():
            for batch in val_loader:
                pose_delta = batch["pose_delta"].to(device)
                action = batch["action"].to(device)

                with autocast_ctx:
                    pred = model(pose_delta)
                loss = F.mse_loss(pred, action)
                val_loss += float(loss) * len(action)
                val_n += len(action)

                # Per-sample absolute error
                abs_err = (pred.float() - action.float()).abs()  # (B, 6)
                val_abs_errors.extend(abs_err.mean(dim=-1).cpu().numpy().tolist())

                # Relative error: |pred - target| / (|target| + eps)
                rel_err = abs_err.mean(dim=-1) / (action.float().abs().mean(dim=-1) + 1e-6)
                val_rel_errors.extend(rel_err.cpu().numpy().tolist())

        val_mse = val_loss / max(val_n, 1)
        val_mae = float(np.mean(val_abs_errors)) if val_abs_errors else 0.0
        val_rel = float(np.mean(val_rel_errors)) if val_rel_errors else 0.0
        # Accuracy: fraction of samples with relative error < 0.1 (10%)
        val_acc = float(np.mean([1.0 if e < 0.1 else 0.0 for e in val_rel_errors])) if val_rel_errors else 0.0

        # Log
        log_entry = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_rel_error": val_rel,
            "val_accuracy_10pct": val_acc,
            "epoch_time_sec": time.time() - t_epoch,
        }
        log_fp.write(json.dumps(log_entry) + "\n")
        log_fp.flush()

        print(
            f"  ep {epoch:3d}  train_mse={train_mse:.6f}  "
            f"val_mse={val_mse:.6f}  val_mae={val_mae:.4f}  "
            f"val_rel={val_rel:.3f}  val_acc(10%)={val_acc:.3f}  "
            f"({time.time() - t_epoch:.1f}s)"
        )

        # Save best
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {**config, "val_mse": val_mse, "epoch": epoch},
            }, best_ckpt)
            print(f"    ↳ new best (val_mse={val_mse:.6f}), saved to {best_ckpt}")

    # Save last
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {**config, "val_mse": val_mse, "epoch": epochs},
    }, last_ckpt)

    log_fp.close()
    print(f"\nDone. Best val_mse={best_val_mse:.6f}. Best checkpoint: {best_ckpt}")
    return str(best_ckpt)


# ================================================================
# CLI
# ================================================================

def main():
    p = argparse.ArgumentParser(
        description="Train PoseDeltaToAction connector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, nargs="+",
                   help="Path(s) to HDF5 file(s)")
    p.add_argument("--output-dir", default="./checkpoints/pose_to_action")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap dataset size (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    train_pose_to_action(
        data_paths=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        hidden_dim=args.hidden_dim,
        max_samples=args.max_samples,
        seed=args.seed,
        use_bf16=args.bf16,
        device=args.device,
    )


if __name__ == "__main__":
    main()