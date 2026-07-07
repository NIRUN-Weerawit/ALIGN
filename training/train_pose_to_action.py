#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PoseDeltaToAction — training script.

Trains the PoseDeltaToAction connector to map EEF pose deltas (m, rad)
→ OSC actions (normalized). Replaces fixed per-axis scales with a
learned inverse-dynamics model.

Training data: every LIBERO demo step already has
  (pose_delta = poses[t+1] - poses[t], expert_action[t])
No new data collection needed — just mine from HDF5.

Benefits:
  - Learns coupling (rotation affects position and vice versa)
  - Fine-tune only this ~18K param layer for a new robot/environment
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
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pose_to_action import PoseDeltaToAction
from training.wandb_utils import init_wandb, log_metrics


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
    """

    def __init__(self, h5_paths: List[str], max_samples: int = 0):
        self.samples = []
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
                if "poses" in ep:
                    poses = ep["poses"][:]
                elif "noisy_poses" in ep:
                    poses = ep["noisy_poses"][:]
                else:
                    continue
                if "actions" not in ep:
                    continue
                actions = ep["actions"][:]

                N = min(len(poses), len(actions))
                if N < 2:
                    continue

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
    cameras: Optional[List[str]] = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    hidden_dim: int = 128,
    max_samples: int = 0,
    seed: int = 0,
    use_bf16: bool = True,
    enable_wandb: bool = False,
    wandb_project: str = "align-pose-to-action",
    wandb_run: Optional[str] = None,
    device: Optional[str] = None,
    # Bounded output: per-dim action range. If None, auto-compute from data.
    action_min: Optional[List[float]] = None,
    action_max: Optional[List[float]] = None,
    # Margin: how much extra padding around the per-dim training-data range
    # to allow for outliers. Default 1.05 = 5% padding on each side.
    bound_margin: float = 1.05,
) -> str:
    """Train PoseDeltaToAction from HDF5 demo data (bounded output).

    The model's output is hard-bounded to [action_min[d], action_max[d]] per
    dim via tanh × range + mid. If action_min/max are not provided, they
    are auto-computed from the per-dim min/max of the training actions,
    with `bound_margin` padding on each side (1.05 = 5% extra on each
    end of the data range, so the model can slightly exceed the data
    range during training but not blow up).

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

    # ---- Auto-compute per-dim action range from data if not provided ----
    if action_min is None or action_max is None:
        print("  Auto-computing per-dim action range from data...")
        all_actions = []
        for h5_path in data_paths:
            with h5py.File(h5_path, "r") as f:
                for ep_key in sorted(k for k in f.keys() if k.startswith("ep_")):
                    ep = f[ep_key]
                    if "actions" not in ep:
                        continue
                    acts = ep["actions"][:, :6]
                    all_actions.append(acts)
        all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        d_min = all_actions.min(axis=0)
        d_max = all_actions.max(axis=0)
        if action_min is None:
            # Margin extends the bound outward — for positive data_min,
            # margin means: min = data_min * margin (further from 0).
            # For negative data_min, margin means: min = data_min * margin
            # (further from 0, more negative).
            action_min = [float(x) for x in (d_min * bound_margin)]
        if action_max is None:
            action_max = [float(x) for x in (d_max * bound_margin)]
        print(f"  data action min: {d_min.tolist()}")
        print(f"  data action max: {d_max.tolist()}")
        print(f"  bounded action_min: {action_min}")
        print(f"  bounded action_max: {action_max}")

    print(f"=== PoseDeltaToAction Training (bounded) ===")
    print(f"  Run:           {out_dir}")
    print(f"  Data:          {data_paths}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {epochs}")
    print(f"  LR:            {lr}")
    print(f"  Hidden dim:    {hidden_dim}")

    # ---- Config ----
    config = {
        "model": "pose-delta-to-action-bounded",
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
        "cameras": cameras if cameras else ["wrist_image"],
        "action_min": action_min,
        "action_max": action_max,
        "bound_margin": bound_margin,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- W&B ----
    wandb_trainer = init_wandb(
        project=wandb_project, name=wandb_run, config=config,
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:           {'enabled' if wandb_trainer.enabled else 'disabled'}")

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

    # ---- Model (bounded) ----
    model = PoseDeltaToAction(
        pose_dim=6, action_dim=6,
        hidden_dim=hidden_dim,
        action_min=action_min,
        action_max=action_max,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params:        {n_params:,}")
    print(f"  Bounded range: {list(zip(action_min, action_max))}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                # Inverse-range weighted MSE: penalize per-dim squared errors
                # by 1/range_d so a 10% error on any dim is treated equally
                # (rotation dims have ~5x smaller range, so they get ~5x
                # more weight than position dims).
                # loss = mean over batch of [ sum_d (err_d^2 / range_d) ]
                err = pred - action
                loss = (err * err / model.action_range).sum(dim=-1).mean()

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
        val_loss = 0.0            # action-space MSE per element (comparable to old)
        val_wloss = 0.0           # inverse-range weighted MSE (matches training loss)
        val_n = 0
        val_abs_errors = []
        val_rel_errors = []
        # Per-dim MSE accumulator — shows which dims are best/worst predicted
        per_dim_se_sum = torch.zeros(model.action_dim, device=device)
        per_dim_n = 0
        with torch.no_grad():
            for batch in val_loader:
                pose_delta = batch["pose_delta"].to(device)
                action = batch["action"].to(device)

                with autocast_ctx:
                    pred = model(pose_delta)
                # Action-space MSE (unweighted, comparable to old runs)
                loss = F.mse_loss(pred, action)
                val_loss += float(loss) * len(action)
                # Inverse-range weighted MSE (matches training loss)
                err = pred.float() - action.float()
                wloss = (err * err / model.action_range).sum(dim=-1).mean()
                val_wloss += float(wloss) * len(action)
                val_n += len(action)

                abs_err = (pred.float() - action.float()).abs()
                val_abs_errors.extend(abs_err.mean(dim=-1).cpu().numpy().tolist())

                rel_err = abs_err.mean(dim=-1) / (action.float().abs().mean(dim=-1) + 1e-6)
                val_rel_errors.extend(rel_err.cpu().numpy().tolist())

                per_dim_se_sum += ((pred.float() - action.float()) ** 2).sum(dim=0)
                per_dim_n += pred.shape[0]

        val_mse = val_loss / max(val_n, 1)
        val_wmse = val_wloss / max(val_n, 1)
        val_mae = float(np.mean(val_abs_errors)) if val_abs_errors else 0.0
        val_rel = float(np.mean(val_rel_errors)) if val_rel_errors else 0.0
        val_acc = float(np.mean([1.0 if e < 0.1 else 0.0 for e in val_rel_errors])) if val_rel_errors else 0.0
        per_dim_mse = (per_dim_se_sum / max(per_dim_n, 1)).cpu().tolist()

        # Log
        log_entry = {
            "epoch": epoch,
            "train_wmse": train_mse,    # inverse-range weighted MSE (training loss)
            "val_mse": val_mse,         # action-space MSE per element (unweighted)
            "val_wmse": val_wmse,       # inverse-range weighted MSE (matches training)
            "val_mae": val_mae,
            "val_rel_error": val_rel,
            "val_per_dim_mse": per_dim_mse,
            "val_accuracy_10pct": val_acc,
            "epoch_time_sec": time.time() - t_epoch,
        }
        log_fp.write(json.dumps(log_entry) + "\n")
        log_fp.flush()
        log_metrics(wandb_trainer, log_entry, step=epoch)

        print(
            f"  ep {epoch:3d}  train_wmse={train_mse:.6f}  "
            f"val_mse={val_mse:.6f}  val_wmse={val_wmse:.6f}  "
            f"val_mae={val_mae:.4f}  val_rel={val_rel:.3f}  "
            f"val_acc(10%)={val_acc:.3f}  "
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
    wandb_trainer.finish()
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
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Camera views used in the source data (logged for traceability)")
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
    p.add_argument("--enable-wandb", action="store_true")
    p.add_argument("--wandb-project", default="align-pose-to-action")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--action-min", type=float, nargs=6, default=None,
                   metavar=("X", "Y", "Z", "AX", "AY", "AZ"),
                   help="Per-dim lower bound for bounded output. Default: auto from data × bound_margin.")
    p.add_argument("--action-max", type=float, nargs=6, default=None,
                   metavar=("X", "Y", "Z", "AX", "AY", "AZ"),
                   help="Per-dim upper bound for bounded output. Default: auto from data × bound_margin.")
    p.add_argument("--bound-margin", type=float, default=1.05,
                   help="Padding around the auto-computed per-dim data range. "
                        "1.05 = 5%% extra on each side. Ignored if --action-min/max are set.")
    args = p.parse_args()

    train_pose_to_action(
        data_paths=args.data,
        output_dir=args.output_dir,
        cameras=args.cameras,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        hidden_dim=args.hidden_dim,
        max_samples=args.max_samples,
        seed=args.seed,
        use_bf16=args.bf16,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        device=args.device,
        action_min=args.action_min,
        action_max=args.action_max,
        bound_margin=args.bound_margin,
    )


if __name__ == "__main__":
    main()