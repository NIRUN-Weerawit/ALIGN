#!/usr/bin/env python3
"""Evaluate a trained AssistantHead checkpoint.

Loads the encoder+mixer and assistant head, runs the assistant head
on a batch of validation data, and reports:
  - Per-output statistics (mean, std, range)
  - Comparison to targets (MSE, MAE)
  - Per-step cosine similarity (does each goal_k match the target goal_k?)
  - Mode collapse check (output variance over many samples)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from data.align_dataset import ALIGNDataset, head_collate
from torch.utils.data import DataLoader


def load_assistant_head(
    heads_checkpoint: str,
    encoder_checkpoint: str,
    device: torch.device,
):
    """Load encoder+mixer and assistant head weights into ALIGNModel."""
    # Load encoder checkpoint to get config
    enc_ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    enc_cfg = enc_ckpt.get("config", {})

    # Build ALIGNModel with the right mixer_dim
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=5,  # default
        use_text=True,
        device=str(device),
        mixer_dim=enc_cfg.get("mixer_dim", 512),
        num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
    ).to(device)

    # Load encoder weights
    if "trainable_state_dict" in enc_ckpt:
        model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    elif "model_state_dict" in enc_ckpt:
        model.load_state_dict(enc_ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model.freeze_all_encoders()
    model.eval()

    # Load heads checkpoint
    heads_ckpt = torch.load(heads_checkpoint, map_location=device, weights_only=False)
    heads_cfg = heads_ckpt.get("config", {})
    chunk_size = heads_cfg.get("chunk_size", 5)

    # If the heads were trained with a different chunk_size, rebuild ALIGNModel
    if chunk_size != 5:
        model = ALIGNModel(
            embed_dim=256,
            chunk_size=chunk_size,
            use_text=True,
            device=str(device),
            mixer_dim=enc_cfg.get("mixer_dim", 512),
            num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
        ).to(device)
        if "trainable_state_dict" in enc_ckpt:
            model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
        model.freeze_backbone()
        model.freeze_all_encoders()
        model.eval()

    # Load the head weights
    if "trainable_state_dict" in heads_ckpt:
        head_state = heads_ckpt["trainable_state_dict"]
        current_state = model.state_dict()
        compatible = {
            k: v for k, v in head_state.items()
            if k in current_state and v.shape == current_state[k].shape
        }
        skipped = len(head_state) - len(compatible)
        if compatible:
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            print(f"  Loaded {len(compatible)}/{len(head_state)} params (skipped {skipped} shape mismatches)")
    elif "model_state_dict" in heads_ckpt:
        model.load_state_dict(heads_ckpt["model_state_dict"], strict=False)

    print(f"  Heads config: {heads_cfg}")
    print(f"  Heads epoch: {heads_ckpt.get('epoch', '?')}")
    print(f"  Heads loss: {heads_ckpt.get('loss', '?'):.6f}")

    return model, chunk_size


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AssistantHead checkpoint")
    parser.add_argument("--heads-checkpoint", required=True,
                        help="Path to assistant_best.pt or heads_best.pt")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Path to pretrained encoder+mixer checkpoint")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--n-batches", type=int, default=20,
                        help="Number of validation batches to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n=== Assistant Head Evaluation ===")
    print(f"  Heads:     {args.heads_checkpoint}")
    print(f"  Encoder:   {args.encoder_checkpoint}")
    print(f"  Data:      {args.data}")
    print(f"  Device:    {device}")

    # Load model
    print("\nLoading...")
    model, chunk_size = load_assistant_head(
        args.heads_checkpoint, args.encoder_checkpoint, device,
    )
    print(f"  Loaded model with chunk_size={chunk_size}")

    # Load dataset
    ds = ALIGNDataset(args.data, mode="head", traj_window=5)
    val_split = 0.1
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    g = torch.Generator()
    g.manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    print(f"  Dataset: {len(train_ds)} train, {len(val_ds)} val")

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size, vision_window_size=chunk_size),
    )

    # Evaluation
    model.eval()
    n_samples = 0
    sum_se = 0.0  # sum of squared errors
    sum_ae = 0.0  # sum of absolute errors
    all_pred_stds = []  # to check mode collapse
    per_step_cos = [[] for _ in range(chunk_size)]  # cosine sim per step
    per_step_mag_pred = [[] for _ in range(chunk_size)]  # magnitude of prediction per step
    per_step_mag_target = [[] for _ in range(chunk_size)]  # magnitude of target per step
    pred_collected = []  # first few predictions for inspection

    print(f"\nRunning {args.n_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break

            frames = torch.from_numpy(batch["frames"]).to(device)
            texts = batch["texts"]
            noisy_pose = torch.from_numpy(batch["noisy_pose"]).float().to(device)
            current_action = torch.from_numpy(batch["current_action"]).float().to(device)
            traj_view = torch.from_numpy(batch["trajectory"]).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                mixed = model.encode_mixed(frames, traj_view, texts)
                z_v = mixed["z_v"].float()
                z_t = mixed["z_t"].float()
                z_text = mixed["z_text"].float()

                action_pred = model.assistant_head(z_v, z_t, z_text)

            # Compute metrics — single-step action prediction
            # action_pred and current_action are both (B, 6)
            B = action_pred.shape[0]
            n_samples += B
            sum_se += ((action_pred - current_action) ** 2).sum().item()
            sum_ae += (action_pred - current_action).abs().sum().item()

            # Per-dim cosine similarity (since we only have 1 output now)
            cos = F.cosine_similarity(action_pred, current_action, dim=-1)
            per_step_cos[0].extend(cos.cpu().tolist())
            per_step_mag_pred[0].extend(action_pred.norm(dim=-1).cpu().tolist())
            per_step_mag_target[0].extend(current_action.norm(dim=-1).cpu().tolist())

            # Check mode collapse: std of predictions across batch
            all_pred_stds.append(action_pred.std(dim=0).mean().item())

            # Collect first few predictions
            if i == 0:
                pred_collected.append({
                    "current_pose": traj_view[-1,:3].cpu().tolist(),
                    "current_action": current_action[:3].cpu().tolist(),
                    "pred_action": action_pred[:3].cpu().tolist(),
                    "text": texts[:3],
                })

    # Aggregate
    total_elements = n_samples * chunk_size * 6
    mse = sum_se / total_elements
    mae = sum_ae / total_elements
    rmse = np.sqrt(mse)

    print(f"\n{'='*60}")
    print(f"=== Results ({n_samples} samples, {chunk_size} goals × 6 dims) ===")
    print(f"{'='*60}")

    print(f"\nOverall metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    print(f"\nPer-step metrics:")
    print(f"  {'Step':<6}{'Cosine':<12}{'Pred Mag':<14}{'Target Mag':<14}")
    for k in range(chunk_size):
        cos_arr = np.array(per_step_cos[k])
        pmag_arr = np.array(per_step_mag_pred[k])
        tmag_arr = np.array(per_step_mag_target[k])
        print(f"  k={k:<4}"
              f"{cos_arr.mean():<12.4f}"
              f"{pmag_arr.mean():<14.4f}"
              f"{tmag_arr.mean():<14.4f}")

    # Mode collapse check
    avg_pred_std = np.mean(all_pred_stds)
    print(f"\nMode collapse check:")
    print(f"  Avg prediction std across batch: {avg_pred_std:.6f}")
    if avg_pred_std < 0.001:
        print(f"  ⚠️  WARNING: Predictions have near-zero variance. Model may have mode-collapsed.")
        print(f"      (mean prediction magnitude: {np.mean([np.mean(per_step_mag_pred[k]) for k in range(chunk_size)]):.6f})")
        print(f"      (mean target magnitude:    {np.mean([np.mean(per_step_mag_target[k]) for k in range(chunk_size)]):.6f})")
    else:
        print(f"  ✓ Predictions have meaningful variance across samples.")

    # Per-step alignment check
    print(f"\nPer-step alignment with target:")
    print(f"  Step 1 cosine sim is the most important (used at inference).")
    print(f"  Step 1 mean cosine: {np.mean(per_step_cos[0]):.4f}")
    if np.mean(per_step_cos[0]) > 0.7:
        print(f"  ✓ Step 1 alignment is good (cos > 0.7)")
    elif np.mean(per_step_cos[0]) > 0.3:
        print(f"  ⚠️  Step 1 alignment is moderate (0.3 < cos < 0.7)")
    else:
        print(f"  ⚠️  Step 1 alignment is poor (cos < 0.3)")

    # Show sample predictions
    if pred_collected:
        print(f"\nSample predictions (first batch, first 3 samples):")
        sample = pred_collected[0]
        for i in range(min(3, len(sample["text"]))):
            print(f"\n  Sample {i}: task='{sample['text'][i][:50]}...'")
            print(f"    current_pose[0:3]:  {sample['current_pose'][i]}")
            print(f"    current_action[0:3]: {sample['current_action'][i]}")
            print(f"    pred_action[0:3]:   {sample['pred_action'][i]}")


if __name__ == "__main__":
    main()