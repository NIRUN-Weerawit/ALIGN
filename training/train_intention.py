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
  - z_s_seq:           (B, K, state_dim)     — state embedding per step
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
from pyexpat import model
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

from data.align_dataset import ALIGNDataset, MultiALIGNDataset, head_collate, v4_segment_collate
from models.align_intention import ALIGNIntentionModel
from training.wandb_utils import init_wandb


# ================================================================
# Data
# ================================================================

def build_datasets(args):
    """Build train and val datasets from one or more HDF5 files."""
    is_v4 = getattr(args, 'use_intent_tokens', False) or getattr(args, 'use_memory_bank', False)
    if is_v4:
        # V4 collate needs at least H*segment_max_mult frames (default: 20*5=100).
        # __getitem__ returns frames_per_ep + traj_window, so we need enough runway.
        traj_window = max(
            args.history_size * getattr(args, 'segment_max_mult', 5),
            args.chunk_size,
        )
    else:
        traj_window = args.chunk_size

    if len(args.data) == 1:
        ds = ALIGNDataset(
            args.data[0], mode="head",
            traj_window=traj_window, cameras=args.cameras,
        )
    else:
        ds = MultiALIGNDataset(
            args.data, mode="head",
            traj_window=traj_window, cameras=args.cameras,
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
    is_v4 = args.use_intent_tokens or args.use_memory_bank
    if is_v4:
        collate_fn = lambda b: v4_segment_collate(
            b, history_size=args.history_size, chunk_size=args.chunk_size,
            segment_min_mult=args.segment_min_mult,
            segment_max_mult=args.segment_max_mult,
        )
    else:
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
        state_dim=args.state_dim,
        mamba_output_dim=args.mamba_output_dim if args.use_history else 0,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        history_size=args.history_size,
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
        compressed_dim=args.compressed_dim,
        # V4 args
        use_intent_tokens=args.use_intent_tokens,
        num_intent_tokens=args.num_intent_tokens,
        intent_dim=args.intent_dim,
        use_memory_bank=args.use_memory_bank,
        memory_bank_len=args.memory_bank_len,
    )
    model = model.to(device)
    return model


# ================================================================
# V4 Training — Sequential T-loop with persistent memory bank
# ================================================================

def train_v4_epoch(model, loader, optimizer, device, args, max_steps=0):
    """V4 training epoch with sequential T-loop and persistent memory bank.

    Each batch contains variable-length segments. The model processes
    each segment step by step, maintaining a persistent memory bank
    that accumulates across the segment.

    Returns (avg_loss, avg_action_mean).
    """
    model.train()
    losses, actions_pred_list = [], []
    n_steps = min(max_steps, len(loader)) if max_steps else len(loader)
    pbar = tqdm(range(n_steps), total=n_steps, desc="  [train V4]", unit="batch", leave=False)
    step_iter = iter(loader)

    for _ in pbar:
        try:
            batch = next(step_iter)
        except StopIteration:
            break

        Hs = args.history_size
        chunk_size = args.chunk_size

        max_seg_len = batch["frames_segment"].shape[1]
        # print(f"Segment length: {max_seg_len}, batch['frames_segment'].shape: {batch['frames_segment'].shape}")
        frames_seg = torch.from_numpy(batch["frames_segment"]).to(device)  # (B, S, V, H, W, 3)
        states_seg = torch.from_numpy(batch["states_segment"]).float().to(device)  # (B, S, 7)
        actions_seg = torch.from_numpy(batch["actions_segment"]).float().to(device)  # (B, S, 7)
        # print(f"frames_seg shape: {frames_seg.shape}, states_seg shape: {states_seg.shape}, actions_seg shape: {actions_seg.shape}")
        seg_lens = torch.as_tensor(batch["segment_len"], device=device)# (B,)
        # print(f"seg_lens: {seg_lens}")
        # Reset memory bank at start of segment
        if model.use_memory_bank:
            model.memory_module.reset(batch_size=seg_lens.shape[0], device=device)

        # Pre-encode vision for the entire segment (vision is the bottleneck)
        # We process each timestep's frames through vision encoder
        z_v_pooled_all = []
        z_s_all = []
        
        B, S, V, H, W, C = frames_seg.shape 
        z_v_all = model._vision_forward(frames_seg.reshape(B * S * V, H, W, C))   # (B*S*V, P+1, raw_dim=768)
        
        z_v_CLS_all = z_v_all[:, -1]  # (B*S*V, raw_dim=768) — CLS tokens only 
        z_v_CLS_all = z_v_CLS_all.reshape(B, S, V, -1)  # (B, S, V, raw_dim=768)
        
        z_v_all = z_v_all[:, :-1]  # (B*S*V, P, raw_dim=768) — patch tokens only
        
        # print(f"z_v_all shape: {z_v_all.shape}, z_v_CLS_all shape: {z_v_CLS_all.shape}")
        
        _ ,P, raw_dim = z_v_all.shape
        z_v_all = z_v_all.reshape(B, S, V * P, raw_dim)            # (B, S, V*P, raw_dim=768)
        
        _, _, state_dim = states_seg.shape
        z_s_all = model.state_encoder(
            states_seg.reshape(B * S, state_dim)
        ).reshape(B, S, -1)
            
        # Stack: (B, S, V*P, raw_dim) and (B, S, state_dim)
        # z_s_all = torch.stack(z_s_all, dim=1)
        z_v_mod_all = model.intention_encoder.encode_patches(z_v_all, z_s_all)  # (B, S, V*P, comp_dim)
        
        # z_v_mamba_all = model.intention_encoder.encode_patches_for_mamba(z_v_CLS_all, z_s_all)  # (B, S, V, comp_dim)
        # print(f"shapes: z_v_all: {z_v_mod_all.shape}")
        
        # Flatten patch axis into feature dim for head consumption (3D expected)
        # B_seg, S, N_tok, comp_dim = z_v_mod_all.shape
        # z_v_all_stacked = z_v_mod_all.reshape(B_seg, S, N_tok * comp_dim)  # (B, S, V*P*comp_dim)
        # print(f"reshapedshapes: z_v_all: {z_v_all_stacked.shape}")

        total_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        # Sequential T-loop
        last_actions_pred = None
        actions_pred = None
        loss_accum = []
        num_windows = max_seg_len - Hs - chunk_size + 1
        time_ids = torch.arange(max_seg_len, device=device)
        valid_time_mask = time_ids.unsqueeze(0) < seg_lens.unsqueeze(1)
        # print(f"Segment length: {max_seg_len}, num_windows: {num_windows}, H: {Hs}, C: {chunk_size}")
        for n in range(num_windows):
            # Build H-window ending at t+H-1
            # Current time = last frame in the window
            current_t = n + Hs - 1
            valid_mask = seg_lens >= (current_t + chunk_size)
            if not valid_mask.any():
                # No valid samples in this window, skip
                continue

            history_start = current_t - Hs + 1
            history_end = current_t + 1

            z_v_win = z_v_mod_all[:, history_start:history_end]  # (B, H_actual, V*P, comp_dim)
            z_s_win = z_s_all[:, history_start:history_end]  # (B, H_actual, state_dim)

            # Target: C future actions from current time
            target = actions_seg[:, current_t: current_t + chunk_size]
            assert target.shape[1] == chunk_size, f"target shape {target.shape} != chunk_size {chunk_size}"
            
            # Forward through model (uses model's internal z_v_pooled_seq for consistency)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                out = model.forward_intent(z_v_CLS_all[:, history_start:history_end], z_s_win) # input: (B, History, V, raw_dim=768)
                intent_emb = out.get("intent_emb", None)
                h_current = out["h_seq"][:, -1]

                # Memory bank (3-stream: perceptual, cognitive, state)
                if model.use_memory_bank:
                    # Flatten patch axis into feature dim for head consumption (3D expected)
                    B_seg, H_actual, VP, comp_dim = z_v_win.shape
                    z_v_win_stacked = z_v_win.reshape(B_seg, H_actual, VP * comp_dim)  # (B, S, V*P*comp_dim)
                    z_v_current = z_v_win_stacked[:, -1]  # (B, pool_out_dim)
                    z_s_current = z_s_win[:, -1]  # (B, state_dim)
                    if intent_emb is not None:
                        # Active phase: retrieve + fuse + store
                        z_v_fused, z_s_fused, intent_fused = model.memory_module(
                            z_v_current, z_s_current, intent_emb, valid_mask=valid_mask
                        )
                        z_v_win_for_head = z_v_fused
                        z_s_win_for_head = z_s_fused
                        h_for_head = intent_fused
                    else:
                        # Warmup: store perceptual + state, no retrieval
                        model.memory_module.store_perceptual_only(z_v_current, z_s_current)
                        z_v_win_for_head = z_v_win_stacked
                        z_s_win_for_head = z_s_win
                        h_for_head = intent_emb
                else:
                    z_v_win_for_head = z_v_win
                    z_s_win_for_head = z_s_win
                    h_for_head = h_current

                # Loss
                if args.head_type == "diffusion":
                    cond = model.intention_head(
                        z_v_win_for_head, z_s_win_for_head, h_for_head,
                    )
                    actions_pred = model.sample_actions(
                        z_v_win_for_head, z_s_win_for_head, h_for_head, num_steps=chunk_size
                    )
                    assert actions_pred.shape == target.shape, f"actions_pred shape {actions_pred.shape} != target shape {target.shape}"
                    loss = model.intention_head.loss(target, cond)
                    # Mask loss: only valid samples contribute
                    if not valid_mask.all():
                        loss = loss * valid_mask.float().mean()
                else:
                    actions_pred = model.predict_actions(
                        z_v_win_for_head, z_s_win_for_head, h_for_head,
                    )
                    loss = F.mse_loss(actions_pred, target, reduction='none')
                    loss = loss[valid_mask].mean() if valid_mask.any() else loss.mean()

            if args.skip_nan and not torch.isfinite(loss):
                continue

            loss_accum.append(loss)
            last_actions_pred = actions_pred

        # One optimizer step per segment: sum all losses and backward
        if loss_accum:
            total_loss = torch.stack(loss_accum).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()

        avg_step_loss = float(total_loss.item() / max(len(loss_accum), 1)) if loss_accum else 0.0
        losses.append(avg_step_loss)
        actions_pred_list.append(actions_pred.detach().abs().mean().item() if actions_pred is not None else 0.0)

        pbar.set_postfix(mse=f"{avg_step_loss:.5f}")

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    avg_action = float(np.mean(actions_pred_list)) if actions_pred_list else 0.0
    return avg_loss, avg_action

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

        # Forward (BF16 always on for speed; disabled automatically on CPU)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            out = model(frames, state)
            h_current = out["h_seq"][:, -1]  # (B, mamba_output_dim) — latest
            # predict_actions returns actions (direct regression) or cond (flow head)

            # Loss depends on head type
            if args.head_type == "diffusion":
                cond = model.intention_head(
                    out["z_v_pooled_seq"], out["z_s_seq"], h_current,
                )
                actions_pred = model.sample_actions(
                    out["z_v_pooled_seq"], out["z_s_seq"], h_current,
                    num_steps=target.shape[1],
                )
                loss = model.intention_head.loss(target, cond)
            else:
                actions_pred = model.predict_actions(
                    out["z_v_pooled_seq"], out["z_s_seq"], h_current,
                )  # (B, K, action_dim) — may be < target.shape[-1] if model
                    #   doesn't predict gripper

                # Pad model output with target's gripper if needed.
                # Model may output fewer dims (e.g. 6 for OSC deltas) than
                # the dataset's action (7, including gripper). We pad with
                # the target's gripper so the per-dim metrics are comparable.
                if actions_pred.shape[-1] < target.shape[-1]:
                    pad = target[..., actions_pred.shape[-1]:]
                    actions_pred_for_loss = torch.cat([actions_pred, pad], dim=-1)
                else:
                    actions_pred_for_loss = actions_pred
                # Direct regression: MSE on actions (use padded for fair comparison)
                loss = F.mse_loss(actions_pred_for_loss, target)

        if args.skip_nan and not torch.isfinite(loss):
            # Skip NaN/Inf batch — common with high LR + Mamba + BF16
            losses.append(loss.item())
            actions_pred_list.append(0.0 if not torch.isfinite(actions_pred).all()
                                      else actions_pred.detach().abs().mean().item())
            pbar.set_postfix(mse=f"NAN", warn="skip")
            optimizer.zero_grad()
            continue

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
    per_dim_squared = None  # will be sized to target.shape[-1]
    per_dim_abs = None
    n_samples = 0
    # Track whether the model is genuinely predicting gripper or just
    # getting it from the target via padding. This is critical for
    # interpreting the grip_mse / grip_acc metrics.
    padded_gripper_batches = 0     # batches where gripper was padded
    genuine_gripper_batches = 0   # batches where model predicted gripper
    grip_correct_total = 0.0      # # of correct gripper open/close
    grip_total_total = 0          # # of gripper predictions
    pbar = tqdm(loader, desc="  [val]  ", unit="batch", leave=False)
    for batch in pbar:
        
        H = args.history_size
        C = args.chunk_size
        B = len(batch["segment_len"])
        
        frames_seg = torch.from_numpy(batch["frames_segment"]).to(device)  # (B, S, V, H, W, 3)
        states_seg = torch.from_numpy(batch["states_segment"]).float().to(device)  # (B, S, 7)
        target_seg = torch.from_numpy(batch["actions_segment"]).float().to(device)  # (B, S, 7)
       
        seg_lens = torch.as_tensor(batch["segment_len"], device=device)# (B,)
        
        # Reset memory bank at start of segment
        if model.use_memory_bank:
            model.memory_module.reset(batch_size=B, device=device)

        max_seg_len = frames_seg.shape[1]
        # Pre-encode vision for the entire segment (vision is the bottleneck)
        B, S, V, H, W, C = frames_seg.shape 
        z_v_all = model._vision_forward(frames_seg.reshape(B * S * V, H, W, C))   # (B*S*V, P+1, raw_dim=768)
        
        z_v_CLS_all = z_v_all[:, -1]  # (B*S*V, raw_dim=768) — CLS tokens only 
        z_v_CLS_all = z_v_CLS_all.reshape(B, S, V, -1)  # (B, S, V, raw_dim=768)
        
        z_v_all = z_v_all[:, :-1]  # (B*S*V, P, raw_dim=768) — patch tokens only
        
        _ ,P, raw_dim = z_v_all.shape
        z_v_all = z_v_all.reshape(B, S, V * P, raw_dim)            # (B, S, V*P, raw_dim=768)
        
        _, _, state_dim = states_seg.shape
        z_s_all = model.state_encoder(
            states_seg.reshape(B * S, state_dim)
        ).reshape(B, S, -1)
        
        z_v_mod_all = model.intention_encoder.encode_patches(z_v_all, z_s_all)  # (B, S, V*P, comp_dim)
        
        # Sequential T-loop
        last_actions_pred = None
        actions_pred = None
        loss_accum = []
        num_windows = max_seg_len - H - C + 1

        for n in range(num_windows):
            # Build H-window ending at t+H-1
            # Current time = last frame in the window
            current_t = n + H - 1

            history_start = current_t - H + 1
            history_end = current_t + 1

            z_v_win = z_v_mod_all[:, history_start:history_end]  # (B, H_actual, V*P, comp_dim)
            z_s_win = z_s_all[:, history_start:history_end]  # (B, H_actual, state_dim)

            valid_mask = seg_lens >= (current_t + C)

            # Forward through model
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                out = model.forward_intent(z_v_CLS_all[:, history_start:history_end], z_s_win) # input: (B, History, V, raw_dim=768)
                intent_emb = out.get("intent_emb", None)
                h_current = out["h_seq"][:, -1]

                # Memory bank (3-stream: perceptual, cognitive, state)
                if model.use_memory_bank:
                    # Flatten patch axis into feature dim for head consumption (3D expected)
                    B_seg, H_actual, VP, comp_dim = z_v_win.shape
                    z_v_win_stacked = z_v_win.reshape(B_seg, H_actual, VP * comp_dim)  # (B, S, V*P*comp_dim)
                    z_v_current = z_v_win_stacked[:, -1]  # (B, pool_out_dim)
                    z_s_current = z_s_win[:, -1]  # (B, state_dim)
                    if intent_emb is not None:
                        # Active phase: retrieve + fuse + store
                        z_v_fused, z_s_fused, intent_fused = model.memory_module(
                            z_v_current, z_s_current, intent_emb, valid_mask=valid_mask
                        )
                        z_v_win_for_head = z_v_fused
                        z_s_win_for_head = z_s_fused
                        h_for_head = intent_fused
                    else:
                        # Warmup: store perceptual + state, no retrieval
                        model.memory_module.store_perceptual_only(z_v_current, z_s_current, valid_mask=valid_mask)
                        z_v_win_for_head = z_v_win_stacked
                        z_s_win_for_head = z_s_win
                        h_for_head = intent_emb
                else:
                    z_v_win_for_head = z_v_win_stacked
                    z_s_win_for_head = z_s_win
                    h_for_head = h_current

                # Target: C future actions from current time
                target = target_seg[:, current_t:current_t + C]

                # Loss
                if args.head_type == "diffusion":
                    cond = model.intention_head(
                        z_v_win_for_head, z_s_win_for_head, h_for_head,
                    )
                    actions_pred = model.sample_actions(
                        z_v_win_for_head, z_s_win_for_head, h_for_head, num_steps=C
                    )
                    loss = model.intention_head.loss(target, cond)
                    if not valid_mask.all():
                        loss = loss * valid_mask.float().mean()
                else:
                    actions_pred = model.predict_actions(
                        z_v_win_for_head, z_s_win_for_head, h_for_head,
                    )
                    loss = F.mse_loss(actions_pred, target, reduction='none')
                    loss = loss[valid_mask].mean() if valid_mask.any() else loss.mean()

            if args.skip_nan and not torch.isfinite(loss):
                continue

            loss_accum.append(loss)
            last_actions_pred = actions_pred

        # Accumulate metrics across all windows in this segment
        if loss_accum:
            avg_window_loss = torch.stack(loss_accum).mean().item()
            losses.append(avg_window_loss)
            actions_pred_list.append(actions_pred.detach().abs().mean().item() if actions_pred is not None else 0.0)
            pbar.set_postfix(mse=f"{avg_window_loss:.5f}")

            # Per-dim error accumulation (across batch and time)
            # Use the last window's predictions for per-dim metrics
            if actions_pred.shape[-1] < target.shape[-1]:
                pad = target[..., actions_pred.shape[-1]:].detach().float().cpu().numpy()
                actions_pred_for_metric = np.concatenate(
                    [actions_pred.detach().float().cpu().numpy(), pad], axis=-1,
                )
                padded_gripper_batches += 1
            else:
                actions_pred_for_metric = actions_pred.detach().float().cpu().numpy()
            target_np = target.detach().float().cpu().numpy()
            diff = actions_pred_for_metric - target_np  # (B, T, D)
            B_win, T, D = diff.shape
            if per_dim_squared is None:
                per_dim_squared = np.zeros(D, dtype=np.float64)
                per_dim_abs = np.zeros(D, dtype=np.float64)
            per_dim_squared += (diff ** 2).sum(axis=(0, 1))  # (D,)
            per_dim_abs += np.abs(diff).sum(axis=(0, 1))      # (D,)
            n_samples += B_win * T

            # Gripper accuracy (only meaningful if model output has >=7 dims
            # AND the model is actually predicting gripper, not just padded).
            if actions_pred.shape[-1] >= 7 and target.shape[-1] >= 7:
                genuine_gripper_batches += 1
                grip_pred = actions_pred[..., 6]  # model's gripper prediction
                grip_target = target[..., 6]
                grip_pred_binary = (grip_pred > 0).float()
                grip_target_binary = (grip_target > 0).float()
                grip_correct = (grip_pred_binary == grip_target_binary).float().sum().item()
                grip_total = B_win * T
                grip_correct_total += grip_correct
                grip_total_total += grip_total

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    avg_action = float(np.mean(actions_pred_list)) if actions_pred_list else 0.0

    # Per-dim metrics (assumes 7 dims: x, y, z, rx, ry, rz, gripper)
    per_dim_metrics = {}
    if n_samples > 0 and per_dim_squared is not None:
        # Per-dim MSE and MAE
        per_dim_mse = per_dim_squared / n_samples
        per_dim_mae = per_dim_abs / n_samples
        # Defensive: handle different action_dim values
        D = len(per_dim_mse)
        pos_mse = float((per_dim_mse[0] + per_dim_mse[1] + per_dim_mse[2]) / 3) if D >= 3 else 0.0
        rot_mse = float((per_dim_mse[3] + per_dim_mse[4] + per_dim_mse[5]) / 3) if D >= 6 else 0.0
        grip_mse = float(per_dim_mse[6]) if D >= 7 else 0.0
        pos_mae = float((per_dim_mae[0] + per_dim_mae[1] + per_dim_mae[2]) / 3) if D >= 3 else 0.0
        rot_mae = float((per_dim_mae[3] + per_dim_mae[4] + per_dim_mae[5]) / 3) if D >= 6 else 0.0
        # Gripper accuracy (only meaningful if model genuinely predicts
        # gripper, i.e. action_dim >= 7). If the model was padded, the
        # accuracy is meaningless (would always be 100%).
        if genuine_gripper_batches > 0 and grip_total_total > 0:
            grip_acc = float(grip_correct_total / grip_total_total)
        else:
            grip_acc = 0.0
        per_dim_metrics = {
            "pos_mse": pos_mse,
            "rot_mse": rot_mse,
            "grip_mse": grip_mse,
            "pos_mae": pos_mae,
            "rot_mae": rot_mae,
            # Per-axis (for fine-grained debugging)
            "px_mse": float(per_dim_mse[0]) if D >= 1 else 0.0,
            "py_mse": float(per_dim_mse[1]) if D >= 2 else 0.0,
            "pz_mse": float(per_dim_mse[2]) if D >= 3 else 0.0,
            "rx_mse": float(per_dim_mse[3]) if D >= 4 else 0.0,
            "ry_mse": float(per_dim_mse[4]) if D >= 5 else 0.0,
            "rz_mse": float(per_dim_mse[5]) if D >= 6 else 0.0,
            "grip_acc": grip_acc,
            # Diagnostics to disambiguate padded vs genuine
            "gripper_padded_batches": padded_gripper_batches,
            "gripper_genuine_batches": genuine_gripper_batches,
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
    parser.add_argument("--cameras", nargs="+", default=["image","wrist_image"],
                        help="Camera names to load (e.g. 'wrist_image image').")
    # NOTE: --num-cameras removed; auto-derived from --cameras.
    parser.add_argument("--val-split", type=float, default=0.1)
    
    # Model
    parser.add_argument("--chunk-size", type=int, default=10)
    
    # Head selection
    parser.add_argument("--head-type", choices=["transformer", "mamba", "hybrid", "diffusion"],
                        default="mamba",
                        help="Which head architecture: transformer, mamba, hybrid, or diffusion")
    parser.add_argument("--use-history", action="store_true", default=True,
                        help="Include Mamba history component (h) in head input.")
    parser.add_argument("--no-history", dest="use_history", action="store_false",
                        help="Disable Mamba history component.")
    
    # Architecture dimensions
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
    parser.add_argument("--action-dim", type=int, default=7,
                        help="Action output dim (default 7).")
    
    # Patch tokens
    parser.add_argument("--no-patch-tokens", dest="use_patch_tokens",
                        action="store_false", default=True,
                        help="Use CLS token instead of patch tokens from DINOv2.")
    parser.set_defaults(use_patch_tokens=True)
    #
    # Per-patch compressed dim (SEVisualCompressor output)
    parser.add_argument("--compressed-dim", type=int, default=16,
                        help="Per-patch dim after SEVisualCompressor (default 16).")
    
    # V4: Intent tokens
    parser.add_argument("--use-intent-tokens", action="store_true", default=False,
                        help="Enable learnable intent tokens (V4).")
    parser.add_argument("--num-intent-tokens", type=int, default=2,
                        help="Number of intent tokens (default 2).")
    parser.add_argument("--intent-dim", type=int, default=512,
                        help="Intent token output dim (default 512).")
    
    # V4: Memory bank
    parser.add_argument("--use-memory-bank", action="store_true", default=False,
                        help="Enable Perceptual-Cognitive Memory Bank (V4).")
    parser.add_argument("--memory-bank-len", type=int, default=16,
                        help="Max paired entries in bank (default 16).")
    
    # V4: Segment training
    parser.add_argument("--history-size", type=int, default=20,
                        help="Past frames for Mamba window (default 20).")
    parser.add_argument("--segment-min-mult", type=int, default=2,
                        help="Min segment length = history_size * this (default 2).")
    parser.add_argument("--segment-max-mult", type=int, default=5,
                        help="Max segment length = history_size * this (default 5).")
    
    # V4: Semantic anchoring
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="Weight for semantic anchoring loss (0 = disabled).")
    
    # Text modality (optional)
    parser.add_argument("--use-text", action="store_true", default=False,
                        help="Enable text encoder + text-conditioned head.")
    parser.add_argument("--text-dim", type=int, default=256,
                        help="Text encoder output dim (default 256).")
    parser.add_argument("--task-text", type=str, default=None,
                        help="Task description for text conditioning (default: auto from dataset).")
    
    # IntentionTransformerHead params
    parser.add_argument("--head-d-model", type=int, default=512,
                        help="IntentionTransformerHead model dimension (default: 512)")
    parser.add_argument("--head-nhead", type=int, default=8,
                        help="IntentionTransformerHead number of head (default: 8)")
    parser.add_argument("--head-num-layers", type=int, default=6,
                        help="IntentionTransformerHead number of layer (default: 6)")
    parser.add_argument("--head-dim-ff", type=int, default=1024,
                        help="IntentionTransformerHead feed-forward dimension (default: 1024)")
    # Training
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None,
                        help="Custom run folder name (default: run_N auto-incremented).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--skip-nan", dest="skip_nan", action="store_true",
                        default=True,
                        help="Skip batches with NaN/Inf loss (default on).")
    parser.add_argument("--no-skip-nan", dest="skip_nan", action="store_false",
                        help="Disable NaN skipping (will NaN out the run).")
    parser.add_argument("--num-workers", type=int, default=1)
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
    print(f"  Head Type:  {args.head_type} head")
    print(f"  Chunk (K):  {args.chunk_size}")
    # Determine num_cameras from --cameras argument (auto-derived)
    num_cameras = len(args.cameras)
    print(f"  Cameras: {num_cameras} (from --cameras {args.cameras})")
    
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

    # Model
    print("\n  Building model...")
    model = build_model(args, num_cameras, device)

    # Configure trainable parameters
    # - DINOv2 backbone: always frozen (ImageNet-pretrained)
    # - Vision projection: always trainable (small, adapts generic features)
    # - State encoder: always trainable (no pretrained weights available)
    # - Intention encoder + head: always trainable
    for p in model.vision_encoder.backbone.parameters():
        p.requires_grad = False
    print("  DINOv2 backbone: frozen (ImageNet-pretrained)")
    # Enable training for everything else
    if model.intention_encoder is not None:
        for p in model.intention_encoder.parameters():
            p.requires_grad = True
    # Head is lazily built on first forward pass; skip if not yet built
    if model.intention_head is not None:
        for p in model.intention_head.parameters():
            p.requires_grad = True
    # Memory bank is lazily built on first forward pass
    if model.memory_module is not None:
        for p in model.memory_module.parameters():
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
    # Build lazy head/bank before counting params.
    # Probe actual vision output dim by running one real frame through VisionEncoder.
    # Use the dataset's __getitem__ to get the actual resized frame.
    print("  Building head (probing actual vision output shape)...")
    with torch.no_grad():
        # Get one sample from the dataset to see the actual resized frame
        sample = full_ds[0]
        # sample is a dict from v4_segment_collate or head_collate
        # For V4: sample has 'frames' key with shape (S, V, H, W, 3)
        # For V3: sample has 'frames' key with shape (K, V, H, W, 3)
        if isinstance(sample, dict) and 'frames' in sample:
            frames_sample = sample['frames']
        elif isinstance(sample, (list, tuple)):
            # V3 collate returns (frames, states, actions, ...)
            frames_sample = sample[0]
        else:
            frames_sample = sample
        # Take first timestep, add batch dim
        if isinstance(frames_sample, np.ndarray):
            dummy = torch.from_numpy(frames_sample[0:1]).to(device)
        else:
            dummy = frames_sample[0:1].to(device)
        z_v_dummy = model._vision_forward(dummy)
        print(f"  Vision output shape: {z_v_dummy.shape}")
        if z_v_dummy.ndim == 2:
            N_tok_actual = z_v_dummy.shape[0]
        else:
            N_tok_actual = z_v_dummy.shape[1]
    pool_out_dim = (N_tok_actual - num_cameras) * args.compressed_dim 
    print(f"  Pool out dim: {pool_out_dim} (N_tok_actual={N_tok_actual}, compressed_dim={args.compressed_dim})")
    model._build_head_and_bank(pool_out_dim)
    print(f"  Head built: pool_out_dim={pool_out_dim} (VP_tokens={N_tok_actual})")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params:   {n_trainable:,}")
    print(f"  Total model params: {n_total:,}")

    # Warn if LR is high (common cause of NaN with Mamba + BF16)
    if args.lr > 1e-3:  
        print(f"  ⚠️  WARNING: --lr {args.lr:.0e} is HIGH (default: 1e-4).")
        print(f"     Mamba + BF16 can be unstable at this LR. If you see NaN,")
        print(f"     lower --lr to 1e-4 (or 5e-4 for warmup).")
    if args.skip_nan:
        print(f"  NaN batches: SKIPPED (use --no-skip-nan to disable)")
    # Update wandb config with model-derived info
    if wandb_trainer.enabled:
        wandb_trainer.run.config.update({
            "num_cameras": num_cameras,
            "n_params": n_total,
            "n_trainable_params": n_trainable,
        })
        # Log architecture dimensions under clean namespaces
        wandb_trainer.run.config.update({
            "arch/state_dim": args.state_dim,
            "arch/mamba_output_dim": args.mamba_output_dim,
            "arch/mamba_d_state": args.mamba_d_state,
            "arch/mamba_d_conv": args.mamba_d_conv,
            "arch/mamba_expand": args.mamba_expand,
            "arch/action_dim": args.action_dim,
            "arch/compressed_dim": args.compressed_dim,
            "arch/use_patch_tokens": args.use_patch_tokens,
            "arch/use_history": args.use_history,
            "arch/use_text": args.use_text,
            "arch/text_dim": args.text_dim,
            "arch/head_type": args.head_type,
            "arch/history_size": args.history_size,
            "arch/chunk_size": args.chunk_size,
            "v4/use_intent_tokens": args.use_intent_tokens,
            "v4/num_intent_tokens": args.num_intent_tokens,
            "v4/intent_dim": args.intent_dim,
            "v4/use_memory_bank": args.use_memory_bank,
            "v4/memory_bank_len": args.memory_bank_len,
            "v4/anchor_weight": args.anchor_weight,
            "v4/segment_min_mult": args.segment_min_mult,
            "v4/segment_max_mult": args.segment_max_mult,
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
    is_v4 = args.use_intent_tokens or args.use_memory_bank
    train_fn = train_v4_epoch if is_v4 else train_one_epoch
    print(f"  Training for {args.epochs} epochs..." + (" (V4 mode)" if is_v4 else " (V3 mode)"))
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        train_loss, train_action = train_fn(
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
        # Show gripper diagnostic if model is genuinely predicting gripper
        genuine = val_per_dim.get('gripper_genuine_batches', 0)
        padded = val_per_dim.get('gripper_padded_batches', 0)
        grip_acc = val_per_dim.get('grip_acc', 0)
        if genuine > 0:
            print(f"           gripper: genuine prediction, acc={grip_acc:.3f} "
                  f"({genuine} genuine batches)")
        elif padded > 0:
            print(f"           gripper: PAD-PADDED (model output dim < 7)")

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
