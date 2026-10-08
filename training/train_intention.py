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
import math
import os
from pyexpat import model
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
# Disable cuDNN — needed when DINOv2 (with torch.no_grad) and Mamba's CUDA
# kernel interact badly. Without this, you get CUDNN_STATUS_NOT_INITIALIZED
# on the first conv2d inside DINOv2. See setup.sh for the same fix.
torch.backends.cudnn.enabled = False

# Enable TF32 for faster matmuls on Ampere+ GPUs (slight precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.align_dataset import ALIGNDataset, MultiALIGNDataset, head_collate, v4_segment_collate
from models.align_intention import ALIGNIntentionModel
from training.wandb_utils import init_wandb


def distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def is_primary():
    return not distributed_enabled() or dist.get_rank() == 0


def setup_distributed(args):
    """Use one process per GPU when launched with torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return torch.device(args.device)
    local_rank = int(os.environ["LOCAL_RANK"])
    requested = torch.device(args.device)
    if requested.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif requested.type == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("Distributed training supports CUDA or CPU devices")
    dist.init_process_group(backend=backend, init_method="env://")
    args.device = str(device)
    return device


def global_mean(values, device, empty=float("inf")):
    """Average batch metrics across all processes."""
    totals = torch.tensor([sum(values), len(values)], dtype=torch.float64, device=device)
    if distributed_enabled():
        dist.all_reduce(totals)
    return (totals[0] / totals[1]).item() if totals[1].item() else empty


def sync_and_step(model, optimizer, device, grad_clip):
    """Average gradients, including parameters unused on some ranks.

    Training uses model submodules and helper methods outside model.forward(),
    so a DDP wrapper around forward alone would miss those gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if distributed_enabled():
        used = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=device)
        dist.all_reduce(used)
        if not torch.any(used):
            return False
        # Group by dtype so each group needs only one gradient collective.
        groups = {}
        for p, count in zip(params, used.tolist()):
            if count:
                groups.setdefault(p.dtype, []).append(p)
        world_size = dist.get_world_size()

        def reduce_bucket(bucket):
            flat = torch.cat([
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in bucket
            ])
            dist.all_reduce(flat)
            flat.div_(world_size)
            offset = 0
            for p in bucket:
                p.grad = flat.narrow(0, offset, p.numel()).view_as(p)
                offset += p.numel()

        # Bound the temporary communication buffer for large models.
        bucket_limit = 25 * 1024 * 1024
        for group in groups.values():
            bucket = []
            bucket_bytes = 0
            for p in group:
                size = p.numel() * p.element_size()
                if bucket and bucket_bytes + size > bucket_limit:
                    reduce_bucket(bucket)
                    bucket = []
                    bucket_bytes = 0
                bucket.append(p)
                bucket_bytes += size
            if bucket:
                reduce_bucket(bucket)
    elif not any(p.grad is not None for p in params):
        return False
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()
    return True


# ================================================================
# Data
# ================================================================

def build_datasets(args):
    """Build train and val datasets from one or more HDF5 files."""
    # V4 mode: enabled if any V4 feature is on or segment args are set
    is_v4 = (
        getattr(args, "use_intent_tokens", False)
        or getattr(args, "use_memory_bank", False)
        or getattr(args, "segment_min_mult", 0) > 0
        or getattr(args, "segment_max_mult", 0) > 0
    )
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
            dinov2_path=args.dinov2_path if args.use_precomputed_dinov2 else None,
        )
    else:
        ds = MultiALIGNDataset(
            args.data, mode="head",
            traj_window=traj_window, cameras=args.cameras,
            dinov2_paths=([args.dinov2_path] * len(args.data)
                          if args.use_precomputed_dinov2 else None),
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
    pin_memory = torch.device(args.device).type == "cuda"
    # Match the training loop's is_v4 detection logic
    is_v4 = (
        getattr(args, "use_intent_tokens", False)
        or getattr(args, "use_memory_bank", False)
        or getattr(args, "segment_min_mult", 0) > 0
        or getattr(args, "segment_max_mult", 0) > 0
    )
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
    if pin_memory:
        # The ALIGN collators return NumPy arrays. DataLoader only pins torch
        # tensors, so convert them here before its pin-memory thread runs.
        numpy_collate_fn = collate_fn
        transfer_keys = ({"frames_segment", "states_segment", "actions_segment"}
                         if is_v4 else {"frames_window", "robot_state_window", "actions_window"})
        def collate_fn(batch):
            result = numpy_collate_fn(batch)
            return {
                key: torch.from_numpy(value) if key in transfer_keys else value
                for key, value in result.items()
            }
    train_sampler = (DistributedSampler(train_ds, shuffle=True, seed=args.seed)
                     if distributed_enabled() else None)
    if distributed_enabled():
        val_ds = Subset(val_ds, range(dist.get_rank(), len(val_ds), dist.get_world_size()))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
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
    if args.head_type not in ("diffusion", "flow_matching"):
        raise ValueError("Transformer/Mamba/hybrid action heads are future work; use diffusion or flow_matching for V4.")
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
    pbar = tqdm(range(n_steps), total=n_steps, desc="  [train V4]", unit="batch",
                leave=False, disable=not is_primary())
    step_iter = iter(loader)

    for _ in pbar:
        try:
            batch = next(step_iter)
        except StopIteration:
            break
        Hs = args.history_size
        chunk_size = args.chunk_size

        # Per-dimension loss weights: down-weight gripper (dim 6) so it doesn't dominate
        dim_weights = torch.ones(args.action_dim, device=device)
        if args.action_dim >= 7:
            dim_weights[6] = getattr(args, "gripper_loss_weight", 0.01)

        frames_seg = torch.as_tensor(batch["frames_segment"]).to(device, non_blocking=True)  # (B, S, V, H, W, 3)
        states_seg = torch.as_tensor(batch["states_segment"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, S, 7)
        actions_seg = torch.as_tensor(batch["actions_segment"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, S, 7)
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


        B, S = frames_seg.shape[:2]
        # Detect pre-computed DINOv2 features (last dim = 768) vs raw
        # frames (last dim = 3, shape = (B, S, V, H, W, 3)). When features
        # are pre-computed, skip the DINOv2 forward entirely and just
        # unpack the saved tokens.
        frames_are_features = frames_seg.shape[-1] == 768
        if frames_are_features:
            # Pre-computed features: (B, S, V*257, 768) float32
            # Layout per timestep: [cam0_patches(256), cam0_CLS(1), cam1_..., cam1_CLS, ...]
            TOKENS = frames_seg.shape[2]   # V * 257
            V = TOKENS // 257             # number of cameras
            z_v_all = frames_seg.reshape(B * S * V, -1, 768)  # (B*S*V, P+1, raw_dim=768)
            # print(f"Precomputed z_v_all shape: {z_v_all.shape}")
        else:
            V, H, W, C = frames_seg.shape[2:]
            z_v_all = model._vision_forward(frames_seg.reshape(B * S * V, H, W, C))   # (B*S*V, P+1, raw_dim=768)
            # print(f"raw z_v_all shape: {z_v_all.shape}")

        # Extract CLS tokens (last position per camera, NOT the last position overall).
        # Layout: [cam0_patches..., cam0_CLS, cam1_patches..., cam1_CLS, ...]
        P_plus_1 = z_v_all.shape[1]
        P = P_plus_1 - 1
        z_v_all = z_v_all.reshape(B * S, V, P_plus_1, 768)
        z_v_CLS_all = z_v_all[:, :, -1, :]  # (B*S, V, 768)
        z_v_CLS_all = z_v_CLS_all.reshape(B, S, V, -1)  # (B, S, V, raw_dim=768)
        # Extract patches (all positions except the last per camera)
        z_v_all = z_v_all[:, :, :-1, :].reshape(B, S , V * P, 768)   # (B, S, V*P, raw_dim=768)

        _, _, state_dim = states_seg.shape
        z_s_all = model.state_encoder(
            states_seg.reshape(B * S, state_dim)
        ).reshape(B, S, -1)

        # Stack: (B, S, V*P, raw_dim) and (B, S, state_dim)
        # z_s_all = torch.stack(z_s_all, dim=1)
        z_v_mod_all = model.vision_patch_encoder(
            z_v_all.reshape(B * S, V * P, 768),
            z_s_all.reshape(B * S, -1),
        ).reshape(B, S, V * P, -1)  # (B, S, V*P, comp_dim)

        # Flatten patch axis into feature dim for head consumption (3D expected)
        # B_seg, S, N_tok, comp_dim = z_v_mod_all.shape
        # z_v_all_stacked = z_v_mod_all.reshape(B_seg, S, N_tok * comp_dim)  # (B, S, V*P*comp_dim)
        # print(f"reshapedshapes: z_v_all: {z_v_all_stacked.shape}")

        total_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        # Sequential T-loop
        last_actions_pred = None
        actions_pred = None
        loss_accum = []
        valid_counts = []
        max_seg_len = S
        if model.use_history:
            num_windows = max_seg_len - Hs - chunk_size + 2
        else:
            # No history: each timestep is independent
            num_windows = max_seg_len - chunk_size + 1

        # Keep one autocast cache for the whole segment: repeated windows
        # must reuse parameter casts rather than retaining a copy each.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # Scan each observation once; readout at t sees only the prefix through t.
            intent_sequence = None
            if num_windows > 0 and getattr(model, "intention_encoder", None) is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    intent_sequence = model.intention_encoder.forward_sequence(
                        z_v_CLS_all[:, :S - chunk_size + 1], z_s_all[:, :S - chunk_size + 1],
                        readout_start=Hs - 1,
                    )

            for n in range(num_windows):
                if model.use_history:
                    # Build H-window ending at t+H-1
                    current_t = n + Hs - 1
                    history_start = current_t - Hs + 1
                    history_end = current_t + 1
                else:
                    # Single timestep
                    current_t = n
                    history_start = n
                    history_end = n + 1
                valid_mask = seg_lens >= (current_t + chunk_size)
                if not valid_mask.any():
                    # No valid samples in this window, skip
                    continue

                z_v_win = z_v_mod_all[:, history_start:history_end]  # (B, H_actual, V*P, comp_dim)
                z_s_win = z_s_all[:, history_start:history_end]  # (B, H_actual, state_dim)

                # Target: C future actions from current time
                target = actions_seg[:, current_t: current_t + chunk_size]
                assert target.shape[1] == chunk_size, f"target shape {target.shape} != chunk_size {chunk_size}"

                # Forward through model
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    intent_emb = intent_sequence[:, current_t] if intent_sequence is not None else None
                    B_seg, H_actual, VP, comp_dim = z_v_win.shape
                    z_v_win_for_head, z_s_win_for_head, h_for_head = model.condition_actions(
                        z_v_win.reshape(B_seg, H_actual, VP * comp_dim),
                        z_s_win, intent_emb, observed_mask=current_t < seg_lens,
                    )

                    # Loss
                    if args.head_type in ("diffusion", "flow_matching"):
                        if getattr(args, "debug", False):
                            print(f"[DEBUG] z_v_win_for_head: {z_v_win_for_head.shape}, "
                                  f"z_s_win_for_head: {z_s_win_for_head.shape}, "
                                  f"h_for_head: {h_for_head.shape if h_for_head is not None else None}")
                        cond = model.intention_head(
                            z_v_win_for_head, z_s_win_for_head, h_for_head,
                        )
                        if getattr(args, "debug", False):
                            print(f"[DEBUG] cond: {cond.shape}, finite: {torch.isfinite(cond).all().item()}")
                        # Sample actions for the action mean (5-10x slower than loss compute).
                        # Skip in training if --no-sample-during-train is set.
                        if getattr(args, "no_sample_during_train", False):
                            actions_pred = None
                        else:
                            actions_pred = model.sample_actions(
                                z_v_win_for_head, z_s_win_for_head, h_for_head
                            )
                            if getattr(args, "debug", False):
                                print(f"[DEBUG] actions_pred: {actions_pred.shape}, "
                                      f"finite: {torch.isfinite(actions_pred).all().item()}, "
                                      f"abs.mean: {actions_pred.detach().abs().mean().item()}")
                            assert actions_pred.shape == target.shape, f"actions_pred shape {actions_pred.shape} != target shape {target.shape}"
                        loss = model.intention_head.loss(
                            target, cond, dim_weights=dim_weights, sample_mask=valid_mask,
                        )
                        if getattr(args, "debug", False):
                            print(f"[DEBUG] loss: {loss.item()}")
                    else:
                        actions_pred = model.predict_actions(
                            z_v_win_for_head, z_s_win_for_head, h_for_head,
                        )
                        loss = F.mse_loss(actions_pred, target, reduction='none')
                        loss = loss[valid_mask].mean() if valid_mask.any() else loss.mean()

                if args.skip_nan and not torch.isfinite(loss):
                    continue

                loss_accum.append(loss)
                valid_counts.append(int(valid_mask.sum().item()))
                last_actions_pred = actions_pred

        # One optimizer step per segment, averaged over valid sample/windows.
        if loss_accum:
            total_loss = sum(loss * count for loss, count in zip(loss_accum, valid_counts)) / sum(valid_counts)
            total_loss.backward()
        sync_and_step(model, optimizer, device, args.grad_clip)

        avg_step_loss = float(total_loss.item()) if loss_accum else 0.0
        losses.append(avg_step_loss)
        actions_pred_list.append(actions_pred.detach().abs().mean().item() if actions_pred is not None else 0.0)

        pbar.set_postfix(mse=f"{avg_step_loss:.5f}")

    avg_loss = global_mean(losses, device)
    avg_action = global_mean(actions_pred_list, device, empty=0.0)
    return avg_loss, avg_action

def train_one_epoch(model, loader, optimizer, device, args, max_steps=0):
    """Train for one epoch. Returns (avg_loss, avg_action_mean)."""
    model.train()
    losses, actions_pred_list = [], []
    n_steps = min(max_steps, len(loader)) if max_steps else len(loader)
    pbar = tqdm(
        range(n_steps), total=n_steps,
        desc=f"  [train]", unit="step", leave=False, disable=not is_primary(),
    )
    step_iter = iter(loader)
    for _ in pbar:
        try:
            batch = next(step_iter)
        except StopIteration:
            break

        frames = torch.as_tensor(batch["frames_window"]).to(device, non_blocking=True)  # (B, K, H, W, 3) or (B, K, V, H, W, 3)
        state = torch.as_tensor(batch["robot_state_window"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, K, 7)
        target = torch.as_tensor(batch["actions_window"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, K, 7)

        # Per-dimension loss weights: down-weight gripper (dim 6) so it doesn't dominate
        dim_weights = torch.ones(args.action_dim, device=device)
        if args.action_dim >= 7:
            dim_weights[6] = getattr(args, "gripper_loss_weight", 0.01)

        # Forward (BF16 always on for speed; disabled automatically on CPU)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            out = model(frames, state)
            h_current = out["h_seq"][:, -1]  # (B, mamba_output_dim) — latest
            # Get intent_emb if available (V4 with intent tokens)
            intent_emb = out.get("intent_emb", None)
            # predict_actions returns actions (direct regression) or cond (flow head)

            # Loss depends on head type
            if args.head_type in ("diffusion", "flow_matching"):
                # V4 head expects intent_emb (3D or None), not h_current.
                # For V3 backward compat with no Mamba/no intent, pass None.
                head_intent = intent_emb if intent_emb is not None and intent_emb.ndim == 3 else None
                cond = model.intention_head(
                    out["z_v_pooled_seq"], out["z_s_seq"], head_intent,
                )
                actions_pred = model.sample_actions(
                    out["z_v_pooled_seq"], out["z_s_seq"], head_intent,
                )
                loss = model.intention_head.loss(target, cond, dim_weights=dim_weights)
            else:
                # V4 head expects intent_emb (3D or None), not h_current.
                head_intent = intent_emb if intent_emb is not None and intent_emb.ndim == 3 else None
                actions_pred = model.predict_actions(
                    out["z_v_pooled_seq"], out["z_s_seq"], head_intent,
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
            pbar.set_postfix(mse=f"NAN", warn="skip")
            optimizer.zero_grad(set_to_none=True)
            sync_and_step(model, optimizer, device, args.grad_clip)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sync_and_step(model, optimizer, device, args.grad_clip)

        losses.append(loss.item())
        actions_pred_list.append(actions_pred.detach().abs().mean().item())

        pbar.set_postfix(
            mse=f"{loss.item():.5f}",
            a_mean=f"{actions_pred.detach().abs().mean().item():.4f}",
        )

    avg_loss = global_mean(losses, device)
    avg_action = global_mean(actions_pred_list, device, empty=0.0)
    return avg_loss, avg_action


# ================================================================
# V4 Batched Training — Single forward per segment (no T-loop)
# ================================================================

def train_v4_batched_epoch(model, loader, optimizer, device, args, max_steps=0):
    """V4 batched training: single batched forward per segment.

    For pure diffusion (no Mamba recurrence, no memory bank), this is
    5-10x faster than the T-loop version because it vectorizes the
    vision encoding and runs one forward+backward per batch instead
    of num_windows per batch.

    Use when: use_history=False and use_memory_bank=False
    (i.e., no sequential state to maintain).
    """
    model.train()
    losses, actions_pred_list = [], []
    n_steps = min(max_steps, len(loader)) if max_steps else len(loader)
    pbar = tqdm(range(n_steps), total=n_steps,
                desc="  [train V4-batched]", unit="batch", leave=False,
                disable=not is_primary())
    step_iter = iter(loader)

    for _ in pbar:
        try:
            batch = next(step_iter)
        except StopIteration:
            break

        frames_seg = torch.as_tensor(batch["frames_segment"]).to(device, non_blocking=True)  # (B, S, V, H, W, 3)
        states_seg = torch.as_tensor(batch["states_segment"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, S, 7)
        actions_seg = torch.as_tensor(batch["actions_segment"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, S, 7)
        # Segment lengths arrive on the CPU. Checking them there avoids a
        # device-to-host synchronization for every action window.
        seg_lens_cpu = torch.as_tensor(batch["segment_len"])  # (B,)
        Hs = args.history_size if model.use_history else 1
        chunk_size = args.chunk_size
        # Pre-computed features arrive as (B, S, V*257, 768); raw frames as
        # (B, S, V, H, W, 3). The model.forward path handles both via the
        # `_vision_forward` skip below.
        B, S = frames_seg.shape[:2]
        frames_are_features = frames_seg.shape[-1] == 768

        # Forward (batched) through the model — one call for the whole segment
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            if frames_are_features:
                # Skip DINOv2: feed features through forward_intent path manually.
                # Build the same intermediates (z_v_CLS_all, z_v_all, z_s_seq)
                # that the model's forward() would produce, then call head directly.
                TOKENS = frames_seg.shape[2]
                V = TOKENS // 257
                P = 256
                P_plus_1 = 257
                features = frames_seg.float()  # ensure fp32 for vision_patch_encoder
                features_4d = features.reshape(B * S, V, P_plus_1, 768)
                z_v_CLS_all = features_4d[:, :, -1, :].reshape(B, S, V, 768)
                z_v_all = features_4d[:, :, :-1, :].reshape(B, S, V * P, 768)
                z_s_seq = model.state_encoder(states_seg.reshape(B * S, -1)).reshape(B, S, -1)
                z_v_mod_seq = model.vision_patch_encoder(
                    z_v_all.reshape(B * S, V * P, 768),
                    z_s_seq.reshape(B * S, -1),
                ).reshape(B, S, V * P, -1)
                pool_out_dim = V * P * z_v_mod_seq.shape[-1]
                if not model._built:
                    model._build_head_and_bank(pool_out_dim)
                z_v_pooled_seq = z_v_mod_seq.reshape(B, S, pool_out_dim)
                # No Mamba history in batched mode (use_history=False required).
                h_seq = torch.zeros(B, S, 1, device=device)
                intent_emb = None
            else:
                out = model(frames_seg, states_seg)
                z_v_pooled_seq = out["z_v_pooled_seq"]  # (B, S, pool_out_dim)
                z_s_seq = out["z_s_seq"]                # (B, S, state_dim)
                h_seq = out["h_seq"]                    # (B, S, mamba_in_dim) or zeros
                intent_emb = out.get("intent_emb", None)  # (B, N, intent_dim) or None

            # Slide a window of size Hs ending at each timestep t, predict
            # the next chunk_size actions.
            num_windows = S - Hs - chunk_size + 2
            loss_accum = []
            logged_loss_sum = 0.0
            valid_counts = []
            actions_pred = None
            target = None

            for n in range(num_windows):
                current_t = n + Hs - 1
                valid_mask_cpu = seg_lens_cpu >= (current_t + chunk_size)
                if not bool(valid_mask_cpu.any()):
                    continue

                valid_mask = valid_mask_cpu.to(device, non_blocking=True)
                valid_count = int(valid_mask_cpu.sum().item())

                # Extract window from the batched outputs (no Python loop over
                # vision encoding — already done in forward())
                z_v_win = z_v_pooled_seq[:, n:current_t + 1]  # (B, Hs, pool_out_dim)
                z_s_win = z_s_seq[:, n:current_t + 1]        # (B, Hs, state_dim)
                h_current = h_seq[:, current_t]              # (B, mamba_in_dim)
                target = actions_seg[:, current_t:current_t + chunk_size]  # (B, K, 7)

                # Head: cond
                if args.head_type in ("diffusion", "flow_matching"):
                    cond = model.intention_head(z_v_win, z_s_win, intent_emb)
                    if not getattr(args, "no_sample_during_train", False):
                        actions_pred = model.sample_actions(
                            z_v_win, z_s_win, intent_emb,
                        )
                    else:
                        actions_pred = None
                    dim_weights = torch.ones(args.action_dim, device=device)
                    if args.action_dim >= 7:
                        dim_weights[6] = getattr(args, "gripper_loss_weight", 0.01)
                    loss = model.intention_head.loss(
                        target, cond, dim_weights=dim_weights, sample_mask=valid_mask,
                    )
                else:
                    actions_pred = model.predict_actions(z_v_win, z_s_win, intent_emb)
                    loss = F.mse_loss(actions_pred, target, reduction='none')
                    valid_mask = valid_mask_cpu.to(device, non_blocking=True)
                    loss = loss[valid_mask].mean()

                if getattr(args, "skip_nan", True):
                    # Reuse this scalar for logging below. The finite-loss
                    # decision already waits for CUDA, so a second .item()
                    # after the optimizer step would be an extra sync.
                    loss_value = loss.detach().item()
                    if not math.isfinite(loss_value):
                        continue
                    logged_loss_sum += loss_value * valid_count
                loss_accum.append(loss)
                valid_counts.append(valid_count)

            total_loss = (sum(loss * count for loss, count in zip(loss_accum, valid_counts)) / sum(valid_counts)
                          if loss_accum else None)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        if total_loss is not None:
            total_loss.backward()
        sync_and_step(model, optimizer, device, args.grad_clip)
        if total_loss is None:
            continue

        profiler = getattr(args, "_profiler", None)
        if profiler is not None:
            profiler.step()

        if getattr(args, "skip_nan", True):
            losses.append(logged_loss_sum / sum(valid_counts))
        else:
            losses.append(total_loss.item())
        if actions_pred is not None:
            actions_pred_list.append(actions_pred.detach().abs().mean().item())
        else:
            actions_pred_list.append(0.0)
        pbar.set_postfix(mse=f"{losses[-1]:.5f}")

    avg_loss = global_mean(losses, device)
    avg_action = global_mean(actions_pred_list, device, empty=0.0)
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
    pbar = tqdm(loader, desc="  [val]  ", unit="batch", leave=False,
                disable=not is_primary())
    for batch in pbar:

        Hs = args.history_size
        chunk_size = args.chunk_size
        B_s = len(batch["segment_len"])

        # Per-dimension loss weights: down-weight gripper (dim 6) so it doesn't dominate
        dim_weights = torch.ones(args.action_dim, device=device)
        if args.action_dim >= 7:
            dim_weights[6] = getattr(args, "gripper_loss_weight", 0.01)

        frames_seg = torch.as_tensor(batch["frames_segment"]).to(device, non_blocking=True)  # (B, S, V, H, W, 3)
        states_seg = torch.as_tensor(batch["states_segment"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, S, 7)
        target_seg = torch.as_tensor(batch["actions_segment"]).to(device, dtype=torch.float32, non_blocking=True)  # (B, S, 7)

        seg_lens = torch.as_tensor(batch["segment_len"], device=device)# (B,)

        # Reset memory bank at start of segment
        if model.use_memory_bank:
            model.memory_module.reset(batch_size=B_s, device=device)

        # Pre-encode vision for the entire segment (vision is the bottleneck)
        B, S = frames_seg.shape[:2]
        # Detect pre-computed DINOv2 features vs raw frames (see train_v4_epoch)
        frames_are_features = frames_seg.shape[-1] == 768
        if frames_are_features:
            TOKENS = frames_seg.shape[2]
            V = TOKENS // 257
            z_v_all = frames_seg.reshape(B * S * V, -1, 768)  # (B*S*V, P+1, raw_dim=768)
        else:
            V, H, W, C = frames_seg.shape[2:]
            z_v_all = model._vision_forward(frames_seg.reshape(B * S * V, H, W, C))   # (B*S*V, P+1, raw_dim=768)

        # Extract CLS tokens (last position per camera, NOT the last position overall).
        # Layout: [cam0_patches..., cam0_CLS, cam1_patches..., cam1_CLS, ...]
        # Total tokens per camera = P + 1
        P_plus_1 = z_v_all.shape[1]
        P = P_plus_1 - 1
        # Reshape to (B*S*V, P+1, 768) then take last per camera
        z_v_all_reshaped = z_v_all.reshape(B * S, V, P_plus_1, 768)
        z_v_CLS_all = z_v_all_reshaped[:, :, -1, :]  # (B*S, V, 768)
        z_v_CLS_all = z_v_CLS_all.reshape(B, S, V, -1)  # (B, S, V, raw_dim=768)

        # Extract patches (all positions except the last per camera)
        z_v_all = z_v_all_reshaped[:, :, :-1, :].reshape(B, S , V * P, 768)   # (B, S, V*P, raw_dim=768)

        _, _, state_dim = states_seg.shape
        z_s_all = model.state_encoder(
            states_seg.reshape(B * S, state_dim)
        ).reshape(B, S, -1)

        z_v_mod_all = model.vision_patch_encoder(
            z_v_all.reshape(B * S, V * P, 768),
            z_s_all.reshape(B * S, -1),
        ).reshape(B, S, V * P, -1)  # (B, S, V*P, comp_dim)

        # Build head and memory bank on first segment (lazy build)
        if not model._built:
            B_seg, S, VP, comp_dim = z_v_mod_all.shape
            # pool_out_dim = VP * comp_dim (the shape stored in memory bank)
            model._build_head_and_bank(VP * comp_dim)

        # Sequential T-loop
        last_actions_pred = None
        actions_pred = None
        loss_accum = []
        valid_counts = []
        max_seg_len = S
        if model.use_history:
            num_windows = max_seg_len - Hs - chunk_size + 2
        else:
            num_windows = max_seg_len - chunk_size + 1

        # Keep one autocast cache for the whole segment: repeated windows
        # must reuse parameter casts rather than retaining a copy each.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            intent_sequence = None
            if num_windows > 0 and getattr(model, "intention_encoder", None) is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    intent_sequence = model.intention_encoder.forward_sequence(
                        z_v_CLS_all[:, :S - chunk_size + 1], z_s_all[:, :S - chunk_size + 1],
                        readout_start=Hs - 1,
                    )

            for n in range(num_windows):
                if model.use_history:
                    current_t = n + Hs - 1
                    history_start = current_t - Hs + 1
                    history_end = current_t + 1
                else:
                    current_t = n
                    history_start = n
                    history_end = n + 1
                valid_mask = seg_lens >= (current_t + chunk_size)
                if not valid_mask.any():
                    continue

                z_v_win = z_v_mod_all[:, history_start:history_end]  # (B, H_actual, V*P, comp_dim)
                z_s_win = z_s_all[:, history_start:history_end]  # (B, H_actual, state_dim)

                # Forward through model
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    intent_emb = intent_sequence[:, current_t] if intent_sequence is not None else None

                    B_seg, H_actual, VP, comp_dim = z_v_win.shape
                    z_v_win_for_head, z_s_win_for_head, h_for_head = model.condition_actions(
                        z_v_win.reshape(B_seg, H_actual, VP * comp_dim),
                        z_s_win, intent_emb, observed_mask=current_t < seg_lens,
                    )

                    # Target: chunk_size future actions from current time
                    target = target_seg[:, current_t:current_t + chunk_size]

                    # Loss
                    if args.head_type in ("diffusion", "flow_matching"):
                        cond = model.intention_head(
                            z_v_win_for_head, z_s_win_for_head, h_for_head,
                        )
                        actions_pred = model.sample_actions(
                            z_v_win_for_head, z_s_win_for_head, h_for_head
                        )
                        loss = model.intention_head.loss(
                            target, cond, dim_weights=dim_weights, sample_mask=valid_mask,
                        )
                    else:
                        actions_pred = model.predict_actions(
                            z_v_win_for_head, z_s_win_for_head, h_for_head,
                        )
                        loss = F.mse_loss(actions_pred, target, reduction='none')
                        loss = loss[valid_mask].mean() if valid_mask.any() else loss.mean()

                if args.skip_nan and not torch.isfinite(loss):
                    continue

                loss_accum.append(loss)
                valid_counts.append(int(valid_mask.sum().item()))
                last_actions_pred = actions_pred

                # Accumulate every valid prediction window, excluding padded samples.
                actions_pred = actions_pred[valid_mask]
                target = target[valid_mask]
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
                    grip_pred_binary = (grip_pred > getattr(args, "gripper_threshold", 0.0)).float()
                    grip_target_binary = (grip_target > 0).float()
                    grip_correct = (grip_pred_binary == grip_target_binary).float().sum().item()
                    grip_total = B_win * T
                    grip_correct_total += grip_correct
                    grip_total_total += grip_total

        # Accumulate metrics across all windows in this segment
        if loss_accum:
            avg_window_loss = (sum(loss * count for loss, count in zip(loss_accum, valid_counts)) / sum(valid_counts)).item()
            losses.append(avg_window_loss)
            actions_pred_list.append(actions_pred.detach().abs().mean().item() if actions_pred is not None else 0.0)
            pbar.set_postfix(mse=f"{avg_window_loss:.5f}")


    avg_loss = global_mean(losses, device)
    avg_action = global_mean(actions_pred_list, device, empty=0.0)

    if distributed_enabled():
        local_stats = (
            per_dim_squared, per_dim_abs, n_samples,
            padded_gripper_batches, genuine_gripper_batches,
            grip_correct_total, grip_total_total,
        )
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_stats)
        squared = [stats[0] for stats in gathered if stats[0] is not None]
        absolute = [stats[1] for stats in gathered if stats[1] is not None]
        per_dim_squared = sum(squared) if squared else None
        per_dim_abs = sum(absolute) if absolute else None
        n_samples = sum(stats[2] for stats in gathered)
        padded_gripper_batches = sum(stats[3] for stats in gathered)
        genuine_gripper_batches = sum(stats[4] for stats in gathered)
        grip_correct_total = sum(stats[5] for stats in gathered)
        grip_total_total = sum(stats[6] for stats in gathered)

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
    parser.add_argument("--head-type", choices=["transformer", "mamba", "hybrid", "diffusion", "flow_matching"],
                        default="diffusion",
                        help="V4 supports diffusion and flow_matching; transformer/mamba/hybrid are future work. Which head architecture: transformer, mamba, hybrid, "
                             "diffusion (DDPM), or flow_matching (CFM with Euler ODE).")
    parser.add_argument("--use-history", action="store_true", default=True,
                        help="Use observation history; Mamba is constructed only with intent tokens.")
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

    parser.add_argument("--gripper-loss-weight", type=float, default=0.01,
                        help="Gripper loss weight relative to each motion dimension")
    parser.add_argument("--gripper-threshold", type=float, default=0.0,
                        help="Validation cutoff for predicted gripper: 0 for signed commands, 0.5 for binary 0/1 targets.")

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

    # Performance: skip expensive operations during training
    parser.add_argument("--no-sample-during-train", action="store_true", default=False,
                        help="Skip DDIM sampling for action mean during training (5-10x faster).")
    parser.set_defaults(no_sample_during_train=True)  # Default: skip for speed
    parser.add_argument("--torch-compile", action="store_true", default=False,
                        help="Reserved; currently reports that compilation is not wired into this trainer.")

    # V4: Segment training
    parser.add_argument("--history-size", type=int, default=1,
                        help="Action-head observation window and initial warmup (default 1). "
                             "Mamba retains causal context across the training segment.")
    parser.add_argument("--segment-min-mult", type=int, default=2,
                        help="Min segment length = history_size * this (default 2). "
                             "Set to 0 to disable V4 segment training (use V3 instead).")
    parser.add_argument("--segment-max-mult", type=int, default=5,
                        help="Max segment length = history_size * this (default 5). "
                             "Set to 0 to disable V4 segment training (use V3 instead).")

    # V4: Semantic anchoring
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="WIP: reserved for semantic anchoring; currently not connected to training.")

    # Text modality (optional)
    parser.add_argument("--use-text", action="store_true", default=False,
                        help="WIP: constructs a text encoder, but semantic supervision is not connected.")
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
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Use PyTorch fused AdamW (default: on for CUDA, off for CPU).")
    parser.add_argument("--skip-nan", dest="skip_nan", action="store_true",
                        default=True,
                        help="Skip batches with NaN/Inf loss (default on).")
    parser.add_argument("--no-skip-nan", dest="skip_nan", action="store_false",
                        help="Disable NaN skipping (will NaN out the run).")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Keep DataLoader workers alive between epochs (default: on when workers > 0).")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches prefetched by each DataLoader worker (default: 2; ignored for workers=0).")
    # NOTE: --loss-mode removed; always 'action' for v3.
    # NOTE: --bf16 / --no-bf16 removed; BF16 is always on for speed.
    # Pre-computed DINOv2 features (skips DINOv2 forward during training).
    parser.add_argument("--use-precomputed-dinov2", action="store_true", default=False,
                        help="Use pre-computed DINOv2 features from a sidecar "
                             "directory of per-episode .npy files. "
                             "Generate with: python scripts/precompute_dinov2.py")
    parser.add_argument("--dinov2-path", default=None,
                        help="Path to DINOv2 sidecar DIRECTORY (containing "
                             "index.json + per-episode .npy files). If omitted "
                             "and --use-precomputed-dinov2 is set, defaults to "
                             "<data path stem>.dinov2/ in the same directory.")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Cap steps per epoch (0 = use full loader).")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Profile V4-batched CUDA steps after one wait and one warmup step; 0 disables profiling.")
    parser.add_argument("--profile-dir", default="profiles",
                        help="Directory for Chrome/TensorBoard profiler traces when --profile-steps > 0.")
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
    device = setup_distributed(args)
    if not is_primary():
        # Only rank zero reports progress; worker errors still reach stderr.
        sys.stdout = open(os.devnull, "w")
    if args.torch_compile:
        print("  NOTE: --torch-compile is not implemented in train_intention.py; running eager mode.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.fused_adamw is None:
        args.fused_adamw = device.type == "cuda"
    elif args.fused_adamw and device.type != "cuda":
        raise ValueError("--fused-adamw requires --device cuda")
    print(f"\n=== ALIGN Intention (Mamba) Training ===")
    print(f"  Device:     {device}")
    if distributed_enabled():
        print(f"  Processes:  {dist.get_world_size()} (batch size is per process)")
    print(f"  Head Type:  {args.head_type} head")
    print(f"  Chunk (K):  {args.chunk_size}")
    # Determine num_cameras from --cameras argument (auto-derived)
    num_cameras = len(args.cameras)
    print(f"  Cameras: {num_cameras} (from --cameras {args.cameras})")

    # Default DINOv2 sidecar path: <data stem>.dinov2/ next to data file(s)
    if args.use_precomputed_dinov2 and args.dinov2_path is None:
        if len(args.data) == 1:
            default_dinov2 = Path(args.data[0]).with_name(
                Path(args.data[0]).stem + ".dinov2"
            )
            args.dinov2_path = str(default_dinov2)
        else:
            raise ValueError(
                "--dinov2-path is required when using multiple --data files "
                "with --use-precomputed-dinov2"
            )
        print(f"  DINOv2 sidecar dir: {args.dinov2_path}")

    # Output dir
    out_dir = Path(args.output_dir)
    if len(args.data) == 1:
        ds_name = Path(args.data[0]).stem
    else:
        ds_name = "+".join(Path(p).stem for p in args.data)
    out_dir = out_dir / ds_name
    if is_primary():
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.run_name:
            out_dir = out_dir / args.run_name
        else:
            # Select a run folder once, then share it with all workers.
            existing_runs = sorted(
                int(p.name.split("_")[1])
                for p in out_dir.glob("run_*")
                if p.is_dir() and p.name.split("_")[1].isdigit()
            )
            next_run = (existing_runs[-1] + 1) if existing_runs else 1
            out_dir = out_dir / f"run_{next_run}"
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed_enabled():
        selected_dir = [str(out_dir) if is_primary() else None]
        dist.broadcast_object_list(selected_dir, src=0)
        out_dir = Path(selected_dir[0])
    print(f"  Output dir: {out_dir}")

    # Wandb
    wandb_trainer = init_wandb(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args),
        enabled=args.wandb and is_primary(),
    )
    print(f"  W&B:        {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # Data
    print("\n  Loading data...")
    full_ds, train_ds, val_ds = build_datasets(args)
    train_loader, val_loader = build_loaders(train_ds, val_ds, args)
    if not len(train_loader):
        raise ValueError("Training set is too small for the per-process batch size")

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
        print(f" shape of dummy : {dummy.shape}")
        if dummy.ndim == 3:
            z_v_dummy = dummy
        else:
            z_v_dummy = model._vision_forward(dummy)
        print(f"  Vision output shape: {z_v_dummy.shape}")
        if z_v_dummy.ndim == 2:
            N_tok_actual = z_v_dummy.shape[0]
        else:
            N_tok_actual = z_v_dummy.shape[1]
    pool_out_dim = (N_tok_actual - num_cameras) * args.compressed_dim
    print(f"  Pool out dim: {pool_out_dim} (N_tok_actual={N_tok_actual}, compressed_dim={args.compressed_dim})")
    model._build_head_and_bank(pool_out_dim)
    if distributed_enabled():
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
        # Keep initialization identical, then vary training-time randomness.
        torch.manual_seed(args.seed + dist.get_rank())
        np.random.seed(args.seed + dist.get_rank())
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
            "dinov2/use_precomputed": args.use_precomputed_dinov2,
            "dinov2/sidecar_path": args.dinov2_path,
        })
        # Watch gradients (for monitoring)
        wandb_trainer.watch(model, log="gradients", log_freq=200, log_graph=False)
    if model.intention_encoder is not None:
        print("  Training: vision projection + state encoder + intention encoder + head")
        print("  (DINOv2 backbone frozen)")
    else:
        print("  Training: vision projection + state encoder + head (no Mamba history)")
        print("  (DINOv2 backbone frozen)")

    # Optimizer — single LR group (only trainable params receive gradients).
    # Fused AdamW keeps parameters and optimizer state in FP32; it changes only
    # kernel fusion/order. The resolved setting is recorded in config.json.
    adamw_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.fused_adamw:
        adamw_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(trainable, **adamw_kwargs)
    print(f"  Optimizer: 1 LR group (lr={args.lr:.2e}, {n_trainable:,} trainable params; "
          f"fused={args.fused_adamw})")

    # Save config snapshot
    config_snapshot = vars(args).copy()
    config_snapshot["num_cameras"] = num_cameras
    config_snapshot["n_trainable_params"] = n_trainable
    config_snapshot["n_total_params"] = n_total
    config_snapshot["model_class"] = "ALIGNIntentionModel"
    if is_primary():
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)

    # Log file
    log_path = out_dir / "intention_log.jsonl"
    log_fp = open(log_path, "w") if is_primary() else None

    # Training loop
    # is_v4 is auto-detected: V4 features on OR any segment-* arg is set
    is_v4 = (
        args.use_intent_tokens
        or args.use_memory_bank
        or getattr(args, "segment_min_mult", 0) > 0
        or getattr(args, "segment_max_mult", 0) > 0
    )

    # Choose the most efficient training function:
    # - V4 batched: 5-10x faster than V4 T-loop, but only works without
    #   sequential state (no Mamba, no memory bank)
    # - V4 T-loop: needed when Mamba or memory bank is active
    # - V3: old V3 mode (for backward compatibility)
    if is_v4 and not args.use_history and not args.use_memory_bank:
        train_fn = train_v4_batched_epoch
        mode_str = "(V4 batched mode)"
    elif is_v4:
        train_fn = train_v4_epoch
        mode_str = "(V4 T-loop mode)"
    else:
        train_fn = train_one_epoch
        mode_str = "(V3 mode)"
    print(f"  Training for {args.epochs} epochs... {mode_str}")
    best_val_loss = float("inf")
    profiler = None
    profile_dir = None
    if args.profile_steps:
        if train_fn is not train_v4_batched_epoch:
            raise ValueError("--profile-steps currently supports V4-batched mode only "+
                             "(requires --no-history and no memory bank).")
        if args.max_steps_per_epoch and args.max_steps_per_epoch < args.profile_steps + 2:
            raise ValueError("--max-steps-per-epoch must be at least --profile-steps + 2 "+
                             "for profiler wait and warmup steps.")
        if is_primary():
            profile_dir = Path(args.profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1,
                                                 active=args.profile_steps, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            profiler.__enter__()
            args._profiler = profiler
            print(f"  CUDA profiler: {args.profile_steps} active steps → {profile_dir}")
    else:
        args._profiler = None
    for epoch in range(1, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        t_start = time.time()
        train_loss, train_action = train_fn(
            model, train_loader, optimizer, device, args,
            max_steps=args.max_steps_per_epoch,
        )
        if profiler is not None:
            assert profile_dir is not None
            profiler.__exit__(None, None, None)
            summary_path = profile_dir / "operator_summary.txt"
            summary_path.write_text(
                profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
            )
            print(f"  CUDA profiler summary: {summary_path}")
            args._profiler = None
            profiler = None
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
        if log_fp is not None:
            log_fp.write(json.dumps(log_record) + "\n")
            log_fp.flush()

        # Save best (based on val_loss)
        if is_primary() and val_loss < best_val_loss:
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

    if log_fp is not None:
        log_fp.close()
    wandb_trainer.finish()
    print(f"\n  Done. Best val_loss={best_val_loss:.5f}")
    print(f"  Best checkpoint: {out_dir / 'intention_best.pt'}")
    print(f"  Logs:            {log_path}")
    if distributed_enabled():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
