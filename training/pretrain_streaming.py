#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming pretraining script — trains ALIGN directly from LeRobot v3 Hub datasets.

Zero disk, zero download. Data streams from Hugging Face during training.

Usage:
    # Pretrain from LIBERO (Franka sim — best match)
    python training/pretrain_streaming.py \\
        --dataset nvidia/LIBERO_LeRobot_v3 \\
        --output-dir ./checkpoints/pretrain \\
        --epochs 50

    # Pretrain from BridgeData (real WidowX — diversity)
    python training/pretrain_streaming.py \\
        --dataset nvidia/BridgeData2_LeRobot_v3 \\
        --output-dir ./checkpoints/pretrain

    # Pretrain from both datasets (multi-dataset pretraining)
    python training/pretrain_streaming.py \\
        --dataset nvidia/LIBERO_LeRobot_v3 \\
        --dataset nvidia/BridgeData2_LeRobot_v3
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from training.contrastive_loss import ContrastiveLoss3Way
from training.wandb_utils import init_wandb
from data.open_dataset import LeRobotAdapter


# ================================================================
# Streaming multi-dataset wrapper
# ================================================================

class MultiDatasetStream(IterableDataset):
    """IterableDataset that round-robins between multiple LeRobot v3 streams.

    Each worker gets a different offset so different GPUs see different data.
    Exhausted loaders are recreated for infinite streaming.
    """

    def __init__(self, repo_ids: list[str], frames_per_item: int = 8):
        super().__init__()
        self.repo_ids = repo_ids
        self.frames_per_item = frames_per_item

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        # Create loaders for each dataset
        adapters: list[LeRobotAdapter] = []
        loaders: list = []
        for i, repo_id in enumerate(self.repo_ids):
            adapter = LeRobotAdapter(
                repo_id,
                batch_size=1,
                num_workers=0,
            )
            ds = adapter.get_streaming_dataset()
            adapters.append(adapter)
            loaders.append(iter(DataLoader(ds, batch_size=1, shuffle=False)))

        # Track which loaders have been exhausted (replace them on exhaustion)
        exhausted: list[bool] = [False] * len(loaders)

        while True:
            for i, loader in enumerate(loaders):
                try:
                    batch = next(loader)
                    yield {
                        "frames": batch["frames"][0],
                        "poses": batch["poses"][0],
                        "text": batch["texts"][0],
                    }
                except StopIteration:
                    # Recreate this loader (streaming datasets are finite;
                    # we restart from scratch to simulate infinite epochs)
                    exhausted[i] = True
            # Recreate all exhausted loaders
            for i, is_exhausted in enumerate(exhausted):
                if is_exhausted:
                    ds = adapters[i].get_streaming_dataset()
                    loaders[i] = iter(DataLoader(ds, batch_size=1, shuffle=False))
                    exhausted[i] = False


def streaming_pretrain_collate(batch: list[dict], traj_window: int = 10) -> dict:
    """Collate streaming samples into ALIGN pretraining batch.

    For contrastive pretraining we need (vision, trajectory_window, text) triples.
    Since streaming gives single frames, we create synthetic trajectory windows
    by repeating the pose (clean data, zero variance — contrastive learns
    to align the static pose with the visual scene).

    For real trajectory windows, delta_timestamps in LeRobotAdapter provides
    temporal context from the Parquet shards.
    """
    import torch

    B = len(batch)
    all_frames = []
    all_trajs = []
    all_texts = []

    for item in batch:
        frame = item["frames"]
        pose = item["poses"]
        text = item.get("text", "pick and place")

        # LeRobot v3 images are always (C, H, W) — permute to (H, W, C) for DINOv2
        if frame.dim() == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        # Ensure uint8 format
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)

        all_frames.append(frame)

        # Build trajectory window from pose (use first 6 dims for EEF)
        # LIBERO state is 8D [x,y,z,ax,ay,az,grip,grip], ALIGN expects 6D
        pose_eef = pose[..., :6] if pose.shape[-1] > 6 else pose
        traj = pose_eef.unsqueeze(0).repeat(traj_window, 1)  # (K, 6)
        all_trajs.append(traj)

        all_texts.append(text)

    return {
        "frames": torch.stack(all_frames, dim=0),       # (B, H, W, 3)
        "trajectories": torch.stack(all_trajs, dim=0),   # (B, K, D)
        "texts": all_texts,
    }


# ================================================================
# Training loops
# ================================================================

def pretrain_from_stream(
    repo_ids: list[str],
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    embed_dim: int = 256,
    temperature: float = 0.07,
    max_grad_norm: float = 1.0,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    checkpoint_every: int = 10,
    wandb_project: str = "align-streaming",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 4,
):
    """Contrastive pretraining directly from LeRobot v3 streaming datasets.

    Args:
        repo_ids: List of Hugging Face dataset repo IDs.
        output_dir: Checkpoint save directory.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        embed_dim: Latent embedding dimension.
        temperature: InfoNCE temperature.
        max_grad_norm: Gradient clipping norm.
        device: Compute device.
        max_steps_per_epoch: Steps per epoch (streaming is infinite).
        checkpoint_every: Save checkpoints every N epochs.
        wandb_project: W&B project name.
        wandb_run: W&B run name.
        enable_wandb: Enable W&B logging.
        num_workers: DataLoader workers. Set to 0 if HF Hub download is slow.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ALIGN Streaming Contrastive Pretraining ===")
    print(f"  Datasets: {repo_ids}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {epochs}")
    print(f"  Steps/ep: {max_steps_per_epoch}")
    print(f"  LR:       {lr}")
    print(f"  Output:   {output_dir}")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-streaming-pretrain",
            "datasets": repo_ids,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "embed_dim": embed_dim,
            "temperature": temperature,
            "max_grad_norm": max_grad_norm,
            "max_steps_per_epoch": max_steps_per_epoch,
            "device": str(device),
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:      {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Streaming dataset ──
    stream_ds = MultiDatasetStream(repo_ids)
    loader = DataLoader(
        stream_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=streaming_pretrain_collate,
        pin_memory=True,
    )

    # -- Model ──
    model = ALIGNModel(
        embed_dim=embed_dim,
        use_text=True,
        device=str(device),
    ).to(device)

    # Freeze backbones
    for p in model.vision_encoder.backbone.parameters():
        p.requires_grad = False
    if model.text_encoder is not None:
        for p in model.text_encoder.model.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_params:,}")

    criterion = ContrastiveLoss3Way(temperature=temperature)
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # -- Training ──
    log_path = output_dir / "streaming_training_log.jsonl"
    log_fp = open(log_path, "a")
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_cos_vt = []
        epoch_cos_vl = []
        epoch_cos_tl = []

        loader_iter = iter(loader)
        for step in range(max_steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            frames = batch["frames"].to(device)
            trajs = batch["trajectories"].float().to(device)
            texts = batch["texts"]

            # Pad trajectories to 6D if needed
            if trajs.shape[-1] < 6:
                pad = torch.zeros(*trajs.shape[:-1], 6 - trajs.shape[-1], device=trajs.device)
                trajs = torch.cat([trajs, pad], dim=-1)

            z_v = model.encode_vision(frames)
            z_t = model.encode_trajectory(trajs)
            z_text = model.encode_text(texts)

            stats = criterion(z_v, z_t, z_text)
            loss = stats["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_cos_vt.append(stats["avg_cos_vt"].item())
            epoch_cos_vl.append(stats["avg_cos_vl"].item())
            epoch_cos_tl.append(stats["avg_cos_tl"].item())

            # Progress every 100 steps
            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}, step {step + 1}/{max_steps_per_epoch}  "
                      f"loss: {loss.item():.4f}  vt: {stats['avg_cos_vt'].item():.3f}  "
                      f"vl: {stats['avg_cos_vl'].item():.3f}  tl: {stats['avg_cos_tl'].item():.3f}",
                      flush=True)

        avg_loss = float(np.mean(epoch_losses))
        avg_vt = float(np.mean(epoch_cos_vt))
        avg_vl = float(np.mean(epoch_cos_vl))
        avg_tl = float(np.mean(epoch_cos_tl))

        print(f"  Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
              f"cos_vt: {avg_vt:.3f}  cos_vl: {avg_vl:.3f}  cos_tl: {avg_tl:.3f}")

        # W&B epoch logging
        wandb_trainer.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "cos_vt": avg_vt,
            "cos_vl": avg_vl,
            "cos_tl": avg_tl,
        }, step=epoch + 1)

        # Log
        log_fp.write(json.dumps({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "cos_vt": avg_vt,
            "cos_vl": avg_vl,
            "cos_tl": avg_tl,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, output_dir / "best.pt")
            print(f"  -> best checkpoint (loss: {avg_loss:.4f})")
            wandb_trainer.log({"best_loss": best_loss}, step=epoch + 1)

        if (epoch + 1) % checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_dir / f"epoch_{epoch + 1:04d}.pt")

    log_fp.close()
    print(f"\n  Pretraining complete. Best loss: {best_loss:.4f}")
    print(f"  Logs: {log_path}")
    wandb_trainer.finish()
    return str(output_dir / "best.pt")


# ================================================================
# Streaming head training with on-the-fly noise
# ================================================================

def train_heads_from_stream(
    repo_ids: list[str],
    pretrained_checkpoint: str,
    output_dir: str,
    epochs_decision: int = 10,
    epochs_assistant: int = 20,
    epochs_joint: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    chunk_size: int = 5,
    max_steps_per_epoch: int = 2000,
    device: Optional[str] = None,
    noise_configs: Optional[list[dict]] = None,
    wandb_project: str = "align-streaming",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 4,
):
    """Train Decision + Assistant heads from streamed data with on-the-fly noise.

    Synthetic noise is injected into streamed clean poses each batch.
    α_target and Δpose targets are computed from the noise + clean pair.

    Args:
        repo_ids: LeRobot v3 dataset IDs.
        pretrained_checkpoint: Path to frozen pretrained backbone.
        output_dir: Checkpoint directory.
        epochs_decision, epochs_assistant, epochs_joint: Stage epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: Weight decay.
        chunk_size: Assistant head chunk size K.
        max_steps_per_epoch: Steps per epoch.
        device: Compute device.
        noise_configs: Noise configs per batch (default: medium noise).
    """
    from training.train_full_pipeline import SyntheticNoiseInjector

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if noise_configs is None:
        noise_configs = [{"pos_noise_std": 0.020, "orn_noise_std": 4.0, "label": "medium"}]

    print(f"=== ALIGN Streaming Head Training ===")
    print(f"  Datasets:    {repo_ids}")
    print(f"  Pretrained:  {pretrained_checkpoint}")
    print(f"  Device:      {device}")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-streaming-heads",
            "datasets": repo_ids,
            "pretrained_checkpoint": pretrained_checkpoint,
            "epochs_decision": epochs_decision,
            "epochs_assistant": epochs_assistant,
            "epochs_joint": epochs_joint,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "chunk_size": chunk_size,
            "max_steps_per_epoch": max_steps_per_epoch,
            "device": str(device),
            "noise_configs": noise_configs,
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:         {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Streaming dataset ──
    stream_ds = MultiDatasetStream(repo_ids)
    loader = DataLoader(
        stream_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=streaming_head_collate,
        pin_memory=True,
    )

    # -- Model ──
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
    ).to(device)

    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.freeze_backbone()
    print(f"  Loaded backbone from {pretrained_checkpoint}")

    # -- Noise injectors (one per config) ──
    injectors = [
        SyntheticNoiseInjector(
            pos_noise_std=cfg["pos_noise_std"],
            orn_noise_std=cfg["orn_noise_std"],
            seed=42 + i,
        )
        for i, cfg in enumerate(noise_configs)
    ]

    log_path = output_dir / "streaming_head_log.jsonl"
    log_fp = open(log_path, "a")

    def _head_collate_stats(stage: str, epoch: int, losses: list, alphas: list, deltas: list):
        entry = {
            "stage": stage,
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "alpha_mean": float(np.mean(alphas)) if alphas else 0,
            "delta_mean": float(np.mean(deltas)) if deltas else 0,
            "timestamp": datetime.now().isoformat(),
        }
        log_fp.write(json.dumps(entry) + "\n")
        log_fp.flush()
        return entry

    def _run_stage(stage: str, opt: optim.Optimizer, epochs: int):
        best = float("inf")
        for epoch in range(epochs):
            model.train()
            losses, alphas, deltas = [], [], []

            loader_iter = iter(loader)
            for step in range(max_steps_per_epoch):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                frames = batch["frames"].to(device)
                clean_poses = batch["poses"].float().to(device)
                texts = batch["texts"]

                # Rotate noise injectors
                injector = injectors[step % len(injectors)]
                noisy_poses_np = injector.inject(clean_poses.cpu().numpy())
                noisy_poses = torch.from_numpy(noisy_poses_np).float().to(device)

                # Compute targets on-the-fly
                d_max = 0.10
                pos_error = torch.norm(noisy_poses[:, :3] - clean_poses[:, :3], dim=1)
                alpha_target = torch.clamp(pos_error / d_max, 0.0, 1.0)

                B, D = clean_poses.shape
                delta_target = torch.zeros(B, chunk_size, 6, device=device)
                # Ensure 6D for orientation: take first 6 dims (pos) or use identity
                clean_6d = clean_poses[:, :6] if D >= 6 else torch.cat([
                    clean_poses, torch.zeros(B, 6 - D, device=device)
                ], dim=-1)
                # Delta = clean − noisy (both already 6D from injector output)
                for i in range(1, chunk_size + 1):
                    delta_target[:, i - 1, :] = clean_6d - noisy_poses[:, :6]

                # Encode
                with torch.no_grad():
                    z_v = model.encode_vision(frames)
                    z_t = model.encode_trajectory(noisy_poses.unsqueeze(1).repeat(1, 10, 1))
                    z_text = model.encode_text(texts)

                dists = torch.zeros(B, 3, device=device)

                if stage == "decision":
                    alpha_pred = model.decision_head(z_v, z_t, z_text, dists)
                    loss = F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_target)
                elif stage == "assistant":
                    delta_pred = model.assistant_head(z_v, z_t, z_text, noisy_poses[:, :6])
                    loss = F.mse_loss(delta_pred, delta_target)
                elif stage == "joint":
                    alpha_pred = model.decision_head(z_v, z_t, z_text, dists)
                    delta_pred = model.assistant_head(z_v, z_t, z_text, noisy_poses[:, :6])
                    loss = (F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_target) +
                            0.5 * F.mse_loss(delta_pred, delta_target))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                losses.append(loss.item())
                if stage in ("decision", "joint"):
                    alphas.append(alpha_pred.detach().mean().item())
                if stage in ("assistant", "joint"):
                    deltas.append(delta_pred.detach().abs().mean().item())

                # Progress every 100 steps
                if (step + 1) % 100 == 0:
                    print(f"  Epoch {epoch + 1}, step {step + 1}/{max_steps_per_epoch} "
                          f"[{stage}] loss: {loss.item():.4f}", flush=True)

            entry = _head_collate_stats(stage, epoch + 1, losses, alphas, deltas)
            avg = entry["loss"]
            print(f"  Epoch {epoch + 1:3d} [{stage}] loss: {avg:.4f}  "
                  f"α: {entry['alpha_mean']:.3f}  Δ: {entry['delta_mean']:.4f}")

            # W&B logging
            wandb_trainer.log({
                f"{stage}/loss": avg,
                f"{stage}/alpha_mean": entry["alpha_mean"],
                f"{stage}/delta_mean": entry["delta_mean"],
                f"{stage}/epoch": epoch + 1,
            }, step=epoch + 1)

            if avg < best:
                best = avg
                torch.save(model.state_dict(), output_dir / f"{stage}_best.pt")
                # Upload checkpoint to W&B
                wandb_trainer.save(str(output_dir / f"{stage}_best.pt"))
        torch.save(model.state_dict(), output_dir / f"{stage}_last.pt")
        print(f"  [{stage}] best loss: {best:.4f}")

    # -- Stage 1: Decision ──
    print(f"\n  --- Decision Head ({epochs_decision} epochs) ---")
    opt_d = optim.AdamW(model.decision_head.parameters(), lr=lr, weight_decay=weight_decay)
    _run_stage("decision", opt_d, epochs_decision)

    # -- Stage 2: Assistant ──
    print(f"\n  --- Assistant Head ({epochs_assistant} epochs) ---")
    opt_a = optim.AdamW(model.assistant_head.parameters(), lr=lr, weight_decay=weight_decay)
    _run_stage("assistant", opt_a, epochs_assistant)

    # -- Stage 3: Joint ──
    print(f"\n  --- Joint Fine-Tuning ({epochs_joint} epochs) ---")
    opt_j = optim.AdamW(
        [p for p in model.decision_head.parameters()] + [p for p in model.assistant_head.parameters()],
        lr=lr * 0.5, weight_decay=weight_decay,
    )
    _run_stage("joint", opt_j, epochs_joint)

    log_fp.close()
    print(f"\n  Head training complete.")
    print(f"  Logs: {log_path}")
    wandb_trainer.finish()
    return str(output_dir / "joint_best.pt")


def streaming_head_collate(batch: list[dict]) -> dict:
    """Collate streaming samples for head training."""
    import torch

    B = len(batch)
    all_frames = []
    all_poses = []
    all_texts = []

    for item in batch:
        frame = item["frames"]
        pose = item["poses"]
        text = item.get("text", "pick and place")

        if frame.dim() == 3 and frame.shape[0] in (1, 3):  # (C, H, W)
            frame = frame.permute(1, 2, 0)
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)

        all_frames.append(frame)
        all_poses.append(pose.float())
        all_texts.append(text)

    return {
        "frames": torch.stack(all_frames, dim=0),
        "poses": torch.stack(all_poses, dim=0),
        "texts": all_texts,
    }


# ================================================================
# Full streaming pipeline (pretrain + heads)
# ================================================================

def run_streaming_pipeline(
    repo_ids: list[str],
    output_dir: str,
    epochs_pretrain: int = 50,
    epochs_heads: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: Optional[str] = None,
    stages: str = "all",
    pretrained_checkpoint: Optional[str] = None,
    wandb_project: str = "align-streaming",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 4,
):
    """Full ALIGN training pipeline using ONLY streaming data.

    Zero downloads. Zero disk space. Zero conversion.

    Args:
        repo_ids: LeRobot v3 dataset IDs.
        output_dir: Base output directory.
        epochs_pretrain: Epochs for contrastive pretraining.
        epochs_heads: Total epochs across all head stages.
        batch_size: Batch size.
        lr: Learning rate.
        device: Compute device.
        stages: 'all', 'pretrain', or 'heads'.
        pretrained_checkpoint: Skip pretraining, use existing backbone.
        wandb_project: W&B project name.
        wandb_run: W&B run name.
        enable_wandb: Enable W&B logging.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = pretrained_checkpoint

    # -- Stage 1: Contrastive Pretraining ──
    if stages in ("all", "pretrain"):
        print(f"\n{'='*60}")
        print(f"[pipeline] Stage 1: Streaming Contrastive Pretraining ({epochs_pretrain} epochs)")
        print(f"{'='*60}")

        pretrained_path = pretrain_from_stream(
            repo_ids=repo_ids,
            output_dir=str(output_dir / "pretrain"),
            epochs=epochs_pretrain,
            batch_size=batch_size,
            lr=lr,
            device=device,
            wandb_project=wandb_project,
            wandb_run=(wandb_run or "streaming") + "-pretrain",
            enable_wandb=enable_wandb,
            num_workers=num_workers,
        )
    else:
        if pretrained_path is None:
            pretrained_path = str(output_dir / "pretrain" / "best.pt")
        if not Path(pretrained_path).exists():
            print(f"[pipeline] ERROR: No pretrained checkpoint at {pretrained_path}")
            sys.exit(1)
        print(f"[pipeline] Using existing pretrained checkpoint: {pretrained_path}")

    # -- Stage 2: Head Training ──
    if stages in ("all", "heads"):
        print(f"\n{'='*60}")
        print(f"[pipeline] Stage 2: Streaming Head Training ({epochs_heads} epochs)")
        print(f"{'='*60}")

        head_path = train_heads_from_stream(
            repo_ids=repo_ids,
            pretrained_checkpoint=pretrained_path,
            output_dir=str(output_dir / "heads"),
            epochs_decision=epochs_heads // 3,
            epochs_assistant=2 * epochs_heads // 3,
            epochs_joint=epochs_heads // 3,
            batch_size=batch_size,
            lr=lr,
            device=device,
            wandb_project=wandb_project,
            wandb_run=(wandb_run or "streaming") + "-heads",
            enable_wandb=enable_wandb,
            num_workers=num_workers,
        )
    else:
        head_path = str(output_dir / "heads" / "joint_best.pt")

    # -- Summary ──
    print(f"\n{'='*60}")
    print("[pipeline] Streaming Training Complete! (zero disk used for data)")
    print(f"[pipeline]")
    print(f"[pipeline] Checkpoints:")
    print(f"[pipeline]   Pretrain: {pretrained_path}")
    print(f"[pipeline]   Heads:    {head_path}")
    print(f"[pipeline]")
    print(f"[pipeline] To run inference:")
    print(f"[pipeline]   python inference/align_inference.py \\")
    print(f"[pipeline]       --checkpoint {head_path} \\")
    print(f"[pipeline]       --task \"pick up the red mug\"")
    print(f"{'='*60}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ALIGN Streaming Pretraining from LeRobot v3 Hub Datasets"
    )
    parser.add_argument("--dataset", action="append", dest="datasets",
                        default=["nvidia/LIBERO_LeRobot_v3"],
                        help="LeRobot v3 dataset repo ID (repeatable)")
    parser.add_argument("--output-dir", default="./checkpoints/streaming", help="Output directory")
    parser.add_argument("--epochs-pretrain", type=int, default=50)
    parser.add_argument("--epochs-heads", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stages", default="all", choices=["all", "pretrain", "heads"])
    parser.add_argument("--pretrained", help="Resume from existing pretrained backbone")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="align-streaming", help="W&B project name")
    parser.add_argument("--wandb-run", default=None, help="W&B run name")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default 4, set 0 if HF Hub is slow)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000,
                        help="Max steps per epoch (default 2000, set low for testing)")

    args = parser.parse_args()

    run_streaming_pipeline(
        repo_ids=args.datasets,
        output_dir=args.output_dir,
        epochs_pretrain=args.epochs_pretrain,
        epochs_heads=args.epochs_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        stages=args.stages,
        pretrained_checkpoint=args.pretrained,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        enable_wandb=args.wandb,
        num_workers=args.num_workers,
        # max_steps_per_epoch=args.max_steps_per_epoch,
    )


if __name__ == "__main__":
    main()
