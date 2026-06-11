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
from data.open_dataset import LeRobotAdapter, _lerobot_align_collate
from training.train_full_pipeline import SyntheticNoiseInjector


# ================================================================
# Streaming multi-dataset wrapper
# ================================================================

class MultiDatasetStream(IterableDataset):
    """IterableDataset that round-robins between multiple LeRobot v3 streams.

    Each worker gets a different offset so different GPUs see different data.
    Exhausted loaders are recreated for infinite streaming.
    """

    def __init__(
        self,
        repo_ids: list[str],
        frames_per_item: int = 8,
        data_dir: Optional[str] = None,
        traj_window: int = 20,
        fps: int = 20,
        chunk_size: int = 5,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.frames_per_item = frames_per_item
        self.data_dir = data_dir
        self.traj_window = traj_window
        self.fps = fps
        self.chunk_size = chunk_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        # Compute local data path for each repo
        repo_data_dirs = []
        for repo_id in self.repo_ids:
            if self.data_dir:
                candidate = Path(self.data_dir) / repo_id
                if candidate.exists():
                    repo_data_dirs.append(str(candidate))
                else:
                    if Path(self.data_dir).exists():
                        repo_data_dirs.append(self.data_dir)
                    else:
                        repo_data_dirs.append(None)
            else:
                repo_data_dirs.append(None)

        # Create loaders for each dataset
        adapters: list[LeRobotAdapter] = []
        loaders: list = []
        fps = self.fps
        dt_seconds = [(i - (self.traj_window - 1)) / fps for i in range(self.traj_window)]
        chunk_size = self.chunk_size
        dt_future = [i / fps for i in range(1, chunk_size + 1)]
        delta_timestamps = {
            "observation.state": dt_seconds,
            "observation.images.wrist_image": dt_seconds,
            "action": dt_future,
        }
        for i, repo_id in enumerate(self.repo_ids):
            adapter = LeRobotAdapter(
                repo_id,
                data_dir=repo_data_dirs[i],
                batch_size=1,
                num_workers=0,
                delta_timestamps=delta_timestamps,
            )
            ds = adapter.get_streaming_dataset()
            adapters.append(adapter)
            loaders.append(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_lerobot_align_collate)))

        exhausted: list[bool] = [False] * len(loaders)

        while True:
            for i, loader in enumerate(loaders):
                try:
                    batch = next(loader)
                    actions = batch.get("actions", None)
                    if isinstance(actions, list):
                        actions = actions[0] if actions else None
                    yield {
                        "frames": batch["frames"][0],
                        "poses": batch["poses"][0],
                        "text": batch["texts"][0],
                        "actions": actions,
                    }
                except StopIteration:
                    exhausted[i] = True
            for i, is_exhausted in enumerate(exhausted):
                if is_exhausted:
                    ds = adapters[i].get_streaming_dataset()
                    loaders[i] = iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_lerobot_align_collate))
                    exhausted[i] = False


def streaming_pretrain_collate(batch: list[dict], traj_window: int = 10) -> dict:
    """Collate streaming samples into ALIGN pretraining batch.

    If LeRobot returned a real trajectory window (shape (K, D) on 'poses'),
    use it directly. Otherwise (e.g. adapter didn't request delta_timestamps),
    fall back to repeating the single pose to fill the window.

    For real trajectory windows, delta_timestamps in LeRobotAdapter provides
    temporal context from the Parquet shards — e.g. [-0.3, -0.2, -0.1, 0.0]
    gives a 4-frame window @ 20fps = 200ms of motion history.
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
        # Convert to uint8 [0, 255]. LeRobot returns float32 [0, 1] — .to(uint8)
        # would TRUNCATE to 0 or 1 (all-black). Multiply first when float.
        if frame.dtype == torch.float32:
            if frame.max() <= 1.0:
                frame = (frame * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                frame = frame.clamp(0, 255).to(torch.uint8)
        elif frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)

        all_frames.append(frame)

        # Build trajectory window from pose.
        # If LeRobot returned a window (K, D), use it. Else repeat single pose.
        if pose.dim() == 2 and pose.shape[0] > 1:
            # Real trajectory window from delta_timestamps
            # Pose is (K, D_state), take first 6 dims (EEF pos + axis_angle)
            pose_eef = pose[..., :6] if pose.shape[-1] >= 6 else torch.cat(
                [pose, torch.zeros(pose.shape[0], 6 - pose.shape[-1], device=pose.device)],
                dim=-1,
            )
            all_trajs.append(pose_eef)
        else:
            # Single pose, repeat to window size
            pose_eef = pose[..., :6] if pose.shape[-1] >= 6 else pose
            if pose_eef.dim() == 0:
                pose_eef = pose_eef.unsqueeze(0)
            all_trajs.append(pose_eef.unsqueeze(0).repeat(traj_window, 1))

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
    epochs_encoder: int = 40,
    epochs_mixer: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    embed_dim: int = 256,
    temperature: float = 0.07,
    max_grad_norm: float = 1.0,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    checkpoint_every: int = 10,
    data_dir: Optional[str] = None,
    wandb_project: str = "align-streaming",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 4,
    traj_window: int = 20,
    fps: int = 20,
    chunk_size: int = 5,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    encoder_checkpoint: Optional[str] = None,
):
    """Contrastive pretraining with two sub-phases.

    Phase 1a (encoder pretrain): InfoNCE on raw encoder outputs.
        Mixer is frozen (identity init ≈ pass-through).
        Trains: vision_proj, traj_encoder, text_proj.

    Phase 1b (mixer warm-up): InfoNCE on mixer outputs.
        Mixer unfrozen, learns cross-modal features on converged encoders.
        Trains: encoders + mixer.

    Args:
        repo_ids: LeRobot v3 dataset IDs.
        output_dir: Checkpoint directory.
        epochs_encoder: Phase 1a epochs (default 40).
        epochs_mixer: Phase 1b epochs (default 10).
        encoder_checkpoint: Resume Phase 1b from existing encoder checkpoint.
            If None, runs Phase 1a from scratch.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ALIGN Contrastive Pretraining ===")
    print(f"  Datasets: {repo_ids}")
    if data_dir:
        print(f"  Data dir: {data_dir}")
    print(f"  Device:   {device}")
    print(f"  Phase 1a (encoder): {epochs_encoder} epochs")
    print(f"  Phase 1b (mixer):   {epochs_mixer} epochs")
    print(f"  Output:   {output_dir}")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-pretrain",
            "datasets": repo_ids,
            "data_dir": data_dir,
            "epochs_encoder": epochs_encoder,
            "epochs_mixer": epochs_mixer,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "embed_dim": embed_dim,
            "temperature": temperature,
            "max_grad_norm": max_grad_norm,
            "max_steps_per_epoch": max_steps_per_epoch,
            "device": str(device),
            "traj_window": traj_window,
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:      {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Streaming dataset ──
    stream_ds = MultiDatasetStream(repo_ids, data_dir=data_dir, traj_window=traj_window, fps=fps, chunk_size=chunk_size)
    loader = DataLoader(
        stream_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: streaming_pretrain_collate(b, traj_window=traj_window),
        pin_memory=True,
    )

    # -- Model ──
    model = ALIGNModel(
        embed_dim=embed_dim,
        use_text=True,
        device=str(device),
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
    ).to(device)
    model.freeze_backbone()

    # -- Loss ──
    criterion = ContrastiveLoss3Way(temperature=temperature)

    # -- Config for checkpoint ──
    config = {
        "embed_dim": embed_dim,
        "traj_window": traj_window,
        "fps": fps,
        "chunk_size": chunk_size,
        "mixer_dim": mixer_dim,
        "num_mixer_blocks": num_mixer_blocks,
        "temperature": temperature,
    }

    log_path = output_dir / "pretrain_log.jsonl"
    log_fp = open(log_path, "a")

    # ================================================================
    # Phase 1a: Encoder Pretrain (mixer frozen, InfoNCE on raw outputs)
    # ================================================================
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        print(f"\n  Resuming from encoder checkpoint: {encoder_checkpoint}")
        ckpt = torch.load(encoder_checkpoint, map_location=device)
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = float("inf")

    if epochs_encoder > 0:
        # Freeze mixer — InfoNCE sees raw encoder outputs
        model.freeze_mixer()
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"\n  Phase 1a — Trainable: {n_params:,}")
        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        for epoch in range(start_epoch, epochs_encoder):
            model.train()
            epoch_losses = []
            epoch_cos_vt = []
            epoch_cos_vl = []
            epoch_cos_tl = []
            import time
            _epoch_start = time.time()
            _step_start = time.time()

            loader_iter = iter(loader)
            for step in range(max_steps_per_epoch):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                frames = batch["frames"].to(device)
                trajs = batch["trajectories"].float().to(device)
                texts = batch["texts"]

                if trajs.shape[-1] < 6:
                    pad = torch.zeros(*trajs.shape[:-1], 6 - trajs.shape[-1], device=trajs.device)
                    trajs = torch.cat([trajs, pad], dim=-1)

                # Raw encoder outputs (no mixer)
                raw = model.encode_raw_all(frames, trajs, texts)
                stats = criterion(raw["z_v"], raw["z_t"], raw["z_text"])
                loss = stats["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_cos_vt.append(stats["avg_cos_vt"].item())
                epoch_cos_vl.append(stats["avg_cos_vl"].item())
                epoch_cos_tl.append(stats["avg_cos_tl"].item())

                if (step + 1) % 100 == 0:
                    _now = time.time()
                    _step_time = (_now - _step_start) / 100
                    _remaining = _step_time * (max_steps_per_epoch - step)
                    print(f"  [1a] Epoch {epoch + 1}, step {step + 1}/{max_steps_per_epoch}  "
                          f"loss: {loss.item():.4f}  vt: {stats['avg_cos_vt'].item():.3f}  "
                          f"vl: {stats['avg_cos_vl'].item():.3f}  tl: {stats['avg_cos_tl'].item():.3f}  "
                          f"{_step_time*1000:.0f}ms/step  ETA:{_remaining/60:.1f}min",
                          flush=True)
                    _step_start = _now

            avg_loss = float(np.mean(epoch_losses))
            avg_vt = float(np.mean(epoch_cos_vt))
            avg_vl = float(np.mean(epoch_cos_vl))
            avg_tl = float(np.mean(epoch_cos_tl))

            print(f"  [1a] Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
                  f"cos_vt: {avg_vt:.3f}  cos_vl: {avg_vl:.3f}  cos_tl: {avg_tl:.3f}")

            wandb_trainer.log({
                "phase": "1a_encoder",
                "epoch": epoch + 1,
                "loss": avg_loss,
                "cos_vt": avg_vt,
                "cos_vl": avg_vl,
                "cos_tl": avg_tl,
            }, step=epoch + 1)

            log_fp.write(json.dumps({
                "phase": "1a", "epoch": epoch + 1, "loss": avg_loss,
                "cos_vt": avg_vt, "cos_vl": avg_vl, "cos_tl": avg_tl,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")
            log_fp.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrain_checkpoint(
                    str(output_dir / "encoder_best.pt"), epoch, avg_loss,
                    "encoder", optimizer.state_dict(), config)
                print(f"  -> encoder_best.pt (loss: {avg_loss:.4f})")

            if (epoch + 1) % checkpoint_every == 0:
                model.save_pretrain_checkpoint(
                    str(output_dir / f"encoder_epoch_{epoch + 1:04d}.pt"),
                    epoch, avg_loss, "encoder", optimizer.state_dict(), config)

        print(f"  Phase 1a complete. Best loss: {best_loss:.4f}")

    # ================================================================
    # Phase 1b: Mixer Warm-Up (InfoNCE on mixer outputs)
    # ================================================================
    if epochs_mixer > 0:
        # Load best encoder checkpoint if available
        encoder_ckpt_path = str(output_dir / "encoder_best.pt")
        if Path(encoder_ckpt_path).exists():
            ckpt = torch.load(encoder_ckpt_path, map_location=device)
            model.load_trainable_state_dict(ckpt["trainable_state_dict"])
            print(f"\n  Loaded encoder checkpoint for Phase 1b")

        # Unfreeze mixer — InfoNCE now flows through mixer
        model.unfreeze_mixer()
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"\n  Phase 1b — Trainable: {n_params:,} (encoders + mixer)")
        optimizer = optim.AdamW(trainable, lr=lr * 0.5, weight_decay=weight_decay)
        best_loss = float("inf")

        for epoch in range(epochs_mixer):
            model.train()
            epoch_losses = []
            epoch_cos_vt = []
            epoch_cos_vl = []
            epoch_cos_tl = []
            import time
            _step_start = time.time()

            loader_iter = iter(loader)
            for step in range(max_steps_per_epoch):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                frames = batch["frames"].to(device)
                trajs = batch["trajectories"].float().to(device)
                texts = batch["texts"]

                if trajs.shape[-1] < 6:
                    pad = torch.zeros(*trajs.shape[:-1], 6 - trajs.shape[-1], device=trajs.device)
                    trajs = torch.cat([trajs, pad], dim=-1)

                # Mixer outputs
                mixed = model.encode_mixed(frames, trajs, texts)
                stats = criterion(mixed["z_v"], mixed["z_t"], mixed["z_text"])
                loss = stats["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_cos_vt.append(stats["avg_cos_vt"].item())
                epoch_cos_vl.append(stats["avg_cos_vl"].item())
                epoch_cos_tl.append(stats["avg_cos_tl"].item())

                if (step + 1) % 100 == 0:
                    _now = time.time()
                    _step_time = (_now - _step_start) / 100
                    _remaining = _step_time * (max_steps_per_epoch - step)
                    print(f"  [1b] Epoch {epoch + 1}, step {step + 1}/{max_steps_per_epoch}  "
                          f"loss: {loss.item():.4f}  vt: {stats['avg_cos_vt'].item():.3f}  "
                          f"vl: {stats['avg_cos_vl'].item():.3f}  tl: {stats['avg_cos_tl'].item():.3f}  "
                          f"{_step_time*1000:.0f}ms/step  ETA:{_remaining/60:.1f}min",
                          flush=True)
                    _step_start = _now

            avg_loss = float(np.mean(epoch_losses))
            avg_vt = float(np.mean(epoch_cos_vt))
            avg_vl = float(np.mean(epoch_cos_vl))
            avg_tl = float(np.mean(epoch_cos_tl))

            print(f"  [1b] Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
                  f"cos_vt: {avg_vt:.3f}  cos_vl: {avg_vl:.3f}  cos_tl: {avg_tl:.3f}")

            wandb_trainer.log({
                "phase": "1b_mixer",
                "epoch": epoch + 1,
                "loss": avg_loss,
                "cos_vt": avg_vt,
                "cos_vl": avg_vl,
                "cos_tl": avg_tl,
            }, step=epochs_encoder + epoch + 1)

            log_fp.write(json.dumps({
                "phase": "1b", "epoch": epoch + 1, "loss": avg_loss,
                "cos_vt": avg_vt, "cos_vl": avg_vl, "cos_tl": avg_tl,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")
            log_fp.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrain_checkpoint(
                    str(output_dir / "best.pt"), epoch, avg_loss,
                    "full", optimizer.state_dict(), config)
                print(f"  -> best.pt (loss: {avg_loss:.4f})")

            if (epoch + 1) % checkpoint_every == 0:
                model.save_pretrain_checkpoint(
                    str(output_dir / f"epoch_{epoch + 1:04d}.pt"),
                    epoch, avg_loss, "full", optimizer.state_dict(), config)

        print(f"  Phase 1b complete. Best loss: {best_loss:.4f}")

    log_fp.close()
    print(f"\n  Pretraining complete. Logs: {log_path}")
    wandb_trainer.finish()
    return str(output_dir / "best.pt")


# ================================================================
# Streaming head training with on-the-fly noise
# ================================================================

def train_heads_from_stream(
    repo_ids: list[str],
    pretrained_checkpoint: str,
    output_dir: str,
    epochs_heads: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    chunk_size: int = 5,
    max_steps_per_epoch: int = 2000,
    device: Optional[str] = None,
    noise_configs: Optional[list[dict]] = None,
    data_dir: Optional[str] = None,
    wandb_project: str = "align-streaming",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 4,
    traj_window: int = 20,
    fps: int = 20,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    temperature: float = 0.07,
):
    """Train Decision + Assistant heads from streamed data with on-the-fly noise.

    Single joint loss: BCE(α_pred, α_target) + 0.5 × MSE(Δpred, Δtarget).
    All encoders + mixer are frozen — only heads train.

    Args:
        repo_ids: LeRobot v3 dataset IDs.
        pretrained_checkpoint: Path to frozen pretrained backbone (Phase 1b output).
        output_dir: Checkpoint directory.
        epochs_heads: Total epochs for joint head training (default 30).
        noise_configs: Noise configs per batch (default: medium noise).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if noise_configs is None:
        noise_configs = [{"pos_noise_std": 0.020, "orn_noise_std": 4.0, "label": "medium"}]

    print(f"=== ALIGN Head Training ===")
    print(f"  Datasets:    {repo_ids}")
    print(f"  Pretrained:  {pretrained_checkpoint}")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {epochs_heads} (joint BCE + MSE)")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-heads",
            "datasets": repo_ids,
            "pretrained_checkpoint": pretrained_checkpoint,
            "epochs_heads": epochs_heads,
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
    stream_ds = MultiDatasetStream(repo_ids, data_dir=data_dir, traj_window=traj_window, fps=fps, chunk_size=chunk_size)
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
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
    ).to(device)

    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    model.freeze_backbone()
    model.freeze_all_encoders()  # ← explicit: no gradient leaks to encoders or mixer
    print(f"  Loaded backbone from {pretrained_checkpoint}")
    print(f"  All encoders + mixer frozen — only heads train")

    # -- Noise injectors (one per config) ──
    injectors = [
        SyntheticNoiseInjector(
            pos_noise_std=cfg["pos_noise_std"],
            orn_noise_std=cfg["orn_noise_std"],
            seed=42 + i,
        )
        for i, cfg in enumerate(noise_configs)
    ]

    log_path = output_dir / "head_log.jsonl"
    log_fp = open(log_path, "a")

    # -- Single joint head optimizer ──
    optimizer = optim.AdamW(
        model.get_head_params(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_loss = float("inf")

    for epoch in range(epochs_heads):
        model.train()
        losses = []
        alphas = []
        deltas = []
        import time
        _step_start = time.time()

        loader_iter = iter(loader)
        for step in range(max_steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            frames = batch["frames"].to(device)
            clean_poses = batch["poses"].float().to(device)
            texts = batch["texts"]
            actions_window = batch.get("actions", None)

            # Rotate noise injectors
            injector = injectors[step % len(injectors)]
            noisy_poses_np = injector.inject(clean_poses.cpu().numpy())
            noisy_poses = torch.from_numpy(noisy_poses_np).float().to(device)

            # Compute targets on-the-fly (position + orientation error)
            d_max_pos = 0.10
            d_max_orn = 0.52  # ~30° axis-angle
            pos_error = torch.norm(noisy_poses[:, :3] - clean_poses[:, :3], dim=1)
            orn_error = torch.norm(noisy_poses[:, 3:6] - clean_poses[:, 3:6], dim=1)
            alpha_target = torch.clamp(
                torch.maximum(pos_error / d_max_pos, orn_error / d_max_orn),
                0.0, 1.0,
            )

            B, D = clean_poses.shape

            # Build delta_target from future action window
            if actions_window is not None and isinstance(actions_window, torch.Tensor) and actions_window.dim() == 3:
                delta_target = actions_window[:, :chunk_size, :6].to(device).float()
            else:
                # Fallback: single-step delta
                delta_target = torch.zeros(B, chunk_size, 6, device=device)
                clean_6d = clean_poses[:, :6] if D >= 6 else torch.cat([
                    clean_poses, torch.zeros(B, 6 - D, device=device)
                ], dim=-1)
                for i in range(1, chunk_size + 1):
                    delta_target[:, i - 1, :] = clean_6d - noisy_poses[:, :6]

            # Encode through frozen encoders + mixer
            with torch.no_grad():
                mixed = model.encode_mixed(frames, noisy_poses.unsqueeze(1).repeat(1, traj_window, 1), texts)
                z_v = mixed["z_v"]
                z_t = mixed["z_t"]
                z_text = mixed["z_text"]

            # Joint loss: BCE(α) + 0.5 × MSE(Δ)
            alpha_pred = model.decision_head(z_v, z_t, z_text)
            delta_pred = model.assistant_head(z_v, z_t, z_text, noisy_poses[:, :6])

            loss = (F.binary_cross_entropy(alpha_pred.squeeze(-1), alpha_target) +
                    0.5 * F.mse_loss(delta_pred, delta_target))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_head_params(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            alphas.append(alpha_pred.detach().mean().item())
            deltas.append(delta_pred.detach().abs().mean().item())

            if (step + 1) % 100 == 0:
                _now = time.time()
                _step_time = (_now - _step_start) / 100
                _remaining = _step_time * (max_steps_per_epoch - step)
                print(f"  [2] Epoch {epoch + 1}, step {step + 1}/{max_steps_per_epoch} "
                      f"loss: {loss.item():.4f}  α: {alpha_pred.mean().item():.3f}  "
                      f"Δ: {delta_pred.abs().mean().item():.4f}  "
                      f"{_step_time*1000:.0f}ms/step  ETA:{_remaining/60:.1f}min", flush=True)
                _step_start = _now

        avg_loss = float(np.mean(losses))
        avg_alpha = float(np.mean(alphas))
        avg_delta = float(np.mean(deltas))

        print(f"  Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
              f"α: {avg_alpha:.3f}  Δ: {avg_delta:.4f}")

        # W&B logging
        wandb_trainer.log({
            "head/loss": avg_loss,
            "head/alpha_mean": avg_alpha,
            "head/delta_mean": avg_delta,
            "head/epoch": epoch + 1,
        }, step=epoch + 1)

        # Log
        log_fp.write(json.dumps({
            "stage": "joint", "epoch": epoch + 1,
            "loss": avg_loss, "alpha_mean": avg_alpha, "delta_mean": avg_delta,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_heads_checkpoint(
                str(output_dir / "best.pt"), epoch, avg_loss,
                optimizer.state_dict(), {"chunk_size": chunk_size})
            wandb_trainer.save(str(output_dir / "best.pt"))
            print(f"  -> best.pt (loss: {avg_loss:.4f})")

    log_fp.close()
    print(f"\n  Head training complete. Best loss: {best_loss:.4f}")
    print(f"  Logs: {log_path}")
    wandb_trainer.finish()
    return str(output_dir / "best.pt")


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
        # Convert to uint8 [0, 255] without truncating float [0,1] to 0/1
        if frame.dtype == torch.float32:
            if frame.max() <= 1.0:
                frame = (frame * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                frame = frame.clamp(0, 255).to(torch.uint8)
        elif frame.dtype != torch.uint8:
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
    epochs_pretrain_encoder: int = 40,
    epochs_pretrain_mixer: int = 10,
    epochs_heads: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: Optional[str] = None,
    stages: str = "all",
    pretrained_checkpoint: Optional[str] = None,
    encoder_checkpoint: Optional[str] = None,
    data_dir: Optional[str] = None,
    wandb_project: str = "align-streaming",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 4,
    max_steps_per_epoch: int = 2000,
    traj_window: int = 20,
    fps: int = 20,
    chunk_size: int = 5,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    temperature: float = 0.07,
):
    """Full ALIGN training pipeline using streaming or local data.

    Three sub-phases:
      1a. Encoder pretrain (mixer frozen, InfoNCE on raw outputs)
      1b. Mixer warm-up (InfoNCE on mixer outputs)
      2.  Head training (BCE + MSE, all encoders frozen)

    Args:
        repo_ids: LeRobot v3 dataset IDs.
        output_dir: Base output directory.
        epochs_pretrain_encoder: Phase 1a epochs (default 40).
        epochs_pretrain_mixer: Phase 1b epochs (default 10).
        epochs_heads: Phase 2 epochs (default 30).
        stages: 'all', 'pretrain', 'heads', 'encoder'.
        pretrained_checkpoint: Skip Phase 1, use existing pretrain checkpoint.
        encoder_checkpoint: Resume Phase 1b from existing encoder checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = pretrained_checkpoint

    # -- Phase 1a + 1b: Contrastive Pretraining ──
    if stages in ("all", "pretrain", "encoder"):
        print(f"\n{'='*60}")
        print(f"[pipeline] Phase 1: Contrastive Pretraining")
        print(f"  Phase 1a (encoder): {epochs_pretrain_encoder} epochs")
        print(f"  Phase 1b (mixer):   {epochs_pretrain_mixer} epochs")
        print(f"{'='*60}")

        pretrained_path = pretrain_from_stream(
            repo_ids=repo_ids,
            output_dir=str(output_dir / "pretrain"),
            epochs_encoder=epochs_pretrain_encoder,
            epochs_mixer=epochs_pretrain_mixer,
            batch_size=batch_size,
            lr=lr,
            device=device,
            data_dir=data_dir,
            wandb_project=wandb_project,
            wandb_run=(wandb_run or "streaming") + "-pretrain",
            enable_wandb=enable_wandb,
            num_workers=num_workers,
            max_steps_per_epoch=max_steps_per_epoch,
            traj_window=traj_window,
            fps=fps,
            chunk_size=chunk_size,
            mixer_dim=mixer_dim,
            num_mixer_blocks=num_mixer_blocks,
            encoder_checkpoint=encoder_checkpoint,
        )
    else:
        if pretrained_path is None:
            pretrained_path = str(output_dir / "pretrain" / "best.pt")
        if not Path(pretrained_path).exists():
            print(f"[pipeline] ERROR: No pretrained checkpoint at {pretrained_path}")
            sys.exit(1)
        print(f"[pipeline] Using existing pretrained checkpoint: {pretrained_path}")

    # -- Phase 2: Head Training ──
    if stages in ("all", "heads"):
        print(f"\n{'='*60}")
        print(f"[pipeline] Phase 2: Head Training ({epochs_heads} epochs)")
        print(f"{'='*60}")

        head_path = train_heads_from_stream(
            repo_ids=repo_ids,
            pretrained_checkpoint=pretrained_path,
            output_dir=str(output_dir / "heads"),
            epochs_heads=epochs_heads,
            batch_size=batch_size,
            lr=lr,
            device=device,
            data_dir=data_dir,
            wandb_project=wandb_project,
            wandb_run=(wandb_run or "streaming") + "-heads",
            enable_wandb=enable_wandb,
            num_workers=num_workers,
            max_steps_per_epoch=max_steps_per_epoch,
            traj_window=traj_window,
            fps=fps,
            chunk_size=chunk_size,
            mixer_dim=mixer_dim,
            num_mixer_blocks=num_mixer_blocks,
            temperature=temperature,
        )
    else:
        head_path = str(output_dir / "heads" / "best.pt")

    # -- Summary ──
    print(f"\n{'='*60}")
    print("[pipeline] Training Complete!")
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
        description="ALIGN Training Pipeline — 3-phase: encoder pretrain → mixer warm-up → head training"
    )
    parser.add_argument("--dataset", action="append", dest="datasets",
                        default=["nvidia/LIBERO_LeRobot_v3"],
                        help="LeRobot v3 dataset repo ID (repeatable)")
    parser.add_argument("--output-dir", default="./checkpoints/streaming", help="Output directory")
    parser.add_argument("--epochs-pretrain-encoder", type=int, default=40,
                        help="Phase 1a: encoder pretrain epochs (default 40)")
    parser.add_argument("--epochs-pretrain-mixer", type=int, default=10,
                        help="Phase 1b: mixer warm-up epochs (default 10)")
    parser.add_argument("--epochs-heads", type=int, default=30,
                        help="Phase 2: joint head training epochs (default 30)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stages", default="all",
                        choices=["all", "pretrain", "heads", "encoder"],
                        help="Which phases to run: 'all', 'pretrain' (1a+1b), 'heads' (2), 'encoder' (1a only)")
    parser.add_argument("--pretrained", help="Skip Phase 1, use existing pretrain checkpoint")
    parser.add_argument("--encoder-checkpoint",
                        help="Resume Phase 1b from existing encoder checkpoint (skips Phase 1a)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="align-streaming", help="W&B project name")
    parser.add_argument("--wandb-run", default=None, help="W&B run name")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default 4, set 0 if HF Hub is slow)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000,
                        help="Max steps per epoch (default 2000, set low for testing)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to local data directory (pre-downloaded). Overrides Hub streaming.")
    parser.add_argument("--traj-window", type=int, default=20,
                        help="Trajectory encoder window size K_traj")
    parser.add_argument("--fps", type=int, default=20,
                        help="Data FPS (LIBERO=20, default real teleop=30)")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="Assistant head chunk size K (default 5)")
    parser.add_argument("--mixer-dim", type=int, default=512,
                        help="Cross-attention mixer hidden dim (default 512)")
    parser.add_argument("--num-mixer-blocks", type=int, default=2,
                        help="Number of cross-attention mixer blocks (default 2)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature (default 0.07)")

    args = parser.parse_args()

    run_streaming_pipeline(
        repo_ids=args.datasets,
        output_dir=args.output_dir,
        epochs_pretrain_encoder=args.epochs_pretrain_encoder,
        epochs_pretrain_mixer=args.epochs_pretrain_mixer,
        epochs_heads=args.epochs_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        stages=args.stages,
        pretrained_checkpoint=args.pretrained,
        encoder_checkpoint=args.encoder_checkpoint,
        data_dir=args.data_dir,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        enable_wandb=args.wandb,
        num_workers=args.num_workers,
        max_steps_per_epoch=args.max_steps_per_epoch,
        traj_window=args.traj_window,
        fps=args.fps,
        chunk_size=args.chunk_size,
        mixer_dim=args.mixer_dim,
        num_mixer_blocks=args.num_mixer_blocks,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
