#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN World Model training — action-conditioned transition model.

Trains the WorldModel (s, a) -> s' on top of a FROZEN pretrained
encoder + mixer. Used for counterfactual imagination in the alpha
pipeline.

Usage:
    python training/train_world_model.py \\
        --data /path/to/libero.h5 \\
        --pretrained checkpoints/pretrain/libero_90/run_7/best.pt \\
        --output-dir ./checkpoints/world_model \\
        --epochs 20 --batch-size 64 --lr 1e-3

The world model is a SEPARATE component from the existing FuturePredictionHead:
  - FuturePredictionHead: predicts K parallel future embeddings (no action)
  - WorldModel: predicts 1 next embedding from current state + action
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.world_model import create_world_model, world_model_loss
from training.wandb_utils import init_wandb

# Try to import world_model_collate (assumed to be added to data/align_dataset.py).
# If it isn't available yet, we provide a small fallback below so the script can
# still smoke-test on this machine while the collate function is being added.
try:
    from data.align_dataset import world_model_collate  # noqa: F401
    _HAS_WM_COLLATE = True
except ImportError:
    _HAS_WM_COLLATE = False

from data.align_dataset import ALIGNDataset, MultiALIGNDataset


# ================================================================
# Fallback collate (used only if world_model_collate is not yet
# available in data/align_dataset.py). Mirrors the head_collate
# pattern but produces (state_t, action_t, state_t+1) triples.
# ================================================================

def _fallback_world_model_collate(batch: list, chunk_size: int = 1) -> dict:
    """Minimal collate that yields state/action/next_state triples.

    For each item in `batch`, picks a random anchor timestep t within
    the available frames. Returns:
      - frames_t:        (B, H, W, 3)         vision at time t
      - trajectory_t:    (B, K, 6)             traj window ending at t
      - frames_next:     (B, H, W, 3)         vision at time t+1
      - trajectory_next: (B, K, 6)             traj window ending at t+1
      - action:          (B, 6)               the OSC_POSE delta at t
      - texts:           list[str]

    Pads frames at the episode boundary by replicating the last frame.
    """
    import numpy as _np

    all_frames_t, all_traj_t = [], []
    all_frames_next, all_traj_next = [], []
    all_actions, all_texts = [], []
    rng = _np.random.default_rng()

    for item in batch:
        frames = item["frames"]
        poses = item["poses"][..., :6]   # (N, 6)
        actions = item.get("actions", None)
        text = item["text"]

        N = len(frames)
        # Anchor t such that both t and t+1 exist
        max_t = max(0, N - 2)
        t = int(rng.integers(0, max_t + 1)) if max_t > 0 else 0

        # Frame window ending at t (length chunk_size)
        start_f = max(0, t - chunk_size + 1)
        frame_window = frames[start_f:t + 1]
        if len(frame_window) < chunk_size:
            pad = _np.zeros((chunk_size - len(frame_window), *frames.shape[1:]), dtype=frames.dtype)
            frame_window = _np.concatenate([pad, frame_window], axis=0)
        all_frames_t.append(frame_window)

        # Trajectory window ending at t (length chunk_size)
        start_t = max(0, t - chunk_size + 1)
        traj_t = poses[start_t:t + 1]
        if len(traj_t) < chunk_size:
            pad = _np.zeros((chunk_size - len(traj_t), 6), dtype=_np.float32)
            traj_t = _np.concatenate([pad, traj_t], axis=0)
        all_traj_t.append(traj_t.astype(_np.float32))

        all_frames_next.append(frames[t + 1])
        start_n = max(0, t + 1 - chunk_size + 1)
        traj_next = poses[start_n:t + 2]
        if len(traj_next) < chunk_size:
            pad = _np.zeros((chunk_size - len(traj_next), 6), dtype=_np.float32)
            traj_next = _np.concatenate([pad, traj_next], axis=0)
        all_traj_next.append(traj_next.astype(_np.float32))

        if actions is not None and t < len(actions):
            act = actions[t, :6].astype(_np.float32)
        else:
            act = _np.zeros(6, dtype=_np.float32)
        all_actions.append(act)
        all_texts.append(text)

    return {
        "frames_t": _np.stack(all_frames_t, axis=0),       # (B, K, H, W, 3)
        "trajectory_t": _np.stack(all_traj_t, axis=0),
        "frames_next": _np.stack(all_frames_next, axis=0),
        "trajectory_next": _np.stack(all_traj_next, axis=0),
        "action": _np.stack(all_actions, axis=0),
        "texts": all_texts,
    }


# ================================================================
# Training
# ================================================================

def train_world_model(
    data_paths: List[str],
    pretrained_checkpoint: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    wandb_project: str = "align-world-model",
    cameras: Optional[List[str]] = None,
    wandb_run: Optional[str] = None,
    enable_wandb: bool = False,
    num_workers: int = 0,
    traj_window: int = 20,
    chunk_size: int = 1,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    use_bf16: bool = True,
    resume: Optional[str] = None,
    # World model arch + kwargs
    arch: str = "mlp",
    action_dim: int = 6,
    embed_dim: int = 256,
    mlp_hidden: int = 512,
    mlp_layers: int = 3,
    window_size: int = 5,
    transformer_layers: int = 2,
    transformer_d_model: int = 384,
    transformer_nhead: int = 4,
    transformer_dropout: float = 0.0,
    transformer_dim_ff: int = 1024,
    seed: int = 42,
) -> str:
    """Train a WorldModel on top of a frozen pretrained encoder+mixer.

    Returns the path to the best world_model checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- Pick the right collate function ----------------------
    if _HAS_WM_COLLATE:
        from data.align_dataset import world_model_collate as wm_collate
        # world_model_collate takes traj_window (length of the K-step past window)
        collate_fn = lambda b: wm_collate(b, traj_window=traj_window)
        print(f"  Collate: data.align_dataset.world_model_collate(traj_window={traj_window})")
    else:
        print("  WARN: world_model_collate not in data/align_dataset.py — using fallback.")
        collate_fn = lambda b: _fallback_world_model_collate(b, chunk_size=chunk_size)

    # -- Output dir: output_dir/{dataset_stem}/run_N/ ---------
    if len(data_paths) == 1:
        ds_name = Path(data_paths[0]).stem
    else:
        ds_name = "+".join(Path(p).stem for p in data_paths)
    base_dir = Path(output_dir) / ds_name

    existing = sorted(base_dir.glob("run_*")) if base_dir.exists() else []
    max_run = 0
    for d in existing:
        try:
            n = int(d.name.split("_")[-1])
            if n > max_run:
                max_run = n
        except (ValueError, IndexError):
            pass
    next_run = max_run + 1

    out_dir = base_dir / f"run_{next_run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ALIGN World Model Training ===")
    print(f"  Run:        {out_dir}")
    print(f"  Data ({len(data_paths)}): {data_paths}")
    print(f"  Pretrained: {pretrained_checkpoint}")
    print(f"  Arch:       {arch}  (action_dim={action_dim}, embed_dim={embed_dim})")
    print(f"  Epochs:     {epochs}  lr={lr}  bs={batch_size}  wd={weight_decay}")
    print(f"  Device:     {device}")

    # -- W&B -------------------------------------------------
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run or out_dir.name,
        config={
            "model": "align-world-model",
            "output_dir": str(output_dir),
            "data": [str(p) for p in data_paths],
            "pretrained_checkpoint": pretrained_checkpoint,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_split": val_split,
            "max_steps_per_epoch": max_steps_per_epoch,
            "num_workers": num_workers,
            "arch": arch,
            "action_dim": action_dim,
            "embed_dim": embed_dim,
            "mixer_dim": mixer_dim,
            "num_mixer_blocks": num_mixer_blocks,
            "traj_window": traj_window,
            "chunk_size": chunk_size,
            "mlp_hidden": mlp_hidden,
            "mlp_layers": mlp_layers,
            "window_size": window_size,
            "transformer_layers": transformer_layers,
            "transformer_d_model": transformer_d_model,
            "transformer_nhead": transformer_nhead,
            "transformer_dropout": transformer_dropout,
            "transformer_dim_ff": transformer_dim_ff,
            "device": str(device),
            "use_bf16": use_bf16,
            "seed": seed,
            "cameras": cameras if cameras else ["wrist_image"],
        },
    )
    print(f"  W&B:        {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Dataset --------------------------------------------
    if len(data_paths) == 1:
        full_ds = ALIGNDataset(data_paths[0], mode="head", traj_window=traj_window,
                               cameras=cameras)
    else:
        full_ds = MultiALIGNDataset(
            data_paths, mode="head", traj_window=traj_window, cameras=cameras
        )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    print(f"  Samples:    {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # -- Frozen ALIGNModel (encoder + mixer only) ------------
    print(f"\n  Loading ALIGNModel from {pretrained_checkpoint} ...")
    # Determine num_cameras: must match pretrain's --cameras
    num_cameras = len(cameras) if cameras else 1
    align = ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
        num_cameras=num_cameras,
    ).to(device)

    # Load encoder + mixer weights from the pretrained checkpoint
    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    if "trainable_state_dict" in ckpt:
        align.load_trainable_state_dict(ckpt["trainable_state_dict"])
    else:
        align.load_state_dict(ckpt, strict=False)
    align.freeze_backbone()
    align.freeze_all_encoders()
    align.eval()  # always in eval mode — encoders are frozen
    print(f"  ALIGNModel loaded + frozen (epoch={ckpt.get('epoch', '?')}, "
          f"phase={ckpt.get('phase', '?')})")

    # -- World Model head ------------------------------------
    wm_kwargs: dict = {}
    if arch == "mlp":
        wm_kwargs = {"hidden_dim": mlp_hidden, "num_layers": mlp_layers, "window_size": window_size}
    elif arch == "rnn":
        wm_kwargs = {"hidden_dim": mlp_hidden, "num_rnn_layers": mlp_layers}
    elif arch == "transformer":
        wm_kwargs = {
            "d_model": transformer_d_model,
            "nhead": transformer_nhead,
            "num_layers": transformer_layers,
            "dim_feedforward": transformer_dim_ff,
            "dropout": transformer_dropout,
        }
    else:
        raise ValueError(f"Unknown arch: {arch} (expected 'mlp', 'rnn', or 'transformer')")

    world_model = create_world_model(
        arch=arch,
        embed_dim=embed_dim,
        action_dim=action_dim,
        **wm_kwargs,
    ).to(device)

    trainable = list(world_model.parameters())
    print(f"  WorldModel ({arch}): {sum(p.numel() for p in trainable):,} trainable params")

    # -- Resume from a previous world model checkpoint ----------
    # When --resume is set, load the world model weights and continue
    # training from the next epoch. The encoder+mixer stay frozen.
    resume_start_epoch = 0
    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        print(f"\n  Resuming from {resume_path} ...")
        wm_ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        if "world_model_state" in wm_ckpt:
            wm_sd = wm_ckpt["world_model_state"]
        elif "model_state_dict" in wm_ckpt:
            wm_sd = wm_ckpt["model_state_dict"]
        else:
            wm_sd = wm_ckpt  # raw state dict
        missing, unexpected = world_model.load_state_dict(wm_sd, strict=False)
        if unexpected:
            print(f"  Resume: ignored {len(unexpected)} unexpected keys: {unexpected[:3]}...")
        if missing:
            print(f"  Resume: {len(missing)} keys not loaded (using init): {missing[:3]}...")
        # Continue epoch numbering from the resumed checkpoint
        prev_epoch = wm_ckpt.get("epoch", 0)
        prev_loss = wm_ckpt.get("loss", float("inf"))
        resume_start_epoch = prev_epoch
        print(f"  Resumed at epoch {prev_epoch} (prev loss: {prev_loss:.4f})")

    opt = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    log_path = out_dir / "world_model_log.jsonl"
    log_fp = open(log_path, "w")

    # -- Checkpointing helper --------------------------------
    config_snapshot = {
        "arch": arch,
        "embed_dim": embed_dim,
        "action_dim": action_dim,
        "mlp_hidden": mlp_hidden,
        "mlp_layers": mlp_layers,
        "transformer_layers": transformer_layers,
        "transformer_d_model": transformer_d_model,
        "transformer_nhead": transformer_nhead,
        "transformer_dropout": transformer_dropout,
        "transformer_dim_ff": transformer_dim_ff,
        "chunk_size": chunk_size,
        "traj_window": traj_window,
        "mixer_dim": mixer_dim,
        "num_mixer_blocks": num_mixer_blocks,
        "pretrained_checkpoint": pretrained_checkpoint,
    }

    def save_checkpoint(path: Path, epoch: int, loss: float) -> None:
        torch.save({
            "world_model_state": world_model.state_dict(),
            "config": config_snapshot,
            "epoch": epoch,
            "loss": loss,
        }, str(path))

    best_loss = float("inf")
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(resume_start_epoch, epochs):
        world_model.train()
        epoch_losses: List[float] = []
        epoch_cos_v: List[float] = []
        epoch_cos_t: List[float] = []

        progress = train_loader
        if max_steps_per_epoch and max_steps_per_epoch < len(train_loader):
            progress = (
                p for p, _ in zip(train_loader, range(max_steps_per_epoch))
            )
        progress_bar = tqdm(
            progress,
            total=min(max_steps_per_epoch, len(train_loader))
                  if max_steps_per_epoch else len(train_loader),
            desc=f"[WM] Epoch {epoch+1}/{epochs}",
            unit="step",
        )

        for step, batch in enumerate(progress_bar):
            if step >= max_steps_per_epoch:
                break

            # -- To device --
            # The batch schema is the one produced by world_model_collate:
            #   frame_t, traj_t, action, frame_next, traj_next, text
            # The fallback collate above uses an alias
            # (frames_t, trajectory_t, ...). Normalize here.
            if "frame_t" in batch:
                frames_t = torch.from_numpy(batch["frame_t"]).to(device)
                traj_t = torch.from_numpy(batch["traj_t"]).float().to(device)
                frames_next = torch.from_numpy(batch["frame_next"]).to(device)
                traj_next = torch.from_numpy(batch["traj_next"]).float().to(device)
                action = torch.from_numpy(batch["action"]).float().to(device)
                texts = batch["text"]
            else:
                # Fallback collate schema (frames_t, trajectory_t, ...)
                frames_t = torch.from_numpy(batch["frames_t"]).to(device)
                traj_t = torch.from_numpy(batch["trajectory_t"]).float().to(device)
                frames_next = torch.from_numpy(batch["frames_next"]).to(device)
                traj_next = torch.from_numpy(batch["trajectory_next"]).float().to(device)
                action = torch.from_numpy(batch["action"]).float().to(device)
                texts = batch["texts"]

            # -- Encode state_t via frozen ALIGNModel --
            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                # frames_t is (B, K, H, W, 3) for single-cam or (B, K, V, H, W, 3) for multi-cam
                # Use encode_raw_vision_window which handles both cases
                z_v_window = align.encode_raw_vision_window(frames_t)  # (B, K, D)

                # Encode trajectory and text
                z_t_tokens = align.encode_raw_trajectory_tokens(traj_t)  # (B, K, D)
                z_text = align.encode_raw_text(texts)                    # (B, D)
                if z_text is None:
                    z_text = torch.zeros_like(z_v_window[:, 0])

                # Through mixer — accepts (B, K, D) for z_v
                z_v_window, z_t_tokens, z_text = align.cross_attention_mixer(
                    z_v_window, z_t_tokens, z_text
                )

                # Slice to world model's window size (use last `window_size` timesteps)
                z_v_window = z_v_window[:, -window_size:]
                z_t_tokens = z_t_tokens[:, -window_size:]

                # -- Encode state_t+1 (target) via frozen ALIGNModel --
                mixed_next = align.encode_mixed(frames_next, traj_next, texts)
                z_v_target = mixed_next["z_v"].float()  # (B, D)
                z_t_target = mixed_next["z_t"].float()  # (B, D)

            # -- Predict next state from window of past states + action --
            z_v_pred, z_t_pred = world_model(z_v_window, z_t_tokens, z_text, action)

            # -- MSE loss (world_model_loss detaches targets internally
            #    via stop-gradient convention — encoder never trains) --
            loss = world_model_loss(z_v_pred, z_v_target, z_t_pred, z_t_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            with torch.no_grad():
                cos_v = torch.nn.functional.cosine_similarity(
                    z_v_pred, z_v_target, dim=-1
                ).mean().item()
                cos_t = torch.nn.functional.cosine_similarity(
                    z_t_pred, z_t_target, dim=-1
                ).mean().item()

            epoch_losses.append(loss.item())
            epoch_cos_v.append(cos_v)
            epoch_cos_t.append(cos_t)

            if step % 10 == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    cos_v=f"{cos_v:.3f}",
                    cos_t=f"{cos_t:.3f}",
                )

        # -- End-of-epoch summary ----------------------------
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        avg_cos_v = float(np.mean(epoch_cos_v)) if epoch_cos_v else 0.0
        avg_cos_t = float(np.mean(epoch_cos_t)) if epoch_cos_t else 0.0
        elapsed = time.time() - t_start

        # -- Validation loop ------------------------------
        # Iterate the val_loader once, computing the same loss/metrics
        # as training but on held-out data. The world model is trained
        # on (z_v_window, z_t_tokens, z_text, action) -> (z_v_target, z_t_target)
        # where targets are encoded state at t+1.
        val_losses: list = []
        val_cos_v: list = []
        val_cos_t: list = []
        with torch.no_grad():
            for val_batch in val_loader:
                # Normalize batch schema (frame_t vs frames_t etc.)
                if "frame_t" in val_batch:
                    vf_t = torch.from_numpy(val_batch["frame_t"]).to(device)
                    vt_t = torch.from_numpy(val_batch["traj_t"]).float().to(device)
                    vf_next = torch.from_numpy(val_batch["frame_next"]).to(device)
                    vt_next = torch.from_numpy(val_batch["traj_next"]).float().to(device)
                    v_action = torch.from_numpy(val_batch["action"]).float().to(device)
                    v_texts = val_batch["text"]
                else:
                    vf_t = torch.from_numpy(val_batch["frames_t"]).to(device)
                    vt_t = torch.from_numpy(val_batch["trajectory_t"]).float().to(device)
                    vf_next = torch.from_numpy(val_batch["frames_next"]).to(device)
                    vt_next = torch.from_numpy(val_batch["trajectory_next"]).float().to(device)
                    v_action = torch.from_numpy(val_batch["action"]).float().to(device)
                    v_texts = val_batch["texts"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    # Encode state_t (input)
                    z_v_w = align.encode_raw_vision_window(vf_t)
                    z_t_tok = align.encode_raw_trajectory_tokens(vt_t)
                    z_t_text = align.encode_raw_text(v_texts)
                    if z_t_text is None:
                        z_t_text = torch.zeros_like(z_v_w[:, 0])
                    z_v_w, z_t_tok, z_t_text = align.cross_attention_mixer(
                        z_v_w, z_t_tok, z_t_text
                    )
                    z_v_w = z_v_w[:, -window_size:]
                    z_t_tok = z_t_tok[:, -window_size:]

                    # Encode state_t+1 (target) — same convention as training
                    mixed_next = align.encode_mixed(vf_next, vt_next, v_texts)
                    z_v_tgt = mixed_next["z_v"].float()
                    z_t_tgt = mixed_next["z_t"].float()

                # Predict + loss
                z_v_p, z_t_p = world_model(z_v_w, z_t_tok, z_t_text, v_action)
                v_loss = world_model_loss(z_v_p, z_v_tgt, z_t_p, z_t_tgt)
                v_cos_v = torch.nn.functional.cosine_similarity(
                    z_v_p, z_v_tgt, dim=-1
                ).mean().item()
                v_cos_t = torch.nn.functional.cosine_similarity(
                    z_t_p, z_t_tgt, dim=-1
                ).mean().item()
                val_losses.append(v_loss.item())
                val_cos_v.append(v_cos_v)
                val_cos_t.append(v_cos_t)

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_val_cos_v = float(np.mean(val_cos_v)) if val_cos_v else 0.0
        avg_val_cos_t = float(np.mean(val_cos_t)) if val_cos_t else 0.0

        print(
            f"  Epoch {epoch+1:3d}/{epochs}  "
            f"train: loss={avg_loss:.4f} cos_v={avg_cos_v:.3f} cos_t={avg_cos_t:.3f}  |  "
            f"val: loss={avg_val_loss:.4f} cos_v={avg_val_cos_v:.3f} cos_t={avg_val_cos_t:.3f}  "
            f"elapsed: {elapsed:.0f}s"
        )

        wandb_trainer.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/cos_v": avg_cos_v,
            "train/cos_t": avg_cos_t,
            "val/loss": avg_val_loss,
            "val/cos_v": avg_val_cos_v,
            "val/cos_t": avg_val_cos_t,
        }, step=epoch + 1)

        log_fp.write(json.dumps({
            "epoch": epoch + 1,
            "stage": "epoch",
            "train_loss": avg_loss,
            "train_cos_v": avg_cos_v,
            "train_cos_t": avg_cos_t,
            "val_loss": avg_val_loss,
            "val_cos_v": avg_val_cos_v,
            "val_cos_t": avg_val_cos_t,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        # -- Best checkpoint: use val/loss as the selection metric ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = out_dir / "best.pt"
            torch.save({
                "world_model_state": world_model.state_dict(),
                "config": config_snapshot,
                "epoch": epoch + 1,
                "loss": avg_val_loss,
                "val_cos_v": avg_val_cos_v,
                "val_cos_t": avg_val_cos_t,
            }, str(best_path))
            print(f"  ↳ new best (val_loss={avg_val_loss:.4f}), saved to {best_path}")

        # -- Per-epoch checkpoint -----------------------------
        epoch_path = out_dir / f"world_model_epoch_{epoch+1:04d}.pt"
        save_checkpoint(epoch_path, epoch + 1, avg_loss)

        # -- Best checkpoint ----------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = out_dir / "world_model_best.pt"
            save_checkpoint(best_path, epoch + 1, avg_loss)
            print(f"  -> world_model_best.pt (loss: {avg_loss:.4f})")
            # Upload best checkpoint to wandb as an artifact
            wandb_trainer.save(str(best_path))

    log_fp.close()
    wandb_trainer.finish()

    print(f"\n  World Model training complete.")
    print(f"    Best loss:  {best_loss:.4f}")
    print(f"    Best ckpt:  {out_dir / 'world_model_best.pt'}")
    print(f"    Epoch ckpts:{out_dir}/world_model_epoch_NNNN.pt")
    print(f"    Log:        {log_path}")

    return str(out_dir / "world_model_best.pt")


# ================================================================
# CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALIGN World Model training (frozen encoder + mixer; "
                    "trains WorldModel head for s' = f(s, a))"
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s). Pass multiple to "
                             "train on the concatenation.")
    parser.add_argument("--pretrained", required=True,
                        help="Phase 1b pretrained checkpoint (encoder + mixer).")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views to use (e.g. 'wrist_image image'). "
                             "Must match the cameras used during pretrain. "
                             "Default: auto-detect a single camera from the dataset.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a world_model_best.pt (or world_model_epoch_NNNN.pt) "
                             "to continue training from. Loads the world model weights only; "
                             "encoder+mixer are still loaded from --pretrained. Epoch numbering "
                             "continues from the resumed checkpoint.")
    parser.add_argument("--output-dir", default="./checkpoints/world_model",
                        help="Directory under which "
                             "{dataset_stem}/run_N/ will be created.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Trajectory window length used to embed state.")
    parser.add_argument("--mixer-dim", type=int, default=512)
    parser.add_argument("--num-mixer-blocks", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--seed", type=int, default=42)

    # World model arch + kwargs
    parser.add_argument("--arch", default="mlp", choices=["mlp", "rnn", "transformer"],
                        help="WorldModel architecture.")
    parser.add_argument("--action-dim", type=int, default=6,
                        help="Action dim (default 6 for OSC_POSE delta).")
    parser.add_argument("--embed-dim", type=int, default=256,
                        help="Per-modality embedding dim (must match pretrained).")
    parser.add_argument("--mlp-hidden", type=int, default=512,
                        help="MLP world model hidden dim.")
    parser.add_argument("--mlp-layers", type=int, default=3,
                        help="MLP world model num layers.")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Number of past timesteps for the world model (default 5). Sliced from traj_window.")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-d-model", type=int, default=384)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-dim-ff", type=int, default=1024)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-world-model")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--compare-archs", action="store_true",
                        help="Train both 'mlp' and 'rnn' architectures sequentially, "
                             "saving best checkpoints for each.")

    args = parser.parse_args()

    if args.compare_archs:
        print(f"\n{'='*70}")
        print("COMPARISON MODE: training MLP and RNN sequentially")
        print(f"{'='*70}\n")
        for comp_arch in ["mlp", "rnn"]:
            print(f"\n>>> Training arch={comp_arch} <<<\n")
            train_world_model(
                data_paths=args.data,
                pretrained_checkpoint=args.pretrained,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                val_split=args.val_split,
                device=args.device,
                max_steps_per_epoch=args.max_steps_per_epoch,
                wandb_project=args.wandb_project,
                wandb_run=f"{args.wandb_run or ''}_{comp_arch}" if args.wandb_run else comp_arch,
                enable_wandb=args.wandb,
                num_workers=args.num_workers,
                traj_window=args.traj_window,
                chunk_size=args.chunk_size,
                mixer_dim=args.mixer_dim,
                num_mixer_blocks=args.num_mixer_blocks,
                use_bf16=args.bf16,
                resume=args.resume,
                arch=comp_arch,
                action_dim=args.action_dim,
                embed_dim=args.embed_dim,
                mlp_hidden=args.mlp_hidden,
                mlp_layers=args.mlp_layers,
                window_size=args.window_size,
                transformer_layers=args.transformer_layers,
                transformer_d_model=args.transformer_d_model,
                transformer_nhead=args.transformer_nhead,
                transformer_dropout=args.transformer_dropout,
                transformer_dim_ff=args.transformer_dim_ff,
                cameras=args.cameras,
                seed=args.seed,
            )
        print(f"\n{'='*70}")
        print("COMPARISON COMPLETE — checkpoints in:")
        print(f"  {args.output_dir}/{{dataset}}/run_N/  (MLP)")
        print(f"  {args.output_dir}/{{dataset}}/run_N+1/  (RNN)")
        print(f"{'='*70}")
        return

    train_world_model(
        data_paths=args.data,
        pretrained_checkpoint=args.pretrained,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        enable_wandb=args.wandb,
        num_workers=args.num_workers,
        traj_window=args.traj_window,
        chunk_size=args.chunk_size,
        mixer_dim=args.mixer_dim,
        num_mixer_blocks=args.num_mixer_blocks,
        use_bf16=args.bf16,
        resume=args.resume,
        arch=args.arch,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        mlp_hidden=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        window_size=args.window_size,
        transformer_layers=args.transformer_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_dropout=args.transformer_dropout,
        transformer_dim_ff=args.transformer_dim_ff,
        cameras=args.cameras,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
