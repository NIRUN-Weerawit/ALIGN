#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN Assistant Head training — action prediction.

Encoders + mixer are frozen. Only the assistant head trains.

Usage:
    python training/train_heads.py \\
        --data ./align.h5 \\
        --pretrained ./checkpoints/pretrain/best.pt \\
        --output-dir ./checkpoints/heads \\
        --epochs-assistant 10

────────────────────────────────────────────────────────────────────────
TRAINING CONTRACT — train_heads.py (Phase 3: Assistant Head)
────────────────────────────────────────────────────────────────────────
INPUT  (per sample, B = batch):
  - MLP arch (default):
      - z_v:     (B, 256)            — current visual embedding (mean-pooled)
      - z_s:     (B, 256)            — one-step state embedding (from robot_state: B, 7)
      - z_sext:  (B, 256)            — text embedding
  - Transformer arch (--assistant-arch transformer):
      - z_v_window: (B, K, 256)       — K past vision embeddings (K = --chunk-size)
      - z_s_window: (B, K, 256)       — K-step state window (replicated from
                                       one-step z_s in v2)
      - z_sext:     (B, 256)          — text embedding (broadcast over K)
  - current_action: (B, 6)           — the human's current OSC action (target)

OUTPUT (per sample, B = batch):
  - action_pred:     (B, 6)           — single predicted OSC action

TARGET:
  - current_action   (B, 6)           — ground-truth current OSC action

LOSS:
  - F.mse_loss(action_pred, current_action)  — single-step MSE regression

METRICS (per epoch, logged to wandb + JSONL):
  - loss        (float, OSC²):         MSE between action_pred and current_action (lower=better)
  - action_mean (float, OSC units):     mean |action_pred| — sanity check for output magnitude

BEST CHECKPOINT:
  - Lowest avg_loss on validation set, saved as assistant_best.pt
────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from training.wandb_utils import init_wandb
from data.align_dataset import ALIGNDataset, head_collate


# ================================================================
# Training
# ================================================================

def train_heads_hdf5(
    data_paths: List[str],
    pretrained_checkpoint: str,
    output_dir: str,
    epochs_assistant: int = 10,
    batch_size: int = 64,
    lr_assistant: float = 1e-3,
    weight_decay: float = 1e-4,
    chunk_size: int = 5,
    val_split: float = 0.1,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    wandb_project: str = "align-heads",
    cameras: Optional[List[str]] = None,
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 0,
    traj_window: int = 20,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    use_bf16: bool = True,
    loss_decay: float = 0.7,
    # Assistant head params
    assistant_hidden: int = 256,
    assistant_layers: int = 2,
    assistant_dropout: float = 0.0,
    # Assistant architecture: "mlp" (default) or "transformer"
    assistant_arch: str = "mlp",
    # Transformer assistant params
    assistant_d_model: int = 384,
    assistant_nhead: int = 4,
    assistant_num_layers: int = 2,
    assistant_dim_ff: int = 1024,
) -> str:
    """Train the Assistant head (delta pose correction) with MSE loss.

    Encoders + mixer are frozen. Only the assistant head trains.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Derive subdirectory
    if len(data_paths) == 1:
        ds_name = Path(data_paths[0]).stem
    else:
        ds_name = "+".join(Path(p).stem for p in data_paths)
    base_dir = Path(output_dir) / ds_name

    # Find next available run number
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

    print(f"=== ALIGN Assistant Head Training ===")
    print(f"  Run:          {out_dir}")
    print(f"  Data ({len(data_paths)}): {data_paths}")
    print(f"  Pretrained:   {pretrained_checkpoint}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {epochs_assistant}")
    print(f"  LR:           {lr_assistant}")
    print(f"  Arch:         {assistant_arch}")

    # -- W&B --
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-assistant-head",
            "data": [str(p) for p in data_paths],
            "pretrained_checkpoint": pretrained_checkpoint,
            "epochs_assistant": epochs_assistant,
            "batch_size": batch_size,
            "lr_assistant": lr_assistant,
            "weight_decay": weight_decay,
            "chunk_size": chunk_size,
            "device": str(device),
            "use_bf16": use_bf16,
            "cameras": cameras if cameras else ["wrist_image"],
            "assistant_arch": assistant_arch,
        },
    )
    print(f"  W&B:          {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Dataset --
    if len(data_paths) == 1:
        full_ds = ALIGNDataset(data_paths[0], mode="head", traj_window=traj_window,
                               cameras=cameras)
    else:
        from data.align_dataset import MultiALIGNDataset
        full_ds = MultiALIGNDataset(
            data_paths, mode="head", traj_window=traj_window, cameras=cameras
        )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    print(f"  {n_train} train, {n_val} val samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size, vision_window_size=chunk_size),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size, vision_window_size=chunk_size),
        num_workers=num_workers,
        pin_memory=True,
    )

    # -- Model --
    num_cameras = len(cameras) if cameras else 1
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
        num_cameras=num_cameras,
        assistant_hidden=assistant_hidden,
        assistant_layers=assistant_layers,
        assistant_dropout=assistant_dropout,
        assistant_arch=assistant_arch,
        assistant_d_model=assistant_d_model,
        assistant_nhead=assistant_nhead,
        assistant_num_layers=assistant_num_layers,
        assistant_dim_ff=assistant_dim_ff,
    ).to(device)

    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])

    # Freeze backbones + encoders + mixer permanently
    model.freeze_backbone()
    model.freeze_all_encoders()
    print(f"  Loaded pretrained backbone from {pretrained_checkpoint}")
    print(f"  All encoders + mixer frozen — only assistant head trains")

    # Unfreeze assistant head
    for p in model.assistant_head.parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=lr_assistant, weight_decay=weight_decay)

    log_path = out_dir / "head_log.jsonl"
    log_fp = open(log_path, "w")
    best_loss = float("inf")

    # ================================================================
    # Training loop
    # ================================================================
    for epoch in range(epochs_assistant):
        model.train()
        losses_b, actions_pred = [], []

        progress = train_loader
        if max_steps_per_epoch and max_steps_per_epoch < len(train_loader):
            progress = (
                p for p, _ in zip(train_loader, range(max_steps_per_epoch))
            )
        progress_bar = tqdm(
            progress,
            total=min(max_steps_per_epoch, len(train_loader)) if max_steps_per_epoch else len(train_loader),
            desc=f"[Assistant] Epoch {epoch+1}/{epochs_assistant}",
            unit="step",
        )

        for step, batch in enumerate(progress_bar):
            if step >= max_steps_per_epoch:
                break

            texts           = batch["texts"]
            frames          = torch.from_numpy(batch["frames"]).to(device)
            delta_t         = torch.from_numpy(batch["delta_target"]).float().to(device)
            current_action  = torch.from_numpy(batch["current_action"]).float().to(device)
            # v2 one-step robot state (B, 7) — preferred for the state encoder.
            # Fall back to the legacy (B, K, 6) trajectory window for
            # backward-compat with older datasets that don't yet include
            # `robot_state`.
            if "robot_state" in batch:
                state = torch.from_numpy(batch["robot_state"]).float().to(device)
            else:
                state = torch.from_numpy(batch["trajectory"]).float().to(device)

            # Frozen encodings via mixer
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frames, state, texts)
                z_v = mixed["z_v"].float()
                z_s = mixed["z_s"].float()
                z_s_tokens = mixed["z_s_tokens"].float()  # (B, K, 256) for transformer
                z_sext = mixed["z_sext"].float()

            # Assistant head: branch by architecture
            if model.assistant_arch == "transformer":
                K = model.assistant_head.chunk_size
                frames_window = torch.from_numpy(batch["frames_window"]).to(device)
                z_v_window_raw = model.encode_raw_vision_window(frames_window)
                # Transformer assistant expects (B, K, D) windows for both
                # z_v and z_s. In v2 we have a one-step state; replicate
                # it K times to fake a K-step window of identical tokens.
                # (The mixer already produces z_s_tokens with shape (B, K, D)
                #  where K=1 — replicate to the assistant's expected K.)
                if z_s_tokens.shape[1] != K:
                    z_s_window = z_s_tokens.expand(-1, K, -1)
                else:
                    z_s_window = z_s_tokens
                z_v_window_mixed, _, _ = model.cross_attention_mixer(
                    z_v_window_raw, z_s_window, z_sext
                )
                action_pred = model.assistant_head(z_v_window_mixed, z_s_window, z_sext)
            else:
                action_pred = model.assistant_head(z_v, z_s, z_sext)

            # Single-step action loss (B, 6) vs (B, 6)
            # action_pred and current_action are both (B, 6) — current OSC action
            # at this timestep. We don't do K-step prediction anymore; the
            # model just predicts the current action from K past frames.
            loss_mse = F.mse_loss(action_pred, current_action)

            optimizer.zero_grad()
            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            losses_b.append(loss_mse.item())
            actions_pred.append(action_pred.detach().abs().mean().item())

            if step % 10 == 0:
                progress_bar.set_postfix(
                    mse=f"{loss_mse.item():.5f}",
                    a_mean=f"{action_pred.detach().abs().mean().item():.4f}",
                )

        avg_loss = float(np.mean(losses_b))
        av_action = float(np.mean(actions_pred))

        # Validation loop
        val_losses, val_actions = [], []
        with torch.no_grad():
            for vbatch in val_loader:
                vframes = torch.from_numpy(vbatch["frames"]).to(device)
                vtexts = vbatch["texts"]
                vcurrent_action = torch.from_numpy(vbatch["current_action"]).float().to(device)
                # v2 one-step robot state; fall back to trajectory window.
                if "robot_state" in vbatch:
                    vstate = torch.from_numpy(vbatch["robot_state"]).float().to(device)
                else:
                    vstate = torch.from_numpy(vbatch["trajectory"]).float().to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    vmixed = model.encode_mixed(vframes, vstate, vtexts)
                    vz_v = vmixed["z_v"].float()
                    vz_s = vmixed["z_s"].float()
                    vz_s_tokens = vmixed["z_s_tokens"].float()
                    vz_sext = vmixed["z_sext"].float()
                    if model.assistant_arch == "transformer":
                        vK = model.assistant_head.chunk_size
                        vframes_window = torch.from_numpy(vbatch["frames_window"]).to(device)
                        vz_v_window_raw = model.encode_raw_vision_window(vframes_window)
                        # v2: replicate one-step state to K-step window
                        if vz_s_tokens.shape[1] != vK:
                            vz_s_window = vz_s_tokens.expand(-1, vK, -1)
                        else:
                            vz_s_window = vz_s_tokens
                        vz_v_window_mixed, _, _ = model.cross_attention_mixer(
                            vz_v_window_raw, vz_s_window, vz_sext
                        )
                        vaction_pred = model.assistant_head(vz_v_window_mixed, vz_s_window, vz_sext)
                    else:
                        vaction_pred = model.assistant_head(vz_v, vz_s, vz_sext)
                vloss = F.mse_loss(vaction_pred, vcurrent_action)
                val_losses.append(vloss.item())
                val_actions.append(vaction_pred.detach().abs().mean().item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_val_action = float(np.mean(val_actions)) if val_actions else 0.0

        print(f"  [Δ] Epoch {epoch+1:3d}/{epochs_assistant}  "
              f"train: MSE={avg_loss:.4f} a_mean={av_action:.4f}  |  "
              f"val: MSE={avg_val_loss:.4f} a_mean={avg_val_action:.4f}")

        wandb_trainer.log({
            "stage": "assistant",
            "loss": avg_loss,
            "action_mean": av_action,
            "val_loss": avg_val_loss,
            "val_action_mean": avg_val_action,
            "epoch": epoch + 1,
        }, step=epoch + 1)

        # Best checkpoint: use val_loss for selection
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.save_heads_checkpoint(
                str(out_dir / "assistant_best.pt"), epoch, avg_val_loss,
                optimizer.state_dict(), {"chunk_size": chunk_size,
                                          "val_loss": avg_val_loss,
                                          "val_action_mean": avg_val_action})
            print(f"  ↳ new best (val_loss={avg_val_loss:.4f}), saved to assistant_best.pt")

        log_fp.write(json.dumps({
            "stage": "assistant", "epoch": epoch + 1,
            "loss": avg_loss, "action_mean": av_action,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

    # Save final checkpoint
    model.train()
    model.save_heads_checkpoint(
        str(out_dir / "heads_best.pt"), epochs_assistant - 1,
        best_loss, optimizer.state_dict(), {"chunk_size": chunk_size})

    log_fp.close()
    wandb_trainer.finish()
    print(f"\n  Assistant head training complete.")
    print(f"    Best:       {out_dir / 'assistant_best.pt'}")
    print(f"    Combined:   {out_dir / 'heads_best.pt'}")
    print(f"    Logs:       {log_path}")
    return str(out_dir / "heads_best.pt")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ALIGN Assistant Head Training (OSC action prediction)")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to align.h5 dataset(s).")
    parser.add_argument("--pretrained", required=True, help="Phase 1 pretrained checkpoint")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views to use (e.g. 'wrist_image image'). "
                             "Must match the cameras used during pretrain.")
    parser.add_argument("--output-dir", default="./checkpoints/heads")
    parser.add_argument("--epochs-assistant", type=int, default=10,
                        help="Assistant (delta-pose) training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-assistant", type=float, default=1e-3,
                        help="Learning rate for Assistant head")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--mixer-dim", type=int, default=512)
    parser.add_argument("--num-mixer-blocks", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 autocast (on by default)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable BF16 autocast")
    parser.add_argument("--loss-decay", type=float, default=0.7,
                        help="Decay for weighted mean of K-step prediction errors")
    parser.add_argument("--assistant-hidden", type=int, default=256,
                        help="Assistant head hidden layer width")
    parser.add_argument("--assistant-layers", type=int, default=2,
                        help="Assistant head number of hidden layers")
    parser.add_argument("--assistant-dropout", type=float, default=0.0,
                        help="Assistant head dropout")
    parser.add_argument("--assistant-arch", type=str, default="mlp",
                        choices=["mlp", "transformer"],
                        help="Assistant head architecture: 'mlp' (default) or 'transformer'")
    parser.add_argument("--assistant-d-model", type=int, default=384,
                        help="Transformer assistant: hidden dim")
    parser.add_argument("--assistant-nhead", type=int, default=4,
                        help="Transformer assistant: number of attention heads")
    parser.add_argument("--assistant-num-layers", type=int, default=2,
                        help="Transformer assistant: number of transformer layers")
    parser.add_argument("--assistant-dim-ff", type=int, default=1024,
                        help="Transformer assistant: FFN hidden dim")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-heads")
    parser.add_argument("--wandb-run", default=None)

    args = parser.parse_args()

    train_heads_hdf5(
        data_paths=args.data,
        pretrained_checkpoint=args.pretrained,
        output_dir=args.output_dir,
        epochs_assistant=args.epochs_assistant,
        batch_size=args.batch_size,
        lr_assistant=args.lr_assistant,
        weight_decay=args.weight_decay,
        chunk_size=args.chunk_size,
        val_split=args.val_split,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        enable_wandb=args.wandb,
        num_workers=args.num_workers,
        traj_window=args.traj_window,
        mixer_dim=args.mixer_dim,
        num_mixer_blocks=args.num_mixer_blocks,
        use_bf16=args.bf16,
        loss_decay=args.loss_decay,
        assistant_hidden=args.assistant_hidden,
        assistant_layers=args.assistant_layers,
        assistant_dropout=args.assistant_dropout,
        assistant_arch=args.assistant_arch,
        assistant_d_model=args.assistant_d_model,
        assistant_nhead=args.assistant_nhead,
        assistant_num_layers=args.assistant_num_layers,
        assistant_dim_ff=args.assistant_dim_ff,
        cameras=args.cameras,
    )


if __name__ == "__main__":
    main()