#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN contrastive pretraining on HDF5 data.

Two sub-phases:
  Phase 1a (encoder): InfoNCE on raw encoder outputs, mixer frozen.
  Phase 1b (mixer):   InfoNCE on mixer outputs, mixer trainable.

Usage:
    # Full training (Phase 1a + 1b)
    python training/pretrain.py \\
        --data ./align.h5 \\
        --output-dir ./checkpoints/pretrain \\
        --epochs-encoder 40 \\
        --epochs-mixer 10

    # Phase 1a only (then check training quality)
    python training/pretrain.py --data ./align.h5 --stages encoder

    # Resume Phase 1b from existing encoder checkpoint
    python training/pretrain.py --data ./align.h5 \\
        --encoder-checkpoint ./checkpoints/pretrain/encoder_best.pt \\
        --epochs-mixer 10
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- Lightweight GPU monitor (cached, non-blocking) ----------
_gpu_stats_cache: Optional[dict] = None
_gpu_sample_interval = 5  # seconds between samples
_gpu_last_sample_time = 0.0


def _get_gpu_stats() -> Optional[dict]:
    """Sample nvidia-smi every few seconds (cached)."""
    global _gpu_stats_cache, _gpu_last_sample_time
    now = time.time()
    if now - _gpu_last_sample_time >= _gpu_sample_interval:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "-u", ",", "--format=csv,noheader"],
                timeout=2, stderr=subprocess.DEVNULL,
            ).decode().strip()
            parts = out.split(",")
            gpu_util_str = parts[0].replace("%", "").strip()
            # Memory may have internal commas (e.g. "14,256 MiB") — rebuild from remaining parts
            mem_str = "".join(parts[1:]).replace("MiB", "").replace("GiB", "").replace(",", "").strip()
            _gpu_stats_cache = {
                "gpu_util": int(gpu_util_str),
                "mem_gb": round(int(mem_str) / 1024, 1),
            }
        except Exception:
            _gpu_stats_cache = None
        _gpu_last_sample_time = now
    return _gpu_stats_cache


from models.align_model import ALIGNModel
from training.contrastive_loss import ContrastiveLoss3Way
from training.wandb_utils import init_wandb
from data.align_dataset import ALIGNDataset, pretrain_collate


# ================================================================
# Training
# ================================================================

def pretrain_hdf5(
    data_path: str,
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
    val_split: float = 0.1,
    val_every: int = 5,
    wandb_project: str = "align-pretrain",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = True,
    num_workers: int = 0,
    traj_window: int = 20,
    frames_per_ep: int = 8,
    episodes_per_batch: int = 8,
    use_text: bool = True,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    use_bf16: bool = True,
    encoder_checkpoint: Optional[str] = None,
    resume: Optional[str] = None,
    stages: str = "all",
):
    """Contrastive pretraining from HDF5 data with two sub-phases."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Derive subdirectory from dataset name: e.g. --output-dir checkpoints/pretrain_local
    # + --data libero_align.h5 → checkpoints/pretrain_local/libero_align/
    ds_name = Path(data_path).stem
    output_dir = Path(output_dir) / ds_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ALIGN Contrastive Pretraining (HDF5) ===")
    print(f"  Dataset:  {data_path}")
    print(f"  Device:   {device}")
    print(f"  Phase 1a (encoder): {epochs_encoder} epochs")
    print(f"  Phase 1b (mixer):   {epochs_mixer} epochs")
    print(f"  Output:   {output_dir}")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-pretrain-hdf5",
            "dataset": str(data_path),
            "epochs_encoder": epochs_encoder,
            "epochs_mixer": epochs_mixer,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "embed_dim": embed_dim,
            "temperature": temperature,
            "max_grad_norm": max_grad_norm,
            "device": str(device),
            "traj_window": traj_window,
            "mixer_dim": mixer_dim,
            "num_mixer_blocks": num_mixer_blocks,
            "use_bf16": use_bf16,
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:      {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Dataset ──
    full_ds = ALIGNDataset(
        data_path,
        mode="pretrain",
        frames_per_ep=frames_per_ep,
        traj_window=traj_window,
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
        collate_fn=lambda b: pretrain_collate(b, traj_window=traj_window),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda b: pretrain_collate(b, traj_window=traj_window),
        num_workers=num_workers,
    )

    # -- Model ──
    model = ALIGNModel(
        embed_dim=embed_dim,
        use_text=use_text,
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
        "frames_per_ep": frames_per_ep,
        "mixer_dim": mixer_dim,
        "num_mixer_blocks": num_mixer_blocks,
        "temperature": temperature,
    }

    log_path = output_dir / "pretrain_log.jsonl"
    log_fp = open(log_path, "a")

    # ================================================================
    # Phase 1a: Encoder Pretrain (mixer frozen, InfoNCE on raw outputs)
    # ================================================================
    start_epoch = 0
    best_loss = float("inf")

    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        print(f"\n  Resuming from encoder checkpoint: {encoder_checkpoint}")
        ckpt = torch.load(encoder_checkpoint, map_location=device)
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}")

    if stages in ("all", "encoder") and epochs_encoder > 0:
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
            # Tqdm progress bar (Phase 1a)
            pbar = tqdm(
                enumerate(train_loader),
                total=min(max_steps_per_epoch, len(train_loader)),
                desc=f"[1a] Ep {epoch + 1}",
                leave=False,
            )

            for step, batch in pbar:
                if step >= max_steps_per_epoch:
                    break

                frames = torch.from_numpy(batch["frames"]).to(device)
                trajs = torch.from_numpy(batch["trajectories"]).float().to(device)
                texts = batch["texts"]

                if trajs.shape[-1] < 6:
                    pad = torch.zeros(*trajs.shape[:-1], 6 - trajs.shape[-1], device=trajs.device)
                    trajs = torch.cat([trajs, pad], dim=-1)

                # Raw encoder outputs (no mixer), BF16 autocast
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
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

                # Update progress bar display
                gpu = _get_gpu_stats() if str(device) == "cuda" else None
                pbar.set_postfix(
                    L=f"{loss.item():.4f}",
                    vt="{:.3f}".format(stats['avg_cos_vt'].item()),
                    vl="{:.3f}".format(stats['avg_cos_vl'].item()),
                    tl="{:.3f}".format(stats['avg_cos_tl'].item()),
                    gpu=f"{gpu['gpu_util']}" if gpu else "?",
                    mem="{:.1f}G".format(gpu['mem_gb']) if gpu else "?G",
                )

                # Per-step logging (W&B + local JSONL)
                wandb_trainer.log({
                    "loss": loss.item(), "cos_vt": stats["avg_cos_vt"].item(),
                    "cos_vl": stats["avg_cos_vl"].item(), "cos_tl": stats["avg_cos_tl"].item(),
                    "lr": lr,
                }, step=epoch * max_steps_per_epoch + step)

                log_fp.write(json.dumps({
                    "phase": "1a_step", "epoch": epoch + 1, "step": step + 1,
                    "loss": loss.item(), "cos_vt": stats["avg_cos_vt"].item(),
                    "cos_vl": stats["avg_cos_vl"].item(), "cos_tl": stats["avg_cos_tl"].item(),
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")

            pbar.close()


            avg_loss = float(np.mean(epoch_losses))
            avg_vt = float(np.mean(epoch_cos_vt))
            avg_vl = float(np.mean(epoch_cos_vl))
            avg_tl = float(np.mean(epoch_cos_tl))

            print(f"  [1a] Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
                  f"cos_vt: {avg_vt:.3f}  cos_vl: {avg_vl:.3f}  cos_tl: {avg_tl:.3f}")

            wandb_trainer.log({
                "phase": "1a_encoder", "epoch": epoch + 1, "loss": avg_loss,
                "cos_vt": avg_vt, "cos_vl": avg_vl, "cos_tl": avg_tl,
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

            # Validation (only every val_every epochs)
            if (epoch + 1) % val_every == 0:
                model.eval()
                val_losses, val_vt, val_vl, val_tl = [], [], [], []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="[1a] Val", leave=False):
                        frames = torch.from_numpy(batch["frames"]).to(device)
                        trajs = torch.from_numpy(batch["trajectories"]).float().to(device)
                        texts = batch["texts"]
                        if trajs.shape[-1] < 6:
                            pad = torch.zeros(*trajs.shape[:-1], 6 - trajs.shape[-1], device=trajs.device)
                            trajs = torch.cat([trajs, pad], dim=-1)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            raw = model.encode_raw_all(frames, trajs, texts)
                        raw = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
                        stats = criterion(raw["z_v"], raw["z_t"], raw["z_text"])
                        val_losses.append(stats["loss"].item())
                        val_vt.append(stats["avg_cos_vt"].item())
                        val_vl.append(stats["avg_cos_vl"].item())
                        val_tl.append(stats["avg_cos_tl"].item())

                avg_val_loss = float(np.mean(val_losses))
                print(f"  [1a val] Epoch {epoch + 1:3d}  loss: {avg_val_loss:.4f}  "
                      f"cos_vt: {float(np.mean(val_vt)):.3f}  "
                      f"cos_vl: {float(np.mean(val_vl)):.3f}  "
                      f"cos_tl: {float(np.mean(val_tl)):.3f}")

                wandb_trainer.log({
                    "val_loss": avg_val_loss, "val_cos_vt": float(np.mean(val_vt)),
                    "val_cos_vl": float(np.mean(val_vl)), "val_cos_tl": float(np.mean(val_tl)),
                }, step=epoch + 1)

                # Log validation to JSONL
                log_fp.write(json.dumps({
                    "phase": "1a_val", "epoch": epoch + 1,
                    "loss": avg_val_loss,
                    "cos_vt": float(np.mean(val_vt)), "cos_vl": float(np.mean(val_vl)),
                    "cos_tl": float(np.mean(val_tl)),
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")
                log_fp.flush()

        print(f"  Phase 1a complete. Best loss: {best_loss:.4f}")

    # ================================================================
    # Phase 1b: Mixer Warm-Up (InfoNCE on mixer outputs)
    # ================================================================
    if stages in ("all", "mixer") and epochs_mixer > 0:
        # Load best encoder checkpoint if we just ran Phase 1a
        encoder_ckpt_path = str(output_dir / "encoder_best.pt")
        if Path(encoder_ckpt_path).exists():
            ckpt = torch.load(encoder_ckpt_path, map_location=device)
            model.load_trainable_state_dict(ckpt["trainable_state_dict"])
            print(f"\n  Loaded encoder checkpoint for Phase 1b")

        model.unfreeze_mixer()
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"\n  Phase 1b — Trainable: {n_params:,} (encoders + mixer)")
        optimizer = optim.AdamW(trainable, lr=lr * 0.5, weight_decay=weight_decay)
        best_loss = float("inf")
        start_epoch = 0

        for epoch in range(start_epoch, epochs_mixer):
            model.train()
            epoch_losses = []
            epoch_cos_vt = []
            epoch_cos_vl = []
            epoch_cos_tl = []

            pbar = tqdm(
                enumerate(train_loader),
                total=min(max_steps_per_epoch, len(train_loader)),
                desc=f"[1b] Ep {epoch + 1}",
                leave=False,
            )

            for step, batch in pbar:
                if step >= max_steps_per_epoch:
                    break

                frames = torch.from_numpy(batch["frames"]).to(device)
                trajs = torch.from_numpy(batch["trajectories"]).float().to(device)
                texts = batch["texts"]

                if trajs.shape[-1] < 6:
                    pad = torch.zeros(*trajs.shape[:-1], 6 - trajs.shape[-1], device=trajs.device)
                    trajs = torch.cat([trajs, pad], dim=-1)

                # Mixer outputs, BF16 autocast
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
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

                gpu = _get_gpu_stats() if str(device) == "cuda" else None
                pbar.set_postfix(
                    L=f"{loss.item():.4f}",
                    vt="{:.3f}".format(stats['avg_cos_vt'].item()),
                    vl="{:.3f}".format(stats['avg_cos_vl'].item()),
                    tl="{:.3f}".format(stats['avg_cos_tl'].item()),
                    gpu=f"{gpu['gpu_util']}" if gpu else "?",
                    mem="{:.1f}G".format(gpu['mem_gb']) if gpu else "?G",
                )

                # Per-step logging (W&B + local JSONL)
                wandb_trainer.log({
                    "loss": loss.item(), "cos_vt": stats["avg_cos_vt"].item(),
                    "cos_vl": stats["avg_cos_vl"].item(), "cos_tl": stats["avg_cos_tl"].item(),
                    "lr": lr * 0.5,
                }, step=epochs_encoder + epoch * max_steps_per_epoch + step)

                log_fp.write(json.dumps({
                    "phase": "1b_step", "epoch": epoch + 1, "step": step + 1,
                    "loss": loss.item(), "cos_vt": stats["avg_cos_vt"].item(),
                    "cos_vl": stats["avg_cos_vl"].item(), "cos_tl": stats["avg_cos_tl"].item(),
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")

            pbar.close()
            avg_loss = float(np.mean(epoch_losses))
            avg_vt = float(np.mean(epoch_cos_vt))
            avg_vl = float(np.mean(epoch_cos_vl))
            avg_tl = float(np.mean(epoch_cos_tl))

            print(f"  [1b] Epoch {epoch + 1:3d}  loss: {avg_loss:.4f}  "
                  f"cos_vt: {avg_vt:.3f}  cos_vl: {avg_vl:.3f}  cos_tl: {avg_tl:.3f}")

            wandb_trainer.log({
                "phase": "1b_mixer", "epoch": epoch + 1, "loss": avg_loss,
                "cos_vt": avg_vt, "cos_vl": avg_vl, "cos_tl": avg_tl,
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
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ALIGN Contrastive Pretraining — HDF5 data, 2-phase: encoder → mixer"
    )
    parser.add_argument("--data", required=True, help="Path to align.h5 dataset")
    parser.add_argument("--output-dir", default="./checkpoints/pretrain")
    parser.add_argument("--epochs-encoder", type=int, default=40,
                        help="Phase 1a: encoder pretrain epochs")
    parser.add_argument("--epochs-mixer", type=int, default=10,
                        help="Phase 1b: mixer warm-up epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--no-text", action="store_true", dest="no_text",
                        help="Disable text modality")
    parser.add_argument("--traj-window", type=int, default=20,
                        help="Trajectory window size K_traj")
    parser.add_argument("--frames-per-ep", type=int, default=8)
    parser.add_argument("--episodes-per-batch", type=int, default=8)
    parser.add_argument("--mixer-dim", type=int, default=512)
    parser.add_argument("--num-mixer-blocks", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 for HDF5 works fine)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 autocast (on by default)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable BF16 autocast")
    parser.add_argument("--encoder-checkpoint",
                        help="Resume Phase 1b from encoder checkpoint (skips Phase 1a)")
    parser.add_argument("--stages", default="all", choices=["all", "encoder", "mixer"],
                        help="Which phases to run")
    parser.add_argument("--resume", help="Resume from full checkpoint (deprecated, use --encoder-checkpoint)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-pretrain")
    parser.add_argument("--wandb-run", default=None)

    args = parser.parse_args()

    pretrain_hdf5(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs_encoder=args.epochs_encoder,
        epochs_mixer=args.epochs_mixer,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch,
        checkpoint_every=args.checkpoint_every,
        val_split=args.val_split,
        val_every=args.val_every,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        enable_wandb=args.wandb,
        num_workers=args.num_workers,
        traj_window=args.traj_window,
        frames_per_ep=args.frames_per_ep,
        episodes_per_batch=args.episodes_per_batch,
        use_text=not args.no_text,
        mixer_dim=args.mixer_dim,
        num_mixer_blocks=args.num_mixer_blocks,
        use_bf16=args.bf16,
        encoder_checkpoint=args.encoder_checkpoint,
        resume=args.resume,
        stages=args.stages,
    )


if __name__ == "__main__":
    main()