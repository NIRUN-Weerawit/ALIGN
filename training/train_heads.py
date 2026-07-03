#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN head training — independent stages so neither head interferes.

Phase 2a (Decision): Train alpha prediction with BCE only.
Phase 2b (Assistant): Train delta-pose correction with MSE only.

Usage:
    # Both stages in sequence (default)
    python training/train_heads.py \\\n        --data ./align.h5 \\\n        --pretrained ./checkpoints/pretrain/best.pt \\\n        --output-dir ./checkpoints/heads \\\n        --epochs-decision 10 \\\n        --epochs-assistant 10

    # Decision head only
    python training/train_heads.py --data ./align.h5 \\\n        --pretrained ./checkpoints/pretrain/best.pt \\\n        --stage decision \\\n        --epochs-decision 10
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
# Common helper to freeze / unfreeze individual heads
# ================================================================

def _freeze_module(module):
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_module(module):
    """Unfreeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = True


# ================================================================
# Training
# ================================================================

def train_heads_hdf5(
    data_paths: List[str],
    pretrained_checkpoint: str,
    output_dir: str,
    epochs_decision: int = 10,
    epochs_assistant: int = 10,
    batch_size: int = 64,
    lr_decision: float = 5e-4,
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
    decision_noise_std: float = 0.02,
    decision_arch: str = "transformer",
    mlp_hidden_dim: int = 512,
    mlp_layers: int = 3,
    transformer_layers: int = 2,
    transformer_d_model: int = 384,
    transformer_nhead: int = 4,
    transformer_dropout: float = 0.0,
    transformer_dim_ff: int = 1024,
    loss_decay: float = 0.7,
    warmup_epochs: int = 0,
    assistant_hidden: int = 256,
    assistant_layers: int = 2,
    assistant_dropout: float = 0.0,
) -> str:
    """Train Decision and Assistant heads independently.

    Phase 2a (Stage A): Future Prediction (cosine loss) — freezes assistant.
      Past trajectory is noised with decision_noise_std to simulate
      human deviation. Future trajectory is clean (the target).
      The model learns to predict the clean future from the noised past.
      When the past is heavily noised, prediction error is high → α is low
      (model abstains). When the past matches the optimal, prediction is
      accurate → α is high (model helps fully).
    Phase 2b (Stage B): MSE on delta (freezes decision).
    Neither head gradient affects the other.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Derive subdirectory: checkpoints/heads_local/libero_align/run_N/
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

    print(f"=== ALIGN Head Training (independent stages) ===")
    print(f"  Run:          {out_dir}")
    print(f"  Data ({len(data_paths)}): {data_paths}")
    print(f"  Pretrained:   {pretrained_checkpoint}")
    print(f"  Device:       {device}")
    print(f"  Stage A (α):  {epochs_decision} epochs, lr={lr_decision}")
    print(f"  Stage B (Δ):  {epochs_assistant} epochs, lr={lr_assistant}")

    # -- W&B ──
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config={
            "model": "align-heads-independent",
            "data": [str(p) for p in data_paths],
            "pretrained_checkpoint": pretrained_checkpoint,
            "epochs_decision": epochs_decision,
            "epochs_assistant": epochs_assistant,
            "batch_size": batch_size,
            "lr_decision": lr_decision,
            "lr_assistant": lr_assistant,
            "weight_decay": weight_decay,
            "chunk_size": chunk_size,
            "device": str(device),
            "use_bf16": use_bf16,
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:          {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Dataset ──
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
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size),
        num_workers=num_workers,
    )

    # -- Model ──
    # decision_K and decision_arch come from CLI args. The future prediction
    # head receives K past embeddings matching what the dataset provides.
    # Pass arch-specific kwargs to the head constructor.
    head_kwargs = {}
    if decision_arch == "mlp":
        head_kwargs = {
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_num_layers": mlp_layers,
        }
    elif decision_arch == "transformer":
        head_kwargs = {
            "num_layers": transformer_layers,
            "d_model": transformer_d_model,
            "nhead": transformer_nhead,
            "dropout": transformer_dropout,
            "dim_feedforward": transformer_dim_ff,
        }

    num_cameras = len(cameras) if cameras else 1
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
        decision_K=chunk_size,
        decision_arch=decision_arch,
        num_cameras=num_cameras,
        **head_kwargs,
        assistant_hidden=assistant_hidden,
        assistant_layers=assistant_layers,
        assistant_dropout=assistant_dropout,
    ).to(device)

    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    
    # Freeze backbones + encoders + mixer permanently
    model.freeze_backbone()
    model.freeze_all_encoders()  
    print(f"  Loaded pretrained backbone from {pretrained_checkpoint}")
    print(f"  All encoders + mixer frozen — only heads train")

    log_path = out_dir / "head_log.jsonl"
    log_fp = open(log_path, "w")

    # ================================================================
    # Phase 2a: Decision head (Future Prediction) — cosine loss only
    # ================================================================
    print("\n=== Stage A: Training Decision Head (Future Prediction) ===")

    # Freeze assistant, unfreeze decision
    _freeze_module(model.assistant_head)
    _unfreeze_module(model.decision_head)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    opt_a = optim.AdamW(trainable, lr=lr_decision, weight_decay=weight_decay)
    # Optional linear warmup over the first `warmup_epochs` epochs
    scheduler_a = None
    if warmup_epochs > 0:
        warmup_lambda = lambda e: min(1.0, (e + 1) / warmup_epochs)
        scheduler_a = optim.lr_scheduler.LambdaLR(opt_a, warmup_lambda)
    best_loss_a = float("inf")

    for epoch in range(epochs_decision):
        if scheduler_a is not None:
            scheduler_a.step()
        model.train()
        losses_a, errors_v, errors_t = [], [], []

        # Wrap loader with tqdm for progress display
        progress = train_loader
        if max_steps_per_epoch and max_steps_per_epoch < len(train_loader):
            progress = (
                p for p, _ in zip(train_loader, range(max_steps_per_epoch))
            )
        progress_bar = tqdm(
            progress,
            total=min(max_steps_per_epoch, len(train_loader)) if max_steps_per_epoch else len(train_loader),
            desc=f"[Decision] Epoch {epoch+1}/{epochs_decision}",
            unit="step",
        )

        for step, batch in enumerate(progress_bar):
            if step >= max_steps_per_epoch:
                break

            frames = torch.from_numpy(batch["frames"]).to(device)
            texts = batch["texts"]
            traj_view = torch.from_numpy(batch["trajectory"]).float().to(device)
            traj_future = torch.from_numpy(batch["trajectory_future"]).float().to(device)

            # ── Inject noise into the past trajectory ──
            # This simulates a noisy human teleoperator. The model must
            # learn to predict the CLEAN future from the NOISED past.
            # When the past is heavily noised, prediction is hard → α is low
            # (model abstains). When the past matches the optimal, prediction
            # is easy → α is high (model helps fully).
            if decision_noise_std > 0:
                # Per-step Gaussian noise on position; rotation noise scaled
                # to match the libero_spatial action scale.
                noise = torch.randn_like(traj_view) * decision_noise_std
                # Position dims (0-2): noise as-is (meters)
                # Orientation dims (3-5): scale up to match rotation noise
                noise[:, :, 3:6] *= 10.0  # 0.02m → 0.2 rad of axis-angle
                traj_view_noisy = traj_view + noise
            else:
                traj_view_noisy = traj_view

            # Frozen encodings via mixer. We need per-token trajectory
            # embeddings (z_t_tokens) for the future prediction head.
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # Encode the NOISED past (input to the model)
                mixed = model.encode_mixed(frames, traj_view_noisy, texts)
                z_v = mixed["z_v"].float()           # (B, D) — current vision
                z_t_tokens = mixed["z_t_tokens"].float()  # (B, K, D) — per-step
                z_text = mixed["z_text"].float()     # (B, D)

            # For the future prediction head:
            #   Input: K past (z_v, z_t) embeddings + z_text
            #   Output: K predicted (z_v, z_t) embeddings
            #   Target: actual next K (z_v, z_t) embeddings
            B, K, D = z_t_tokens.shape
            z_v_window = z_v.unsqueeze(1).expand(-1, K, -1)  # (B, K, D) — current vision, broadcast
            z_t_target_tokens = z_t_tokens  # (B, K, D) — past K traj tokens (we'll encode future separately)

            # We also need the actual FUTURE trajectory embeddings as targets.
            # Encode the future trajectory through the frozen encoder+mixer.
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # Encode future trajectory separately
                mixed_future = model.encode_mixed(frames, traj_future, texts)
                z_t_future_tokens = mixed_future["z_t_tokens"].float()  # (B, K, D)

            # Vision target is the current vision (broadcast) — since we
            # only have one frame, the model learns to "copy" it
            z_v_target = z_v.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)

            # Run the future prediction head
            predicted_z_v, predicted_z_t = model.decision_head(
                z_v_window, z_t_tokens, z_text
            )

            # Cosine loss (bounded [0, 2])
            loss = ALIGNModel.future_prediction_loss(
                predicted_z_v, predicted_z_t, z_v_target, z_t_future_tokens,
                decay=loss_decay,
            )

            opt_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt_a.step()

            losses_a.append(loss.item())
            with torch.no_grad():
                err_v = (1 - F.cosine_similarity(predicted_z_v, z_v_target, dim=-1)).mean().item()
                err_t = (1 - F.cosine_similarity(predicted_z_t, z_t_future_tokens, dim=-1)).mean().item()
            errors_v.append(err_v)
            errors_t.append(err_t)

            # Update progress bar
            if step % 10 == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    err_v=f"{err_v:.3f}",
                    err_t=f"{err_t:.3f}",
                )

        avg_loss = float(np.mean(losses_a))
        av_err_v = float(np.mean(errors_v))
        av_err_t = float(np.mean(errors_t))

        print(
            f"  [α-future] Epoch {epoch+1:3d}/{epochs_decision}  "
            f"loss: {avg_loss:.4f}  err_v: {av_err_v:.3f}  err_t: {av_err_t:.3f}"
        )

        wandb_trainer.log({
            "stage": "decision",
            "loss": avg_loss,
            "err_v": av_err_v,
            "err_t": av_err_t,
            "epoch": epoch + 1,
        }, step=epoch + 1)

        log_fp.write(json.dumps({
            "stage": "decision", "epoch": epoch + 1,
            "loss": avg_loss, "err_v": av_err_v, "err_t": av_err_t,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        if avg_loss < best_loss_a:
            best_loss_a = avg_loss
            model.save_heads_checkpoint(
                str(out_dir / "decision_best.pt"), epoch, avg_loss,
                opt_a.state_dict(), {"chunk_size": chunk_size, "decision_K": model.decision_K}
            )
            print(f"  -> decision_best.pt (loss: {avg_loss:.4f})")

    print(f"\n  Stage A complete. Best loss: {best_loss_a:.4f}")

    # ================================================================
    # Phase 2b: Assistant head (delta pose) — MSE only
    # ================================================================
    print("\n=== Stage B: Training Assistant Head (Δpose) ===")

    # Freeze decision, unfreeze assistant
    _freeze_module(model.decision_head)
    _unfreeze_module(model.assistant_head)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    opt_b = optim.AdamW(trainable, lr=lr_assistant, weight_decay=weight_decay)
    best_loss_b = float("inf")

    for epoch in range(epochs_assistant):
        model.train()
        losses_b, deltas_pred = [], []

        # Wrap loader with tqdm for progress display
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

            frames = torch.from_numpy(batch["frames"]).to(device)
            texts = batch["texts"]
            delta_t = torch.from_numpy(batch["delta_target"]).float().to(device)
            traj_view = torch.from_numpy(batch["trajectory"]).float().to(device)

            # Frozen encodings via mixer
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frames, traj_view, texts)
                z_v = mixed["z_v"].float()
                z_t = mixed["z_t"].float()
                z_text = mixed["z_text"].float()

            # Assistant loss only (MSE against dynamic delta target)
            # current_action = torch.from_numpy(batch["current_action"]).float().to(device)
            delta_pred = model.assistant_head(z_v, z_t, z_text)
            loss_mse = F.mse_loss(delta_pred, delta_t)

            opt_b.zero_grad()
            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt_b.step()

            losses_b.append(loss_mse.item())
            deltas_pred.append(delta_pred.detach().abs().mean().item())

            # Update progress bar
            if step % 10 == 0:
                progress_bar.set_postfix(
                    mse=f"{loss_mse.item():.5f}",
                    d_mean=f"{delta_pred.detach().abs().mean().item():.4f}",
                )

        avg_loss = float(np.mean(losses_b))
        av_delta = float(np.mean(deltas_pred))

        print(f"  [Δ] Epoch {epoch+1:3d}/{epochs_assistant}  MSE: {avg_loss:.4f}  Δ_mean: {av_delta:.4f}")

        wandb_trainer.log({
            "stage": "assistant",
            "loss": avg_loss,
            "delta mean": av_delta,
            "epoch": epoch + 1,
        }, step=epochs_decision + epoch + 1)

        log_fp.write(json.dumps({
            "stage": "assistant", "epoch": epoch + 1,
            "loss": avg_loss, "delta_mean": av_delta,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        if avg_loss < best_loss_b:
            best_loss_b = avg_loss
            model.save_heads_checkpoint(
                str(out_dir / "assistant_best.pt"), epoch, avg_loss,
                opt_b.state_dict(), {"chunk_size": chunk_size})
            print(f"  -> assistant_best.pt (loss: {avg_loss:.4f})")

    print(f"\n  Stage B complete. Best loss: {best_loss_b:.4f}")

    # ================================================================
    # Save final combined checkpoint (both heads ready for inference)
    # ================================================================
    model.train()  # unfreeze everything for a proper save
    _unfreeze_module(model.decision_head)
    _unfreeze_module(model.assistant_head)
    
    model.save_heads_checkpoint(
        str(out_dir / "heads_best.pt"), epochs_decision + epochs_assistant - 1,
        best_loss_a + best_loss_b, opt_b.state_dict(), {"chunk_size": chunk_size})

    log_fp.close()
    print(f"\n  Head training complete.")
    print(f"    Decision:   {out_dir / 'decision_best.pt'}")
    print(f"    Assistant:  {out_dir / 'assistant_best.pt'}")
    print(f"    Combined:   {out_dir / 'heads_best.pt'}")
    print(f"    Logs:       {log_path}")

    wandb_trainer.finish()
    return str(out_dir / "heads_best.pt")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ALIGN Head Training (independent stages: Decision → Assistant)")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to align.h5 dataset(s). Pass multiple "
                             "to train on the concatenation.")
    parser.add_argument("--pretrained", required=True, help="Phase 1 pretrained checkpoint")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views to use (e.g. 'wrist_image image'). "
                             "Must match the cameras used during pretrain.")
    parser.add_argument("--output-dir", default="./checkpoints/heads")
    parser.add_argument("--epochs-decision", type=int, default=10,
                        help="Decision (alpha) training epochs")
    parser.add_argument("--epochs-assistant", type=int, default=10,
                        help="Assistant (delta-pose) training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-decision", type=float, default=5e-4,
                        help="Learning rate for Decision head")
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
    parser.add_argument("--decision-noise-std", type=float, default=0.02,
                        help="Std of noise injected into the past trajectory "
                             "when training the Decision (future prediction) head. "
                             "Simulates human deviation. 0 = no noise.")
    parser.add_argument("--decision-arch", default="transformer",
                        choices=["mlp", "transformer"],
                        help="Architecture for the Decision (future prediction) head")
    parser.add_argument("--mlp-hidden", type=int, default=512,
                        help="MLP head hidden dim (only used if --decision-arch=mlp)")
    parser.add_argument("--mlp-layers", type=int, default=3,
                        help="MLP head number of layers (only used if --decision-arch=mlp)")
    parser.add_argument("--transformer-layers", type=int, default=2,
                        help="Transformer head number of layers (only used if --decision-arch=transformer)")
    parser.add_argument("--transformer-d-model", type=int, default=384,
                        help="Transformer head model dim (only used if --decision-arch=transformer)")
    parser.add_argument("--transformer-nhead", type=int, default=4,
                        help="Transformer head attention heads (only used if --decision-arch=transformer)")
    parser.add_argument("--transformer-dropout", type=float, default=0.0,
                        help="Transformer head dropout (only used if --decision-arch=transformer)")
    parser.add_argument("--transformer-dim-ff", type=int, default=1024,
                        help="Transformer head feedforward dim (only used if --decision-arch=transformer)")
    parser.add_argument("--loss-decay", type=float, default=0.7,
                        help="Decay for weighted mean of K-step prediction errors")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup over this many epochs")
    parser.add_argument("--assistant-hidden", type=int, default=256,
                        help="Assistant head hidden layer width")
    parser.add_argument("--assistant-layers", type=int, default=2,
                        help="Assistant head number of hidden layers (between input and output)")
    parser.add_argument("--assistant-dropout", type=float, default=0.0,
                        help="Assistant head dropout")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-heads")
    parser.add_argument("--wandb-run", default=None)

    args = parser.parse_args()

    train_heads_hdf5(
        data_paths=args.data,
        pretrained_checkpoint=args.pretrained,
        output_dir=args.output_dir,
        epochs_decision=args.epochs_decision,
        epochs_assistant=args.epochs_assistant,
        batch_size=args.batch_size,
        lr_decision=args.lr_decision,
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
        decision_noise_std=args.decision_noise_std,
        decision_arch=args.decision_arch,
        mlp_hidden_dim=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        transformer_layers=args.transformer_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_dropout=args.transformer_dropout,
        transformer_dim_ff=args.transformer_dim_ff,
        loss_decay=args.loss_decay,
        warmup_epochs=args.warmup_epochs,
        assistant_hidden=args.assistant_hidden,
        assistant_layers=args.assistant_layers,
        assistant_dropout=args.assistant_dropout,
        cameras=args.cameras,
    )


if __name__ == "__main__":
    main()
