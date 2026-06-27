#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN GAIL Discriminator training.

Trains the GAIL discriminator D(s, a) -> P(expert) on top of a FROZEN
pretrained encoder + mixer. Used as the learned reward signal for the
alpha pipeline.

Per docs/ALPHA_INTERVENTION_DESIGN.md, this is Stage 1 of the alpha
pipeline. After training, the reward is:

    r(s, a) = -log(1 - D(s, a))        # GAIL reward (Ho & Ermon, 2016)

This reward is then used to train the value head V(s) in Stage 2,
which in turn drives the alpha intervention score.

Usage:
    python training/train_gail.py \\
        --data /path/to/libero.h5 \\
        --pretrained checkpoints/pretrain/libero_90/run_7/best.pt \\
        --output-dir ./checkpoints/gail \\
        --epochs 20 --batch-size 64 --lr 1e-3

For each batch:
  - Get expert (s, a) from the dataset
  - Get rollout (s, a_random) by sampling random actions
  - Pass both through the discriminator
  - D(expert) -> 1, D(rollout) -> 0
  - BCE loss

NOTE on "rollout" data: This first version samples random actions from
the empirical action distribution (Gaussian with mean/std computed on
the dataset). World-model rollouts are a planned follow-up that should
give a stronger learning signal — the discriminator will then learn to
distinguish expert transitions from model-imagined transitions, not
just from noise.
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
from models.gail_discriminator import (
    create_gail_discriminator,
    gail_loss,
    compute_reward,
)
from training.wandb_utils import init_wandb

# world_model_collate yields the same (frame_t, traj_t, action, text, ...)
# schema we need for the discriminator — its expert (s, a) pairs are
# exactly the LIBERO demonstration transitions.
try:
    from data.align_dataset import world_model_collate  # noqa: F401
    _HAS_WM_COLLATE = True
except ImportError:
    _HAS_WM_COLLATE = False

from data.align_dataset import ALIGNDataset, MultiALIGNDataset


# ================================================================
# Empirical action distribution (for random-rollout generation)
# ================================================================

def compute_action_stats(
    data_paths: List[str],
    action_dim: int = 6,
    max_episodes: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean / std of actions across the dataset(s).

    Used to draw "rollout" actions from a Gaussian fit to the empirical
    action distribution. This is a stronger baseline than uniform random
    because it preserves the action scale and correlation-free marginals
    that the expert actually used.

    Args:
        data_paths: one or more HDF5 dataset paths.
        action_dim: number of action dims to keep (default 6 = OSC_POSE).
        max_episodes: cap episodes scanned per dataset for speed.

    Returns:
        (mean, std) each shape (action_dim,).
    """
    all_actions = []
    for path in data_paths:
        ds = ALIGNDataset(path, mode="head", traj_window=1)
        n = min(len(ds), max_episodes)
        for i in range(n):
            item = ds[i]
            acts = item.get("actions", None)
            if acts is None:
                continue
            acts = np.asarray(acts, dtype=np.float32)
            if acts.ndim == 1:
                acts = acts[None, :]
            if acts.shape[-1] > action_dim:
                acts = acts[:, :action_dim]
            elif acts.shape[-1] < action_dim:
                pad = np.zeros((acts.shape[0], action_dim - acts.shape[-1]),
                               dtype=np.float32)
                acts = np.concatenate([acts, pad], axis=-1)
            all_actions.append(acts)
    if not all_actions:
        # Sensible fallback: OSC_POSE deltas are small.
        return np.zeros(action_dim, dtype=np.float32), \
               np.ones(action_dim, dtype=np.float32) * 0.05
    all_actions = np.concatenate(all_actions, axis=0)
    mean = all_actions.mean(axis=0).astype(np.float32)
    std = all_actions.std(axis=0).astype(np.float32) + 1e-6
    return mean, std


# ================================================================
# Training
# ================================================================

def train_gail(
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
    wandb_project: str = "align-gail",
    wandb_run: Optional[str] = None,
    enable_wandb: bool = False,
    num_workers: int = 0,
    traj_window: int = 5,
    chunk_size: int = 1,
    mixer_dim: int = 512,
    num_mixer_blocks: int = 2,
    use_bf16: bool = True,
    # Discriminator arch + kwargs
    arch: str = "mlp",
    action_dim: int = 6,
    embed_dim: int = 256,
    mlp_hidden: int = 512,
    mlp_layers: int = 3,
    mlp_dropout: float = 0.0,
    transformer_layers: int = 2,
    transformer_d_model: int = 384,
    transformer_nhead: int = 4,
    transformer_dropout: float = 0.0,
    transformer_dim_ff: int = 1024,
    # Random-rollout controls
    rollout_mode: str = "gaussian",     # "gaussian" | "uniform" | "fixed"
    rollout_noise_scale: float = 1.0,  # multiplier on empirical std
    seed: int = 42,
) -> str:
    """Train a GAIL discriminator on top of a frozen pretrained encoder+mixer.

    Returns the path to the best discriminator checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- Pick the right collate function ----------------------
    # world_model_collate yields (frame_t, traj_t, action, frame_next,
    # traj_next, text, ep_idx) — we use the (frame_t, traj_t, action,
    # text) subset for the discriminator. Trajectory window length
    # matches the world model's, which is fine: the discriminator sees
    # the same s encoding that downstream consumers will use.
    if _HAS_WM_COLLATE:
        from data.align_dataset import world_model_collate as wm_collate
        collate_fn = lambda b: wm_collate(b, traj_window=traj_window)
        print(f"  Collate: data.align_dataset.world_model_collate(traj_window={traj_window})")
    else:
        raise ImportError(
            "world_model_collate is required for train_gail.py. "
            "It should be in data/align_dataset.py."
        )

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

    print(f"=== ALIGN GAIL Discriminator Training ===")
    print(f"  Run:        {out_dir}")
    print(f"  Data ({len(data_paths)}): {data_paths}")
    print(f"  Pretrained: {pretrained_checkpoint}")
    print(f"  Arch:       {arch}  (action_dim={action_dim}, embed_dim={embed_dim})")
    print(f"  Epochs:     {epochs}  lr={lr}  bs={batch_size}  wd={weight_decay}")
    print(f"  Device:     {device}")
    print(f"  Rollout:    mode={rollout_mode}  noise_scale={rollout_noise_scale}")

    # -- W&B -------------------------------------------------
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run or out_dir.name,
        config={
            "model": "align-gail",
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
            "mlp_dropout": mlp_dropout,
            "transformer_layers": transformer_layers,
            "transformer_d_model": transformer_d_model,
            "transformer_nhead": transformer_nhead,
            "transformer_dropout": transformer_dropout,
            "transformer_dim_ff": transformer_dim_ff,
            "rollout_mode": rollout_mode,
            "rollout_noise_scale": rollout_noise_scale,
            "device": str(device),
            "use_bf16": use_bf16,
            "seed": seed,
        },
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run or out_dir.name, config={})
    print(f"  W&B:        {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Empirical action distribution ------------------------
    print(f"  Computing empirical action distribution...")
    act_mean, act_std = compute_action_stats(
        data_paths, action_dim=action_dim
    )
    print(f"    mean: {np.round(act_mean, 4).tolist()}")
    print(f"    std:  {np.round(act_std, 4).tolist()}")

    # -- Dataset --------------------------------------------
    if len(data_paths) == 1:
        full_ds = ALIGNDataset(data_paths[0], mode="head", traj_window=traj_window)
    else:
        full_ds = MultiALIGNDataset(
            data_paths, mode="head", traj_window=traj_window
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

    # -- Frozen ALIGNModel (encoder + mixer only) ------------
    print(f"\n  Loading ALIGNModel from {pretrained_checkpoint} ...")
    align = ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=mixer_dim,
        num_mixer_blocks=num_mixer_blocks,
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

    # -- Discriminator head ----------------------------------
    disc_kwargs: dict = {}
    if arch == "mlp":
        disc_kwargs = {
            "hidden_dim": mlp_hidden,
            "num_layers": mlp_layers,
            "dropout": mlp_dropout,
        }
    elif arch == "transformer":
        disc_kwargs = {
            "d_model": transformer_d_model,
            "nhead": transformer_nhead,
            "num_layers": transformer_layers,
            "dim_feedforward": transformer_dim_ff,
            "dropout": transformer_dropout,
        }
    else:
        raise ValueError(f"Unknown arch: {arch} (expected 'mlp' or 'transformer')")

    discriminator = create_gail_discriminator(
        arch=arch,
        embed_dim=embed_dim,
        action_dim=action_dim,
        **disc_kwargs,
    ).to(device)

    trainable = list(discriminator.parameters())
    print(f"  Discriminator ({arch}): {sum(p.numel() for p in trainable):,} trainable params")

    opt = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    log_path = out_dir / "gail_log.jsonl"
    log_fp = open(log_path, "w")

    # -- Checkpointing helper --------------------------------
    config_snapshot = {
        "arch": arch,
        "embed_dim": embed_dim,
        "action_dim": action_dim,
        "mlp_hidden": mlp_hidden,
        "mlp_layers": mlp_layers,
        "mlp_dropout": mlp_dropout,
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
        "rollout_mode": rollout_mode,
        "rollout_noise_scale": rollout_noise_scale,
    }

    def save_checkpoint(path: Path, epoch: int, loss: float) -> None:
        torch.save({
            "discriminator_state": discriminator.state_dict(),
            "config": config_snapshot,
            "epoch": epoch,
            "loss": loss,
        }, str(path))

    def sample_rollout_actions(B: int) -> torch.Tensor:
        """Draw B random actions for the rollout (s, a_random) pair.

        Modes:
          - "gaussian": N(act_mean, act_std * noise_scale)  — empirical
          - "uniform":  U(act_mean - 3*act_std, act_mean + 3*act_std)
          - "fixed":    act_mean (constant) — degenerate sanity test

        The state s is reused from the expert batch — only the action
        is randomized. This is the simplest "expert-incorrect-action"
        baseline; world-model rollouts (state s' is also rolled out) are
        a planned follow-up.
        """
        mean = torch.from_numpy(act_mean).to(device)
        std = torch.from_numpy(act_std).to(device) * rollout_noise_scale
        if rollout_mode == "gaussian":
            eps = torch.randn(B, action_dim, device=device)
            return mean + std * eps
        elif rollout_mode == "uniform":
            lo = (mean - 3 * std).unsqueeze(0).expand(B, -1)
            hi = (mean + 3 * std).unsqueeze(0).expand(B, -1)
            return lo + (hi - lo) * torch.rand(B, action_dim, device=device)
        elif rollout_mode == "fixed":
            return mean.unsqueeze(0).expand(B, -1).clone()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode}")

    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(epochs):
        discriminator.train()
        epoch_losses: List[float] = []
        epoch_exp_acc: List[float] = []
        epoch_rol_acc: List[float] = []
        epoch_reward_exp: List[float] = []
        epoch_reward_rol: List[float] = []

        progress = train_loader
        if max_steps_per_epoch and max_steps_per_epoch < len(train_loader):
            progress = (
                p for p, _ in zip(train_loader, range(max_steps_per_epoch))
            )
        progress_bar = tqdm(
            progress,
            total=min(max_steps_per_epoch, len(train_loader))
                  if max_steps_per_epoch else len(train_loader),
            desc=f"[GAIL] Epoch {epoch+1}/{epochs}",
            unit="step",
        )

        for step, batch in enumerate(progress_bar):
            if step >= max_steps_per_epoch:
                break

            # -- To device --
            # world_model_collate schema: frame_t (B, K, H, W, 3), traj_t (B, K, 6),
            # action, frame_next, traj_next, text, ep_idx.
            # We only need (frame_t, traj_t, action, text) — use the LAST frame
            # in the window (current timestep).
            frame_t = torch.from_numpy(batch["frame_t"][:, -1]).to(device)  # (B, H, W, 3)
            traj_t = torch.from_numpy(batch["traj_t"]).float().to(device)   # (B, K, 6)
            action_exp = torch.from_numpy(batch["action"]).float().to(device)
            texts = batch["text"]
            B = frame_t.shape[0]

            # -- Encode state_t via frozen ALIGNModel --
            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                mixed_t = align.encode_mixed(frame_t, traj_t, texts)
                z_v = mixed_t["z_v"].float()        # (B, D)
                z_t = mixed_t["z_t"].float()        # (B, D)
                z_text = mixed_t["z_text"].float()  # (B, D)

            # -- Sample rollout actions (same states, different a) --
            action_rol = sample_rollout_actions(B)

            # -- Discriminator logits --
            expert_logits = discriminator(z_v, z_t, z_text, action_exp)
            rollout_logits = discriminator(z_v, z_t, z_text, action_rol)

            # -- BCE loss (expert -> 1, rollout -> 0) --
            loss, exp_acc, rol_acc = gail_loss(expert_logits, rollout_logits)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            with torch.no_grad():
                # Diagnostic: track the actual reward magnitudes (post-sigmoid,
                # post-softplus). Both should be > 0 and the expert reward
                # should rise faster than the rollout reward.
                r_exp = compute_reward(expert_logits).mean().item()
                r_rol = compute_reward(rollout_logits).mean().item()

            epoch_losses.append(loss.item())
            epoch_exp_acc.append(exp_acc.item())
            epoch_rol_acc.append(rol_acc.item())
            epoch_reward_exp.append(r_exp)
            epoch_reward_rol.append(r_rol)

            if step % 10 == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    exp_acc=f"{exp_acc.item():.3f}",
                    rol_acc=f"{rol_acc.item():.3f}",
                    r_exp=f"{r_exp:.3f}",
                    r_rol=f"{r_rol:.3f}",
                )

        # -- End-of-epoch summary ----------------------------
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        avg_exp_acc = float(np.mean(epoch_exp_acc)) if epoch_exp_acc else 0.0
        avg_rol_acc = float(np.mean(epoch_rol_acc)) if epoch_rol_acc else 0.0
        avg_r_exp = float(np.mean(epoch_reward_exp)) if epoch_reward_exp else 0.0
        avg_r_rol = float(np.mean(epoch_reward_rol)) if epoch_reward_rol else 0.0
        elapsed = time.time() - t_start

        print(
            f"  Epoch {epoch+1:3d}/{epochs}  "
            f"loss: {avg_loss:.4f}  exp_acc: {avg_exp_acc:.3f}  rol_acc: {avg_rol_acc:.3f}  "
            f"r_exp: {avg_r_exp:.3f}  r_rol: {avg_r_rol:.3f}  "
            f"elapsed: {elapsed:.0f}s"
        )

        wandb_trainer.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/expert_acc": avg_exp_acc,
            "train/rollout_acc": avg_rol_acc,
            "train/reward_expert": avg_r_exp,
            "train/reward_rollout": avg_r_rol,
        }, step=epoch + 1)

        log_fp.write(json.dumps({
            "epoch": epoch + 1,
            "stage": "train",
            "loss": avg_loss,
            "expert_acc": avg_exp_acc,
            "rollout_acc": avg_rol_acc,
            "reward_expert": avg_r_exp,
            "reward_rollout": avg_r_rol,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        log_fp.flush()

        # -- Per-epoch checkpoint -----------------------------
        epoch_path = out_dir / f"gail_epoch_{epoch+1:04d}.pt"
        save_checkpoint(epoch_path, epoch + 1, avg_loss)

        # -- Best checkpoint ----------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = out_dir / "gail_best.pt"
            save_checkpoint(best_path, epoch + 1, avg_loss)
            print(f"  -> gail_best.pt (loss: {avg_loss:.4f})")
            # Upload best checkpoint to wandb as an artifact
            wandb_trainer.save(str(best_path))

    log_fp.close()
    wandb_trainer.finish()

    print(f"\n  GAIL discriminator training complete.")
    print(f"    Best loss:    {best_loss:.4f}")
    print(f"    Best ckpt:    {out_dir / 'gail_best.pt'}")
    print(f"    Epoch ckpts:  {out_dir}/gail_epoch_NNNN.pt")
    print(f"    Log:          {log_path}")

    return str(out_dir / "gail_best.pt")


# ================================================================
# CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALIGN GAIL discriminator training (frozen encoder + mixer; "
                    "trains D(s, a) -> P(expert) head)."
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s). Pass multiple to "
                             "train on the concatenation.")
    parser.add_argument("--pretrained", required=True,
                        help="Phase 1b pretrained checkpoint (encoder + mixer).")
    parser.add_argument("--output-dir", default="./checkpoints/gail",
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

    # Discriminator arch + kwargs
    parser.add_argument("--arch", default="mlp", choices=["mlp", "transformer"],
                        help="Discriminator architecture.")
    parser.add_argument("--action-dim", type=int, default=6,
                        help="Action dim (default 6 for OSC_POSE delta).")
    parser.add_argument("--embed-dim", type=int, default=256,
                        help="Per-modality embedding dim (must match pretrained).")
    parser.add_argument("--mlp-hidden", type=int, default=512,
                        help="MLP discriminator hidden dim.")
    parser.add_argument("--mlp-layers", type=int, default=3,
                        help="MLP discriminator num layers.")
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-d-model", type=int, default=384)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-dim-ff", type=int, default=1024)

    # Random-rollout controls
    parser.add_argument("--rollout-mode", default="gaussian",
                        choices=["gaussian", "uniform", "fixed"],
                        help="How to sample rollout actions. "
                             "'gaussian' uses empirical N(mu, sigma). "
                             "'uniform' uses U(mu - 3*sigma, mu + 3*sigma). "
                             "'fixed' uses mu (sanity check).")
    parser.add_argument("--rollout-noise-scale", type=float, default=1.0,
                        help="Multiplier on empirical action std.")

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-gail")
    parser.add_argument("--wandb-run", default=None)

    args = parser.parse_args()

    train_gail(
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
        arch=args.arch,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        mlp_hidden=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        mlp_dropout=args.mlp_dropout,
        transformer_layers=args.transformer_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_dropout=args.transformer_dropout,
        transformer_dim_ff=args.transformer_dim_ff,
        rollout_mode=args.rollout_mode,
        rollout_noise_scale=args.rollout_noise_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
