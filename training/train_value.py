"""
ALIGN Value Head training.

Trains V(s) on top of a FROZEN pretrained encoder+mixer, using rewards
from a trained GAIL discriminator. The value function is trained with
TD(λ) targets on real expert trajectories.

Per docs/ALPHA_INTERVENTION_DESIGN.md, this is Stage 2 of the alpha
pipeline (after GAIL). The value function V(s) estimates expected
cumulative future reward, and is used to compute alpha:

    alpha = sigmoid((V(s'_m) - V(s'_h)) / tau)

where s'_m = world_model(s, a_m) and s'_h = world_model(s, a_h).

Usage:
    python training/train_value.py \
        --data /path/to/libero.h5 \
        --pretrained checkpoints/pretrain/.../best.pt \
        --gail-checkpoint checkpoints/gail/.../gail_best.pt \
        --output-dir ./checkpoints/value \
        --epochs 30 --batch-size 32 --lr 1e-3 \
        --gamma 0.99 --lam 0.7

Training loop:
  For each batch (transitions from world_model_collate):
    1. Encode s_t and s_{t+1} through frozen encoder+mixer
    2. Compute reward r_t = -log(1 - D(s_t, a_t)) from GAIL
    3. Compute V(s_{t+1}) via the value head
    4. Compute TD target: V_target = r_t + gamma * V(s_{t+1})
    5. Loss = MSE(V(s_t), V_target.detach())

For simplicity this first version uses single-step TD(0) targets.
Full TD(λ) is a follow-up (the helper is in models/value_head.py).

NOTE: The GAIL checkpoint must exist. If you don't have one trained
yet, the script will use a random GAIL (untrained) for smoke testing.
"""

import argparse
import json
import sys
import time
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
from models.value_head import create_value_head, value_loss
from models.gail_discriminator import create_gail_discriminator, compute_reward
from data.align_dataset import ALIGNDataset, MultiALIGNDataset, world_model_collate
from training.wandb_utils import init_wandb, log_metrics


def train_value(
    data_paths: List[str],
    pretrained_checkpoint: str,
    gail_checkpoint: Optional[str],
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gamma: float = 0.99,
    lam: float = 0.7,
    n_steps: int = 1,
    val_split: float = 0.1,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    enable_wandb: bool = False,
    wandb_project: str = "align-value",
    wandb_run: Optional[str] = None,
    num_workers: int = 0,
    traj_window: int = 5,
    chunk_size: int = 1,
    embed_dim: int = 256,
    # Value head arch
    hidden_dim: int = 256,
    num_layers: int = 3,
    use_bf16: bool = True,
    seed: int = 42,
) -> str:
    """Train value head on top of frozen encoder+mixer with GAIL rewards.

    Returns the path to the best value head checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- Build dataset --
    if len(data_paths) == 1:
        ds = ALIGNDataset(data_paths[0], mode="pretrain", traj_window=traj_window)
        ds_name = Path(data_paths[0]).stem
    else:
        ds = MultiALIGNDataset(data_paths, mode="pretrain", traj_window=traj_window)
        ds_name = "+".join(Path(p).stem for p in data_paths)

    # -- Derive output dir --
    base_dir = Path(output_dir) / ds_name
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(base_dir.glob("run_*"))
    max_run = max([int(d.name.split("_")[-1]) for d in existing] + [0])
    out_dir = base_dir / f"run_{max_run + 1}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "value_log.jsonl"
    log_fp = open(log_path, "w")

    print(f"=== ALIGN Value Head Training ===")
    print(f"  Run: {out_dir}")
    print(f"  Data ({len(data_paths)}): {data_paths}")
    print(f"  Encoder: {pretrained_checkpoint}")
    print(f"  GAIL: {gail_checkpoint or '(random init)'}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, gamma={gamma}, lambda={lam}")
    print(f"  Value head: hidden={hidden_dim}, layers={num_layers}")

    # -- W&B -------------------------------------------------
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run or out_dir.name,
        config={
            "model": "align-value",
            "data": [str(p) for p in data_paths],
            "pretrained_checkpoint": pretrained_checkpoint,
            "gail_checkpoint": gail_checkpoint,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "lam": lam,
            "n_steps": n_steps,
            "val_split": val_split,
            "max_steps_per_epoch": max_steps_per_epoch,
            "num_workers": num_workers,
            "traj_window": traj_window,
            "chunk_size": chunk_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "device": str(device),
            "use_bf16": use_bf16,
            "seed": seed,
        },
    ) if enable_wandb else init_wandb(
        project=wandb_project, name=wandb_run or out_dir.name, config={},
    )
    print(f"  W&B:        {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # -- Load encoder+mixer --
    model = ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        traj_d_model=128,
    ).to(device)
    enc_ckpt = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    if "trainable_state_dict" in enc_ckpt:
        model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    elif "model_state_dict" in enc_ckpt:
        model.load_state_dict(enc_ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model.freeze_all_encoders()
    model.eval()
    print(f"  Loaded encoder+mixer from {pretrained_checkpoint}")

    # -- Load GAIL discriminator (for rewards) --
    gail_disc = None
    if gail_checkpoint:
        print(f"  Loading GAIL from {gail_checkpoint}")
        gail_ckpt = torch.load(gail_checkpoint, map_location=device, weights_only=False)
        gail_config = gail_ckpt.get("config", {})
        gail_arch = gail_config.get("arch", "mlp")
        gail_kwargs = {}
        if gail_arch == "mlp":
            gail_kwargs = {
                "hidden_dim": gail_config.get("mlp_hidden_dim", 512),
                "num_layers": gail_config.get("mlp_layers", 3),
            }
        gail_disc = create_gail_discriminator(
            arch=gail_arch,
            embed_dim=gail_config.get("embed_dim", 256),
            action_dim=gail_config.get("action_dim", 6),
            **gail_kwargs,
        ).to(device)
        gail_disc.load_state_dict(gail_ckpt["discriminator_state"])
        gail_disc.eval()
    else:
        print("  WARNING: No GAIL checkpoint provided. Using random GAIL (untrained).")
        gail_disc = create_gail_discriminator(
            arch="mlp", embed_dim=256, action_dim=6,
        ).to(device)
        gail_disc.eval()

    # -- Build value head --
    value_head = create_value_head(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    trainable = [p for p in value_head.parameters() if p.requires_grad]
    print(f"  Value head: {sum(p.numel() for p in trainable):,} trainable params")

    opt = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # -- Build dataloader --
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: world_model_collate(b, traj_window=traj_window),
        num_workers=num_workers,
        drop_last=True,
    )

    # -- Training loop --
    best_loss = float("inf")
    for epoch in range(epochs):
        value_head.train()
        losses, val_v_means, val_r_means = [], [], []

        n_steps = min(max_steps_per_epoch, len(loader)) if max_steps_per_epoch else len(loader)
        progress = tqdm(
            loader,
            total=n_steps,
            desc=f"[Value] Epoch {epoch+1}/{epochs}",
            unit="step",
        )

        for step, batch in enumerate(progress):
            if step >= n_steps:
                break

            # 1. Encode current state s_t
            # world_model_collate returns frame_t as (B, K, H, W, 3) — use last frame
            frame_t = torch.from_numpy(batch["frame_t"][:, -1]).to(device)
            traj_t = torch.from_numpy(batch["traj_t"]).float().to(device)
            action = torch.from_numpy(batch["action"]).float().to(device)
            texts = batch["text"]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed_t = model.encode_mixed(frame_t, traj_t, texts)
            z_v_t = mixed_t["z_v"].float()
            z_t_t = mixed_t["z_t"].float()
            z_text_t = mixed_t["z_text"].float()

            # 2. Compute reward r_t from GAIL
            with torch.no_grad():
                gail_logits = gail_disc(z_v_t, z_t_t, z_text_t, action)
                r_t = compute_reward(gail_logits)  # (B,)

            # 3. Encode next state s_{t+1} (for TD target)
            frame_next = torch.from_numpy(batch["frame_next"]).to(device)
            traj_next = torch.from_numpy(batch["traj_next"]).float().to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed_next = model.encode_mixed(frame_next, traj_next, texts)
            z_v_next = mixed_next["z_v"].float()
            z_t_next = mixed_next["z_t"].float()

            # 4. Compute V(s_t) and the bootstrap V at the look-ahead state
            v_t = value_head(z_v_t, z_t_t, z_text_t)

            # 5. Build the TD target.
            #    - n_steps=1 (default): TD(0) target = r_t + gamma * V(s_{t+1})
            #    - n_steps>1: n-step return, but the world_model_collate only
            #      gives us (s_t, a_t, s_{t+1}) — not s_{t+2..t+n}. For n-step
            #      return we need access to multiple future states, which the
            #      current collate doesn't provide. Fall back to TD(0) and warn.
            if n_steps > 1:
                # Use TD(0) since we don't have multi-step states in the batch.
                # For full TD(lambda) see compute_td_lambda_return in
                # models/value_head.py — requires per-episode data.
                pass  # falls through to TD(0) below
            with torch.no_grad():
                v_next = value_head(z_v_next, z_t_next, z_text_t)
            v_target = r_t + gamma * v_next

            # 6. Loss
            loss = value_loss(v_t, v_target.detach())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            losses.append(loss.item())
            val_v_means.append(v_t.mean().item())
            val_r_means.append(r_t.mean().item())

            progress.set_postfix(
                loss=loss.item(),
                v=v_t.mean().item(),
                r=r_t.mean().item(),
            )

        mean_loss = np.mean(losses)
        mean_v = np.mean(val_v_means)
        mean_r = np.mean(val_r_means)
        log_entry = {
            "stage": "value", "epoch": epoch + 1,
            "loss": mean_loss, "v_mean": mean_v, "r_mean": mean_r,
            "timestamp": datetime.now().isoformat(),
        }
        log_fp.write(json.dumps(log_entry) + "\n")
        log_fp.flush()

        print(f"  Epoch {epoch+1}/{epochs}  loss: {mean_loss:.4f}  V: {mean_v:.4f}  r: {mean_r:.4f}")

        wandb_trainer.log({
            "epoch": epoch + 1,
            "train/loss": mean_loss,
            "train/v_mean": mean_v,
            "train/r_mean": mean_r,
        }, step=epoch + 1)

        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt_path = out_dir / "value_best.pt"
            torch.save({
                "value_head_state": value_head.state_dict(),
                "config": {
                    "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "gamma": gamma,
                    "lam": lam,
                },
                "epoch": epoch + 1,
                "loss": mean_loss,
            }, ckpt_path)
            print(f"  -> value_best.pt (loss: {mean_loss:.4f})")
            # Save the best checkpoint to wandb (uploads as artifact)
            wandb_trainer.save(str(ckpt_path))

    log_fp.close()
    wandb_trainer.finish()
    print(f"\n  Value head training complete.")
    print(f"    Best loss: {best_loss:.4f}")
    print(f"    Best ckpt: {out_dir / 'value_best.pt'}")
    return str(out_dir / "value_best.pt")


def main():
    parser = argparse.ArgumentParser(
        description="ALIGN Value Head training — V(s) on GAIL rewards",
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s)")
    parser.add_argument("--pretrained", required=True,
                        help="Phase 1b encoder+mixer checkpoint (FROZEN)")
    parser.add_argument("--gail-checkpoint", default=None,
                        help="Trained GAIL discriminator (for reward). "
                             "Optional — uses random init if not provided.")
    parser.add_argument("--output-dir", default="./checkpoints/value")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for TD target")
    parser.add_argument("--lam", type=float, default=0.7,
                        help="Trace decay for TD(λ)")
    parser.add_argument("--n-steps", type=int, default=1,
                        help="Look-ahead for the TD target. 1 = TD(0). "
                             "Note: world_model_collate only provides s_{t+1}, "
                             "so n_steps > 1 currently falls back to TD(0). "
                             "For full n-step return, see compute_td_lambda_return.")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--traj-window", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Value head hidden dim")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Value head number of layers")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="align-value",
                        help="W&B project name")
    parser.add_argument("--wandb-run", default=None,
                        help="W&B run name (defaults to run_N)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_value(
        data_paths=args.data,
        pretrained_checkpoint=args.pretrained,
        gail_checkpoint=args.gail_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        lam=args.lam,
        n_steps=args.n_steps,
        val_split=args.val_split,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        num_workers=args.num_workers,
        traj_window=args.traj_window,
        chunk_size=args.chunk_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_bf16=args.bf16,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
