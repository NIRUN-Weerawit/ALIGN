#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate trained Decision and Assistant heads on a validation split.

Usage:
    python eval/eval_heads.py \
        --data /path/to/libero/align.h5 \
        --checkpoint checkpoints/heads_libero/heads_best.pt \
        --traj-window 20 --chunk-size 5 
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from data.align_dataset import ALIGNDataset, head_collate


def evaluate(
    data_paths: List[str],
    heads_checkpoint: str,
    encoder_checkpoint: str = None,
    batch_size: int = 64,
    traj_window: int = 20,
    chunk_size: int = 5,
    val_split: float = 0.1,
    device: str = None,
    use_bf16: bool = True,
    decision_arch: str = "transformer",
    mlp_hidden_dim: int = 512,
    mlp_num_layers: int = 3,
    transformer_layers: int = 2,
    transformer_d_model: int = 384,
    transformer_nhead: int = 4,
    transformer_dropout: float = 0.0,
    transformer_dim_ff: int = 1024,
    assistant_hidden: int = 256,
    assistant_layers: int = 2,
    assistant_dropout: float = 0.0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load the HEADS checkpoint (decision_head + assistant_head only)
    heads_ckpt = torch.load(heads_checkpoint, map_location=device)
    print(f"  Loading heads: {heads_checkpoint}")
    cfg = heads_ckpt.get("config", {})
    if cfg.get("chunk_size"):
        chunk_size = cfg["chunk_size"]
        print(f"  Detected chunk_size={chunk_size} from heads checkpoint config")
    if cfg.get("decision_K"):
        decision_K = cfg["decision_K"]
    else:
        decision_K = chunk_size

    # Load the ENCODER checkpoint (vision_proj + traj_encoder + text_encoder + mixer).
    # This is the Phase 1b best.pt.
    enc_ckpt = None
    if encoder_checkpoint is not None and Path(encoder_checkpoint).exists():
        enc_ckpt = torch.load(encoder_checkpoint, map_location=device)
        print(f"  Loading encoder: {encoder_checkpoint}")
    else:
        # Try to auto-detect in standard location
        heads_path = Path(heads_checkpoint)
        candidate = heads_path.parent.parent / "pretrain" / heads_path.parent.name / "run_2" / "best.pt"
        if candidate.exists():
            encoder_checkpoint = str(candidate)
            enc_ckpt = torch.load(encoder_checkpoint, map_location=device)
            print(f"  Auto-found encoder checkpoint: {encoder_checkpoint}")
        else:
            print("  WARNING: No encoder checkpoint provided.")
            print("           The encoders will be randomly initialized.")
            print("           Eval results will be meaningless. Pass --encoder-checkpoint to fix.")

    # -- Model
    head_kwargs = {}
    if decision_arch == "mlp":
        head_kwargs = {
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_num_layers": mlp_num_layers,
        }
    elif decision_arch == "transformer":
        head_kwargs = {
            "num_layers": transformer_layers,
            "d_model": transformer_d_model,
            "nhead": transformer_nhead,
            "dropout": transformer_dropout,
            "dim_feedforward": transformer_dim_ff,
        }
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=chunk_size,
        use_text=True,
        device=device,
        decision_K=decision_K,
        decision_arch=decision_arch,
        **head_kwargs,
        assistant_hidden=assistant_hidden,
        assistant_layers=assistant_layers,
        assistant_dropout=assistant_dropout,
    ).to(device)
    model.freeze_backbone()
    model.freeze_all_encoders()

    # Step 1: load encoder + mixer weights (only the relevant keys, not the heads)
    if enc_ckpt is not None and "trainable_state_dict" in enc_ckpt:
        enc_state = enc_ckpt["trainable_state_dict"]
        encoder_keys = {
            k: v for k, v in enc_state.items()
            if "vision_encoder.projection" in k
            or "traj_encoder" in k
            or "text_encoder" in k
            or "cross_attention_mixer" in k
        }
        if encoder_keys:
            missing, unexpected = model.load_state_dict(encoder_keys, strict=False)
            unexpected = [u for u in unexpected
                          if "decision_head" not in u
                          and "assistant_head" not in u]
            if unexpected:
                print(f"  WARNING: Unexpected keys: {unexpected[:3]}...")
            print(f"  Loaded {len(encoder_keys)} encoder/mixer params")

    # Step 2: load head weights
    if "trainable_state_dict" in heads_ckpt:
        model.load_trainable_state_dict(heads_ckpt["trainable_state_dict"])
        print(f"  Loaded heads (decision + assistant)")
    elif "model_state_dict" in heads_ckpt:
        model.load_state_dict(heads_ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(heads_ckpt, strict=False)

    model.eval()
    print(f"  Phase from heads checkpoint: {heads_ckpt.get('phase', 'N/A')}, Epoch: {heads_ckpt.get('epoch', '?')}")

    #  -- Dataset (use the last val_split as validation)
    if len(data_paths) == 1:
        ds = ALIGNDataset(data_paths[0], mode="head", traj_window=traj_window)
    else:
        from data.align_dataset import MultiALIGNDataset
        ds = MultiALIGNDataset(
            data_paths, mode="head", traj_window=traj_window
        )
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    indices = list(range(n_total - n_val, n_total))

    print(f"  Dataset ({len(data_paths)}): {data_paths}")
    print(f"  Validation samples: {n_val}")
    print(f"  Device: {device}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=RandomSampler(indices),
        drop_last=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size, vision_window_size=chunk_size),
    )

    # -- Evaluation loop
    alpha_errors = []
    delta_rmses = []
    decision_losses = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            frames = torch.from_numpy(batch["frames"]).to(device)
            # v2: one-step robot state (B, 7) — replaces the (B, K, 6) trajectory window
            state = torch.from_numpy(batch["robot_state"]).float().to(device)
            noisy_pose = torch.from_numpy(batch["noisy_pose"]).float().to(device)
            texts = batch["texts"]
            alpha_need = torch.from_numpy(batch["alpha_need"]).float().to(device)
            delta_t = torch.from_numpy(batch["delta_target"]).float().to(device)

            # Encode via frozen mixer
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frames, state, texts)
                z_v = mixed["z_v"].float()
                z_t_tokens = mixed["z_t_tokens"].float()
                z_text = mixed["z_text"].float()

            # Decision head (now a future prediction head)
            # Predict K future embeddings; compare against current-state embedding
            # (the v2 dataset no longer exposes a future-state window).
            K = model.decision_K
            with torch.no_grad():
                # Re-encode the same current state as the "target" future.
                # In v2 there's no separate future-state field, so we use the
                # current state's embedding as the target for self-prediction.
                mixed_future = model.encode_mixed(frames, state, texts)
                z_t_future_tokens = mixed_future["z_t_tokens"].float()
                z_v_target = z_v.unsqueeze(1).expand(-1, K, -1)
                z_v_window = z_v.unsqueeze(1).expand(-1, K, -1)
                predicted_z_v, predicted_z_t = model.decision_head(
                    z_v_window, z_t_tokens, z_text
                )
            # Cosine loss for future prediction (lower = better)
            decision_loss = ALIGNModel.future_prediction_loss(
                predicted_z_v, predicted_z_t, z_v_target, z_t_future_tokens
            )
            decision_losses.append(decision_loss.item())

            # Assistant head: input is current action, not pose
            z_t = z_t_tokens.mean(dim=1)  # mean-pool for assistant head
            _ca = batch.get("current_action", noisy_pose)
            if not isinstance(_ca, torch.Tensor):
                _ca = torch.tensor(_ca, dtype=torch.float32)
            current_action = _ca.to(device)
            # Single-step action prediction: (B, 6)
            action_pred = model.assistant_head(z_v, z_t, z_text)
            rmse_per_batch = (action_pred - current_action).pow(2).mean(dim=1).sqrt()
            delta_rmses.extend(rmse_per_batch.cpu().tolist())

    avg_decision_loss = float(np.mean(decision_losses)) if decision_losses else 0.0
    delta_rmse = float(np.mean(delta_rmses))

    print(f"\n=== Evaluation Results (N={len(indices)}) ===")
    print(f"  Decision head  future-prediction loss:  {avg_decision_loss:.4f}  (cosine, [0, 2])")
    print(f"  Assistant head Δ RMSE:                  {delta_rmse:.4f}")
    return {
        "decision_future_loss": avg_decision_loss,
        "assistant_delta_rmse": delta_rmse,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALIGN trained heads")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s). Pass multiple to evaluate on the union.")
    parser.add_argument("--checkpoint", required=True,
                        help="Heads checkpoint (.pt) — contains decision_head + assistant_head")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Encoder/mixer checkpoint from Phase 1b. "
                             "Auto-detected if not provided.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    # Decision head arch (must match training to load weights)
    parser.add_argument("--decision-arch", default="transformer",
                        choices=["mlp", "transformer"])
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-d-model", type=int, default=384)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-dim-ff", type=int, default=1024)
    # Assistant head arch
    parser.add_argument("--assistant-hidden", type=int, default=256)
    parser.add_argument("--assistant-layers", type=int, default=2)
    parser.add_argument("--assistant-dropout", type=float, default=0.0)

    args = parser.parse_args()
    evaluate(
        data_paths=args.data,
        heads_checkpoint=args.checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        batch_size=args.batch_size,
        traj_window=args.traj_window,
        chunk_size=args.chunk_size,
        val_split=args.val_split,
        device=args.device,
        decision_arch=args.decision_arch,
        mlp_hidden_dim=args.mlp_hidden,
        mlp_num_layers=args.mlp_layers,
        transformer_layers=args.transformer_layers,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_dropout=args.transformer_dropout,
        transformer_dim_ff=args.transformer_dim_ff,
        assistant_hidden=args.assistant_hidden,
        assistant_layers=args.assistant_layers,
        assistant_dropout=args.assistant_dropout,
    )
