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

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from data.align_dataset import ALIGNDataset, head_collate


def evaluate(
    data_path: str,
    checkpoint_path: str,
    batch_size: int = 64,
    traj_window: int = 20,
    chunk_size: int = 5,
    val_split: float = 0.1,
    device: str = None,
    use_bf16: bool = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -- Model (load both heads from the combined checkpoint)
    model = ALIGNModel(
        embed_dim=256, chunk_size=chunk_size, use_text=True, device=device
    ).to(device)
    model.freeze_backbone()
    model.freeze_all_encoders()
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"  Loading: {checkpoint_path}")
    if "trainable_state_dict" in ckpt:
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        # Try loading directly as a state dict
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    print(f"  Phase from checkpoint: {ckpt.get('phase', 'N/A')}, Epoch: {ckpt.get('epoch', '?')}")

    # Detect chunk_size from checkpoint config
    cfg = ckpt.get("config", {})
    if cfg.get("chunk_size"):
        chunk_size = cfg["chunk_size"]
        print(f"  Detected chunk_size={chunk_size} from checkpoint config")

    #  -- Dataset (use the last val_split as validation)
    ds = ALIGNDataset(data_path, mode="head", traj_window=traj_window)
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    indices = list(range(n_total - n_val, n_total))

    print(f"  Dataset: {data_path}")
    print(f"  Validation samples: {n_val}")
    print(f"  Device: {device}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=RandomSampler(indices),
        drop_last=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size),
    )

    # -- Evaluation loop
    alpha_errors = []
    delta_rmses = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            frames = torch.from_numpy(batch["frames"]).to(device)
            traj_view = torch.from_numpy(batch["trajectory"]).float().to(device)
            noisy_pose = torch.from_numpy(batch["noisy_pose"]).float().to(device)
            texts = batch["texts"]
            alpha_need = torch.from_numpy(batch["alpha_need"]).float().to(device)
            delta_t = torch.from_numpy(batch["delta_target"]).float().to(device)

            # Encode via frozen mixer
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frames, traj_view, texts)
                z_v = mixed["z_v"].float()
                z_t = mixed["z_t"].float()
                z_text = mixed["z_text"].float()

            # Decision head
            alpha_pred = model.decision_head(z_v, z_t, z_text).squeeze(-1)  # (B,)
            alpha_errors.extend((alpha_pred - alpha_need).abs().cpu().tolist())

            # Assistant head
            delta_pred = model.assistant_head(z_v, z_t, z_text, noisy_pose)  # (B, K, 6)
            rmse_per_batch = (delta_pred - delta_t).pow(2).mean(dim=[1, 2]).sqrt()
            delta_rmses.extend(rmse_per_batch.cpu().tolist())

    alpha_mae = float(np.mean(alpha_errors))
    delta_rmse = float(np.mean(delta_rmses))

    print(f"\n=== Evaluation Results (N={len(indices)}) ===")
    print(f"  Decision head  α MAE:     {alpha_mae:.4f}")
    print(f"  Assistant head Δ RMSE:    {delta_rmse:.4f}")
    return {"decision_alpha_mae": alpha_mae, "assistant_delta_rmse": delta_rmse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALIGN trained heads")
    parser.add_argument("--data", required=True, help="Path to ALIGN HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, help="Combined heads checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()
    evaluate(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        traj_window=args.traj_window,
        chunk_size=args.chunk_size,
        val_split=args.val_split,
        device=args.device,
    )
