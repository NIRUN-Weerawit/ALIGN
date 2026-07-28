#!/usr/bin/env python3
"""Compare attention statistics across runs by re-running the test on each checkpoint.

Extracts attention weight statistics (mean, std, entropy, top-5 concentration)
to compare how well each run's StateConditionalCrossAttn is learning.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import h5py

from eval.eval_intention import load_intention_model


def get_attention_stats(model, frames, states, device):
    """Extract attention weight statistics from StateConditionalCrossAttn."""
    vpe = model.vision_patch_encoder
    modulator = vpe.state_modulator

    T = min(frames.shape[0], 20)
    V = frames.shape[1] if frames.ndim == 5 else 1

    model.eval()
    all_entropies = []
    all_max_weights = []
    all_sensitivities = []
    Ppc = None  # will be set on first timestep

    with torch.no_grad():
        for t in range(T):
            f_t = torch.from_numpy(frames[t:t+1]).unsqueeze(0).to(device)
            s_t = torch.from_numpy(states[t:t+1]).float().unsqueeze(0).to(device)

            z_v = model._vision_forward(f_t[:, 0])
            z_v_patches = z_v[:, :-1]
            z_s = model.state_encoder(s_t[:, 0])

            z_v_comp = vpe.se_compressor(z_v_patches)
            B, N_pos, D = z_v_comp.shape

            # Get attention weights
            q = modulator.q_proj(z_s).unsqueeze(1).expand(-1, N_pos, -1)
            _, attn_weights = modulator.cross_attn(
                q, z_v_comp, z_v_comp,
                need_weights=True, average_attn_weights=True,
            )
            if attn_weights.ndim == 3:
                w = attn_weights[0, 0].cpu().numpy()
            else:
                w = attn_weights[0].cpu().numpy()

            # Per-camera stats
            Ppc = N_pos // V
            for c in range(V):
                w_cam = w[c * Ppc:(c + 1) * Ppc]
                # Entropy (lower = more focused)
                p = w_cam / (w_cam.sum() + 1e-8)
                entropy = -np.sum(p * np.log(p + 1e-8)) / np.log(Ppc)  # normalized [0,1]
                all_entropies.append(entropy)
                # Max weight (higher = more focused)
                all_max_weights.append(w_cam.max())

            # Sensitivity: difference between original and zero z_s
            z_s_zero = torch.zeros_like(z_s)
            q_zero = modulator.q_proj(z_s_zero).unsqueeze(1).expand(-1, N_pos, -1)
            _, aw_zero = modulator.cross_attn(
                q_zero, z_v_comp, z_v_comp,
                need_weights=True, average_attn_weights=True,
            )
            if aw_zero.ndim == 3:
                w_zero = aw_zero[0, 0].cpu().numpy()
            else:
                w_zero = aw_zero[0].cpu().numpy()
            sensitivity = np.abs(w - w_zero).mean()
            all_sensitivities.append(sensitivity)

    return {
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "mean_max_weight": float(np.mean(all_max_weights)),
        "mean_sensitivity": float(np.mean(all_sensitivities)),
        "uniform_baseline_entropy": 1.0,  # uniform = 1.0
        "uniform_baseline_max_weight": 1.0 / (Ppc),  # ~0.0039 for 256 patches
    }


def main():
    runs = [
        (15, "checkpoints/v4/libero_spatial/run_15/intention_best_fixed.pt"),
        (17, "checkpoints/v4/libero_spatial/run_17/intention_best_fixed.pt"),
        (18, "checkpoints/v4/libero_spatial/run_18/intention_best_fixed.pt"),
        (21, "checkpoints/v4/libero_spatial/run_21/intention_best.pt"),
        (22, "checkpoints/v4/libero_spatial/run_22/intention_best.pt"),
        (23, "checkpoints/v4/libero_spatial/run_23/intention_best.pt"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load one episode for testing
    data_path = "data/libero_spatial.h5"
    with h5py.File(data_path, "r") as f:
        ep_keys = sorted([k for k in f.keys() if k.startswith("ep_")])
    ep_key = ep_keys[0]
    print(f"Testing on: {ep_key}\n")

    # Load frames and states
    with h5py.File(data_path, "r") as f:
        ep = f[ep_key]
        frames_list = []
        for cam in ["image", "wrist_image"]:
            frames_list.append(ep["frames"][cam][:20])
        frames = np.stack(frames_list, axis=1)
        poses = ep["poses"][:20]
        gripper = np.zeros((20, 1), dtype=np.float32)
        states = np.concatenate([poses, gripper], axis=1)

    print(f"{'Run':>6} | {'Config':<50} | {'Entropy':>8} | {'MaxW':>8} | {'Sens':>8}")
    print("-" * 90)

    for run_id, ckpt_path in runs:
        try:
            model, cfg = load_intention_model(ckpt_path, device)
            stats = get_attention_stats(model, frames, states, device)

            config_str = (
                f"H={cfg.get('use_history', '?')} "
                f"I={cfg.get('use_intent_tokens', '?')} "
                f"M={cfg.get('use_memory_bank', '?')} "
                f"lr={cfg.get('lr', '?')} "
                f"bs={cfg.get('batch_size', '?')}"
            )

            print(f"run_{run_id:>2} | {config_str:<50} | {stats['mean_entropy']:.3f} | {stats['mean_max_weight']:.4f} | {stats['mean_sensitivity']:.4f}")
        except Exception as e:
            print(f"run_{run_id:>2} | ERROR: {e}")

    print()
    print("Uniform baseline: entropy=1.000, max_weight≈0.0039")
    print("Good attention:   entropy<0.8, max_weight>0.01, sensitivity>0.1")


if __name__ == "__main__":
    main()
