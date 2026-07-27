#!/usr/bin/env python3
"""Test StateConditionalCrossAttn sensitivity and visualize attention.

Tests:
  1. StateConditionalCrossAttn sensitivity — does z_s change the output?
  2. Attention visualization — what spatial regions does the model attend to?

Usage:
    python tests/test_attention_modules.py \
        --data data/libero_spatial.h5 \
        --checkpoint checkpoints/v4/libero_spatial/run_22/intention_best.pt \
        --cameras image wrist_image \
        --n-samples 3 \
        --out-dir test_output
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_intention import load_intention_model


def load_episode(h5_path: str, ep_key: str, cameras: list, n_frames: int = 50):
    """Load a single episode's frames and states."""
    with h5py.File(h5_path, "r") as f:
        ep = f[ep_key]
        frames_list = []
        for cam in cameras:
            if cam in ep["frames"]:
                frames_list.append(ep["frames"][cam][:n_frames])
        if len(frames_list) == 1:
            frames = frames_list[0]  # (N, H, W, 3)
        else:
            frames = np.stack(frames_list, axis=1)  # (N, V, H, W, 3)
        poses = ep["poses"][:n_frames]
        actions = ep["actions"][:n_frames, :6]
        gripper = np.zeros((n_frames, 1), dtype=np.float32)
        states = np.concatenate([poses, gripper], axis=1)  # (N, 7)
        return frames, states, actions


def test_state_conditional_sensitivity(model, frames, states, device):
    """Test that StateConditionalCrossAttn actually uses z_s.

    Method:
      1. Run vision_patch_encoder with original z_s
      2. Run with perturbed z_s (z_s + noise)
      3. Run with zero z_s
      Compare outputs — if sensitive to z_s, outputs differ.
    """
    print("\n=== Test 1: StateConditionalCrossAttn sensitivity ===")

    # Get vision_patch_encoder
    vpe = model.vision_patch_encoder
    if vpe is None:
        print("  SKIP: model has no vision_patch_encoder")
        return

    # Prepare inputs: (B, VP, raw_dim) and (B, state_dim)
    f_t = torch.from_numpy(frames[:1]).unsqueeze(0).to(device)  # (1, 1, V, H, W, 3) or (1, 1, H, W, 3)
    s_t = torch.from_numpy(states[:1]).float().unsqueeze(0).to(device)  # (1, 1, 7)

    model.eval()
    with torch.no_grad():
        # Get DINOv2 patches for the first frame
        z_v = model._vision_forward(f_t[:, 0])  # (1, V*P+1, 768) or (1, P+1, 768)
        # Split CLS and patches
        z_v_patches = z_v[:, :-1]  # (1, V*P, 768)
        z_s = model.state_encoder(s_t[:, 0])  # (1, state_dim)

        # 1. Original z_s
        out_orig = vpe(z_v_patches, z_s)

        # 2. Perturbed z_s
        z_s_pert = z_s + torch.randn_like(z_s) * 0.5
        out_pert = vpe(z_v_patches, z_s_pert)

        # 3. Zero z_s
        z_s_zero = torch.zeros_like(z_s)
        out_zero = vpe(z_v_patches, z_s_zero)

    diff_pert = (out_orig - out_pert).norm(dim=-1).mean().item()
    diff_zero = (out_orig - out_zero).norm(dim=-1).mean().item()

    print(f"  ||orig - perturbed||  = {diff_pert:.4f}")
    print(f"  ||orig - zero||       = {diff_zero:.4f}")

    if diff_pert > 0.05 and diff_zero > 0.05:
        print("  ✓ StateConditionalCrossAttn IS sensitive to z_s")
    elif diff_pert > 0.01:
        print("  ~ Weak sensitivity to z_s")
    else:
        print("  ✗ StateConditionalCrossAttn is NOT sensitive to z_s")
        print("    (cross-attn may have learned identity despite state input)")

    return diff_pert, diff_zero


def visualize_attention(model, frames, states, device, out_dir: str = None):
    """Visualize attention weights from StateConditionalCrossAttn.

    Extracts cross-attention weights from the state-conditioned cross-attention
    module and overlays them on the camera images as heatmaps.

    The attention shows which spatial patches the model focuses on given
    the current robot state.
    """
    print("\n=== Test 2: Attention visualization ===")

    vpe = model.vision_patch_encoder
    if vpe is None:
        print("  SKIP: no vision_patch_encoder")
        return

    modulator = vpe.state_modulator
    if modulator is None:
        print("  SKIP: no state_modulator in vision_patch_encoder")
        return

    T = min(frames.shape[0], 20)  # limit to 20 timesteps for speed
    V = frames.shape[1] if frames.ndim == 5 else 1

    model.eval()
    all_weights = []  # list of (T, V, P) per timestep

    with torch.no_grad():
        for t in range(T):
            # Get frame and state for this timestep
            f_t = torch.from_numpy(frames[t:t+1]).unsqueeze(0).to(device)  # (1, 1, V, H, W, 3) or (1, 1, H, W, 3)
            s_t = torch.from_numpy(states[t:t+1]).float().unsqueeze(0).to(device)

            # DINOv2
            z_v = model._vision_forward(f_t[:, 0])  # (1, V*P+1, 768)
            z_v_patches = z_v[:, :-1]  # (1, V*P, 768)
            z_s = model.state_encoder(s_t[:, 0])  # (1, state_dim)

            # SE compress first
            z_v_comp = vpe.se_compressor(z_v_patches)  # (1, V*P, comp_dim)

            # StateConditionalCrossAttn with attention weights
            B, N_pos, D = z_v_comp.shape
            q = modulator.q_proj(z_s).unsqueeze(1).expand(-1, N_pos, -1)  # (1, N_pos, D)
            k = v = z_v_comp

            # Get attention weights
            # MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights shape: (B, num_heads, L, S) or (B, L, S) with average_attn_weights=True
            _, attn_weights = modulator.cross_attn(
                q, k, v, need_weights=True, average_attn_weights=True,
            )
            # attn_weights: (1, N_pos) with average_attn_weights=True
            if attn_weights.ndim == 3:
                # (1, 1, N_pos) — still has head dim
                weights = attn_weights[0, 0].cpu().numpy()  # (N_pos,)
            else:
                weights = attn_weights[0].cpu().numpy()  # (N_pos,)

            # Split by camera
            P_per_cam = N_pos // V
            cam_weights = []
            for c in range(V):
                start = c * P_per_cam
                end = (c + 1) * P_per_cam
                cam_weights.append(weights[start:end])

            all_weights.append(cam_weights)

    # Print per-timestep top patches
    P_per_cam = len(all_weights[0][0])
    grid_dim = int(np.sqrt(P_per_cam))
    print(f"  Patches per camera: {P_per_cam} ({grid_dim}x{grid_dim} grid)")

    for t in range(min(T, 5)):  # print first 5 timesteps
        for c in range(V):
            w = all_weights[t][c]
            top5 = np.argsort(w)[-5:][::-1]
            top_str = " | ".join([f"idx={int(i)} w={w[i]:.3f}" for i in top5])
            print(f"  t={t} cam_{c}: top-5 patches → {top_str}")

    # Save visualizations
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from PIL import Image as PILImage

            # Per-timestep heatmap grid (static)
            n_cols = min(T, 10)
            for c in range(V):
                fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
                if n_cols == 1:
                    axes = [axes]
                for t in range(n_cols):
                    ax = axes[t]
                    if frames.ndim == 5:
                        img = frames[t, c]
                    else:
                        img = frames[t]
                    w = all_weights[t][c].reshape(grid_dim, grid_dim)
                    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
                    w_resized = np.array(
                        PILImage.fromarray(w_norm).resize(
                            (img.shape[1], img.shape[0]), PILImage.BILINEAR
                        )
                    )
                    heat_rgb = cm.hot(w_resized)[:, :, :3]
                    alpha = np.stack([w_resized] * 3, axis=-1)
                    overlay = np.clip(
                        img.astype(np.float32) * (1.0 - alpha * 0.5)
                        + heat_rgb * 255 * (alpha * 0.5),
                        0, 255,
                    ).astype(np.uint8)
                    ax.imshow(overlay)
                    ax.set_title(f"t={t}")
                    ax.axis("off")
                fig.suptitle(f"Camera {c}: state-conditioned attention over time")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"attention_timeline_cam{c}.png"), dpi=80)
                plt.close(fig)

            # Per-camera attention video (full episode)
            for c in range(V):
                try:
                    import imageio
                    vid_path = os.path.join(out_dir, f"attention_video_cam{c}.mp4")
                    writer = imageio.get_writer(vid_path, fps=10, codec="libx264", quality=8)
                    for t in range(T):
                        if frames.ndim == 5:
                            img = frames[t, c]
                        else:
                            img = frames[t]
                        w = all_weights[t][c].reshape(grid_dim, grid_dim)
                        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
                        w_resized = np.array(
                            PILImage.fromarray(w_norm).resize(
                                (img.shape[1], img.shape[0]), PILImage.BILINEAR
                            )
                        )
                        heat_rgb = cm.hot(w_resized)[:, :, :3]
                        alpha = np.stack([w_resized] * 3, axis=-1)
                        overlay = np.clip(
                            img.astype(np.float32) * (1.0 - alpha * 0.5)
                            + heat_rgb * 255 * (alpha * 0.5),
                            0, 255,
                        ).astype(np.uint8)
                        writer.append_data(overlay)
                    writer.close()
                    print(f"  Saved attention video: {vid_path}")
                except ImportError:
                    print(f"  Skipping video: imageio not installed")

            # Combined video (all cameras side-by-side)
            if V > 1:
                try:
                    import imageio
                    vid_path = os.path.join(out_dir, "attention_video_combined.mp4")
                    writer = imageio.get_writer(vid_path, fps=10, codec="libx264", quality=8)
                    for t in range(T):
                        cam_frames = []
                        for c in range(V):
                            if frames.ndim == 5:
                                img = frames[t, c]
                            else:
                                img = frames[t]
                            w = all_weights[t][c].reshape(grid_dim, grid_dim)
                            w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
                            w_resized = np.array(
                                PILImage.fromarray(w_norm).resize(
                                    (img.shape[1], img.shape[0]), PILImage.BILINEAR
                                )
                            )
                            heat_rgb = cm.hot(w_resized)[:, :, :3]
                            alpha = np.stack([w_resized] * 3, axis=-1)
                            overlay = np.clip(
                                img.astype(np.float32) * (1.0 - alpha * 0.5)
                                + heat_rgb * 255 * (alpha * 0.5),
                                0, 255,
                            ).astype(np.uint8)
                            cam_frames.append(overlay)
                        combined = np.concatenate(cam_frames, axis=1)
                        writer.append_data(combined)
                    writer.close()
                    print(f"  Saved combined attention video: {vid_path}")
                except ImportError:
                    pass

            # Side-by-side: original vs zero vs perturbed z_s on first frame
            f_t0 = torch.from_numpy(frames[0:1]).unsqueeze(0).to(device)
            s_t0 = torch.from_numpy(states[0:1]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_v0 = model._vision_forward(f_t0[:, 0])
                z_v_patches0 = z_v0[:, :-1]
                z_v_comp0 = vpe.se_compressor(z_v_patches0)
                z_s0 = model.state_encoder(s_t0[:, 0])

                variants = {
                    "original": z_s0,
                    "zero": torch.zeros_like(z_s0),
                    "perturbed": z_s0 + torch.randn_like(z_s0) * 0.3,
                }

                fig, axes = plt.subplots(V, len(variants), figsize=(5 * len(variants), 4 * V))
                if V == 1:
                    axes = axes.reshape(1, -1)

                for c in range(V):
                    for j, (label, z_s_v) in enumerate(variants.items()):
                        ax = axes[c, j] if V > 1 else axes[0, j]
                        q_v = modulator.q_proj(z_s_v).unsqueeze(1).expand(-1, z_v_comp0.shape[1], -1)
                        _, aw = modulator.cross_attn(
                            q_v, z_v_comp0, z_v_comp0,
                            need_weights=True, average_attn_weights=True,
                        )
                        if aw.ndim == 3:
                            w = aw[0, 0].cpu().numpy()
                        else:
                            w = aw[0].cpu().numpy()
                        Ppc = w.shape[0] // V
                        w_cam = w[c * Ppc:(c + 1) * Ppc].reshape(grid_dim, grid_dim)
                        w_norm = (w_cam - w_cam.min()) / (w_cam.max() - w_cam.min() + 1e-8)

                        if frames.ndim == 5:
                            img = frames[0, c]
                        else:
                            img = frames[0]

                        from PIL import Image as PILImage
                        w_resized = np.array(
                            PILImage.fromarray(w_norm).resize(
                                (img.shape[1], img.shape[0]), PILImage.BILINEAR
                            )
                        )
                        heat_rgb = cm.hot(w_resized)[:, :, :3]
                        alpha = np.stack([w_resized] * 3, axis=-1)
                        overlay = np.clip(
                            img.astype(np.float32) * (1.0 - alpha * 0.5)
                            + heat_rgb * 255 * (alpha * 0.5),
                            0, 255,
                        ).astype(np.uint8)
                        ax.imshow(overlay)
                        ax.set_title(f"cam {c}, z_s={label}")
                        ax.axis("off")

                fig.suptitle("State-conditioned attention: rows=cameras, cols=z_s variants")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "attention_comparison.png"), dpi=80)
                plt.close(fig)

            print(f"  Saved visualizations to {out_dir}/")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Failed to save visualizations: {e}")

    return all_weights


def main():
    parser = argparse.ArgumentParser(description="Test attention modules")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to intention_best.pt")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"],
                        help="Camera names")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of episodes to test")
    parser.add_argument("--n-frames", type=int, default=50,
                        help="Frames per episode")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None,
                        help="Save attention heatmaps to this dir")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n=== Attention Module Tests ===")
    print(f"  Data:       {args.data}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Cameras:    {args.cameras}")
    print(f"  Device:     {device}")

    model, cfg = load_intention_model(args.checkpoint, device)
    print(f"  Loaded: chunk_size={cfg['chunk_size']}, "
          f"head={cfg.get('head_type', 'mamba')}, "
          f"cameras={cfg.get('num_cameras', 1)}")

    # Find episodes
    with h5py.File(args.data, "r") as f:
        ep_keys = sorted([k for k in f.keys() if k.startswith("ep_")])
    if not ep_keys:
        print("No episodes found!")
        return
    print(f"  Found {len(ep_keys)} episodes, testing first {args.n_samples}")

    for i in range(min(args.n_samples, len(ep_keys))):
        ep_key = ep_keys[i]
        print(f"\n--- Sample {i+1}/{args.n_samples}: {ep_key} ---")
        frames, states, actions = load_episode(args.data, ep_key, args.cameras, args.n_frames)
        print(f"  Frames: {frames.shape}, States: {states.shape}")

        test_state_conditional_sensitivity(model, frames, states, device)
        if i == 0:
            visualize_attention(model, frames, states, device, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
