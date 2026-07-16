"""Test script for cross-camera and state-conditioned attention modules.

Verifies that:
  1. State-conditioned attention actually uses z_t (changes when z_t changes)
  2. Cross-camera attention actually uses both cameras
  3. The pooled output can be decoded back to z_t (info roundtrip)
  4. Attention weights are sensible (focus on task-relevant patches)

Usage:
    python tests/test_attention_modules.py \
        --data data/libero_object.h5 \
        --checkpoint checkpoints/v3/libero_object/run_2/intention_best.pt \
        --n-samples 5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_intention import load_intention_model


def load_batch(h5_path: str, ep_key: str, cameras: list, n_frames: int = 5):
    """Load a single episode's frames and states."""
    with h5py.File(h5_path, "r") as f:
        ep = f[ep_key]
        # Frames: stack multiple cameras
        frames_list = []
        for cam in cameras:
            if cam in ep["frames"]:
                frames_list.append(ep["frames"][cam][:n_frames])
        if len(frames_list) == 1:
            frames = frames_list[0]  # (N, H, W, 3)
        else:
            frames = np.stack(frames_list, axis=1)  # (N, V, H, W, 3)
        # States
        poses = ep["poses"][:n_frames]
        actions = ep["actions"][:n_frames, :6]
        gripper = np.zeros((n_frames, 1), dtype=np.float32)
        states = np.concatenate([poses, gripper], axis=1)  # (N, 7)
        return frames, states, actions


def test_state_conditioned_pool(model, frames, states, device):
    """Test that the state-conditioned attention pool actually uses z_t.

    Method:
      1. Run pool with original z_t
      2. Run pool with perturbed z_t (z_t + noise)
      3. Run pool with zero z_t
      Compare the outputs - if pool is sensitive to z_t, outputs differ.
    """
    print("\n=== Test 1: State-conditioned pool uses z_t ===")
    f_t = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, K, V, H, W, 3)
    s_t = torch.from_numpy(states).float().unsqueeze(0).to(device)  # (1, K, 7)
    model.eval()
    intention_encoder = model.intention_encoder
    if intention_encoder is None:
        print("  SKIP: model has no intention encoder")
        return
    with torch.no_grad():
        # Get raw patch tokens (per timestep — vision encoder takes 5D or 4D)
        B, K = f_t.shape[:2]
        z_v_patches_per_step = []
        for t in range(K):
            z_v_t = model._vision_forward(f_t[:, t])  # (1, [V,] P, vision_dim)
            z_v_patches_per_step.append(z_v_t)
        # z_v_patches_seq: (1, K, V, P, vision_dim) or (1, K, P, vision_dim)
        z_v_patches_seq = torch.stack(z_v_patches_per_step, dim=1)
        # Encode states
        z_t_seq = model.state_encoder(s_t)  # (1, K, state_dim)

        # Helper: split concatenated (V*P, vision_dim) → (B, V, P, vision_dim)
        num_cams_cfg = (cfg or {}).get("num_cameras", 1) if False else 1  # not available
        # We don't have cfg here; detect num_cams from the model
        num_cams_cfg = intention_encoder.pool.num_cameras

        def to_per_cam(z_v_t, B_inner):
            # z_v_t: (B, P, vision_dim) or (B, V, P, vision_dim) or (B, V*P, vision_dim)
            if z_v_t.ndim == 4:
                return z_v_t  # already (B, V, P, vision_dim)
            elif z_v_t.ndim == 3:
                # Could be (B, P, vision_dim) (single cam) or (B, V*P, vision_dim) (multi)
                P = z_v_t.shape[1]
                if num_cams_cfg == 1:
                    return z_v_t.unsqueeze(1)  # (B, 1, P, vision_dim)
                else:
                    # multi: split into V chunks
                    P_per_cam = P // num_cams_cfg
                    return z_v_t.reshape(B_inner, num_cams_cfg, P_per_cam, -1)
            return z_v_t

        # 1. Original
        z_v_pooled_orig = []
        B, T = z_v_patches_seq.shape[:2]
        for t in range(T):
            z_v_t = to_per_cam(z_v_patches_seq[:, t], B)
            z_t_t = z_t_seq[:, t]
            pooled = intention_encoder.pool_patches(z_v_t, z_t_t)
            z_v_pooled_orig.append(pooled)
        z_v_pooled_orig = torch.stack(z_v_pooled_orig, dim=1)
        # 2. Perturbed z_t
        z_t_perturbed = z_t_seq + torch.randn_like(z_t_seq) * 0.5
        z_v_pooled_perturbed = []
        for t in range(T):
            z_v_t = to_per_cam(z_v_patches_seq[:, t], B)
            z_t_t = z_t_perturbed[:, t]
            pooled = intention_encoder.pool_patches(z_v_t, z_t_t)
            z_v_pooled_perturbed.append(pooled)
        z_v_pooled_perturbed = torch.stack(z_v_pooled_perturbed, dim=1)
        # 3. Zero z_t
        z_t_zero = torch.zeros_like(z_t_seq)
        z_v_pooled_zero = []
        for t in range(T):
            z_v_t = to_per_cam(z_v_patches_seq[:, t], B)
            z_t_t = z_t_zero[:, t]
            pooled = intention_encoder.pool_patches(z_v_t, z_t_t)
            z_v_pooled_zero.append(pooled)
        z_v_pooled_zero = torch.stack(z_v_pooled_zero, dim=1)
    # Compare
    diff_perturbed = (z_v_pooled_orig - z_v_pooled_perturbed).norm(dim=-1).mean().item()
    diff_zero = (z_v_pooled_orig - z_v_pooled_zero).norm(dim=-1).mean().item()
    print(f"  ||z_v_pooled(orig) - z_v_pooled(perturbed)||  = {diff_perturbed:.4f}")
    print(f"  ||z_v_pooled(orig) - z_v_pooled(zero)||     = {diff_zero:.4f}")
    if diff_perturbed > 0.1 and diff_zero > 0.1:
        print("  ✓ Pool IS sensitive to z_t (state-conditioned)")
    else:
        print("  ✗ Pool is NOT sensitive to z_t (state-conditioning may be broken)")
    return diff_perturbed, diff_zero


def test_cross_camera_attention(model, frames, states, device):
    """Test that the cross-camera attention actually uses both cameras.

    Method:
      1. Run with both cameras
      2. Run with one camera zeroed out
      3. Run with cameras swapped
    Compare outputs - if both cameras contribute, the outputs differ.
    """
    print("\n=== Test 2: Cross-camera attention uses both cameras ===")
    if frames.ndim != 5:
        print("  SKIP: only 1 camera in dataset (no cross-camera to test)")
        return
    V = frames.shape[1]
    if V < 2:
        print(f"  SKIP: only {V} camera, need >= 2 for cross-camera test")
        return
    f_t = torch.from_numpy(frames).unsqueeze(0).to(device)
    s_t = torch.from_numpy(states).float().unsqueeze(0).to(device)
    model.eval()
    intention_encoder = model.intention_encoder
    if intention_encoder is None:
        print("  SKIP: no intention encoder")
        return
    with torch.no_grad():
        B, K = f_t.shape[:2]
        z_v_patches_per_step = []
        for t in range(K):
            z_v_t = model._vision_forward(f_t[:, t])
            z_v_patches_per_step.append(z_v_t)
        z_v_patches_seq = torch.stack(z_v_patches_per_step, dim=1)
        z_t_seq = model.state_encoder(s_t)
        # Reshape to per-camera format (B, K, V, P, vision_dim) for proper pooling
        num_cams = intention_encoder.pool.num_cameras
        # z_v_patches_seq: (1, K, V*P, vision_dim) for multi-cam
        if num_cams > 1 and z_v_patches_seq.ndim == 4:
            P_per_cam = z_v_patches_seq.shape[2] // num_cams
            z_v_patches_seq = z_v_patches_seq.reshape(
                z_v_patches_seq.shape[0], z_v_patches_seq.shape[1],
                num_cams, P_per_cam, -1,
            )
        B, T = z_v_patches_seq.shape[:2]
        # 1. Original (both cameras)
        z_v_pooled_orig = []
        for t in range(T):
            pooled = intention_encoder.pool_patches(z_v_patches_seq[:, t], z_t_seq[:, t])
            z_v_pooled_orig.append(pooled)
        z_v_pooled_orig = torch.stack(z_v_pooled_orig, dim=1)
        # 2. Zero out camera 0
        z_v_patches_zero_cam0 = z_v_patches_seq.clone()
        z_v_patches_zero_cam0[:, :, 0] = 0.0
        z_v_pooled_zero_cam0 = []
        for t in range(T):
            pooled = intention_encoder.pool_patches(
                z_v_patches_zero_cam0[:, t], z_t_seq[:, t]
            )
            z_v_pooled_zero_cam0.append(pooled)
        z_v_pooled_zero_cam0 = torch.stack(z_v_pooled_zero_cam0, dim=1)
        # 3. Swap cameras
        z_v_patches_swap = z_v_patches_seq.clone()
        z_v_patches_swap[:, :, [0, 1]] = z_v_patches_swap[:, :, [1, 0]]
        z_v_pooled_swap = []
        for t in range(T):
            pooled = intention_encoder.pool_patches(
                z_v_patches_swap[:, t], z_t_seq[:, t]
            )
            z_v_pooled_swap.append(pooled)
        z_v_pooled_swap = torch.stack(z_v_pooled_swap, dim=1)
    # Compare
    diff_zero_cam0 = (z_v_pooled_orig - z_v_pooled_zero_cam0).norm(dim=-1).mean().item()
    diff_swap = (z_v_pooled_orig - z_v_pooled_swap).norm(dim=-1).mean().item()
    print(f"  ||z_v_pooled(orig) - z_v_pooled(zero_cam0)||  = {diff_zero_cam0:.4f}")
    print(f"  ||z_v_pooled(orig) - z_v_pooled(swapped)||   = {diff_swap:.4f}")
    if diff_zero_cam0 > 0.1:
        print("  ✓ Camera 0 contributes to pooled output")
    else:
        print("  ✗ Camera 0 does NOT contribute (cross-camera may be broken)")
    if diff_swap > 0.1:
        print("  ✓ Different camera order produces different output")
    else:
        print("  ✗ Order-invariant (no order info preserved)")


def test_z_t_recovery(model, frames, states, device):
    """Test that z_t info is preserved in the pooled output.

    Method:
      1. Pool with original z_t → get z_v_pooled
      2. Train a small linear probe to predict z_t from z_v_pooled
      3. If probe works (low MSE), z_t info was preserved
    """
    print("\n=== Test 3: z_t info preservation in pooled output ===")
    f_t = torch.from_numpy(frames).unsqueeze(0).to(device)
    s_t = torch.from_numpy(states).float().unsqueeze(0).to(device)
    model.eval()
    intention_encoder = model.intention_encoder
    if intention_encoder is None:
        print("  SKIP: no intention encoder")
        return
    with torch.no_grad():
        B, K = f_t.shape[:2]
        z_v_patches_per_step = []
        for t in range(K):
            z_v_t = model._vision_forward(f_t[:, t])
            z_v_patches_per_step.append(z_v_t)
        z_v_patches_seq = torch.stack(z_v_patches_per_step, dim=1)
        z_t_seq = model.state_encoder(s_t)
        # Reshape to per-camera format
        num_cams = intention_encoder.pool.num_cameras
        if num_cams > 1 and z_v_patches_seq.ndim == 4:
            P_per_cam = z_v_patches_seq.shape[2] // num_cams
            z_v_patches_seq = z_v_patches_seq.reshape(
                z_v_patches_seq.shape[0], z_v_patches_seq.shape[1],
                num_cams, P_per_cam, -1,
            )
        B, T = z_v_patches_seq.shape[:2]
        z_v_pooled_list = []
        for t in range(T):
            pooled = intention_encoder.pool_patches(z_v_patches_seq[:, t], z_t_seq[:, t])
            z_v_pooled_list.append(pooled)
        z_v_pooled = torch.stack(z_v_pooled_list, dim=1)  # (1, T, V*vision_dim)
        z_t_target = z_t_seq
    # Train a small linear probe (with holdout to avoid overfitting)
    pool_dim = z_v_pooled.shape[-1]
    state_dim = z_t_target.shape[-1]
    probe = torch.nn.Linear(pool_dim, state_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    z_v_flat = z_v_pooled.reshape(B * T, pool_dim)
    z_t_flat = z_t_target.reshape(B * T, state_dim)
    # 80/20 train/val split
    n_train = max(1, int(0.8 * len(z_v_flat)))
    perm = torch.randperm(len(z_v_flat))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:] if n_train < len(z_v_flat) else perm[:1]
    z_v_train, z_t_train = z_v_flat[train_idx], z_t_flat[train_idx]
    z_v_val, z_t_val = z_v_flat[val_idx], z_t_flat[val_idx]
    for step in range(500):
        pred = probe(z_v_train)
        loss = F.mse_loss(pred, z_t_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_loss = F.mse_loss(probe(z_v_train), z_t_train).item()
        val_loss = F.mse_loss(probe(z_v_val), z_t_val).item()
    # Baseline: predict the mean (from training set)
    z_t_mean = z_t_train.mean(dim=0, keepdim=True)
    with torch.no_grad():
        baseline_loss = F.mse_loss(
            z_t_mean.expand_as(z_t_val), z_t_val
        ).item()
    recovery_ratio = val_loss / max(baseline_loss, 1e-6)
    print(f"  Train probe MSE:   {train_loss:.6f}")
    print(f"  Val probe MSE:     {val_loss:.6f}")
    print(f"  Baseline MSE:      {baseline_loss:.6f} (predict train mean)")
    print(f"  Recovery ratio:    {recovery_ratio:.2%} (lower = better info recovery)")
    if recovery_ratio < 0.3:
        print("  ✓ z_t info IS well-preserved in pooled output")
    elif recovery_ratio < 0.7:
        print("  ~ z_t info is partially preserved")
    else:
        print("  ✗ z_t info is NOT preserved (probe can't beat mean)")


def test_attention_patterns(model, frames, states, device, cfg: Optional[dict] = None,
                             out_dir: Optional[str] = None):
    """Visualize attention weights from the state-conditioned pool.

    Shows which patches the model attends to for different z_t.
    Optionally saves heatmap visualizations overlaid on the image.

    Args:
        model: the model
        frames: (K, V, H, W, 3) or (K, H, W, 3) uint8
        states: (K, 7) float32
        device: torch device
        cfg: model config (used for num_cameras)
        out_dir: if set, save heatmap images here
    """
    print("\n=== Test 4: Attention pattern visualization ===")
    f_t = torch.from_numpy(frames[:1]).unsqueeze(0).to(device)  # just 1 frame
    s_t = torch.from_numpy(states[:1]).float().unsqueeze(0).to(device)
    model.eval()
    intention_encoder = model.intention_encoder
    if intention_encoder is None:
        print("  SKIP: no intention encoder")
        return
    with torch.no_grad():
        B, K = f_t.shape[:2]
        z_v_patches_per_step = []
        for t in range(K):
            z_v_t = model._vision_forward(f_t[:, t])
            z_v_patches_per_step.append(z_v_t)
        z_v_patches_seq = torch.stack(z_v_patches_per_step, dim=1)  # (1, 1, V, P, vision_dim)
        z_t_seq = model.state_encoder(s_t)  # (1, 1, state_dim)
    # Get the first pool
    if intention_encoder.pool.pools is None or len(intention_encoder.pool.pools) == 0:
        print("  SKIP: no pool layers")
        return
    pool = intention_encoder.pool.pools[0]
    # Try to get attention weights by calling the cross-attention directly
    # z_v_patches_seq may be (1, 1, V*P, vision_dim) if concatenated, or
    # (1, 1, V, P, vision_dim) if separate cameras.
    # The PerCameraStateConditionedPool takes (B, V, P, vision_dim) input.
    z_v_seq = z_v_patches_seq[0, 0]  # (V*P, vision_dim) or (V, P, vision_dim)
    num_cams = (cfg or {}).get("num_cameras", 1)
    # We need a single camera's patches for visualization: (P, vision_dim)
    if z_v_seq.ndim == 2:
        # Concatenated format: (V*P, vision_dim) — take first P (=total/num_cams)
        P = z_v_seq.shape[0] // num_cams
        z_v_for_pool = z_v_seq[:P]  # (P, vision_dim)
    else:
        # Separate format: (V, P, vision_dim) — take camera 0
        z_v_for_pool = z_v_seq[0]  # (P, vision_dim)
    print(f"  z_v_for_pool shape: {z_v_for_pool.shape}")
    print(f"  z_t shape: {z_t_seq[0, 0].shape}")
    # Try to capture attention weights
    saved_heatmaps = []  # for visualization
    for label, z_t_test in [
        ("original", z_t_seq[0, 0]),
        ("zero", torch.zeros_like(z_t_seq[0, 0])),
        ("perturbed", z_t_seq[0, 0] + torch.randn_like(z_t_seq[0, 0]) * 0.3),
    ]:
        z_t_test = z_t_test.detach()
        z_t_proj = pool.state_proj(z_t_test.unsqueeze(0))  # (1, vision_dim)
        q = z_t_proj.unsqueeze(1)  # (1, 1, vision_dim)
        k = v = z_v_for_pool.unsqueeze(0)  # (1, P, vision_dim)
        try:
            with torch.no_grad():
                attn_out, attn_w = pool.cross_attn(
                    q, k, v, need_weights=True, average_attn_weights=False
                )
            # attn_w: (B, num_heads, 1, P)
            # Average over heads for cleaner visualization
            w_per_head = attn_w[0, :, 0].cpu().numpy()  # (num_heads, P)
            w = w_per_head.mean(axis=0)  # (P,) averaged over heads
            top_5 = np.argsort(w)[-5:][::-1]
            print(f"  z_t={label}: top-5 attended patches = {top_5.tolist()}, "
                  f"weights = {[f'{w[i]:.3f}' for i in top_5]}")
            # Save for visualization
            saved_heatmaps.append((label, w))
        except Exception as e:
            print(f"  z_t={label}: failed to get attn weights ({e})")
            return

    # If out_dir specified, save visualizations
    if out_dir and saved_heatmaps:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(out_dir, exist_ok=True)
            # Get per-camera attention weights (the user wants DIFFERENT
            # heatmaps per camera, not the same heatmap duplicated)
            saved_heatmaps_per_cam = []  # (label, [w_cam0, w_cam1, ...])
            for label, z_t_test_for_viz in [
                ("original", z_t_seq[0, 0]),
                ("zero", torch.zeros_like(z_t_seq[0, 0])),
                ("perturbed", z_t_seq[0, 0] + torch.randn_like(z_t_seq[0, 0]) * 0.3),
            ]:
                z_t_v = z_t_test_for_viz.detach()
                z_t_proj_v = pool.state_proj(z_t_v.unsqueeze(0))  # (1, vision_dim)
                q_v = z_t_proj_v.unsqueeze(1)  # (1, 1, vision_dim)
                # Per-camera attention: run on each camera's patches separately
                per_cam_weights = []
                if z_v_seq.ndim == 2:
                    # Concatenated: (V*P, vision_dim) — split per camera
                    total = z_v_seq.shape[0]
                    P_per_cam = total // num_cams
                    for v_idx in range(num_cams):
                        z_v_cam = z_v_seq[v_idx * P_per_cam : (v_idx + 1) * P_per_cam]
                        k_v = v_v = z_v_cam.unsqueeze(0)
                        with torch.no_grad():
                            _, attn_w_v = pool.cross_attn(
                                q_v, k_v, v_v, need_weights=True, average_attn_weights=False
                            )
                        w_v = attn_w_v[0, :, 0].mean(dim=0).cpu().numpy()  # (P,)
                        per_cam_weights.append(w_v)
                else:
                    # Separate: (V, P, vision_dim)
                    for v_idx in range(z_v_seq.shape[0]):
                        z_v_cam = z_v_seq[v_idx]  # (P, vision_dim)
                        k_v = v_v = z_v_cam.unsqueeze(0)
                        with torch.no_grad():
                            _, attn_w_v = pool.cross_attn(
                                q_v, k_v, v_v, need_weights=True, average_attn_weights=False
                            )
                        w_v = attn_w_v[0, :, 0].mean(dim=0).cpu().numpy()  # (P,)
                        per_cam_weights.append(w_v)
                saved_heatmaps_per_cam.append((label, per_cam_weights))
            # Save per-camera visualizations
            orig_frame = frames[0]
            if orig_frame.ndim == 4:  # (V, H, W, 3) multi-cam
                for cam_idx, cam_name in enumerate(["cam0", "cam1"][:orig_frame.shape[0]]):
                    # Use original z_t's attention for this camera
                    img = orig_frame[cam_idx]
                    self_attn_grid = saved_heatmaps_per_cam[0][1][cam_idx]
                    visualize_attention(
                        img=img,
                        attn_weights=self_attn_grid,
                        out_path=os.path.join(
                            out_dir, f"attention_{cam_name}_z_t_original.png"
                        ),
                        title=f"Attention ({cam_name}, z_t=original)",
                    )
                # Side-by-side: rows = cameras, cols = z_t variants
                if orig_frame.shape[0] >= 2:
                    # Build a per-(cam, z_t) heatmap grid
                    n_cams = orig_frame.shape[0]
                    n_labels = len(saved_heatmaps_per_cam)
                    fig, axes = plt.subplots(
                        n_cams, n_labels,
                        figsize=(4 * n_labels, 4 * n_cams),
                    )
                    if n_cams == 1 and n_labels == 1:
                        axes = np.array([[axes]])
                    elif n_cams == 1:
                        axes = axes.reshape(1, -1)
                    elif n_labels == 1:
                        axes = axes.reshape(-1, 1)
                    for cam_idx in range(n_cams):
                        img = orig_frame[cam_idx]
                        for label_idx, (label, per_cam_w) in enumerate(saved_heatmaps_per_cam):
                            ax = axes[cam_idx, label_idx]
                            w = per_cam_w[cam_idx]
                            P_here = len(w)
                            grid_size = int(np.sqrt(P_here))
                            if grid_size * grid_size != P_here:
                                ax.text(0.5, 0.5, f"P={P_here} not square",
                                        ha="center", va="center")
                                ax.axis("off")
                                continue
                            attn_grid = w.reshape(grid_size, grid_size)
                            attn_grid_norm = (attn_grid - attn_grid.min()) / (
                                attn_grid.max() - attn_grid.min() + 1e-8
                            )
                            ax.imshow(img, alpha=0.6)
                            ax.imshow(
                                attn_grid_norm, cmap="hot", alpha=0.5,
                                interpolation="bilinear",
                                extent=(0, img.shape[1], img.shape[0], 0),
                            )
                            ax.set_title(f"cam {cam_idx}, z_t={label}")
                            ax.axis("off")
                    fig.suptitle("Per-camera state-conditioned attention: rows=cameras, cols=z_t")
                    fig.tight_layout()
                    out_path = os.path.join(out_dir, "attention_comparison.png")
                    fig.savefig(out_path, dpi=80, bbox_inches="tight")
                    plt.close(fig)
            else:  # (H, W, 3) single camera
                self_attn_grid = saved_heatmaps_per_cam[0][1][0]
                visualize_attention(
                    img=orig_frame,
                    attn_weights=self_attn_grid,
                    out_path=os.path.join(out_dir, "attention_z_t_original.png"),
                    title="Attention (z_t=original)",
                )
            print(f"  Saved attention visualizations to {out_dir}")
        except Exception as e:
            print(f"  Failed to save visualizations: {e}")


def visualize_attention(img: np.ndarray, attn_weights: np.ndarray,
                         out_path: str, title: str = "Attention"):
    """Overlay attention weights on the image as a heatmap.

    Args:
        img: (H, W, 3) uint8 image
        attn_weights: (P,) flat attention weights, P = grid_size^2
        out_path: where to save the visualization
        title: plot title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = len(attn_weights)
    grid_size = int(np.sqrt(P))
    if grid_size * grid_size != P:
        print(f"  Skipping: P={P} is not a perfect square")
        return
    # Reshape to grid
    attn_grid = attn_weights.reshape(grid_size, grid_size)
    # Normalize for visualization
    attn_grid_norm = (attn_grid - attn_grid.min()) / (
        attn_grid.max() - attn_grid.min() + 1e-8
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # 1. Original image
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")
    # 2. Attention heatmap
    im = axes[1].imshow(
        attn_grid_norm, cmap="hot", interpolation="bilinear",
        extent=(0, img.shape[1], img.shape[0], 0),
    )
    axes[1].set_title("Attention (heatmap)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    # 3. Overlay
    axes[2].imshow(img, alpha=0.6)
    axes[2].imshow(
        attn_grid_norm, cmap="hot", alpha=0.5, interpolation="bilinear",
        extent=(0, img.shape[1], img.shape[0], 0),
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def side_by_side_attention(frames: np.ndarray, attn_weights_list: list,
                              labels: list, out_path: str):
    """Side-by-side attention heatmaps for multiple z_t values.

    Args:
        frames: (V, H, W, 3) — V cameras
        attn_weights_list: list of (P,) attention arrays, one per label
        labels: list of strings (e.g. ['original', 'zero', 'perturbed'])
        out_path: where to save
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V = frames.shape[0]
    n_labels = len(labels)
    fig, axes = plt.subplots(V, n_labels, figsize=(4 * n_labels, 4 * V))
    if V == 1 and n_labels == 1:
        axes = np.array([[axes]])
    elif V == 1:
        axes = axes.reshape(1, -1)
    elif n_labels == 1:
        axes = axes.reshape(-1, 1)
    for cam_idx in range(V):
        img = frames[cam_idx]  # (H, W, 3)
        for label_idx, (label, w) in enumerate(zip(labels, attn_weights_list)):
            ax = axes[cam_idx, label_idx]
            P = len(w)
            grid_size = int(np.sqrt(P))
            attn_grid = w.reshape(grid_size, grid_size)
            attn_grid_norm = (attn_grid - attn_grid.min()) / (
                attn_grid.max() - attn_grid.min() + 1e-8
            )
            ax.imshow(img, alpha=0.6)
            ax.imshow(
                attn_grid_norm, cmap="hot", alpha=0.5, interpolation="bilinear",
                extent=(0, img.shape[1], img.shape[0], 0),
            )
            ax.set_title(f"cam {cam_idx}, z_t={label}")
            ax.axis("off")
    fig.suptitle("State-conditioned attention: rows=cameras, cols=z_t")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test attention modules")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to intention_best.pt")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"],
                        help="Cameras to use (>=2 to test cross-camera)")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of samples to test")
    parser.add_argument("--n-frames", type=int, default=5,
                        help="Number of frames per sample")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None,
                        help="Save attention heatmap visualizations to this dir")
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

    # Find an episode
    with h5py.File(args.data, "r") as f:
        ep_keys = sorted([k for k in f.keys() if k.startswith("ep_")])
    if not ep_keys:
        print("No episodes found!")
        return
    print(f"  Found {len(ep_keys)} episodes, testing first {args.n_samples}")

    # Test on multiple episodes
    for i in range(min(args.n_samples, len(ep_keys))):
        ep_key = ep_keys[i]
        print(f"\n--- Sample {i+1}/{args.n_samples}: {ep_key} ---")
        frames, states, actions = load_batch(args.data, ep_key, args.cameras, args.n_frames)
        print(f"  Frames shape: {frames.shape}")
        print(f"  States shape: {states.shape}")

        # Run tests
        test_state_conditioned_pool(model, frames, states, device)
        test_cross_camera_attention(model, frames, states, device)
        test_z_t_recovery(model, frames, states, device)
        if i == 0:
            # Only run attention viz once (it's slow)
            test_attention_patterns(model, frames, states, device, cfg=cfg,
                                     out_dir=args.out_dir)


if __name__ == "__main__":
    main()
