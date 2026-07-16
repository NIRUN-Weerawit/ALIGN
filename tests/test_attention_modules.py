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
        # 1. Original
        z_v_pooled_orig = []
        B, T = z_v_patches_seq.shape[:2]
        for t in range(T):
            z_v_t = z_v_patches_seq[:, t]
            z_t_t = z_t_seq[:, t]
            pooled = intention_encoder.pool_patches(z_v_t, z_t_t)
            z_v_pooled_orig.append(pooled)
        z_v_pooled_orig = torch.stack(z_v_pooled_orig, dim=1)
        # 2. Perturbed z_t
        z_t_perturbed = z_t_seq + torch.randn_like(z_t_seq) * 0.5
        z_v_pooled_perturbed = []
        for t in range(T):
            z_v_t = z_v_patches_seq[:, t]
            z_t_t = z_t_perturbed[:, t]
            pooled = intention_encoder.pool_patches(z_v_t, z_t_t)
            z_v_pooled_perturbed.append(pooled)
        z_v_pooled_perturbed = torch.stack(z_v_pooled_perturbed, dim=1)
        # 3. Zero z_t
        z_t_zero = torch.zeros_like(z_t_seq)
        z_v_pooled_zero = []
        for t in range(T):
            z_v_t = z_v_patches_seq[:, t]
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


def test_attention_patterns(model, frames, states, device):
    """Visualize attention weights from the state-conditioned pool.

    Shows which patches the model attends to for different z_t.
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
    z_v_for_pool = z_v_patches_seq[0, 0, 0] if z_v_patches_seq.ndim == 5 else z_v_patches_seq[0, 0]
    print(f"  z_v_for_pool shape: {z_v_for_pool.shape}")
    print(f"  z_t shape: {z_t_seq[0, 0].shape}")
    # Try to capture attention weights
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
            w = attn_w[0, 0, 0].cpu().numpy()  # head 0, (P,)
            top_5 = np.argsort(w)[-5:][::-1]
            print(f"  z_t={label}: top-5 attended patches = {top_5.tolist()}, "
                  f"weights = {[f'{w[i]:.3f}' for i in top_5]}")
        except Exception as e:
            print(f"  z_t={label}: failed to get attn weights ({e})")
            return


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
            test_attention_patterns(model, frames, states, device)


if __name__ == "__main__":
    main()
