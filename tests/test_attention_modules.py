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
    """Visualize attention weights from the state-conditioned pool across an episode.

    Instead of a single frame snapshot, this processes every frame in the sample and 
    produces a timeline grid so you can see exactly how the state-conditioned focus 
    shifts as the robot moves through the task.

    Args:
        model: the intention model
        frames: (T, V?, H, W, 3) uint8 images across time steps T.
        states: (T, 7) float32 EEF pose + gripper values over the horizon T
        device: torch device for inference.
        cfg: optional model config dict used to look up camera count or patch layout.
        out_dir: target folder if you want generated heatmaps saved to disk.
    """
    print("\n=== Test 4: Attention pattern visualization across timeline ===")

    # ---------------------------------------------------------------------------
    # 1. Forward pass through frozen DINOv2 + state encoder on the FULL horizon T 
    # ---------------------------------------------------------------------------
    T_ep = min(frames.shape[0], states.shape[0])
    all_frames_t = torch.from_numpy(frames[:T_ep]).unsqueeze(0).to(device)   # (1, T, ...)
    all_states_t = torch.from_numpy(states[:T_ep] ).float().unsqueeze(0).to(device)

    model.eval()
    intention_encoder = model.intention_encoder
    if intention_encoder is None:
        print("  SKIP: no intention encoder")
        return
    if intention_encoder.pool.pools is None or len(intention_encoder.pool.pools) == 0:
        print("  SKIP: no pool layers")
        return

    # First pool to extract per-step cross-attention weights from.
    pool = intention_encoder.pool.pools[0]
    num_cams_cfg = intention_encoder.pool.num_cameras

    with torch.no_grad():
        z_v_patches_per_t = []
        for t_idx in range(T_ep):
            # If input frames have camera dim stacked DINOv2 handles it.
            z_v_t = model._vision_forward(all_frames_t[:, t_idx])
            z_v_patches_per_t.append(z_v_t)
        # patch tokens (1, T, total_patches_p_or_V_total, vision_dim).
        patches_seq = torch.stack(z_v_patches_per_t, dim=1)
        states_enc  = model.state_encoder(all_states_t)  # (1, T, state_dim).

    # Detect per-camera geometry: how many raw patches each camera contributes?  
    first_frame_raw = patches_seq[0, 0]  # shape either (total_patches, dim) or (V, P, dim)
    if first_frame_raw.ndim == 2:        # concatenated multi-cam case
        P_per_cam   = first_frame_raw.shape[0] // num_cams_cfg
        is_concat   = True
    else:                                 # separate per-camera case.
        P_per_cam   = first_frame_raw.shape[1] if first_frame_raw.ndim == 3 else first_frame_raw.shape[0]
        is_concat   = False

    grid_dim = int(np.sqrt(P_per_cam))
    print(f"  T = {T_ep} steps | patches_per_cam = {P_per_cam} (grid {grid_dim}x{grid_dim})")

    # ---------------------------------------------------------------------------
    # 2. Extract per-timestep cross-attention weights (original z_t queries)      #
    # ---------------------------------------------------------------------------

    def _extract_weights_for_step(pt: torch.Tensor, st: torch.Tensor):
        """Return (num_cams, P_per_cam) averaged cross-attn from ``pool`."""
        # pt could be concatenated or multi-camera depending on upstream encoder.
        if is_concat:
            # split back into camera chunks.
            cam_patches = []
            for c in range(num_cams_cfg):
                start = c * P_per_cam
                end   = (c + 1) * P_per_cam
                cam_patches.append(pt[start:end])                       # (P, D)
        else:
            cam_patches = [pt[c] for c in range(num_cams_cfg)]         # (P, D).

        # Query is state embedding projected through pool.state_proj.
        q_proj = pool.state_proj(st.unsqueeze(0)).unsqueeze(1)  # (1, 1, D).

        cam_weights = []  # one array per camera of shape (P,)
        for c_idx in range(num_cams_cfg):
            k_v = cam_patches[c_idx].unsqueeze(0)              # (1, P, D).
            _, aw = pool.cross_attn(q_proj, k_v, k_v,
                                    need_weights=True, average_attn_weights=False)
            # aw : (B=1, heads, 1, P) -> avg over heads:
            w_avg = aw[0, :, 0].mean(dim=0).detach().cpu().numpy()   # (P,)
            cam_weights.append(w_avg.astype(np.float32))

        return np.stack(cam_weights, axis=0)   # (num_cams, P_per_cam)

    # Store every step's heatmaps so we can plot the grid later.
    timeline_weights = []  # list of (C, P) arrays; len == T_ep.

    print("  Extracting attention per timestep ...")
    for t_idx in range(T_ep):
        pt_t = patches_seq[0, t_idx].detach()   # (total_patches,) or (V,P,D)
        st_t = states_enc[0, t_idx].detach()      # (state_dim,)
        wts  = _extract_weights_for_step(pt_t, st_t)     # (num_cams, P)
        timeline_weights.append(wts)

    # Print per-timestep top-attended patches to console for quick inspection.
    for t_idx in range(T_ep):
        for c_idx in range(num_cams_cfg):
            w         = timeline_weights[t_idx][c_idx]      # (P,)
            top5      = np.argsort(w)[-5:][::-1]
            top_str   = " | ".join([f"idx={int(i)} w={w[i]:.3f}" for i in top5])
            print(f"  t={t_idx} cam_{c_idx}: top-5 patches → {top_str}")

    # ---------------------------------------------------------------------------
    # 3. Visualisation (optional grid of overlaid images over time)               #
    # ---------------------------------------------------------------------------
    if out_dir and timeline_weights:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(out_dir, exist_ok=True)

            # Build one big image per camera so that columns = time steps.
            for cam_idx in range(num_cams_cfg):
                # Extract frames correctly handling V-axis for multi-cam setups.
                if frames[0].ndim == 4:  # Multi-cam: shape is (T, V, H, W, 3) -> grab cam_idx properly
                    img_rows = np.array([frames[t, cam_idx] for t in range(T_ep)])  # (T, H, W, 3)
                else:  # Single-cam: shape is (T, H, W, 3)
                    img_rows = np.array([frames[t] for t in range(T_ep)])

                fig, axes = plt.subplots(1, T_ep, figsize=(5 * T_ep, 5))
                if T_ep == 1:
                    axes = np.array([axes])       # make it iterable for uniform loop.

                for t_idx in range(T_ep):
                    ax   = axes[t_idx] if T_ep > 1 else axes[0]
                    img  = img_rows[t_idx]           # (H, W, 3)
                    att  = timeline_weights[t_idx][cam_idx].reshape(grid_dim, grid_dim).astype(np.float64)

                    # Normalise attention to [0,1] per-step.
                    att_min, att_max = att.min(), att.max()
                    if att_max - att_min > 1e-8:
                        norm_att = (att - att_min) / (att_max - att_min)
                    else:
                        norm_att = np.zeros_like(att)

                    ax.imshow(img)
                    ax.imshow(
                        norm_att, cmap="hot", alpha=0.5, interpolation="bilinear",
                        extent=(0, img.shape[1], img.shape[0], 0),
                    )
                    ax.set_title(f"t={t_idx}")
                    ax.axis("off")

                fig.suptitle(f"Camera {cam_idx}: state-conditioned attention over episode timeline")
                fig.tight_layout()
                save_path = os.path.join(out_dir, f"attention_timeline_cam{cam_idx}.png")
                fig.savefig(save_path, dpi=80, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved timeline grid → {save_path}")

                # ---- Also stitch individual frames into an MP4 video --------
                _save_timeline_video(
                    out_dir=out_dir,
                    cam_idx=cam_idx,
                    img_rows=img_rows,
                    timeline_weights=timeline_weights,
                    T_ep=T_ep,
                    grid_dim=grid_dim,
                )

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  Failed to save visualisations: {e}")


def _save_timeline_video(out_dir, cam_idx, img_rows, timeline_weights, T_ep, grid_dim):
    """Render an MP4 video where each frame is a full-size image with a heatmap overlay."""
    import os, tempfile, shutil, subprocess
    from PIL import Image
    import numpy as np
    
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import cm

    tmp_dir = tempfile.mkdtemp()
    try:
        for t in range(T_ep):
            img = img_rows[t]                              # (H, W, 3) uint8
            att = timeline_weights[t][cam_idx].reshape(grid_dim, grid_dim)
            H, W = img.shape[:2]

            # Resize attention map to exact frame dimensions (bilinear interpolation)
            att_resized = np.array(Image.fromarray(att, 'F').resize((W, H), Image.BILINEAR))
            min_v, max_v = att_resized.min(), att_resized.max()
            norm_att = (att_resized - min_v) / (max_v - min_v + 1e-8)

            # Map normalized attention to the "hot" colormap 
            heat_rgb = cm.hot(norm_att)[:,:,:3] * 255.0     # (H, W, 3) float32
            alpha = np.stack([norm_att]*3, axis=-1)         # (H, W, 3)
            
            # Blend original image and heatmap based on attention intensity 
            frame_out = np.clip( img.astype(np.float32)*(1.0 - alpha*0.6) + heat_rgb*(alpha*0.6), 0, 255 ).astype(np.uint8)
            
            Image.fromarray(frame_out).save(f"{tmp_dir}/f_{t:04d}.png")

        # Stitch individual overlayed frames into a single MP4 
        vid_path = os.path.join(out_dir, f"attention_video_cam{cam_idx}.mp4")
        if len(os.listdir(tmp_dir)) > 0:
            subprocess.run(
                f"ffmpeg -y -framerate {max(1, T_ep)} "
                f"-i {tmp_dir}/f_%04d.png -c:v mpeg4 -pix_fmt yuv420p -qscale 0 {vid_path}",
                shell=True, capture_output=True
            )
            if os.path.exists(vid_path):
                print(f"  Saved attention video -> {vid_path}")
    finally:
        shutil.rmtree(tmp_dir)


def visualize_attention(img: np.ndarray, attn_weights: np.ndarray,
                         out_path: str, title: str = "Attention"):
    """Overlay attention weights on a single image as a heatmap."""
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
    parser.add_argument("--n-frames", type=int, default=50,
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
