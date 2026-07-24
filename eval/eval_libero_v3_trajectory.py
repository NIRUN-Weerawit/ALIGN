#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate ALIGN v3 intention model on LIBERO trajectory data.

This script replays a LIBERO trajectory (frames + states + actions) and
compares the v3 intention model's predicted actions to:
  1. Expert actions (ground truth)
  2. Noised human actions (simulated user mistakes)

For each timestep, the v3 model takes the K past (frames, states) and
predicts K future actions. We compare:
  - prediction[k=0]   vs.  expert[k=0]    (next action)
  - prediction[k=0]   vs.  noised[k=0]    (noisy human action)

Metrics:
  - Per-dim MSE/RMSE/MAE between predicted and expert
  - Per-step MSE for k=0..K-1 (action chunking quality)
  - Step-1 cosine alignment
  - Error reduction: how much better is the prediction vs. the noised action?

Usage:
    # Single checkpoint
    python eval/eval_libero_v3_trajectory.py \
        --data data/libero_object.h5 \
        --checkpoint checkpoints/v3/libero_object/run_1/intention_best.pt \
        --n-episodes 5 --noise-std 0.05

    # With text
    python eval/eval_libero_v3_trajectory.py \
        --data data/libero_object.h5 \
        --checkpoint checkpoints/v3/libero_object/run_1/intention_best.pt \
        --task-text "pick up the cup"

Outputs:
  - Per-episode summary printed to stdout
  - Per-dim, per-step, error-reduction metrics
  - JSON summary saved to <checkpoint>.traj_eval.json
  - Trajectory plots (predicted vs. expert vs. noised)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Disable cuDNN — same fix as train_intention.py
torch.backends.cudnn.enabled = False
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from data.align_dataset import ALIGNDataset, head_collate
from eval.eval_intention import load_intention_model

# MuJoCo / LIBERO imports (optional — only needed for --use-mujoco)
try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    OffScreenRenderEnv = None
    get_libero_path = None

try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:
    _Rotation = None


# ================================================================
# Trajectory loading
# ================================================================

def load_trajectory(h5_path: str, episode_key: str,
                     cameras: List[str]) -> Optional[Dict]:
    """Load a single episode from the HDF5.

    Returns:
        dict with keys: frames, states, actions, poses, text, cam_name
        - frames: (N, V, H, W, 3) uint8 — multi-cam
        - states: (N, 7) float32 — robot states [pos(3), euler(3), gripper(1)]
        - actions: (N, 7) float32 — expert actions (pose deltas + gripper)
        - poses: (N, 6+) float32 — expert EEF poses
        - text: str — task description
    """
    with h5py.File(h5_path, "r") as h5:
        if episode_key not in h5:
            return None
        group = h5[episode_key]
        # Frames: multi-cam (N, V, H, W, 3)
        frames_group = group.get("frames", None)
        if frames_group is None:
            return None
        available = list(frames_group.keys()) if hasattr(frames_group, "keys") else []
        cam_list = [c for c in cameras if c in available]
        if not cam_list:
            cam_list = [available[0]] if available else None
            if cam_list is None:
                return None
        if len(cam_list) == 1:
            frames = frames_group[cam_list[0]][:]
        else:
            per_cam = [frames_group[c][:] for c in cam_list]
            frames = np.stack(per_cam, axis=1)  # (N, V, H, W, 3)
        # Poses (6-D) and Actions (7-D, with gripper as last column)
        poses = None
        if "poses" in group:
            poses = group["poses"][:]
        elif "noisy_poses" in group:
            poses = group["noisy_poses"][:]
        actions = group["actions"][:]  # (N, 7)
        # Build states: concat[poses, gripper] = (N, 7)
        # matches the v3 model's expected state format:
        # [pos_x, pos_y, pos_z, roll, pitch, yaw, gripper]
        if poses is not None:
            gripper = actions[:, -1:]  # (N, 1)
            states = np.concatenate([poses, gripper], axis=1).astype(np.float32)  # (N, 7)
        else:
            return None
        # Text
        text = ""
        if "texts" in group:
            try:
                text = json.loads(group["texts"][()])[0]
            except Exception:
                text = ""
        return {
            "frames": frames,
            "states": states,
            "actions": actions,
            "poses": poses,
            "text": text,
            "cam_name": cam_list[0] if len(cam_list) == 1 else cam_list,
        }


def list_episodes(h5_path: str) -> List[str]:
    """List all episode keys in the HDF5."""
    with h5py.File(h5_path, "r") as h5:
        return sorted([k for k in h5.keys() if k.startswith("ep_")])


# ================================================================
# Noise injection
# ================================================================

def inject_action_noise(actions: np.ndarray, std: float = 0.05,
                         rng: np.random.Generator = None) -> np.ndarray:
    """Add Gaussian noise to actions (simulating human mistakes).

    Args:
        actions: (N, 6) or (N, 7) array (last column is gripper, not noised)
        std: standard deviation of Gaussian noise
    Returns:
        (N, 6) or (N, 7) noised actions (gripper preserved)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noised = actions.copy()
    # Only noise the first 6 dimensions (pose deltas); keep gripper
    D = min(6, actions.shape[1])
    noised[:, :D] += rng.normal(0, std, size=actions[:, :D].shape).astype(np.float32)
    return noised


# ================================================================
# Sliding-window prediction
# ================================================================

# ================================================================
# Metrics
# ================================================================
# ================================================================

def compute_metrics(predicted: np.ndarray, expert: np.ndarray,
                     noised: np.ndarray) -> Dict:
    """Compute per-dim and overall metrics.

    Args:
        predicted: (N, 6) or (N, 7) — model predictions (NaN for early steps)
        expert:    (N, 6) or (N, 7) — ground truth actions
        noised:    (N, 6) or (N, 7) — noised human actions
    Returns:
        dict with per-dim and overall metrics
    """
    # Use only first 6 dims (pose deltas); gripper has its own metric
    D = min(6, predicted.shape[1], expert.shape[1], noised.shape[1])
    # Filter to valid (non-NaN) steps
    valid = ~np.isnan(predicted[:, :D]).any(axis=1)
    p = predicted[valid, :D]
    e = expert[valid, :D]
    n = noised[valid, :D]
    if len(p) == 0:
        return {}

    # Per-dim
    diff_pred = p - e
    diff_noised = n - e
    per_dim_mse = (diff_pred ** 2).mean(axis=0).tolist()
    per_dim_mae = np.abs(diff_pred).mean(axis=0).tolist()
    per_dim_rmse = np.sqrt(per_dim_mse).tolist()
    noised_per_dim_mse = (diff_noised ** 2).mean(axis=0).tolist()

    # Overall
    overall_mse = float(np.mean(per_dim_mse))
    overall_mae = float(np.mean(per_dim_mae))
    noised_overall_mse = float(np.mean(noised_per_dim_mse))

    # Step-1 alignment
    cos = np.sum(p * e, axis=1) / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(e, axis=1) + 1e-8
    )
    step1_cos = float(np.mean(cos))

    # Error reduction: how much does the model improve over noised?
    error_reduction = 1.0 - (overall_mse / max(noised_overall_mse, 1e-8))

    metrics = {
        "n_valid_steps": int(len(p)),
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "noised_overall_mse": noised_overall_mse,
        "error_reduction": error_reduction,
        "per_dim_mse": per_dim_mse,
        "per_dim_mae": per_dim_mae,
        "per_dim_rmse": per_dim_rmse,
        "noised_per_dim_mse": noised_per_dim_mse,
        "step1_cos": step1_cos,
    }

    # Gripper accuracy (if actions have 7 dims)
    if expert.shape[1] >= 7 and predicted.shape[1] >= 7:
        g_expert = expert[valid, 6] > 0
        g_pred = predicted[valid, 6] > 0
        g_noised = noised[valid, 6] > 0
        gripper_acc = float((g_pred == g_expert).mean())
        noised_gripper_acc = float((g_noised == g_expert).mean())
        metrics["gripper_acc"] = gripper_acc
        metrics["noised_gripper_acc"] = noised_gripper_acc

    return metrics


# ================================================================
# Plotting
# ================================================================

def plot_trajectory(episode_key: str, expert: np.ndarray,
                     noised: np.ndarray, predicted: np.ndarray,
                     out_dir: str):
    """Plot predicted vs. expert vs. noised actions over time."""
    os.makedirs(out_dir, exist_ok=True)
    valid = ~np.isnan(predicted).any(axis=1)
    t = np.arange(len(expert))
    t_valid = t[valid]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, ax in enumerate(axes.flat):
        ax.plot(t, expert[:, i], label="Expert", color="C1", linestyle="--", alpha=0.7)
        ax.plot(t, noised[:, i], label="Noised (human)", color="C2", alpha=0.5)
        ax.plot(t_valid, predicted[valid, i], label="V3 prediction", color="C0")
        ax.set_ylabel(dim_names[i])
        ax.legend(loc="upper right", fontsize=8)
    axes[2, 0].set_xlabel("timestep")
    axes[2, 1].set_xlabel("timestep")
    fig.suptitle(f"Episode {episode_key} — V3 action prediction")
    fig.tight_layout()
    fname = f"{episode_key.replace('/', '_')}_traj.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
# MuJoCo helpers
# ================================================================

def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle (rx,ry,rz)."""
    if _Rotation is not None:
        return _Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def get_sim_frame(env, key: str = "agentview_image",
                  render_size: int = 256,
                  flip_vertical: bool = True,
                  flip_horizontal: bool = False) -> np.ndarray:
    """Render a single frame from the MuJoCo sim via observation dict.

    LIBERO with use_camera_obs=True puts frames in obs as
    obs["agentview_image"] and obs["robot0_eye_in_hand_image"].

    Args:
        env: LIBERO env
        key: which camera key to use (image/agentview_image/wrist_image/...)
        render_size: not used (kept for compat)
        flip_vertical: if True, np.flipud() the frame (default True,
            matches the old eval_libero_trajectory.py convention)
        flip_horizontal: if True, np.fliplr() the frame (default False)

    Returns (H, W, 3) uint8.
    """
    # Map sim key names
    sim_key_map = {
        "image": "agentview_image",
        "agentview_image": "agentview_image",
        "wrist_image": "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_image": "robot0_eye_in_hand_image",
    }
    sim_key = sim_key_map.get(key, key)
    # Get observation
    try:
        obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    except Exception:
        img = np.zeros((render_size, render_size, 3), dtype=np.uint8)
    else:
        img = obs.get(sim_key)
        if img is None:
            # Fallback to any available camera
            for k in ["agentview_image", "robot0_eye_in_hand_image"]:
                img = obs.get(k)
                if img is not None:
                    break
        if img is None:
            img = np.zeros((render_size, render_size, 3), dtype=np.uint8)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 4:
            img = img[0]  # remove batch dim
        img = img.astype(np.uint8)
    # Apply orientation flip (match old eval_libero_trajectory.py)
    if flip_vertical:
        img = np.flipud(img).copy()
    if flip_horizontal:
        img = np.fliplr(img).copy()
    return img


def get_sim_eef_pose(obs: dict) -> np.ndarray:
    """Get current EEF pose from sim observation.

    Returns (6,) array: [x, y, z, rx, ry, rz] in world frame.
    """
    pos = obs.get("robot0_eef_pos", np.zeros(3))
    quat = obs.get("robot0_eef_quat", np.zeros(4))  # (x,y,z,w)
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    if isinstance(quat, torch.Tensor):
        quat = quat.cpu().numpy()
    aa = quat_to_axisangle(quat)
    return np.concatenate([pos, aa]).astype(np.float32)


def get_bddl_path(suite_name: str, task_name: str) -> str:
    """Get path to BDDL file for LIBERO task."""
    if get_libero_path is None:
        raise ImportError("libero not installed")
    # LIBERO task names are like "pick up the black bowl..."
    # BDDL files use lowercase with underscores only.
    safe_name = "".join(
        c if c.isalnum() else "_" for c in task_name.lower()
    ).strip("_")
    return os.path.join(get_libero_path("bddl_files"), suite_name, f"{safe_name}.bddl")


# Map of LIBERO suite name to task names
LIBERO_SUITE_TASKS = {
    "libero_spatial": [],
    "libero_object": [],
    "libero_goal": [],
    "libero_10": [],
    "libero_90": [],
}


def _try_load_libero_task_list(suite_name: str) -> List[str]:
    """Try to load the task list for a LIBERO suite.

    Falls back to empty list if libero_task_map can't be loaded.
    """
    if suite_name in LIBERO_SUITE_TASKS and LIBERO_SUITE_TASKS[suite_name]:
        return LIBERO_SUITE_TASKS[suite_name]
    try:
        import importlib.util as _util
        import os as _os
        libero_dir = _os.path.dirname(__import__("libero").__file__)
        bench_file = _os.path.join(libero_dir, "libero", "benchmark",
                                    "libero_suite_task_map.py")
        if not _os.path.exists(bench_file):
            return []
        spec = _util.spec_from_file_location("_libero_task_map", bench_file)
        mod = _util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "libero_task_map") and suite_name in mod.libero_task_map:
            LIBERO_SUITE_TASKS[suite_name] = mod.libero_task_map[suite_name]
            return LIBERO_SUITE_TASKS[suite_name]
    except Exception:
        pass
    return []


def run_replay_in_sim(
    env,
    expert_actions: np.ndarray,
    expert_poses: Optional[np.ndarray] = None,
    max_steps: int = 200,
    render_size: int = 256,
    use_camera: str = "agentview_image",
    flip_vertical: bool = True,
    flip_horizontal: bool = False,
) -> Dict:
    """Replay dataset's expert actions in MuJoCo sim. Record frames.

    This shows what the sim looks like if you apply the expert actions
    from the dataset. The sim state is reset, then each step applies
    the expert action (with gripper sign).

    Returns:
        dict with:
          - frames: list of (H, W, 3) uint8 sim frames (agentview)
          - sim_positions: (T, 6) array of EEF poses
          - errors: (T,) EEF position error vs dataset expert (if poses given)
          - n_steps: number of steps run
    """
    obs = env.reset()
    frames = []
    sim_positions = []
    errors = []

    # Pad expert_actions to (N, 7) if needed
    n_actions = len(expert_actions)
    actions = expert_actions[:max_steps].copy()
    if actions.shape[1] < 7:
        # Pad with zero gripper
        pad = np.zeros((actions.shape[0], 7 - actions.shape[1]), dtype=actions.dtype)
        actions = np.concatenate([actions, pad], axis=1)

    for step in range(min(len(actions), max_steps)):
        # Get current frame BEFORE step
        frame = get_sim_frame(env, key=use_camera, render_size=render_size,
                                flip_vertical=flip_vertical,
                                flip_horizontal=flip_horizontal)
        
        if frame is not None and frame.size > 0:
            frames.append(frame.copy())
        sim_eef = get_sim_eef_pose(obs)
        sim_positions.append(sim_eef)
        # Step sim with expert action
        action = actions[step].copy()
        # Clamp gripper to {-1, +1}
        if action.shape[0] >= 7:
            action[6] = 1.0 if action[6] <= 0.5 else -1.0
        obs, reward, done, info = env.step(action)
        # Get sim_eef AFTER step
        sim_eef_after = get_sim_eef_pose(obs)
        sim_positions[-1] = sim_eef_after
        # Compute EEF error vs expert (if poses available)
        if expert_poses is not None and step < len(expert_poses):
            expert_eef = expert_poses[step]
            err = float(np.linalg.norm(sim_eef_after[:3] - expert_eef[:3]))
            errors.append(err)
        else:
            errors.append(0.0)
        if done:
            break

    return {
        "frames": frames,
        "sim_positions": np.array(sim_positions),
        "errors": np.array(errors),
        "n_steps": len(frames),
    }


def run_model_in_sim(
    env,
    model: torch.nn.Module,
    device: torch.device,
    expert_actions: np.ndarray,
    expert_poses: Optional[np.ndarray] = None,
    chunk_size: int = 5,
    max_steps: int = 200,
    alpha: float = 1.0,
    action_scale: float = 1.0,
    switch_at: float = 0.5,
    render_size: int = 256,
    use_camera: str = "agentview_image",
    flip_vertical: bool = True,
    flip_horizontal: bool = False,
    debug: bool = False,
) -> Dict:
    """Run V4 model in MuJoCo sim. Record frames.

    Phase 1 (steps < switch_at * max_steps):
      Expert controls 100%. Model runs in background to build
      Mamba state, memory bank, and intent tokens.

    Phase 2 (steps >= switch_at * max_steps):
      Model controls 100%. Uses accumulated context to continue
      the task autonomously.

    At each step:
      1. Render sim frame
      2. Build K-window of past (frames, states) from sim
      3. Run V4 model -> a_model
      4. Apply: action = expert (phase 1) or a_model (phase 2)
      5. Step sim
      6. Compute EEF position error vs dataset expert (if poses given)

    Returns:
        dict with:
          - frames: list of (H, W, 3) uint8 sim frames
          - sim_positions: (T, 6) array of EEF poses
          - errors: (T,) EEF position error vs dataset expert (if poses given)
          - stored_actions: (T, 6) array of model predictions
          - n_steps: number of steps run
          - switch_step: int — step at which model took over
    """
    # Pad expert_actions to (N, 7) for the "noised human" baseline (alpha-blend)
    n_actions = len(expert_actions)
    actions = expert_actions[:max_steps].copy()
    if actions.shape[1] < 7:
        pad = np.zeros((actions.shape[0], 7 - actions.shape[1]), dtype=actions.dtype)
        actions = np.concatenate([actions, pad], axis=1)
    # Pad actions to max_steps by repeating last action (safe for Phase 1 indexing)
    if len(actions) < max_steps:
        pad = np.tile(actions[-1:], (max_steps - len(actions), 1))
        actions = np.concatenate([actions, pad], axis=0)
    obs = env.reset()
    frames = []
    sim_positions = []
    errors = []
    stored_actions = []

    # Reset memory bank at start of episode
    if getattr(model, 'use_memory_bank', False) and model.memory_module is not None:
        model.memory_module.reset(batch_size=1, device=device)

    # State buffer (K-window) — each entry is a (7,) state vector
    pose_buffer = []
    # Frame buffer (K-window) — each entry is (V, H, W, 3) uint8, V = num cameras
    frame_buffer = []

    def _normalize_frame(f):
        """Make sure frame is (H, W, 3) uint8."""
        if f is None:
            return None
        if f.ndim == 4:
            f = f[0]
        return f.astype(np.uint8)

    def _render_all_cameras():
        """Render all cameras and return (V, H, W, 3) uint8 stack."""
        per_cam = []
        for camera_view in use_camera:
            f = _normalize_frame(get_sim_frame(env, key=camera_view,
                                              render_size=render_size,
                                              flip_vertical=flip_vertical,
                                              flip_horizontal=flip_horizontal))
            per_cam.append(f)
        return np.stack(per_cam, axis=0)  # (V, H, W, 3)

    # Get initial sim state to populate buffer
    init_eef = get_sim_eef_pose(obs)
    init_state = np.concatenate([init_eef, [0.0]]).astype(np.float32)  # (7,)
    init_frame_stack = _render_all_cameras()  # (V, H, W, 3)

    # Pad initial buffers with K copies of the initial state/frame
    for k in range(chunk_size):
        pose_buffer.append(init_state.copy())
        frame_buffer.append(init_frame_stack.copy())

    n_steps = max_steps
    switch_step = int(min(len(actions), max_steps) * switch_at)

    for step in range(n_steps):
        # 1. Render current sim frame BEFORE step
        current_frame_stack = _render_all_cameras()  # (V, H, W, 3)
        # Record the first camera's view for video output
        frames.append(current_frame_stack[0].copy())

        sim_eef = get_sim_eef_pose(obs)
        sim_positions.append(sim_eef)

        # 2. Update sliding windows: pop oldest, push newest
        sim_state = np.concatenate([sim_eef, [0.0]]).astype(np.float32)
        pose_buffer.append(sim_state)
        pose_buffer.pop(0)
        frame_buffer.append(current_frame_stack)
        frame_buffer.pop(0)

        # 3. Build K-window tensors
        win_states = np.stack(pose_buffer, axis=0).astype(np.float32)  # (K, 7)
        win_frames = np.stack(frame_buffer, axis=0)  # (K, V, H, W, 3) uint8
        f_t = torch.from_numpy(win_frames).unsqueeze(0).to(device)  # (1, K, V, H, W, 3)
        s_t = torch.from_numpy(win_states).float().unsqueeze(0).to(device)

        # 4. Run V4 model (always — builds Mamba state, memory bank, intent tokens)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    out = model(f_t, s_t)
                    h_current = out["h_seq"][:, -1]
                    intent_emb = out.get("intent_emb", None)

                    # Memory bank step (if enabled)
                    if getattr(model, 'use_memory_bank', False) and intent_emb is not None:
                        z_v_current = out["z_v_pooled_seq"][:, -1]
                        z_s_current = out["z_s_seq"][:, -1]
                        z_v_fused, z_s_fused, intent_fused = model.memory_module(
                            z_v_current, z_s_current, intent_emb,
                        )
                        h_for_head = intent_fused
                    else:
                        h_for_head = intent_emb if intent_emb is not None else h_current

                    if model.head_type == "diffusion":
                        a_model_full = model.sample_actions(
                            out["z_v_pooled_seq"], out["z_s_seq"],
                            h_for_head,
                        )
                    else:
                        a_model_full = model.predict_actions(
                            out["z_v_pooled_seq"], out["z_s_seq"],
                            h_for_head,
                        )
        a_model = a_model_full[0, 0, :].float().cpu().numpy()  # (6,) or (7,)

        # 5. Build the final action
        a_model_scaled = a_model * action_scale

        if step < switch_step:
            # Phase 1: expert controls, model observes
            final_action = actions[step].copy()
        else:
            # Phase 2: model controls
            final_action = a_model_scaled.copy()
            # Gripper: use model's prediction
            if a_model.shape[0] >= 7:
                final_action[6] = 1.0 if a_model_scaled[6] >= 0.5 else -1.0
            elif final_action.shape[0] >= 7:
                final_action[6] = -1.0  # fallback: close gripper

        if debug:
            print(f"Step {step}: phase={'expert' if step < switch_step else 'model'} "
                  f"action={a_model}")

        stored_actions.append(a_model_scaled.copy())

        # 6. Step sim
        obs, reward, done, info = env.step(final_action)
        sim_eef_after = get_sim_eef_pose(obs)
        sim_positions[-1] = sim_eef_after
        # 7. Compute EEF error vs dataset expert (only while expert data exists)
        if expert_poses is not None and step < len(expert_poses):
            expert_eef = expert_poses[step]
            err = float(np.linalg.norm(sim_eef_after[:3] - expert_eef[:3]))
            errors.append(err)
        else:
            errors.append(0.0)
        if done:
            break

    return {
        "frames": frames,
        "sim_positions": np.array(sim_positions),
        "errors": np.array(errors),
        "stored_actions": np.array(stored_actions) if stored_actions else np.zeros((0, 6)),
        "n_steps": len(frames),
        "switch_step": switch_step,
        "action_magnitude_model": float(np.mean(np.linalg.norm(
            np.array(stored_actions), axis=1
        ))) if stored_actions else 0.0,
        "action_magnitude_dataset": float(np.mean(np.linalg.norm(
            actions[:, :6], axis=1
        ))),
    }


def _extract_dataset_frames(
    traj: Dict, max_steps: int = 200,
    target_camera: str = "image",
    flip_vertical: bool = True,
    flip_horizontal: bool = False,
) -> List[np.ndarray]:
    """Extract dataset frames for the given camera (default: agentview).

    Args:
        traj: trajectory dict with 'frames' key
        max_steps: max number of frames to extract
        target_camera: which camera to extract (default 'image' = agentview)
        flip_vertical: if True, np.flipud() each frame (match sim convention)
        flip_horizontal: if True, np.fliplr() each frame

    Returns a list of (H, W, 3) uint8 frames, up to max_steps.
    """
    frames_all = traj.get("frames")
    if frames_all is None:
        return []
    # traj["frames"] is (N, V, H, W, 3) or (N, H, W, 3)
    if frames_all.ndim == 5:
        # (N, V, H, W, 3) — find target camera
        cameras = traj.get("cam_name")
        if isinstance(cameras, list) and target_camera in cameras:
            cam_idx = cameras.index(target_camera)
        else:
            # Try to find by name in cam_name
            cam_name = traj.get("cam_name", "wrist_image")
            if isinstance(cam_name, list) and target_camera in cam_name:
                cam_idx = cam_name.index(target_camera)
            else:
                cam_idx = 0  # default to first view
        frames = frames_all[:, cam_idx]  # (N, H, W, 3)
    else:
        frames = frames_all  # (N, H, W, 3)
    out = []
    for i in range(min(len(frames), max_steps)):
        f = frames[i].astype(np.uint8)
        if flip_vertical:
            f = np.flipud(f).copy()
        if flip_horizontal:
            f = np.fliplr(f).copy()
        out.append(f)
    return out


def save_video_3panel(
    dataset_frames: List[np.ndarray],
    replay_frames: List[np.ndarray],
    model_frames: List[np.ndarray],
    out_path: str,
    fps: int = 20,
) -> None:
    """Save a 3-panel MP4 video: [dataset agentview, replay sim, model sim].

    All three lists are stacked horizontally. If lengths differ, the
    shortest is used (with looping for the dataset).
    """
    try:
        import imageio
    except ImportError:
        print(f"    ⚠️  imageio not installed; cannot save video")
        return

    n = min(len(dataset_frames), len(replay_frames), len(model_frames))
    if n == 0:
        return

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for i in range(n):
        # Resize all to same H, W
        d = dataset_frames[i]
        r = replay_frames[i]
        m = model_frames[i]
        # Resize if needed
        target_h, target_w = min(d.shape[0], r.shape[0], m.shape[0]), \
                              min(d.shape[1], r.shape[1], m.shape[1])
        # Ensure all are 3-channel
        if d.ndim == 2:
            d = np.stack([d] * 3, axis=-1)
        if r.ndim == 2:
            r = np.stack([r] * 3, axis=-1)
        if m.ndim == 2:
            m = np.stack([m] * 3, axis=-1)
        # Crop to same size
        d = d[:target_h, :target_w]
        r = r[:target_h, :target_w]
        m = m[:target_h, :target_w]
        panel = np.concatenate([d, r, m], axis=1)  # horizontal stack
        writer.append_data(panel)
    writer.close()


# ================================================================
# Main evaluation
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate v3 intention model on LIBERO trajectory data."
    )
    parser.add_argument("--data", required=True,
                        help="Path to HDF5 dataset.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to intention_best.pt")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"],
                        help="Camera names (default: wrist_image). "
                             "MUST match the cameras used during training "
                             "(e.g. 'image wrist_image' for 2-cam checkpoints).")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--noise-std", type=float, default=0.00001,
                        help="Gaussian noise std for noised actions.")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode.")
    parser.add_argument("--task-text", type=str, default=None,
                        help="Task text (default: read from HDF5 if available).")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for plots (default: alongside checkpoint).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Save trajectory plots (default: on).")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    # MuJoCo / LIBERO options
    parser.add_argument("--use-mujoco", action="store_true", default=True,
                        help="Run episodes in MuJoCo sim (default on).")
    parser.add_argument("--no-mujoco", dest="use_mujoco", action="store_false",
                        help="Skip MuJoCo evaluation (offline only).")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save 3-panel side-by-side video (default on).")
    parser.add_argument("--no-video", dest="save_video", action="store_false",
                        help="Skip video saving.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Blend factor: action = (1-alpha) * a_human + alpha * a_model. "
                             "Default 1.0 (use model only).")
    parser.add_argument("--switch-at", type=float, default=0.5,
                        help="Fraction of episode after which model takes over. "
                             "0.5 = switch at halfway. 0.0 = model from start. 1.0 = expert only.")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Scale factor applied to model actions before applying to sim. "
                             "Useful if model outputs are too small/large. "
                             "Example: 10.0 multiplies model action by 10. "
                             "Default: 1.0 (no scaling).")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step model action values.")
    parser.add_argument("--libero-suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_object",
                                 "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO benchmark suite.")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Frame size for MuJoCo rendering.")
    parser.add_argument("--no-flip-vertical", action="store_true",
                        help="Skip vertical flip on sim and dataset frames "
                             "(default: flip vertical like old eval).")
    parser.add_argument("--no-flip-horizontal", action="store_true",
                        help="Skip horizontal flip on sim and dataset frames "
                             "(default: don't flip horizontal).")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n=== ALIGN v3 Trajectory Evaluation ===")
    print(f"  Data:        {args.data}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {device}")
    print(f"  Noise std:   {args.noise_std}")
    print(f"  Cameras:     {args.cameras}")

    # Load model
    model, cfg = load_intention_model(args.checkpoint, device)
    chunk_size = cfg["chunk_size"]
    print(f"  Chunk (K):   {chunk_size}")
    print(f"  Head:        {cfg.get('head_type', 'mamba')}")
    # Verify camera count matches the model
    model_cameras = cfg.get("num_cameras", 1)
    args_cameras = len(args.cameras)
    if model_cameras != args_cameras:
        print(f"  ⚠️  Camera count mismatch!")
        print(f"      Model expects {model_cameras} cameras, but you passed {args_cameras} ({args.cameras})")
        print(f"      Use --cameras to match the training config.")
    if cfg.get("use_text", False):
        print(f"  Text:        enabled (dim={cfg.get('text_dim', 256)})")

    # List episodes
    episodes = list_episodes(args.data)[:args.n_episodes]
    if not episodes:
        print(f"  No episodes found in {args.data}")
        return
    print(f"  Episodes:    {len(episodes)} (out of {len(list_episodes(args.data))} total)")

    # Output directory for plots
    if args.out_dir is None:
        args.out_dir = str(Path(args.checkpoint).parent)
        
        # Make sure it exists
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    else:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Skip MuJoCo if --no-mujoco
    if not args.use_mujoco:
        print("  ⚠️  --no-mujoco passed; skipping MuJoCo sim eval.")
        print("      (Offline-only eval was removed; this script is sim-focused.)")
        return

    if not LIBERO_AVAILABLE:
        print(f"\n  ⚠️  libero is not installed. Run: pip install libero")
        return

    print(f"\n{'='*68}")
    print(f"=== MuJoCo Sim Evaluation (alpha={args.alpha}) ===")
    print(f"{'='*68}")

    # Try to load the LIBERO task list for this suite
    task_list = _try_load_libero_task_list(args.libero_suite)
    if not task_list:
        print(f"  ⚠️  Could not load task list for suite '{args.libero_suite}'")
        print(f"      Falling back to using trajectories from the HDF5")
    else:
        print(f"  Suite: {args.libero_suite} ({len(task_list)} tasks)")

    # Compute flip flags (same convention as old eval_libero_trajectory.py)
    flip_vertical = not args.no_flip_vertical
    flip_horizontal = not args.no_flip_horizontal
    print(f"  Flip vertical:   {flip_vertical}  (--no-flip-vertical to disable)")
    print(f"  Flip horizontal: {flip_horizontal}  (--no-flip-horizontal to enable)")

    mujoco_results = []
    for ep_idx, ep_key in enumerate(episodes):
        traj = load_trajectory(args.data, ep_key, args.cameras)
        if traj is None:
            continue
        # Find a matching LIBERO task
        task_name = traj.get("text", ep_key)
        if task_list and task_name not in task_list:
            best_match = None
            for t in task_list:
                if task_name.lower() in t.lower() or t.lower() in task_name.lower():
                    best_match = t
                    break
            if best_match:
                task_name = best_match
        bddl_path = get_bddl_path(args.libero_suite, task_name)
        if not os.path.exists(bddl_path):
            print(f"\n  [{ep_idx+1}/{len(episodes)}] {ep_key}: BDDL not found: {bddl_path}")
            continue

        print(f"\n  [{ep_idx+1}/{len(episodes)}] {ep_key} → task='{task_name}'")
        print(f"    BDDL: {bddl_path}")

        try:
            if OffScreenRenderEnv is None:
                raise ImportError("OffScreenRenderEnv not available")
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                use_camera_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_widths=args.render_size,
                camera_heights=args.render_size,
                reward_shaping=False,
                control_freq=20,
                initialization_noise=None,
            )
        except Exception as e:
            print(f"    ⚠️  Failed to create env: {e}")
            continue

        # ── Run 1: Replay expert actions in sim ──
        t0 = time.time()
        replay_result = run_replay_in_sim(
            env=env,
            expert_actions=traj["actions"],
            expert_poses=traj["poses"] if traj["poses"] is not None else None,
            max_steps=args.max_steps,
            render_size=args.render_size,
            use_camera="agentview_image",
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
        )
        t_replay = time.time() - t0

        # ── Run 2: Model rollout in sim ──
        t0 = time.time()
        model_result = run_model_in_sim(
            env=env,
            model=model,
            device=device,
            expert_actions=traj["actions"],
            expert_poses=traj["poses"] if traj["poses"] is not None else None,
            chunk_size=chunk_size,
            max_steps=args.max_steps,
            alpha=args.alpha,
            action_scale=args.action_scale if args.action_scale is not None else 1.0,
            switch_at=args.switch_at,
            render_size=args.render_size,
            use_camera=args.cameras,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            debug=args.debug,
        )
        t_model = time.time() - t0

        # ── Compute metrics ──
        # EEF error for model rollout vs expert
        eef_err_model = float(np.mean(model_result["errors"])) if len(model_result["errors"]) > 0 else float("nan")
        eef_err_replay = float(np.mean(replay_result["errors"])) if len(replay_result["errors"]) > 0 else float("nan")

        print(f"    Replay run:  {replay_result['n_steps']:3d} steps  "
              f"EEF err: {eef_err_replay:.4f} m  ({t_replay:.1f}s)")
        switch_step = model_result.get("switch_step", 0)
        print(f"    Model run:   {model_result['n_steps']:3d} steps  "
              f"EEF err: {eef_err_model:.4f} m  ({t_model:.1f}s)  "
              f"switch at step {switch_step}")
        # Diagnostic: action magnitude comparison
        mag_model = model_result.get("action_magnitude_model", 0.0)
        mag_dataset = model_result.get("action_magnitude_dataset", 0.0)
        if mag_dataset > 0:
            scale_ratio = mag_model / mag_dataset
            print(f"    Action magnitudes: model={mag_model:.4f}  "
                  f"dataset={mag_dataset:.4f}  ratio={scale_ratio:.3f}")
            if scale_ratio < 0.5:
                print(f"    ⚠️  Model output is {scale_ratio:.1%} of dataset action scale")
                print(f"      Try: --action-scale {1.0/max(scale_ratio, 0.01):.1f}")

        # ── Save video (3-panel side-by-side) ──
        if args.save_video:
            try:
                # Get dataset frames (agentview camera, numpy uint8)
                dataset_frames = _extract_dataset_frames(
                    traj, max_steps=args.max_steps,
                    target_camera="image",  # agentview
                    flip_vertical=False,
                    flip_horizontal=False,
                )
                video_path = os.path.join(
                    args.out_dir, f"{ep_key}_{args.libero_suite}.mp4",
                )
                save_video_3panel(
                    dataset_frames=dataset_frames,
                    replay_frames=replay_result["frames"],
                    model_frames=model_result["frames"],
                    out_path=video_path,
                    fps=20,
                )
                print(f"    Video: {video_path}")
            except Exception as e:
                print(f"    ⚠️  Failed to save video: {e}")

        # ── Save trajectory plot ──
        if args.plot and len(model_result["sim_positions"]) > 0:
            try:
                t_arr = np.arange(len(model_result["sim_positions"]))
                # fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
                # for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
                #     ax.plot(t_arr, model_result["sim_positions"][:, i],
                #             label="model_sim", color="C0")
                #     if len(replay_result["sim_positions"]) == len(t_arr):
                #         ax.plot(t_arr, replay_result["sim_positions"][:, i],
                #                 label="replay_sim", color="C2", alpha=0.6)
                #     if traj["poses"] is not None:
                #         ax.plot(t_arr[:len(traj["poses"])],
                #                 traj["poses"][:len(t_arr), i],
                #                 label="expert (dataset)", color="C1", linestyle="--")
                #     ax.set_ylabel(f"pos_{name}")
                #     ax.legend()
                # axes[0].set_title(
                #     f"{ep_key} — {task_name} (alpha={args.alpha})"
                # )
                # axes[-1].set_xlabel("timestep")
                
                fig, axes = plt.subplots(6, 1, figsize=(10, 6), sharex=True)
                for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z", "ax", "ay", "az"])):
                    ax.plot(t_arr, model_result["stored_actions"][:, i],
                            label="model_sim", color="C0")
                    if len(traj["actions"]) == len(t_arr):
                        ax.plot(t_arr, traj["actions"][:, i],
                                label="replay_sim", color="C2", alpha=0.6)
                    ax.set_ylabel(f"action_{name}")
                    ax.legend()  
                axes[0].set_title(
                    f"{ep_key} — {task_name} (alpha={args.alpha})"
                )
                axes[-1].set_xlabel("timestep")      
                fig.tight_layout()
                plot_path = os.path.join(args.out_dir, f"{ep_key}_mujoco_traj.png")
                fig.savefig(plot_path)
                plt.close(fig)
                print(f"    Plot: {plot_path}")
            except Exception as e:
                print(f"    ⚠️  Failed to save plot: {e}")

        # Track aggregate
        mujoco_results.append({
            "episode": ep_key,
            "task_name": task_name,
            "n_steps": model_result["n_steps"],
            "mean_error_replay": eef_err_replay,
            "mean_error_model": eef_err_model,
        })

        try:
            if env is not None and hasattr(env, "close"):
                env.close()
        except Exception:
            pass

    # ── Aggregate MuJoCo metrics ──
    if mujoco_results:
        replay_errs = [r["mean_error_replay"] for r in mujoco_results
                        if not np.isnan(r["mean_error_replay"])]
        model_errs = [r["mean_error_model"] for r in mujoco_results
                       if not np.isnan(r["mean_error_model"])]
        print(f"\n{'='*68}")
        print(f"=== MuJoCo Aggregate (alpha={args.alpha}) ===")
        print(f"{'='*68}")
        if replay_errs:
            print(f"  Replay EEF err:  {np.mean(replay_errs):.4f} m  (n={len(replay_errs)})")
        if model_errs:
            print(f"  Model  EEF err:  {np.mean(model_errs):.4f} m  (n={len(model_errs)})")
        print(f"\n  Per-episode:")
        for r in mujoco_results:
            tn = r.get("task_name", "?")
            print(f"    {tn[:50]:<50}  "
                  f"replay={r['mean_error_replay']:.4f}  "
                  f"model={r['mean_error_model']:.4f}  "
                  f"steps={r['n_steps']}")

        # Save JSON summary
        summary = {
            "checkpoint": args.checkpoint,
            "data": args.data,
            "alpha": args.alpha,
            "n_episodes": len(mujoco_results),
            "episodes": mujoco_results,
            "aggregate": {
                "mean_replay_err": float(np.mean(replay_errs)) if replay_errs else None,
                "mean_model_err": float(np.mean(model_errs)) if model_errs else None,
            },
        }
        summary_path = Path(args.checkpoint).with_suffix(".mujoco_eval.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
