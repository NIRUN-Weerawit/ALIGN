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

def predict_trajectory(model: torch.nn.Module, frames: np.ndarray,
                        states: np.ndarray, K: int, chunk_size: int,
                        device: torch.device,
                        z_text: Optional[torch.Tensor] = None) -> np.ndarray:
    """Run the v3 model in sliding-window mode over a trajectory.

    For each timestep t (where t >= K-1), take the K frames
    [t-K+1, t] and K states, predict K future actions, and use
    actions[0] as the next-step prediction.

    Args:
        frames: (N, V, H, W, 3) uint8
        states: (N, 7) float32
        K: chunk size (model's K)
        chunk_size: same as K (passed through)
        device: torch device
        z_text: (B, text_dim) or None
    Returns:
        (N,) array of predictions (NaN for early steps where we don't
        have enough history)
    """
    N = frames.shape[0]
    predictions = np.full((N, 6), np.nan, dtype=np.float32)
    model.eval()

    for t in range(K - 1, N):
        # Window of K past frames/states
        win_frames = frames[t - K + 1: t + 1]  # (K, V, H, W, 3)
        win_states = states[t - K + 1: t + 1]  # (K, 7)
        # To tensor: (1, K, V, H, W, 3) and (1, K, 7)
        f = torch.from_numpy(win_frames).unsqueeze(0).to(device)
        s = torch.from_numpy(win_states).float().unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    out = model(f, s)
                    h_current = out["h_seq"][:, -1]
                    # Use sample_actions for flow head, predict_actions otherwise
                    if model.head_type == "flow":
                        actions_pred = model.sample_actions(
                            out["z_v_pooled_seq"], out["z_t_seq"],
                            h_current, z_text=z_text,
                        )
                    else:
                        actions_pred = model.predict_actions(
                            out["z_v_pooled_seq"], out["z_t_seq"],
                            h_current, z_text=z_text,
                        )
        # Use the FIRST predicted action (k=0) as the next-step prediction
        predictions[t] = actions_pred[0, 0, :].float().cpu().numpy()
    return predictions


# ================================================================
# Metrics
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
                  render_size: int = 256) -> np.ndarray:
    """Render a single frame from the MuJoCo sim via observation dict.

    LIBERO with use_camera_obs=True puts frames in obs as
    obs["agentview_image"] and obs["robot0_eye_in_hand_image"].

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
        return np.zeros((render_size, render_size, 3), dtype=np.uint8)
    img = obs.get(sim_key)
    if img is None:
        # Fallback to any available camera
        for k in ["agentview_image", "robot0_eye_in_hand_image"]:
            img = obs.get(k)
            if img is not None:
                break
    if img is None:
        return np.zeros((render_size, render_size, 3), dtype=np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0]  # remove batch dim
    return img.astype(np.uint8)


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


def run_episode_in_sim(
    env,
    model: torch.nn.Module,
    device: torch.device,
    expert_actions: np.ndarray,
    expert_poses: np.ndarray,
    task_description: str,
    z_text: Optional[torch.Tensor],
    noise_std: float = 0.05,
    chunk_size: int = 5,
    max_steps: int = 200,
    alpha: float = 1.0,
    render_size: int = 256,
    use_camera: str = "agentview_image",
) -> Dict:
    """Run one episode in MuJoCo with the v3 model.

    Pipeline (alpha fixed to `alpha`, default 1.0):
      1. Reset sim
      2. At each step:
         a. Render current sim frame
         b. Build K-window of past (frames, states) from sim
         c. Run v3 model → predict K future actions (a_model)
         d. Apply: action = (1 - alpha) * a_human + alpha * a_model
         e. Step sim
      3. Track EEF position error vs expert

    Returns:
        dict with metrics: sim_positions, errors, etc.
    """
    n_expert = min(len(expert_actions), max_steps, len(expert_poses))

    # Inject noise into expert actions (the "noisy human")
    rng = np.random.default_rng(42)
    if noise_std > 0:
        noisy_actions = inject_action_noise(
            expert_actions[:n_expert], std=noise_std, rng=rng,
        )
    else:
        noisy_actions = expert_actions[:n_expert].copy()

    # State buffer (K-window of past states for the v3 model)
    pose_buffer = []  # list of (7,) states [pos(3), euler(3), gripper(1)]
    frame_buffer = []  # list of (H, W, 3) frames

    # Tracking
    sim_positions = []  # list of (6,) sim EEF poses
    errors_with_align = []  # EEF error vs expert at each step
    stored_actions = []  # model predictions (a_model)
    alpha_vals = []  # alpha used at each step

    # Initialize
    obs = env.reset()
    step = 0
    done = False

    # Pad the buffer for the first K-1 steps with the initial state
    init_pose = get_sim_eef_pose(obs)
    init_state = np.concatenate([init_pose, [0.0]])  # (7,) — gripper=0
    init_frame = get_sim_frame(env, key=use_camera, render_size=render_size)

    while not done and step < n_expert:
        # Update buffers with current sim state
        sim_pose = get_sim_eef_pose(obs)
        sim_state = np.concatenate([sim_pose, [0.0]]).astype(np.float32)  # (7,)
        sim_frame = get_sim_frame(env, key=use_camera, render_size=render_size)

        pose_buffer.append(sim_state)
        frame_buffer.append(sim_frame)
        if len(pose_buffer) > chunk_size:
            pose_buffer.pop(0)
        if len(frame_buffer) > chunk_size:
            frame_buffer.pop(0)
        # Pad if not enough history
        while len(pose_buffer) < chunk_size:
            pose_buffer.insert(0, sim_state.copy())
        while len(frame_buffer) < chunk_size:
            frame_buffer.insert(0, sim_frame.copy())

        # Stack to (K, 7) and (K, H, W, 3)
        win_states = np.stack(pose_buffer, axis=0).astype(np.float32)  # (K, 7)
        win_frames = np.stack(frame_buffer, axis=0)  # (K, H, W, 3) uint8

        # To torch: (1, K, 7) and (1, K, H, W, 3)
        f_t = torch.from_numpy(win_frames).unsqueeze(0).to(device)
        s_t = torch.from_numpy(win_states).float().unsqueeze(0).to(device)

        # Run v3 model
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    out = model(f_t, s_t)
                    h_current = out["h_seq"][:, -1]
                    if model.head_type == "flow":
                        a_model_full = model.sample_actions(
                            out["z_v_pooled_seq"], out["z_t_seq"],
                            h_current, z_text=z_text,
                        )
                    else:
                        a_model_full = model.predict_actions(
                            out["z_v_pooled_seq"], out["z_t_seq"],
                            h_current, z_text=z_text,
                        )
        # a_model_full: (1, K, 6) — take the first step's prediction
        a_model = a_model_full[0, 0, :].float().cpu().numpy()  # (6,)

        # Build the final action
        # Base action: noised human action (with gripper)
        base_action = noisy_actions[step].copy()  # (7,)
        # Apply alpha: action = (1 - alpha) * a_human + alpha * a_model
        # a_human has 7 dims (pose deltas + gripper); a_model has 6 dims
        final_action = (1.0 - alpha) * base_action.copy()
        final_action[:6] = final_action[:6] + alpha * a_model  # 7-dim
        # Keep gripper sign: clamp to {-1, +1}
        if final_action[6] <= 0.5:
            final_action[6] = 1.0
        else:
            final_action[6] = -1.0

        # Track
        stored_actions.append(a_model.copy())
        alpha_vals.append(alpha)

        # Step sim
        obs, reward, done, info = env.step(final_action)
        sim_eef = get_sim_eef_pose(obs)
        sim_positions.append(sim_eef)
        step += 1

        # Compute EEF error vs expert
        if step <= len(expert_poses):
            expert_eef = expert_poses[step - 1]  # (6,) at this step
            err = float(np.linalg.norm(sim_eef[:3] - expert_eef[:3]))
            errors_with_align.append(err)

    return {
        "sim_positions": np.array(sim_positions) if sim_positions else np.zeros((0, 6)),
        "errors_with_align": np.array(errors_with_align),
        "stored_actions": np.array(stored_actions) if stored_actions else np.zeros((0, 6)),
        "alpha_vals": np.array(alpha_vals),
        "n_steps": step,
    }


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
                        help="Camera names (default: wrist_image).")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Gaussian noise std for noised actions.")
    parser.add_argument("--max-steps", type=int, default=200,
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
    parser.add_argument("--use-mujoco", action="store_true", default=False,
                        help="Run episodes in MuJoCo sim (requires libero + mujoco).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Blend factor: action = (1-alpha) * a_human + alpha * a_model. "
                             "Default 1.0 (use model only).")
    parser.add_argument("--libero-suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_object",
                                 "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO benchmark suite.")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Frame size for MuJoCo rendering.")
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
    print(f"  Output dir:  {args.out_dir}")

    # Run evaluation per episode
    all_metrics = []
    for ep_idx, ep_key in enumerate(episodes):
        print(f"\n  Episode {ep_idx + 1}/{len(episodes)}: {ep_key}")
        traj = load_trajectory(args.data, ep_key, args.cameras)
        if traj is None:
            print(f"    ⚠️  Skipping (could not load)")
            continue

        frames = traj["frames"][:args.max_steps]
        states = traj["states"][:args.max_steps]
        expert = traj["actions"][:args.max_steps]
        text = traj["text"]
        print(f"    Frames:     {frames.shape}")
        print(f"    States:     {states.shape}")
        print(f"    Expert:     {expert.shape}")
        print(f"    Text:       {text[:80] if text else '(none)'}")

        # Inject noise
        rng = np.random.default_rng(42)
        noised = inject_action_noise(expert, std=args.noise_std, rng=rng)

        # Encode text (if model has text encoder)
        z_text = None
        if getattr(model, "text_encoder", None) is not None:
            task = args.task_text or text or "default task"
            z_text = model.text_encoder([task] * 1)

        # Run sliding-window prediction
        t0 = time.time()
        predicted = predict_trajectory(
            model, frames, states, chunk_size, chunk_size, device, z_text=z_text,
        )
        elapsed = time.time() - t0
        print(f"    Prediction: {predicted.shape} ({elapsed:.1f}s)")

        # Compute metrics
        metrics = compute_metrics(predicted, expert, noised)
        metrics["episode"] = ep_key
        metrics["text"] = text
        metrics["n_total_steps"] = len(expert)
        all_metrics.append(metrics)

        print(f"    MSE (pred):  {metrics['overall_mse']:.5f}")
        print(f"    MSE (noised): {metrics['noised_overall_mse']:.5f}")
        print(f"    Error reduction: {metrics['error_reduction'] * 100:+.1f}%")
        print(f"    Step-1 cos:  {metrics['step1_cos']:.4f}")
        print(f"    Per-dim RMSE: {[f'{r:.4f}' for r in metrics['per_dim_rmse']]}")
        if "gripper_acc" in metrics:
            print(f"    Gripper acc:  pred={metrics['gripper_acc']:.3f}  "
                  f"noised={metrics['noised_gripper_acc']:.3f}")

        # Save plot
        if args.plot:
            plot_path = plot_trajectory(
                ep_key, expert, noised, predicted, args.out_dir,
            )
            print(f"    Plot:       {plot_path}")

    # ================================================================
    # Optional: MuJoCo sim evaluation
    # ================================================================
    mujoco_results = []
    if args.use_mujoco:
        if not LIBERO_AVAILABLE:
            print(f"\n  ⚠️  --use-mujoco requested but libero is not installed.")
            print(f"      Run: pip install libero")
        else:
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

            for ep_idx, ep_key in enumerate(episodes):
                traj = load_trajectory(args.data, ep_key, args.cameras)
                if traj is None:
                    continue
                # Find a matching LIBERO task
                task_name = traj.get("text", ep_key)
                if task_list and task_name not in task_list:
                    # Try to find a partial match
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

                # Encode text (if model uses text)
                z_text_eval = None
                if getattr(model, "text_encoder", None) is not None:
                    z_text_eval = model.text_encoder([task_name] * 1)

                # Run the sim
                t0 = time.time()
                result = run_episode_in_sim(
                    env=env,
                    model=model,
                    device=device,
                    expert_actions=traj["actions"],
                    expert_poses=traj["poses"] if traj["poses"] is not None else np.zeros((len(traj["actions"]), 6)),
                    task_description=task_name,
                    z_text=z_text_eval,
                    noise_std=args.noise_std,
                    chunk_size=chunk_size,
                    max_steps=args.max_steps,
                    alpha=args.alpha,
                    render_size=args.render_size,
                    use_camera="agentview_image",
                )
                elapsed = time.time() - t0

                mean_err = float(np.mean(result["errors_with_align"])) if len(result["errors_with_align"]) > 0 else float("nan")
                result["task_name"] = task_name
                result["mean_error_with_align"] = mean_err
                result["elapsed"] = elapsed
                mujoco_results.append(result)

                print(f"    Steps:        {result['n_steps']}")
                print(f"    Mean EEF err: {mean_err:.4f} m ({elapsed:.1f}s)")

                # Save plot
                if args.plot and len(result["sim_positions"]) > 0:
                    try:
                        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                        t_arr = np.arange(len(result["sim_positions"]))
                        for i, (ax, name) in enumerate(zip(axes, ["x", "y"])):
                            ax.plot(t_arr, result["sim_positions"][:, i], label="sim", color="C0")
                            if traj["poses"] is not None:
                                ax.plot(t_arr[:len(traj["poses"])], traj["poses"][:len(t_arr), i],
                                        label="expert", color="C1", linestyle="--")
                            ax.set_ylabel(f"pos_{name}")
                            ax.legend()
                        axes[0].set_title(f"{ep_key} — {task_name} (alpha={args.alpha})")
                        axes[-1].set_xlabel("timestep")
                        fig.tight_layout()
                        plot_path = os.path.join(args.out_dir, f"{ep_key}_mujoco_traj.png")
                        fig.savefig(plot_path)
                        plt.close(fig)
                        print(f"    Plot: {plot_path}")
                    except Exception as e:
                        print(f"    ⚠️  Failed to save plot: {e}")

                try:
                    if env is not None and hasattr(env, "close"):
                        env.close()
                except Exception:
                    pass

            if mujoco_results:
                mean_errs = [r["mean_error_with_align"] for r in mujoco_results
                             if not np.isnan(r["mean_error_with_align"])]
                if mean_errs:
                    print(f"\n  MuJoCo aggregate (alpha={args.alpha}):")
                    print(f"    Mean EEF err: {np.mean(mean_errs):.4f} m  (n={len(mean_errs)})")
                    print(f"    Per-episode:")
                    for r in mujoco_results:
                        tn = r.get("task_name", "?")
                        ep = r.get("n_steps", 0)
                        err = r.get("mean_error_with_align", float("nan"))
                        print(f"      {tn[:50]:<50}  err={err:.4f}  steps={ep}")

    # Aggregate metrics
    if not all_metrics:
        print(f"\n  No episodes evaluated.")
        return

    print(f"\n{'='*68}")
    print(f"=== Aggregate over {len(all_metrics)} episodes ===")
    print(f"{'='*68}")
    keys = ["overall_mse", "noised_overall_mse", "error_reduction", "step1_cos"]
    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean_val = np.mean(vals)
        if "reduction" in k or "cos" in k:
            print(f"  {k:25s}: {mean_val:+.4f}  (min={min(vals):+.4f}, max={max(vals):+.4f})")
        else:
            print(f"  {k:25s}: {mean_val:.5f}  (min={min(vals):.5f}, max={max(vals):.5f})")

    # Per-dim aggregate
    print(f"\n  Per-dim RMSE (averaged across episodes):")
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    per_dim_avg = np.mean([m["per_dim_rmse"] for m in all_metrics], axis=0)
    for i, d in enumerate(dim_names):
        print(f"    {d:<6}: {per_dim_avg[i]:.5f}")

    # Save JSON summary
    summary = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "noise_std": args.noise_std,
        "n_episodes": len(all_metrics),
        "episodes": all_metrics,
        "aggregate": {
            "overall_mse": float(np.mean([m["overall_mse"] for m in all_metrics])),
            "noised_overall_mse": float(np.mean([m["noised_overall_mse"] for m in all_metrics])),
            "error_reduction": float(np.mean([m["error_reduction"] for m in all_metrics])),
            "step1_cos": float(np.mean([m["step1_cos"] for m in all_metrics])),
            "per_dim_rmse": per_dim_avg.tolist(),
        },
    }
    summary_path = Path(args.checkpoint).with_suffix(".traj_eval.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
