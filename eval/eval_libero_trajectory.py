#!/usr/bin/env python3
"""Evaluate ALIGN in LIBERO simulation using expert trajectories from the dataset.

Loads a LIBERO task + its expert trajectory from the pre-decoded HDF5,
replays the trajectory in MuJoCo with synthetic noise, and measures
whether ALIGN's correction brings the robot back toward the expert path.

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/eval_libero_trajectory.py \
        --data ./data/libero_10.h5 \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --encoder-checkpoint ./checkpoints/pretrain/pretrain/best.pt \
        --suite libero_10 --n-episodes 3 --noise-std 0.03

    # With video
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/eval_libero_trajectory.py \
        --data ./data/libero_10.h5 \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --encoder-checkpoint ./checkpoints/pretrain/pretrain/best.pt \
        --suite libero_10 --n-episodes 1 --noise-std 0.03 --record-video
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# MuJoCo EGL corrupts PyTorch's cuDNN state. Re-init CUDA after env creation.
os.environ.setdefault("MUJOCO_GPU_RENDERING", "0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel

try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
except ImportError:
    raise ImportError("libero not installed. Run: pip install libero")

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


# ================================================================
# Task mapping (same as eval_libero.py)
# ================================================================

LIBERO_TASK_MAP = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
    "libero_90": "libero_90",
}

SUITE_TASK_LISTS = {}

_benchmark_file = os.path.join(
    os.path.dirname(__import__("libero").__file__),
    "libero", "benchmark", "libero_suite_task_map.py"
)
if os.path.exists(_benchmark_file):
    import importlib.util as _util
    _spec = _util.spec_from_file_location("_libero_task_map", _benchmark_file)
    if _spec and _spec.loader:
        _mod = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for suite_key in LIBERO_TASK_MAP:
            if hasattr(_mod, 'libero_task_map') and suite_key in _mod.libero_task_map:
                SUITE_TASK_LISTS[suite_key] = _mod.libero_task_map[suite_key]


def get_bddl_path(suite_name: str, task_name: str) -> str:
    return os.path.join(get_libero_path("bddl_files"), suite_name, f"{task_name}.bddl")


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_quat(quat).as_rotvec()
    return np.zeros(3, dtype=np.float32)


def axisangle_to_quat(axisangle: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_rotvec(axisangle).as_quat()
    return np.array([1, 0, 0, 0])


# ================================================================
# Auto-detect architecture from checkpoint
# ================================================================

def _detect_arch_from_checkpoint(ckpt: dict) -> dict:
    """Auto-detect model architecture from checkpoint state dict keys and shapes.

    Returns a dict of kwargs to pass to ALIGNModel.__init__.
    """
    state = ckpt.get("trainable_state_dict", ckpt)
    cfg = ckpt.get("config", {})
    kwargs = {}

    # chunk_size
    kwargs["chunk_size"] = cfg.get("chunk_size", 5)

    # decision_K
    kwargs["decision_K"] = cfg.get("decision_K", kwargs["chunk_size"])

    # Detect decision head architecture
    has_transformer = any("transformer" in k for k in state)
    has_mlp = any(k.startswith("decision_head.mlp.") for k in state)

    if has_transformer:
        kwargs["decision_arch"] = "transformer"
        pe = state.get("decision_head.pos_encoding", None)
        if pe is not None:
            kwargs["d_model"] = pe.shape[1]
        layer_keys = [k for k in state if "transformer.layers." in k]
        layer_ids = set()
        for k in layer_keys:
            for p in k.split("."):
                if p.isdigit():
                    layer_ids.add(int(p))
        kwargs["num_layers"] = max(layer_ids) + 1 if layer_ids else 2
        l1 = state.get("decision_head.transformer.layers.0.linear1.weight", None)
        if l1 is not None:
            kwargs["dim_feedforward"] = l1.shape[0]
        d_model = kwargs.get("d_model", 384)
        for n in [8, 4, 2, 1]:
            if d_model % n == 0:
                kwargs["nhead"] = n
                break
        kwargs["dropout"] = 0.0
    elif has_mlp:
        kwargs["decision_arch"] = "mlp"
        w0 = state.get("decision_head.mlp.0.weight", None)
        if w0 is not None:
            kwargs["mlp_hidden_dim"] = w0.shape[0]
        mlp_weights = sorted([k for k in state if k.startswith("decision_head.mlp.") and "weight" in k])
        kwargs["mlp_num_layers"] = max(len(mlp_weights) - 1, 1)
    else:
        print("  WARNING: Could not detect decision head architecture from checkpoint keys.")

    # Detect assistant head architecture
    ah_w0 = state.get("assistant_head.mlp.0.weight", None)
    if ah_w0 is not None:
        kwargs["assistant_hidden"] = ah_w0.shape[0]
    ah_weights = sorted([k for k in state if k.startswith("assistant_head.mlp.") and "weight" in k])
    kwargs["assistant_layers"] = max(len(ah_weights) - 1, 1)
    kwargs["assistant_dropout"] = 0.0

    return kwargs


# ================================================================
# Load expert trajectory from HDF5
# ================================================================

def find_episode_for_task(h5_path: str, task_name: str,
                            use_first: bool = True) -> Optional[dict]:
    """Find a matching episode in the HDF5 for a given task name.

    Args:
        h5_path: Path to the HDF5 dataset.
        task_name: LIBERO task name (BDDL stem, e.g.
                   "LIVING_ROOM_SCENE5_put_the_white_mug..." or
                   "pick_up_the_black_bowl_on_the_cookie_box..." for libero_spatial).
        use_first: If True, return the first matching episode. If False, return
                   the best match (most word overlap).

    Returns dict with frames, poses, actions, text, or None if not found.
    """
    # Use the full BDDL name (don't strip scene prefix). This works for
    # both naming conventions:
    #   - libero_10/90: "LIVING_ROOM_SCENE5_put_the_white_mug_..."
    #     Text doesn't have the scene prefix words, so the overlap score
    #     is lower but still high enough to match correctly.
    #   - libero_spatial: "pick_up_the_black_bowl_..."
    #     Text has all the same words, so overlap is 100%.
    # Strip scene prefix (e.g. "LIVING_ROOM_SCENE2_put_both_..." → "put_both_...")
    # so libero_10/90 task names match the HDF5 text which has no prefix.
    import re as _re
    task_stripped = _re.sub(r'^[A-Z_]+_SCENE\d+_', '', task_name)
    task_words = set(task_stripped.lower().replace("_", " ").split())

    best_match = None
    best_score = -1
    task_words = set(task_stripped.lower().replace("_", " ").split())
    # Normalize the task name for exact comparison
    task_normalized = task_stripped.lower().replace("_", " ").strip()

    with h5py.File(h5_path, "r") as h5:
        ep_keys = sorted([k for k in h5.keys() if k.startswith("ep_")])
        for ep_key in ep_keys:
            group = h5[ep_key]
            texts_raw = group.get("texts", None)
            if texts_raw is None:
                continue
            import json as _json
            text = _json.loads(texts_raw[()])[0]
            text_normalized = text.lower().replace(",", "").replace(".", "").strip()

            # Tier 1: Exact match (after normalization). BDDL names like
            # "pick_up_the_black_bowl_next_to_the_plate..." should match
            # exactly to the stored text.
            if text_normalized == task_normalized:
                return _extract_episode(group, text)

            # Tier 2: Word overlap. For libero_10/90 with scene prefixes,
            # the text won't have those prefix words, so we use overlap
            # of task words IN the text. Require 95% to disambiguate
            # between similar tasks (e.g. "next_to_plate" vs
            # "next_to_cookie_box").
            text_words = set(text_normalized.split())
            if not task_words or not text_words:
                continue
            overlap = len(task_words & text_words)
            score = overlap / len(task_words)
            if use_first and score >= 0.95:
                return _extract_episode(group, text)
            elif score > best_score and score >= 0.95:
                best_score = score
                best_match = _extract_episode(group, text)
    return best_match


def _extract_episode(group, text: str) -> Optional[dict]:
    """Extract frames, poses, actions from a HDF5 episode group."""
    frames_group = group.get("frames", None)
    if frames_group is None:
        return None
    # Try wrist first, then front
    cam_name = None
    for c in ["wrist_image", "image"]:
        if c in frames_group:
            cam_name = c
            break
    if cam_name is None:
        cam_name = list(frames_group.keys())[0]
    frames = frames_group[cam_name][:]
    # Use the stored actions (already in OSC_POSE delta format)
    if "actions" in group:
        poses = group["actions"][:, :6]
    else:
        poses = group["noisy_poses"][:]
    return {
        "frames": frames,
        "poses": poses,
        "actions": group["actions"][:],
        "text": text,
        "cam_name": cam_name,
    }


# ================================================================
# Noise injection
# ================================================================

def inject_noise(poses: np.ndarray, std: float = 0.02, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    noisy = poses.copy()
    noisy[:, :3] += rng.normal(0, std, size=poses[:, :3].shape)
    noisy[:, 3:6] += rng.normal(0, std * 10, size=poses[:, 3:6].shape)
    return noisy


# ================================================================
# Episode runner in MuJoCo
# ================================================================

def run_episode_in_sim(
    env,
    model: ALIGNModel,
    device: torch.device,
    expert_frames: np.ndarray,
    expert_poses: np.ndarray,
    expert_actions: np.ndarray,
    task_description: str,
    z_text: torch.Tensor,
    noise_std: float = 0.0,
    chunk_size: int = 10,
    traj_window: int = 10,  # overridden by model.decision_K at runtime
    max_steps: int = 500,
    use_bf16: bool = True,
    record_video: bool = False,
    no_flip_vertical: bool = False,
    no_flip_horizontal: bool = False,
    fixed_alpha: float = None,
) -> dict:
    """Run one episode in MuJoCo with expert trajectory + synthetic noise.

    Runs TWO parallel trajectories for comparison:
      - No ALIGN: raw stored actions → step sim (baseline)
      - With ALIGN: stored actions + alignment correction → step sim

    Both are compared against the expert trajectory.
    When record_video is True, saves a 3-panel MP4:
      [DATASET GT] | [NO ALIGN] | [WITH ALIGN]
    """
    n_expert = min(len(expert_poses), max_steps, len(expert_actions))

    # ── Inject noise into expert poses / actions for BOTH branches ──
    # This ensures the comparison is fair: both branches start from the
    # SAME noised input. The "No ALIGN" baseline just ignores the noise.
    rng = np.random.default_rng(42)
    if noise_std > 0:
        noisy_poses = inject_noise(expert_poses[:n_expert], std=noise_std, rng=rng)
        # Also inject noise into the actions (delta in OSC_POSE format)
        # Noise on actions = noise on the intended motion
        noisy_actions = inject_noise(expert_actions[:n_expert], std=noise_std, rng=rng)
    else:
        noisy_poses = expert_poses[:n_expert].copy()
        noisy_actions = expert_actions[:n_expert].copy()

    # ── Run "No ALIGN" trajectory first ──
    # Uses the SAME noised actions as "With ALIGN" — fair comparison.
    # Tracks per-step error for comparison.
    obs = env.reset()
    step = 0
    done = False
    frames_no_align = []
    error_no_align_raw = []  # per-step errors for No ALIGN branch

    while not done and step < max_steps and step < n_expert:
        frame = _get_sim_frame(env)

        # Use the noised action (not the clean expert action)
        action = noisy_actions[step].copy()
        # Remap gripper: dataset 0/1 → LIBERO env -1/1
        if len(action) >= 7:
            if action[6] <= 0.5:
                action[6] = 1.0
            else:
                action[6] = -1.0
        obs, reward, done, info = env.step(action)
        step += 1

        # Per-step error: sim's EEF vs expert EEF (post-step, same timing as With ALIGN)
        sim_eef = obs.get("robot0_eef_pos", np.zeros(3))
        if isinstance(sim_eef, torch.Tensor):
            sim_eef = sim_eef.cpu().numpy()
        # After stepping with noisy_actions[step-1] (since step was already incremented),
        # sim should be AT expert_poses[step-1]
        if step > 0 and step - 1 < len(expert_poses):
            clean_pose = expert_poses[step - 1]
            err = float(np.linalg.norm(sim_eef - clean_pose[:3]))
            error_no_align_raw.append(err)

        # Capture frame AFTER stepping (so it shows the result of this action)
        frame_post = _get_sim_frame(env)
        if record_video:
            frames_no_align.append(frame_post.copy())
    # ── Run "With ALIGN" trajectory ──
    obs = env.reset()
    step = 0
    done = False
    pose_buffer = []
    chunk_cache = None
    alpha_vals = []
    delta_norms = []
    delta_vectors = []  # (dx, dy, dz) for quiver visualization
    error_no_align = []
    error_with_align = []
    frames_with_align = []

    while not done and step < max_steps and step < n_expert:
        # Get the ACTUAL sim pose BEFORE stepping (this is what the sim actually produced)
        sim_pose_before = np.concatenate([
            obs.get("robot0_eef_pos", np.zeros(3)),
            obs.get("robot0_eef_quat", np.zeros(4))[:3] if "robot0_eef_quat" in obs else np.zeros(3),
        ])
        frame = _get_sim_frame(env)
        clean_pose = expert_poses[step]  # ground truth for comparison
        base_action = noisy_actions[step]  # noised delta (same noise as "No ALIGN")

        # Build pose buffer from ACTUAL sim poses (not noisy_poses from the dataset)
        # Buffer size = model.decision_K so the future prediction head
        # receives exactly K past embeddings.
        traj_window = model.decision_K
        pose_buffer.append(sim_pose_before.copy())
        if len(pose_buffer) > traj_window:
            pose_buffer.pop(0)
        while len(pose_buffer) < traj_window:
            pose_buffer.insert(0, sim_pose_before.copy())

        # ALIGN inference: future prediction (Decision head) + corrective delta (Assistant head)
        with torch.no_grad():
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            traj_t = torch.from_numpy(np.stack(pose_buffer)).unsqueeze(0).float().to(device)

            # Get the actual future K poses (ground truth, what we want to predict)
            # These are the CLEAN expert poses from the dataset (the "true" future)
            K = model.decision_K
            future_start = min(step + 1, n_expert - 1)
            future_end = min(step + 1 + K, n_expert)
            future_poses = expert_poses[future_start:future_end]  # (K, 6) clean
            if len(future_poses) < K:
                # Pad with the last available pose
                pad = np.tile(future_poses[-1], (K - len(future_poses), 1))
                future_poses = np.concatenate([future_poses, pad], axis=0)
            traj_future_t = torch.from_numpy(future_poses).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # Encode current (noised) trajectory
                mixed = model.encode_mixed(frame_t, traj_t, [""])
                z_v = mixed["z_v"]
                z_t_tokens = mixed["z_t_tokens"]  # (1, K, D)
                z_text = mixed["z_text"]
                # Encode the actual future (clean) trajectory as targets
                mixed_future = model.encode_mixed(frame_t, traj_future_t, [""])
                z_t_future_tokens = mixed_future["z_t_tokens"]  # (1, K, D)

            # Decision head: predict K future embeddings from K past
            z_v_window = z_v.unsqueeze(1).expand(-1, K, -1)  # (1, K, D)
            z_v_target = z_v.unsqueeze(1).expand(-1, K, -1)  # (1, K, D) — same as input
            predicted_z_v, predicted_z_t = model.decision_head(
                z_v_window, z_t_tokens, z_text
            )

            # Compute α from prediction error, or use fixed value if specified
            if fixed_alpha is not None:
                alpha_val = fixed_alpha
            else:
                alpha = ALIGNModel.compute_alpha_from_predictions(
                    predicted_z_v, predicted_z_t,
                    z_v_target, z_t_future_tokens,
                    aggregation="weighted_mean", decay=0.7,
                )
                alpha_val = float(alpha.squeeze().cpu())

            # Assistant head: input is the current human action (delta),
            # not the current pose. The current EEF pose is encoded in z_t.
            current_action_t = torch.from_numpy(base_action[:6]).unsqueeze(0).float().to(device)
            chunk = model.assistant_head(z_v, z_t_tokens.mean(dim=1), z_text, current_action_t)
            chunk_np = chunk.squeeze(0).cpu().numpy()

            if chunk_cache is not None:
                corrective = 0.7 * chunk_np[0] + 0.3 * chunk_cache[-1]
            else:
                corrective = chunk_np[0]
            chunk_cache = chunk_np

        alpha_vals.append(alpha_val)
        delta_norms.append(float(np.linalg.norm(chunk_np[0])))
        delta_vectors.append(corrective[:3].copy())  # 3D position delta

        # Apply the dataset action plus the ALIGN correction
        # base_action is already in delta format (OSC_POSE), so we just add
        # the corrective term to it
        action = base_action.copy()
        action[:6] = base_action[:6] + alpha_val * corrective[:6]
        # Remap gripper
        if action[6] <= 0.5:
            action[6] = 1.0
        else:
            action[6] = -1.0

        obs, reward, done, info = env.step(action)
        step += 1

        # Get post-step frame
        frame_post = _get_sim_frame(env)

        # Error metrics: compare sim's EEF to expert EEF
        sim_eef = obs.get("robot0_eef_pos", np.zeros(3))
        if isinstance(sim_eef, torch.Tensor):
            sim_eef = sim_eef.cpu().numpy()
        # Error metrics: compare sim's EEF to expert EEF
        # "No ALIGN" error: use the pre-computed per-step error from the
        # "No ALIGN" branch (same timing: post-step, after the noised action).
        # "With ALIGN" error: post-step after the ALIGN-corrected action.
        err_no_align = error_no_align_raw[step - 1] if step <= len(error_no_align_raw) else 0.0
        err_with_align = float(np.linalg.norm(sim_eef - clean_pose[:3]))
        error_no_align.append(err_no_align)
        error_with_align.append(err_with_align)

        if record_video:
            # Store raw frame (text drawn AFTER flipping in video assembly
            # so text isn't flipped)
            frames_with_align.append(frame_post.copy())

    # The 'info' dict was last returned by env.step() in the ALIGN run
    success = False
    if 'info' in locals() and isinstance(info, dict):
        success = bool(info.get("success", False))

    # Summary
    avg_alpha = float(np.mean(alpha_vals)) if alpha_vals else 0.0
    avg_delta = float(np.mean(delta_norms)) if delta_norms else 0.0
    avg_err_no_align = float(np.mean(error_no_align)) if error_no_align else 0.0
    avg_err_with_align = float(np.mean(error_with_align)) if error_with_align else 0.0
    improvement = avg_err_no_align - avg_err_with_align
    improvement_pct = (improvement / avg_err_no_align * 100) if avg_err_no_align > 0 else 0.0

    result = {
        "success": success,
        "steps": step,
        "mean_alpha": avg_alpha,
        "mean_delta_norm": avg_delta,
        "mean_error_no_align": avg_err_no_align,
        "mean_error_with_align": avg_err_with_align,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "frames_buffer": None,
    }

    # ── Create 4-panel video: [DATASET GT] | [NO ALIGN] | [WITH ALIGN] | [Δ VECTOR] ──
    if record_video and frames_no_align and frames_with_align:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import cv2

        n_frames = min(len(frames_no_align), len(frames_with_align))
        side_by_side = []
        # Use the sim's natural resolution for the panel
        h, w = frames_no_align[0].shape[:2]

        # Pre-create figure for delta vector panel
        fig = plt.figure(figsize=(2.5, 2.5 * h / w))
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for i in range(n_frames):
            # Dataset ground truth frame (raw, no text — we'll add text after flip)
            gt_idx = i
            if gt_idx < len(expert_frames):
                gt = expert_frames[gt_idx]
                if gt.shape[:2] != (h, w):
                    from PIL import Image as _PIL
                    gt = np.array(_PIL.fromarray(gt).resize((w, h)))
                gt_raw = gt
            else:
                gt_raw = np.zeros((h, w, 3), dtype=np.uint8)

            f_no = frames_no_align[i]
            f_with = frames_with_align[i]

            # Apply flip if requested (BEFORE drawing text so text isn't flipped)
            if not no_flip_vertical:
                f_no = np.flipud(f_no).copy()
                f_with = np.flipud(f_with).copy()
            if not no_flip_horizontal:
                f_no = np.fliplr(f_no).copy()
                f_with = np.fliplr(f_with).copy()

            # Now draw text on the flipped frames (text will appear right-side up)
            f_no = _overlay_text(f_no, f"NO ALIGN  step={i+1}", color=(255, 0, 0))
            # Include the per-step metrics in the WITH ALIGN panel
            step_idx = i + 5  # warmup offset used in the run loop
            if i < len(error_with_align) and i < len(alpha_vals):
                f_with = _overlay_text(
                    f_with,
                    f"WITH ALIGN  a={alpha_vals[i]:.2f}  step={i+1}",
                    color=(0, 255, 0),
                )
                f_with = _overlay_text(
                    f_with,
                    f"err={error_with_align[i]:.3f}",
                    pos=(10, 30),
                    color=(0, 255, 0),
                )
            else:
                f_with = _overlay_text(f_with, f"WITH ALIGN  step={i+1}", color=(0, 255, 0))
            gt_display = _overlay_text(gt_raw, f"GT  step={gt_idx}", color=(255, 255, 255))

            # ── 4th panel: 3D delta vector quiver ──
            ax.clear()
            if i < len(delta_vectors):
                dv = delta_vectors[i]
                scale = 0.05
                ax.quiver(0, 0, 0, dv[0], dv[1], dv[2],
                          color='r', arrow_length_ratio=0.3, linewidth=2)
                # Axes reference
                ax.quiver(0, 0, 0, scale, 0, 0, color='gray', alpha=0.3, linewidth=1)
                ax.quiver(0, 0, 0, 0, scale, 0, color='gray', alpha=0.3, linewidth=1)
                ax.quiver(0, 0, 0, 0, 0, scale, color='gray', alpha=0.3, linewidth=1)
                ax.text(scale, 0, 0, 'X', color='gray', fontsize=6)
                ax.text(0, scale, 0, 'Y', color='gray', fontsize=6)
                ax.text(0, 0, scale, 'Z', color='gray', fontsize=6)
                # Set limits symmetric around origin
                max_abs = max(abs(dv).max(), 0.02)
                lim = max_abs * 1.5
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
            else:
                ax.set_xlim(-0.02, 0.02)
                ax.set_ylim(-0.02, 0.02)
                ax.set_zlim(-0.02, 0.02)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            # Title
            ax.text2D(0.5, 0.95, f"Δ  α={alpha_vals[i]:.2f}" if i < len(alpha_vals) else "Δ",
                      transform=ax.transAxes, ha='center', color='white', fontsize=8)

            fig.canvas.draw()
            buf = np.array(fig.canvas.buffer_rgba())
            # Convert RGBA → RGB, resize to match panel height
            delta_panel = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
            delta_panel = cv2.resize(delta_panel, (int(w * 0.8), h))

            combined = np.concatenate([gt_display, f_no, f_with, delta_panel], axis=1)
            side_by_side.append(combined)

        plt.close(fig)

        # White divider lines between panels
        for i in range(n_frames):
            side_by_side[i][:, w-2:w+2] = [255, 255, 255]
            side_by_side[i][:, 2*w-2:2*w+2] = [255, 255, 255]
            side_by_side[i][:, 3*w-2:3*w+2] = [255, 255, 255]

        result["frames_buffer"] = side_by_side

    return result


def _get_sim_frame(env, camera_name: str = "agentview") -> np.ndarray:
    """Extract and preprocess a camera frame from sim observation.

    Uses robosuite's _get_observations to match the dataset's pipeline.
    Returns (H, W, 3) uint8 numpy array, vertically flipped and horizontally
    mirrored to match the dataset's orientation.
    """
    try:
        obs = env.env._get_observations()
        frame = obs[camera_name + "_image"]
        # robosuite returns (H, W, C) float32 [0,1] — convert to uint8
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.transpose(1, 2, 0)
        # Flip to match dataset orientation (upside-down + mirrored by default
        # in robosuite's _get_observations; the dataset was captured via the
        # same pipeline so no flip is needed here).
        return frame
    except Exception:
        return np.zeros((256, 256, 3), dtype=np.uint8)


def _overlay_text(frame: np.ndarray, text: str, pos=(10, 10), color=(0, 255, 0)):
    """Draw text overlay on a frame."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=font)
    return np.array(img)


# ================================================================
# Main evaluation
# ================================================================

def evaluate_suite(
    suite_name: str,
    task_list: list,
    data_path: str,
    checkpoint_path: str,
    encoder_checkpoint: str = None,
    output_dir: str = "./eval/libero_traj_results",
    device: str = None,
    n_episodes: int = 3,
    noise_std: float = 0.0,
    max_steps: int = 500,
    record_video: bool = False,
    render_size: int = 256,
    no_flip_vertical: bool = False,
    no_flip_horizontal: bool = False,
    fixed_alpha: float = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir) / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    chunk_size = ckpt.get("config", {}).get("chunk_size", 10)

    # Auto-detect architecture from checkpoint
    detected = _detect_arch_from_checkpoint(ckpt)
    print(f"  Auto-detected architecture: {detected}")

    model = ALIGNModel(
        embed_dim=256, chunk_size=detected["chunk_size"], use_text=True, device=DEVICE,
        decision_K=detected["decision_K"],
        decision_arch=detected["decision_arch"],
        mlp_hidden_dim=detected.get("mlp_hidden_dim", 512),
        mlp_num_layers=detected.get("mlp_num_layers", 3),
        num_layers=detected.get("num_layers", 2),
        d_model=detected.get("d_model", 384),
        nhead=detected.get("nhead", 4),
        dropout=detected.get("dropout", 0.0),
        dim_feedforward=detected.get("dim_feedforward", 1024),
        assistant_hidden=detected["assistant_hidden"],
        assistant_layers=detected["assistant_layers"],
        assistant_dropout=detected.get("assistant_dropout", 0.0),
    ).to(DEVICE)

    if encoder_checkpoint:
        enc_ckpt = torch.load(encoder_checkpoint, map_location=device)
        if "trainable_state_dict" in enc_ckpt:
            # Load only encoder/mixer keys, skip head keys
            enc_state = enc_ckpt["trainable_state_dict"]
            encoder_keys = {
                k: v for k, v in enc_state.items()
                if "decision_head" not in k and "assistant_head" not in k
            }
            if encoder_keys:
                missing, unexpected = model.load_state_dict(encoder_keys, strict=False)
                print(f"  Loaded {len(encoder_keys)} encoder/mixer params from {encoder_checkpoint}")
        print(f"  Loaded encoder: {encoder_checkpoint}")

    if "trainable_state_dict" in ckpt:
        # Load only the keys that exist in the current model (skip mismatched head weights)
        head_state = ckpt["trainable_state_dict"]
        current_state = model.state_dict()
        compatible = {
            k: v for k, v in head_state.items()
            if k in current_state and v.shape == current_state[k].shape
        }
        skipped = len(head_state) - len(compatible)
        if compatible:
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            print(f"  Loaded {len(compatible)}/{len(head_state)} head params (skipped {skipped} mismatched)")
        if skipped > 0:
            print(f"  WARNING: {skipped} head keys had shape mismatch — likely old architecture. Heads are at random init.")
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    print(f"\n{'='*60}")
    print(f"Suite: {suite_name} ({len(task_list)} tasks)")
    print(f"  Noise std: {noise_std}")
    if fixed_alpha is not None:
        print(f"  Fixed α: {fixed_alpha} (bypassed Decision head)")
    print(f"{'='*60}")

    all_results = []

    for task_idx, task_name in enumerate(task_list):
        for ep in range(n_episodes):
            print(f"  [{task_idx+1}/{len(task_list)}] {task_name[:100]}  ep {ep+1}/{n_episodes}")

            try:
                # Find matching episode in HDF5
                # Use best-match (most word overlap) — not first-match.
                # First-match can return an unrelated task if its first 40
                # chars happen to overlap with the requested task.
                expert = find_episode_for_task(data_path, task_name, use_first=False)
                if expert is None:
                    print(f"    WARNING: No matching episode found in HDF5")
                    continue
                # Debug: print which episode was matched
                print(f"    [match] {expert['text'][:60]} (frames: {expert['frames'].shape[0]})")

                # Precompute text embedding
                z_text = model.encode_text([task_name])

                # Create simulation environment
                bddl_path = get_bddl_path(suite_name, task_name)
                if not os.path.exists(bddl_path):
                    print(f"    WARNING: BDDL not found: {bddl_path}")
                    continue

                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    use_camera_obs=True,
                    camera_names=["agentview", "robot0_eye_in_hand"],
                    camera_widths=render_size,
                    camera_heights=render_size,
                    reward_shaping=True,
                    control_freq=20,
                    initialization_noise=None,
                )

                # Re-init CUDA after MuJoCo/EGL grabs GPU context
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force cuDNN re-initialization with a small conv
                    _ = torch.nn.functional.conv2d(
                        torch.zeros(1, 3, 10, 10, device="cuda"),
                        torch.zeros(3, 3, 3, 3, device="cuda"),
                    )

                result = run_episode_in_sim(
                    env=env, model=model, device=DEVICE,
                    expert_frames=expert["frames"],
                    expert_poses=expert["poses"],
                    expert_actions=expert["actions"],
                    task_description=task_name,
                    z_text=z_text,
                    noise_std=noise_std,
                    chunk_size=chunk_size,
                    max_steps=max_steps,
                    record_video=record_video,
                    no_flip_vertical=no_flip_vertical,
                    no_flip_horizontal=no_flip_horizontal,
                    fixed_alpha=fixed_alpha,
                )

                result["task_name"] = task_name
                result["task_idx"] = task_idx
                result["episode"] = ep
                all_results.append(result)

                status = "✓" if result["improvement"] > 0 else "✗"
                print(f"    {status}  α={result['mean_alpha']:.3f}  "
                      f"Δ={result['mean_delta_norm']:.4f}  "
                      f"no_align={result['mean_error_no_align']:.4f}  align={result['mean_error_with_align']:.4f}  "
                      f"{result['improvement_pct']:+.1f}%")

                # Save video
                if record_video and result.get("frames_buffer"):
                    try:
                        import imageio
                        video_path = out_dir / f"task_{task_idx:03d}_ep{ep}.mp4"
                        writer = imageio.get_writer(str(video_path), fps=20, codec="libx264", quality=8)
                        for f in result["frames_buffer"]:
                            writer.append_data(f)
                        writer.close()
                        print(f"    Video: {video_path}")
                    except ImportError:
                        pass

                env.close()

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    if all_results:
        avg_alpha = float(np.mean([r["mean_alpha"] for r in all_results]))
        avg_delta = float(np.mean([r["mean_delta_norm"] for r in all_results]))
        avg_err_no_align = float(np.mean([r["mean_error_no_align"] for r in all_results]))
        avg_err_with_align = float(np.mean([r["mean_error_with_align"] for r in all_results]))
        avg_improvement = float(np.mean([r["improvement"] for r in all_results]))
        n_improved = sum(1 for r in all_results if r["improvement"] > 0)

        print(f"\n  --- {suite_name} Summary ---")
        print(f"  Avg α:              {avg_alpha:.3f}")
        print(f"  Avg Δ:              {avg_delta:.4f}")
        print(f"  No ALIGN error:     {avg_err_no_align:.4f}")
        print(f"  With ALIGN error:   {avg_err_with_align:.4f}")
        print(f"  Avg improvement:    {avg_improvement:.4f} ({avg_improvement/avg_err_no_align*100:+.1f}%)")
        print(f"  Episodes improved:  {n_improved}/{len(all_results)} ({n_improved/len(all_results):.0%})")

        results = {
            "suite": suite_name,
            "noise_std": noise_std,
            "n_episodes": len(all_results),
            "avg_alpha": avg_alpha,
            "avg_delta": avg_delta,
            "avg_error_no_align": avg_err_no_align,
            "avg_error_with_align": avg_err_with_align,
            "avg_improvement": avg_improvement,
            "avg_improvement_pct": avg_improvement / avg_err_no_align * 100 if avg_err_no_align > 0 else 0.0,
            "n_improved": n_improved,
            "details": all_results,
        }

        with open(out_dir / "results.json", "w") as f:
            json.dump(json.loads(json.dumps(results, default=str)), f, indent=2)

        return results

    return None


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ALIGN in LIBERO sim with expert trajectories from dataset"
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, help="Heads checkpoint (.pt)")
    parser.add_argument("--encoder-checkpoint", default=None, help="Phase 1 backbone")
    parser.add_argument("--output-dir", default="./eval/libero_traj_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--suite", default=None, choices=list(LIBERO_TASK_MAP.keys()),
                        help="Run only one suite (default: all)")
    parser.add_argument("--n-episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="Synthetic noise std (0 = clean replay)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--fixed-alpha", type=float, default=None,
                        help="Override α to a fixed value (e.g. 0.5, 1.0). "
                             "Omit to use Decision head's prediction.")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Sim camera render resolution (default 256, matches dataset)")
    parser.add_argument("--no-flip-vertical", action="store_true",
                        help="Skip vertical flip on sim frames (default: flip applied)")
    parser.add_argument("--no-flip-horizontal", action="store_true",
                        help="Skip horizontal flip on sim frames (default: flip applied)")
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else list(LIBERO_TASK_MAP.keys())

    all_summaries = {}
    for suite_name in suites_to_run:
        if suite_name not in SUITE_TASK_LISTS:
            print(f"WARNING: No task list found for {suite_name}, skipping.")
            continue

        result = evaluate_suite(
            suite_name=suite_name,
            task_list=SUITE_TASK_LISTS[suite_name],
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            encoder_checkpoint=args.encoder_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            n_episodes=args.n_episodes,
            noise_std=args.noise_std,
            max_steps=args.max_steps,
            record_video=args.record_video,
            render_size=args.render_size,
            no_flip_vertical=args.no_flip_vertical,
            no_flip_horizontal=args.no_flip_horizontal,
            fixed_alpha=args.fixed_alpha,
        )
        if result:
            all_summaries[suite_name] = result

    if all_summaries:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        total_eps = sum(s["n_episodes"] for s in all_summaries.values())
        total_improved = sum(s["n_improved"] for s in all_summaries.values())
        print(f"  Total episodes: {total_eps}, Improved: {total_improved} ({total_improved/total_eps:.0%})")
        for name, res in all_summaries.items():
            print(f"  {name:20s}  α={res['avg_alpha']:.3f}  "
                  f"no_align={res['avg_error_no_align']:.4f}  align={res['avg_error_with_align']:.4f}  "
                  f"{res['avg_improvement_pct']:+.1f}%  "
                  f"improved={res['n_improved']}/{res['n_episodes']}")

        with open(Path(args.output_dir) / "summary.json", "w") as f:
            json.dump(json.loads(json.dumps(all_summaries, default=str)), f, indent=2)
        print(f"\nResults: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()