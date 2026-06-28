#!/usr/bin/env python3
"""
Evaluate ALIGN in LIBERO simulation using the new alpha pipeline.

Pipeline (per docs/ALPHA_INTERVENTION_DESIGN.md):
  1. Load expert trajectory from HDF5
  2. Replay in MuJoCo with synthetic noise
  3. At each step:
     a. Encode current (noised) state with frozen encoder+mixer
     b. Apply WORLD MODEL to imagine counterfactual next states:
        s'_h = f(s, a_human)
        s'_m = f(s, a_model)
     c. Apply VALUE HEAD to score the counterfactuals:
        V(s'_h), V(s'_m)
     d. Compute alpha = sigmoid((V(s'_m) - V(s'_h)) / tau)
     e. Apply ASSISTANT HEAD to get corrective delta
     f. Step sim with: action = a_human + alpha * delta

Three branches compared:
  - No ALIGN:    raw noised action
  - With ALIGN:  noised action + alpha * corrective_delta
  - Fixed alpha: same as With ALIGN but alpha is constant (e.g., 0.5, 1.0)

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/eval_libero_trajectory.py \
        --data ./data/libero_10.h5 \
        --encoder-checkpoint ./checkpoints/pretrain_helios/.../best.pt \
        --world-model-checkpoint ./checkpoints/world_model/.../world_model_best.pt \
        --value-head-checkpoint ./checkpoints/value/.../value_best.pt \
        --heads-checkpoint ./checkpoints/heads_libero/.../heads_best.pt \
        --suite libero_10 --n-episodes 3 --noise-std 0.03

Notes:
  - All 4 components must use the same encoder+mixer architecture.
    Pass --encoder-checkpoint to ensure the mixer_dim matches.
  - The world model must be compatible with the encoder (embed_dim=256).
  - The value head is trained with GAIL rewards. Pass --gail-checkpoint
    if you want to also log GAIL reward for diagnostics (not required for alpha).
  - The heads checkpoint provides the AssistantHead (corrective delta).
    If omitted, the script falls back to zero correction (no ALIGN head).
  - alpha can be overridden with --fixed-alpha for ablation studies.
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
from models.world_model import create_world_model
from models.value_head import create_value_head

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
# Task mapping
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
    os.path.dirname(os.__import__("libero").__file__),
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
# Load expert trajectory from HDF5
# ================================================================

def find_episode_for_task(h5_path: str, task_name: str,
                            use_first: bool = True) -> Optional[dict]:
    """Find a matching episode in the HDF5 for a given task name."""
    import re as _re
    task_stripped = _re.sub(r'^[A-Z_]+_SCENE\d+_', '', task_name)
    task_words = set(task_stripped.lower().replace("_", " ").split())
    task_normalized = task_stripped.lower().replace("_", " ").strip()

    best_match = None
    best_score = -1

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

            if text_normalized == task_normalized:
                return _extract_episode(group, text)

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
    cam_name = None
    for c in ["wrist_image", "image"]:
        if c in frames_group:
            cam_name = c
            break
    if cam_name is None:
        cam_name = list(frames_group.keys())[0]
    frames = frames_group[cam_name][:]
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
# Sim frame helper
# ================================================================

def _get_sim_frame(env, key="agentview_image"):
    obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    img = obs.get(key)
    if img is None:
        for k in ["agentview_image", "image", "rgb"]:
            img = obs.get(k)
            if img is not None:
                break
    if img is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


# ================================================================
# World model input preparation helper
# ================================================================

def _prepare_world_model_input(world_model, z_v, z_t, K=5):
    """Add window dimension if the world model needs (B, K, D) input.

    Detects architecture via class name:
      - WorldModelMLP: check window_size attribute
      - WorldModelRNN / WorldModelTransformer: always use window
    """
    cls_name = type(world_model).__name__
    if cls_name == "WorldModelMLP":
        wm_window_size = getattr(world_model, "window_size", 0)
        if wm_window_size > 0:
            z_v_w = z_v.unsqueeze(1).expand(-1, wm_window_size, -1).contiguous()
            z_t_w = z_t.unsqueeze(1).expand(-1, wm_window_size, -1).contiguous()
            return z_v_w, z_t_w
        return z_v, z_t
    elif cls_name in ("WorldModelRNN", "WorldModelTransformer"):
        z_v_w = z_v.unsqueeze(1).expand(-1, K, -1).contiguous()
        z_t_w = z_t.unsqueeze(1).expand(-1, K, -1).contiguous()
        return z_v_w, z_t_w
    else:
        return z_v, z_t


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
    world_model=None,
    value_head=None,
    heads_model=None,
    noise_std: float = 0.0,
    traj_window: int = 5,
    max_steps: int = 500,
    use_bf16: bool = True,
    record_video: bool = False,
    no_flip_vertical: bool = False,
    no_flip_horizontal: bool = False,
    fixed_alpha: Optional[float] = None,
    tau: float = 1.0,
    use_alpha_from_v: bool = True,
) -> dict:
    """Run one episode in MuJoCo with expert trajectory + synthetic noise.

    New alpha pipeline (per docs/ALPHA_INTERVENTION_DESIGN.md):
      alpha = sigmoid((V(s'_m) - V(s'_h)) / tau)
    where s'_m, s'_h are imagined next states via the world model.

    Three branches compared:
      - No ALIGN:    raw noised action
      - With ALIGN:  noised action + alpha * corrective_delta
      - Fixed alpha: same as With ALIGN but alpha is constant
    """
    n_expert = min(len(expert_poses), max_steps, len(expert_actions))

    rng = np.random.default_rng(42)
    if noise_std > 0:
        noisy_poses = inject_noise(expert_poses[:n_expert], std=noise_std, rng=rng)
        noisy_actions = inject_noise(expert_actions[:n_expert], std=noise_std, rng=rng)
    else:
        noisy_poses = expert_poses[:n_expert].copy()
        noisy_actions = expert_actions[:n_expert].copy()

    # ── Run "No ALIGN" trajectory first ──
    obs = env.reset()
    step = 0
    done = False
    frames_no_align = []
    error_no_align_raw = []

    while not done and step < max_steps and step < n_expert:
        frame = _get_sim_frame(env)
        action = noisy_actions[step].copy()
        if len(action) >= 7:
            if action[6] <= 0.5:
                action[6] = 1.0
            else:
                action[6] = -1.0
        obs, reward, done, info = env.step(action)
        step += 1

        sim_eef = obs.get("robot0_eef_pos", np.zeros(3))
        if isinstance(sim_eef, torch.Tensor):
            sim_eef = sim_eef.cpu().numpy()
        if step > 0 and step - 1 < len(expert_poses):
            clean_pose = expert_poses[step - 1]
            err = float(np.linalg.norm(sim_eef - clean_pose[:3]))
            error_no_align_raw.append(err)

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
    delta_vectors = []
    error_no_align = []
    error_with_align = []
    frames_with_align = []

    while not done and step < max_steps and step < n_expert:
        sim_pose_before = np.concatenate([
            obs.get("robot0_eef_pos", np.zeros(3)),
            obs.get("robot0_eef_quat", np.zeros(4))[:3] if "robot0_eef_quat" in obs else np.zeros(3),
        ])
        frame = _get_sim_frame(env)
        clean_pose = expert_poses[step]
        base_action = noisy_actions[step]

        # Build pose buffer
        traj_window = max(traj_window, 5)
        pose_buffer.append(sim_pose_before.copy())
        if len(pose_buffer) > traj_window:
            pose_buffer.pop(0)
        while len(pose_buffer) < traj_window:
            pose_buffer.insert(0, sim_pose_before.copy())

        with torch.no_grad():
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            traj_t = torch.from_numpy(np.stack(pose_buffer)).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frame_t, traj_t, [task_description])
            z_v = mixed["z_v"].float()
            z_t = mixed["z_t"].float()
            z_text_local = mixed["z_text"].float()

            current_action_t = torch.from_numpy(base_action[:6]).unsqueeze(0).float().to(device)

            # ── Compute α via counterfactual imagination ──
            if fixed_alpha is not None:
                alpha_val = fixed_alpha
            elif not use_alpha_from_v or world_model is None or value_head is None:
                alpha_val = 0.5
            else:
                z_v_w, z_t_w = _prepare_world_model_input(world_model, z_v, z_t, K=traj_window)

                # Counterfactual: imagine next state for human action vs model action.
                # In the current single-action design, both use the same action;
                # in a multi-action design, a_model could be the assistant's
                # suggested action instead.
                z_v_h, z_t_h = world_model(z_v_w, z_t_w, z_text_local, current_action_t)
                z_v_m, z_t_m = world_model(z_v_w, z_t_w, z_text_local, current_action_t)

                v_h = value_head(z_v_h, z_t_h, z_text_local)
                v_m = value_head(z_v_m, z_t_m, z_text_local)

                diff = (v_m - v_h) / tau
                alpha_val = float(torch.sigmoid(diff).item())

            # ── Assistant head: K pose-relative goals ──
            # Output is now (B, K, 6) where goal[k] = desired_pose[t+k+1] - noisy_current_pose.
            # Use goal[0] as the model's proposed action (a_model).
            a_model = np.zeros(6, dtype=np.float32)  # default fallback
            if heads_model is not None and hasattr(heads_model, 'assistant_head'):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    goal_5steps = heads_model.assistant_head(
                        z_v, z_t, z_text_local, current_action_t
                    )
                goal_np = goal_5steps.squeeze(0).float().cpu().numpy()
                a_model = goal_np[0]

        alpha_vals.append(alpha_val)
        delta_norms.append(float(np.linalg.norm(a_model)))
        delta_vectors.append(a_model[:3].copy())

        # Apply α-weighted blend of human action and model's proposed action.
        # new: action = (1 - α) * a_human + α * a_model
        # (vs. the old formulation: action = a_human + α * corrective)
        action = (1.0 - alpha_val) * base_action.copy() + alpha_val * np.concatenate(
            [a_model, np.zeros(1, dtype=np.float32)]  # pad to 7-dim (with gripper)
        )
        if action[6] <= 0.5:
            action[6] = 1.0
        else:
            action[6] = -1.0

        obs, reward, done, info = env.step(action)
        step += 1

        frame_post = _get_sim_frame(env)
        sim_eef = obs.get("robot0_eef_pos", np.zeros(3))
        if isinstance(sim_eef, torch.Tensor):
            sim_eef = sim_eef.cpu().numpy()
        err_no_align = error_no_align_raw[step - 1] if step <= len(error_no_align_raw) else 0.0
        err_with_align = float(np.linalg.norm(sim_eef - clean_pose[:3]))
        error_no_align.append(err_no_align)
        error_with_align.append(err_with_align)

        if record_video:
            frames_with_align.append(frame_post.copy())

    success = False
    if 'info' in locals() and isinstance(info, dict):
        success = bool(info.get("success", False))

    avg_alpha = float(np.mean(alpha_vals)) if alpha_vals else 0.0
    avg_delta = float(np.mean(delta_norms)) if delta_norms else 0.0
    avg_err_no_align = float(np.mean(error_no_align)) if error_no_align else 0.0
    avg_err_with_align = float(np.mean(error_with_align)) if error_with_align else 0.0
    improvement = avg_err_no_align - avg_err_with_align
    improvement_pct = (improvement / avg_err_no_align * 100) if avg_err_no_align > 0 else 0.0

    return {
        "success": success,
        "steps": step,
        "mean_alpha": avg_alpha,
        "mean_delta_norm": avg_delta,
        "mean_error_no_align": avg_err_no_align,
        "mean_error_with_align": avg_err_with_align,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "frames_no_align": frames_no_align if record_video else None,
        "frames_with_align": frames_with_align if record_video else None,
        "alpha_vals": alpha_vals,
        "delta_norms": delta_norms,
        "delta_vectors": delta_vectors,
    }


# ================================================================
# Main evaluation
# ================================================================

def evaluate_suite(
    suite_name: str,
    task_list: list,
    data_path: str,
    encoder_checkpoint: str,
    world_model_checkpoint: Optional[str] = None,
    value_head_checkpoint: Optional[str] = None,
    heads_checkpoint: Optional[str] = None,
    gail_checkpoint: Optional[str] = None,
    output_dir: str = "./eval/libero_traj_results",
    device: Optional[str] = None,
    n_episodes: int = 3,
    noise_std: float = 0.02,
    max_steps: int = 500,
    record_video: bool = False,
    render_size: int = 256,
    no_flip_vertical: bool = False,
    no_flip_horizontal: bool = False,
    fixed_alpha: Optional[float] = None,
    tau: float = 1.0,
    use_alpha_from_v: bool = True,
) -> Optional[dict]:
    """Evaluate ALIGN across all tasks in a suite using the new alpha pipeline."""
    device = device or DEVICE
    device = torch.device(device)
    print(f"\n=== ALIGN LIBERO Trajectory Evaluation (NEW alpha pipeline) ===")
    print(f"  Encoder:        {encoder_checkpoint}")
    print(f"  World model:    {world_model_checkpoint or '(none — fixed alpha only)'}")
    print(f"  Value head:     {value_head_checkpoint or '(none — fixed alpha only)'}")
    print(f"  Heads (assistant): {heads_checkpoint or '(none — no corrective delta)'}")
    print(f"  GAIL (diagnostic): {gail_checkpoint or '(none)'}")
    print(f"  Noise std:      {noise_std}")
    print(f"  tau:            {tau}")
    if fixed_alpha is not None:
        print(f"  Fixed alpha:    {fixed_alpha}")
    if not use_alpha_from_v:
        print(f"  Alpha source:   fixed={fixed_alpha or 0.5} (--no-alpha-from-v)")

    out_dir = Path(output_dir) / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load encoder+mixer (frozen) ──
    enc_ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    enc_cfg = enc_ckpt.get("config", {}) if isinstance(enc_ckpt, dict) else {}
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=5,
        use_text=True,
        device=DEVICE,
        mixer_dim=enc_cfg.get("mixer_dim", 512),
        num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
    ).to(device)
    if "trainable_state_dict" in enc_ckpt:
        model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    model.freeze_backbone()
    model.freeze_all_encoders()
    model.eval()
    print(f"  Loaded encoder+mixer (mixer_dim={enc_cfg.get('mixer_dim', 512)})")

    # ── Load world model ──
    world_model = None
    if world_model_checkpoint:
        wm_ckpt = torch.load(world_model_checkpoint, map_location=device, weights_only=False)
        wm_cfg = wm_ckpt.get("config", {})
        wm_arch = wm_cfg.get("arch", "mlp")
        wm_kwargs = {}
        if wm_arch == "mlp":
            first_w = next(iter(wm_ckpt.get("world_model_state", {}).values()), None)
            if first_w is not None and first_w.shape[1] == 774:
                wm_kwargs["window_size"] = 0
            else:
                wm_kwargs["window_size"] = wm_cfg.get("window_size", 5)
            wm_kwargs.update({
                "hidden_dim": wm_cfg.get("mlp_hidden", 512),
                "num_layers": wm_cfg.get("mlp_layers", 3),
            })
        elif wm_arch == "rnn":
            max_l = 0
            for k in wm_ckpt.get("world_model_state", {}).keys():
                if k.startswith("gru.weight_ih_l"):
                    try:
                        l = int(k.split("_l")[-1])
                        max_l = max(max_l, l + 1)
                    except ValueError:
                        pass
            wm_kwargs = {
                "hidden_dim": wm_cfg.get("rnn_hidden_dim", wm_cfg.get("mlp_hidden", 256)),
                "num_rnn_layers": max_l if max_l > 0 else wm_cfg.get("num_rnn_layers", 1),
                "window_size": wm_cfg.get("window_size", 5),
            }
        elif wm_arch == "transformer":
            wm_kwargs = {
                "d_model": wm_cfg.get("transformer_d_model", 384),
                "nhead": wm_cfg.get("transformer_nhead", 4),
                "num_layers": wm_cfg.get("transformer_layers", 2),
                "dim_feedforward": wm_cfg.get("transformer_dim_ff", 1024),
                "dropout": wm_cfg.get("transformer_dropout", 0.0),
                "window_size": wm_cfg.get("window_size", 5),
            }
        world_model = create_world_model(
            arch=wm_arch,
            embed_dim=wm_cfg.get("embed_dim", 256),
            action_dim=wm_cfg.get("action_dim", 6),
            **wm_kwargs,
        ).to(device)
        world_model.load_state_dict(wm_ckpt["world_model_state"])
        world_model.eval()
        print(f"  Loaded world model ({wm_arch})")

    # ── Load value head ──
    value_head = None
    if value_head_checkpoint:
        val_ckpt = torch.load(value_head_checkpoint, map_location=device, weights_only=False)
        val_cfg = val_ckpt.get("config", {})
        value_head = create_value_head(
            embed_dim=val_cfg.get("embed_dim", 256),
            hidden_dim=val_cfg.get("hidden_dim", 256),
            num_layers=val_cfg.get("num_layers", 3),
        ).to(device)
        value_head.load_state_dict(val_ckpt["value_head_state"])
        value_head.eval()
        print(f"  Loaded value head")

    # ── Load heads (Assistant) ──
    heads_model = None
    if heads_checkpoint:
        heads_ckpt = torch.load(heads_checkpoint, map_location=device, weights_only=False)
        heads_model = ALIGNModel(
            embed_dim=256,
            chunk_size=heads_ckpt.get("config", {}).get("chunk_size", 5),
            use_text=True,
            device=DEVICE,
            mixer_dim=enc_cfg.get("mixer_dim", 512),
            num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
        ).to(device)
        if "trainable_state_dict" in heads_ckpt:
            heads_model.load_trainable_state_dict(heads_ckpt["trainable_state_dict"])
        elif "model_state_dict" in heads_ckpt:
            heads_model.load_state_dict(heads_ckpt["model_state_dict"], strict=False)
        heads_model.freeze_backbone()
        heads_model.freeze_all_encoders()
        heads_model.eval()
        print(f"  Loaded heads (assistant)")

    # ── Load GAIL (diagnostic only, not used in pipeline) ──
    if gail_checkpoint:
        print(f"  GAIL checkpoint provided (diagnostic, not loaded)")

    print(f"\n{'='*60}")
    print(f"Suite: {suite_name} ({len(task_list)} tasks)")
    print(f"{'='*60}")

    all_results = []

    for task_idx, task_name in enumerate(task_list):
        for ep in range(n_episodes):
            print(f"  [{task_idx+1}/{len(task_list)}] {task_name[:100]}  ep {ep+1}/{n_episodes}")

            try:
                expert = find_episode_for_task(data_path, task_name, use_first=False)
                if expert is None:
                    print(f"    WARNING: No matching episode found in HDF5")
                    continue
                print(f"    [match] {expert['text'][:60]} (frames: {expert['frames'].shape[0]})")

                z_text = model.encode_text([task_name])

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

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    _ = torch.nn.functional.conv2d(
                        torch.zeros(1, 3, 10, 10, device="cuda"),
                        torch.zeros(3, 3, 3, 3, device="cuda"),
                    )

                result = run_episode_in_sim(
                    env=env, model=model, device=device,
                    world_model=world_model,
                    value_head=value_head,
                    heads_model=heads_model,
                    expert_frames=expert["frames"],
                    expert_poses=expert["poses"],
                    expert_actions=expert["actions"],
                    task_description=task_name,
                    z_text=z_text,
                    noise_std=noise_std,
                    traj_window=5,
                    max_steps=max_steps,
                    record_video=record_video,
                    no_flip_vertical=no_flip_vertical,
                    no_flip_horizontal=no_flip_horizontal,
                    fixed_alpha=fixed_alpha,
                    tau=tau,
                    use_alpha_from_v=use_alpha_from_v,
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

                if record_video and result.get("frames_no_align") and result.get("frames_with_align"):
                    try:
                        import imageio
                        video_path = out_dir / f"task_{task_idx:03d}_ep{ep}.mp4"
                        writer = imageio.get_writer(str(video_path), fps=20, codec="libx264", quality=8)
                        n_frames = min(len(result["frames_no_align"]),
                                       len(result["frames_with_align"]),
                                       len(expert["frames"]))
                        for i in range(n_frames):
                            gt = expert["frames"][i] if i < len(expert["frames"]) else result["frames_with_align"][i]
                            no_align = result["frames_no_align"][i]
                            with_align = result["frames_with_align"][i]
                            panel = np.concatenate([gt, no_align, with_align], axis=1)
                            writer.append_data(panel)
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
        print(f"  Avg improvement:    {avg_improvement:.4f} "
              f"({(avg_improvement/avg_err_no_align*100) if avg_err_no_align > 0 else 0:+.1f}%)")
        print(f"  Episodes improved:  {n_improved}/{len(all_results)} "
              f"({n_improved/len(all_results):.0%})")

        results = {
            "suite": suite_name,
            "noise_std": noise_std,
            "tau": tau,
            "use_alpha_from_v": use_alpha_from_v,
            "fixed_alpha": fixed_alpha,
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
        description="Evaluate ALIGN in LIBERO sim with the new alpha pipeline (world model + V)"
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Phase 1b encoder+mixer checkpoint")
    parser.add_argument("--world-model-checkpoint", default=None,
                        help="World model checkpoint (.pt) — required for α from V")
    parser.add_argument("--value-head-checkpoint", default=None,
                        help="Value head checkpoint (.pt) — required for α from V")
    parser.add_argument("--heads-checkpoint", default=None,
                        help="Heads checkpoint (.pt) with assistant_head — "
                             "optional, falls back to zero correction if omitted")
    parser.add_argument("--gail-checkpoint", default=None,
                        help="GAIL checkpoint (.pt) — diagnostic only")
    parser.add_argument("--output-dir", default="./eval/libero_traj_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--suite", default=None, choices=list(LIBERO_TASK_MAP.keys()),
                        help="Run only one suite (default: all)")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="Synthetic noise std (0 = clean replay)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--fixed-alpha", type=float, default=None,
                        help="Override α to a fixed value (e.g. 0.5, 1.0)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Temperature for sigmoid alpha (default 1.0)")
    parser.add_argument("--no-alpha-from-v", dest="use_alpha_from_v",
                        action="store_false", default=True,
                        help="Disable alpha-from-V (uses 0.5 default)")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Sim camera render resolution")
    parser.add_argument("--no-flip-vertical", action="store_true",
                        help="Skip vertical flip on sim frames")
    parser.add_argument("--no-flip-horizontal", action="store_true",
                        help="Skip horizontal flip on sim frames")
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
            encoder_checkpoint=args.encoder_checkpoint,
            world_model_checkpoint=args.world_model_checkpoint,
            value_head_checkpoint=args.value_head_checkpoint,
            heads_checkpoint=args.heads_checkpoint,
            gail_checkpoint=args.gail_checkpoint,
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
            tau=args.tau,
            use_alpha_from_v=args.use_alpha_from_v,
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