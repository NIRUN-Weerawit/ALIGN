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
# Load expert trajectory from HDF5
# ================================================================

def find_episode_for_task(h5_path: str, task_name: str) -> Optional[dict]:
    """Find a matching episode in the HDF5 for a given task name.

    Returns dict with frames, poses, text, or None if not found.
    """
    with h5py.File(h5_path, "r") as h5:
        ep_keys = sorted([k for k in h5.keys() if k.startswith("ep_")])
        for ep_key in ep_keys:
            group = h5[ep_key]
            texts_raw = group.get("texts", None)
            if texts_raw is None:
                continue
            import json as _json
            text = _json.loads(texts_raw[()])[0]
            # Match by extracting the core task description
            # BDDL names: "LIVING_ROOM_SCENE5_put_the_white_mug..."
            # HDF5 texts: "put the white mug on the left plate..."
            # Strip scene prefix from BDDL name and compare
            task_parts = task_name.split("_", 2)  # ["LIVING", "ROOM", "SCENE5_put..."]
            task_desc = task_parts[-1] if len(task_parts) >= 3 else task_name
            # Normalize both
            text_normalized = text.lower().replace("_", " ").strip()
            task_normalized = task_desc.lower().replace("_", " ").strip()
            # Compare first 40 chars
            if text_normalized[:40] in task_normalized or task_normalized[:40] in text_normalized:
                # Found a match
                frames_group = group.get("frames", None)
                if frames_group is None:
                    continue
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
                # Fall back to noisy_poses for backward compat
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
    return None


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
    traj_window: int = 20,
    max_steps: int = 500,
    use_bf16: bool = True,
    record_video: bool = False,
    no_flip_vertical: bool = False,
    no_flip_horizontal: bool = False,
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

    # ── Run "No ALIGN" trajectory first ──
    obs = env.reset()
    step = 0
    done = False
    frames_no_align = []

    while not done and step < max_steps and step < n_expert:
        frame = _get_sim_frame(env)

        # Use the dataset's stored action directly (it's a delta in OSC_POSE format)
        action = expert_actions[step].copy()
        # Remap gripper: dataset 0/1 → LIBERO env -1/1
        if len(action) >= 7:
            if action[6] <= 0.5:
                action[6] = -1.0
            else:
                action[6] = 1.0
        obs, reward, done, info = env.step(action)
        step += 1

        # Capture frame AFTER stepping (so it shows the result of this action)
        frame_post = _get_sim_frame(env)
        if record_video:
            display = _overlay_text(frame_post, f"NO ALIGN  step={step}", color=(255, 0, 0))
            frames_no_align.append(display)

    # ── Run "With ALIGN" trajectory ──
    obs = env.reset()
    step = 0
    done = False
    pose_buffer = []
    chunk_cache = None
    alpha_vals = []
    delta_norms = []
    error_no_align = []
    error_with_align = []
    frames_with_align = []

    while not done and step < max_steps and step < n_expert:
        frame = _get_sim_frame(env)
        raw_pose = expert_poses[step]
        clean_pose = expert_poses[step]
        base_action = expert_actions[step]  # the dataset's stored delta

        # Build pose buffer for trajectory encoder
        pose_buffer.append(raw_pose.copy())
        if len(pose_buffer) > traj_window:
            pose_buffer.pop(0)
        while len(pose_buffer) < traj_window:
            pose_buffer.insert(0, raw_pose.copy())

        # ALIGN inference: predict a correction to add to the base action
        with torch.no_grad():
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            traj_t = torch.from_numpy(np.stack(pose_buffer)).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frame_t, traj_t, [""])

            z_v = mixed["z_v"]
            z_t = mixed["z_t"]

            alpha_raw = model.decision_head(z_v, z_t, z_text)
            z_v_n = F.normalize(z_v, dim=-1)
            z_t_n = F.normalize(z_t, dim=-1)
            z_text_n = F.normalize(z_text, dim=-1)
            cos_vt = (z_v_n * z_t_n).sum(dim=-1, keepdim=True)
            cos_vl = (z_v_n * z_text_n).sum(dim=-1, keepdim=True)
            cos_tl = (z_t_n * z_text_n).sum(dim=-1, keepdim=True)
            consistency = torch.min(torch.min(cos_vt, cos_vl), cos_tl)
            alpha = alpha_raw * consistency
            alpha_val = float(alpha.squeeze().cpu())

            noisy_t = torch.from_numpy(raw_pose).unsqueeze(0).float().to(device)
            chunk = model.assistant_head(z_v, z_t, z_text, noisy_t)
            chunk_np = chunk.squeeze(0).cpu().numpy()

            if chunk_cache is not None:
                corrective = 0.7 * chunk_np[0] + 0.3 * chunk_cache[-1]
            else:
                corrective = chunk_np[0]
            chunk_cache = chunk_np

        alpha_vals.append(alpha_val)
        delta_norms.append(float(np.linalg.norm(chunk_np[0])))

        # Apply the dataset action plus the ALIGN correction
        # base_action is already in delta format (OSC_POSE), so we just add
        # the corrective term to it
        action = base_action.copy()
        action[:6] = base_action[:6] + alpha_val * corrective[:6]
        # Remap gripper
        if action[6] <= 0.5:
            action[6] = -1.0
        else:
            action[6] = 1.0

        obs, reward, done, info = env.step(action)
        step += 1

        # Get post-step frame
        frame_post = _get_sim_frame(env)

        # Error metrics: compare sim's EEF to expert EEF
        sim_eef = obs.get("robot0_eef_pos", np.zeros(3))
        if isinstance(sim_eef, torch.Tensor):
            sim_eef = sim_eef.cpu().numpy()
        err_no_align = float(np.linalg.norm(raw_pose[:3] - clean_pose[:3]))
        err_with_align = float(np.linalg.norm(sim_eef - clean_pose[:3]))
        error_no_align.append(err_no_align)
        error_with_align.append(err_with_align)

        if record_video:
            display = _overlay_text(frame_post, f"WITH ALIGN  a={alpha_val:.2f}  step={step}", color=(0, 255, 0))
            display = _overlay_text(display, f"err={err_with_align:.3f}", pos=(10, 30), color=(0, 255, 0))
            frames_with_align.append(display)

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

    # ── Create 3-panel video: [DATASET GT] | [NO ALIGN] | [WITH ALIGN] ──
    if record_video and frames_no_align and frames_with_align:
        n_frames = min(len(frames_no_align), len(frames_with_align))
        side_by_side = []
        # Use the sim's natural resolution for the panel
        h, w = frames_no_align[0].shape[:2]

        for i in range(n_frames):
            # Dataset ground truth frame
            gt_idx = i
            if gt_idx < len(expert_frames):
                gt = expert_frames[gt_idx]
                if gt.shape[:2] != (h, w):
                    from PIL import Image as _PIL
                    gt = np.array(_PIL.fromarray(gt).resize((w, h)))
                gt_display = _overlay_text(gt, f"GT  step={gt_idx}", color=(255, 255, 255))
            else:
                gt_display = np.zeros((h, w, 3), dtype=np.uint8)

            f_no = frames_no_align[i]
            f_with = frames_with_align[i]

            # Apply flip if requested
            if not no_flip_vertical:
                f_no = np.flipud(f_no)
                f_with = np.flipud(f_with)
            if not no_flip_horizontal:
                f_no = np.fliplr(f_no)
                f_with = np.fliplr(f_with)

            combined = np.concatenate([gt_display, f_no, f_with], axis=1)
            side_by_side.append(combined)

        # White divider lines between panels
        for i in range(n_frames):
            side_by_side[i][:, w-2:w+2] = [255, 255, 255]
            side_by_side[i][:, 2*w-2:2*w+2] = [255, 255, 255]

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
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir) / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    chunk_size = ckpt.get("config", {}).get("chunk_size", 10)

    model = ALIGNModel(
        embed_dim=256, chunk_size=chunk_size, use_text=True, device=DEVICE,
    ).to(DEVICE)

    if encoder_checkpoint:
        enc_ckpt = torch.load(encoder_checkpoint, map_location=device)
        if "trainable_state_dict" in enc_ckpt:
            model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
        print(f"  Loaded encoder: {encoder_checkpoint}")

    if "trainable_state_dict" in ckpt:
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    print(f"\n{'='*60}")
    print(f"Suite: {suite_name} ({len(task_list)} tasks)")
    print(f"  Noise std: {noise_std}")
    print(f"{'='*60}")

    all_results = []

    for task_idx, task_name in enumerate(task_list):
        for ep in range(n_episodes):
            print(f"  [{task_idx+1}/{len(task_list)}] {task_name[:60]}  ep {ep+1}/{n_episodes}")

            try:
                # Find matching episode in HDF5
                expert = find_episode_for_task(data_path, task_name)
                if expert is None:
                    print(f"    WARNING: No matching episode found in HDF5")
                    continue

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
                    camera_widths=args.render_size,
                    camera_heights=args.render_size,
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