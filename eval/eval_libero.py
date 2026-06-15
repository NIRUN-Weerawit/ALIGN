#!/usr/bin/env python3
"""Evaluate ALIGN on LIBERO simulation tasks with optional video recording.

Tests all LIBERO task suites in simulation, running ALIGN inference
in the loop. Reports α values, Δpose magnitudes, and task success.
Can record videos of episodes for visual inspection.

Usage:
    # Quick test on one task
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/eval_libero.py \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --suite libero_10 --n-episodes 1 --max-steps 200

    # Record video of one episode
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/eval_libero.py \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --suite libero_10 --n-episodes 1 --record-video

    # Full evaluation with videos
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/eval_libero.py \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --output-dir ./eval/libero_results --record-video
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel

try:
    from libero.libero.envs import OffScreenRenderEnv, TASK_MAPPING
except ImportError:
    raise ImportError("libero not installed. Run: pip install libero")

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


# ================================================================
# Task suites mapping
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
    from libero.libero import get_libero_path
    return os.path.join(get_libero_path("bddl_files"), suite_name, f"{task_name}.bddl")


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_quat(quat).as_rotvec()
    return np.zeros(3, dtype=np.float32)


# ================================================================
# Video recording
# ================================================================

def _overlay_text(frame: np.ndarray, text: str, pos=(10, 10), color=(0, 255, 0)):
    """Draw text overlay on a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=font)
    return np.array(img)


def record_episode_video(
    env,
    model: ALIGNModel,
    device: torch.device,
    task_description: str,
    z_text: torch.Tensor,
    output_path: str,
    chunk_size: int = 10,
    traj_window: int = 20,
    max_steps: int = 500,
    fps: int = 20,
    use_bf16: bool = True,
) -> dict:
    """Run one episode with ALIGN inference and record video.

    Saves an MP4 with agentview camera frames overlaid with α, Δ, and step info.
    """
    try:
        import imageio
    except ImportError:
        print("  imageio not installed. Install: pip install imageio[ffmpeg]")
        return run_episode(env, model, device, task_description, z_text,
                          chunk_size, traj_window, max_steps, use_bf16)

    obs = env.reset()
    step = 0
    done = False

    alpha_vals = []
    delta_norms = []
    pose_buffer = []
    chunk_cache = None
    frames_buffer = []

    while not done and step < max_steps:
        # Get camera image
        frame = None
        for cam_name in ["agentview_image", "wrist_camera", "agentview"]:
            if cam_name in obs:
                frame = obs[cam_name]
                break
        if frame is None:
            for k in obs:
                if "image" in k.lower() or "rgb" in k.lower():
                    frame = obs[k]
                    break

        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.shape[0] in (1, 3) and frame.ndim == 3:
                frame = frame.transpose(1, 2, 0)
            if frame.shape[:2] != (224, 224):
                frame = np.array(Image.fromarray(frame).resize((224, 224)))
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Get EEF pose
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0]))
        if isinstance(eef_pos, torch.Tensor):
            eef_pos = eef_pos.cpu().numpy()
        if isinstance(eef_quat, torch.Tensor):
            eef_quat = eef_quat.cpu().numpy()
        axis_angle = quat_to_axisangle(eef_quat)
        raw_pose = np.concatenate([eef_pos, axis_angle]).astype(np.float32)

        # Fill buffer
        pose_buffer.append(raw_pose.copy())
        if len(pose_buffer) > traj_window:
            pose_buffer.pop(0)
        while len(pose_buffer) < traj_window:
            pose_buffer.insert(0, raw_pose.copy())

        if step < 5:
            step += 1
            action = np.zeros(7, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            continue

        # ALIGN inference
        with torch.no_grad():
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            traj_t = torch.from_numpy(np.stack(pose_buffer)).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frame_t, traj_t, [""])

            z_v = mixed["z_v"]
            z_t = mixed["z_t"]

            alpha = model.decision_head(z_v, z_t, z_text)
            alpha_val = float(alpha.squeeze().cpu())

            noisy_t = torch.from_numpy(raw_pose).unsqueeze(0).float().to(device)
            chunk = model.assistant_head(z_v, z_t, z_text, noisy_t)
            chunk_np = chunk.squeeze(0).cpu().numpy()

            if chunk_cache is not None:
                corrective = 0.7 * chunk_np[0] + 0.3 * chunk_cache[-1]
            else:
                corrective = chunk_np[0]
            commanded_pose = raw_pose + alpha_val * corrective
            chunk_cache = chunk_np

        alpha_vals.append(alpha_val)
        delta_norms.append(float(np.linalg.norm(chunk_np[0])))

        # Overlay info on frame
        display = _overlay_text(frame, f"α={alpha_val:.2f}  Δ={np.linalg.norm(chunk_np[0]):.3f}  step={step}")
        display = _overlay_text(display, f"task: {task_description[:50]}", pos=(10, 30), color=(255, 255, 0))
        frames_buffer.append(display)

        # Step env
        action = np.zeros(7, dtype=np.float32)
        action[:6] = commanded_pose
        action[6] = -1.0
        obs, reward, done, info = env.step(action)
        step += 1

    success = info.get("success", False)

    # Write video
    if frames_buffer:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for f in frames_buffer:
            writer.append_data(f)
        writer.close()
        print(f"    Video saved: {output_path} ({len(frames_buffer)} frames)")

    return {
        "success": success,
        "steps": step,
        "mean_alpha": float(np.mean(alpha_vals)) if alpha_vals else 0.0,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
        "video_path": output_path if frames_buffer else None,
    }


def run_episode(
    env,
    model: ALIGNModel,
    device: torch.device,
    task_description: str,
    z_text: torch.Tensor,
    chunk_size: int = 10,
    traj_window: int = 20,
    max_steps: int = 500,
    use_bf16: bool = True,
) -> dict:
    """Run one episode with ALIGN inference (no video)."""
    obs = env.reset()
    step = 0
    done = False

    alpha_vals = []
    delta_norms = []
    pose_buffer = []
    chunk_cache = None

    while not done and step < max_steps:
        frame = None
        for cam_name in ["agentview_image", "wrist_camera", "agentview"]:
            if cam_name in obs:
                frame = obs[cam_name]
                break
        if frame is None:
            for k in obs:
                if "image" in k.lower() or "rgb" in k.lower():
                    frame = obs[k]
                    break

        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.shape[0] in (1, 3) and frame.ndim == 3:
                frame = frame.transpose(1, 2, 0)
            if frame.shape[:2] != (224, 224):
                frame = np.array(Image.fromarray(frame).resize((224, 224)))
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0]))
        if isinstance(eef_pos, torch.Tensor):
            eef_pos = eef_pos.cpu().numpy()
        if isinstance(eef_quat, torch.Tensor):
            eef_quat = eef_quat.cpu().numpy()
        axis_angle = quat_to_axisangle(eef_quat)
        raw_pose = np.concatenate([eef_pos, axis_angle]).astype(np.float32)

        pose_buffer.append(raw_pose.copy())
        if len(pose_buffer) > traj_window:
            pose_buffer.pop(0)
        while len(pose_buffer) < traj_window:
            pose_buffer.insert(0, raw_pose.copy())

        if step < 5:
            step += 1
            action = np.zeros(7, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            continue

        with torch.no_grad():
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            traj_t = torch.from_numpy(np.stack(pose_buffer)).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mixed = model.encode_mixed(frame_t, traj_t, [""])

            z_v = mixed["z_v"]
            z_t = mixed["z_t"]

            alpha = model.decision_head(z_v, z_t, z_text)
            alpha_val = float(alpha.squeeze().cpu())

            noisy_t = torch.from_numpy(raw_pose).unsqueeze(0).float().to(device)
            chunk = model.assistant_head(z_v, z_t, z_text, noisy_t)
            chunk_np = chunk.squeeze(0).cpu().numpy()

            if chunk_cache is not None:
                corrective = 0.7 * chunk_np[0] + 0.3 * chunk_cache[-1]
            else:
                corrective = chunk_np[0]
            commanded_pose = raw_pose + alpha_val * corrective
            chunk_cache = chunk_np

        alpha_vals.append(alpha_val)
        delta_norms.append(float(np.linalg.norm(chunk_np[0])))

        action = np.zeros(7, dtype=np.float32)
        action[:6] = commanded_pose
        action[6] = -1.0
        obs, reward, done, info = env.step(action)
        step += 1

    success = info.get("success", False)

    return {
        "success": success,
        "steps": step,
        "mean_alpha": float(np.mean(alpha_vals)) if alpha_vals else 0.0,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
    }


# ================================================================
# Main evaluation
# ================================================================

def evaluate_suite(
    suite_name: str,
    task_list: list,
    checkpoint_path: str,
    output_dir: str,
    device: str = None,
    n_episodes: int = 3,
    max_steps: int = 500,
    record_video: bool = False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir) / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    chunk_size = ckpt.get("config", {}).get("chunk_size", 10)

    model = ALIGNModel(
        embed_dim=256, chunk_size=chunk_size, use_text=True, device=device,
    ).to(device)

    if "trainable_state_dict" in ckpt:
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    print(f"\n{'='*60}")
    print(f"Suite: {suite_name} ({len(task_list)} tasks)")
    print(f"{'='*60}")

    all_results = []

    for task_idx, task_name in enumerate(task_list):
        for ep in range(n_episodes):
            print(f"  [{task_idx+1}/{len(task_list)}] {task_name[:60]}  ep {ep+1}/{n_episodes}")

            try:
                z_text = model.encode_text([task_name])

                bddl_path = get_bddl_path(suite_name, task_name)
                if not os.path.exists(bddl_path):
                    print(f"    WARNING: BDDL not found: {bddl_path}")
                    continue

                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    use_camera_obs=True,
                    camera_names=["agentview"],
                    camera_widths=224,
                    camera_heights=224,
                    reward_shaping=True,
                    control_freq=20,
                )

                if record_video:
                    video_path = str(out_dir / f"task_{task_idx:03d}_ep{ep}.mp4")
                    result = record_episode_video(
                        env=env, model=model, device=device,
                        task_description=task_name, z_text=z_text,
                        output_path=video_path,
                        chunk_size=chunk_size, max_steps=max_steps,
                    )
                else:
                    result = run_episode(
                        env=env, model=model, device=device,
                        task_description=task_name, z_text=z_text,
                        chunk_size=chunk_size, max_steps=max_steps,
                    )

                result["task_name"] = task_name
                result["task_idx"] = task_idx
                result["episode"] = ep
                all_results.append(result)

                status = "✓" if result["success"] else "✗"
                print(f"    {status}  α={result['mean_alpha']:.3f}  "
                      f"Δ={result['mean_delta_norm']:.4f}  "
                      f"steps={result['steps']}")

                env.close()

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    if all_results:
        success_rate = sum(r["success"] for r in all_results) / len(all_results)
        avg_alpha = float(np.mean([r["mean_alpha"] for r in all_results]))
        avg_delta = float(np.mean([r["mean_delta_norm"] for r in all_results]))

        print(f"\n  --- {suite_name} Summary ---")
        print(f"  Success rate: {success_rate:.1%} ({sum(r['success'] for r in all_results)}/{len(all_results)})")
        print(f"  Avg α:        {avg_alpha:.3f}")
        print(f"  Avg Δ:        {avg_delta:.4f}")

        results = {
            "suite": suite_name,
            "success_rate": success_rate,
            "avg_alpha": avg_alpha,
            "avg_delta": avg_delta,
            "n_tasks": len(task_list),
            "n_episodes": len(all_results),
            "checkpoint": checkpoint_path,
            "chunk_size": chunk_size,
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
    parser = argparse.ArgumentParser(description="Evaluate ALIGN on LIBERO simulation")
    parser.add_argument("--checkpoint", required=True, help="Heads checkpoint (.pt)")
    parser.add_argument("--output-dir", default="./eval/libero_results", help="Output directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--suite", default=None, choices=list(LIBERO_TASK_MAP.keys()),
                        help="Run only one suite (default: all)")
    parser.add_argument("--record-video", action="store_true",
                        help="Record MP4 videos of episodes")
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else list(LIBERO_TASK_MAP.keys())

    all_summaries = {}
    for suite_name in suites_to_run:
        if suite_name not in SUITE_TASK_LISTS:
            print(f"WARNING: No task list found for {suite_name}, skipping.")
            continue

        task_list = SUITE_TASK_LISTS[suite_name]

        result = evaluate_suite(
            suite_name=suite_name,
            task_list=task_list,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            record_video=args.record_video,
        )
        if result:
            all_summaries[suite_name] = result

    if all_summaries:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        total_eps = sum(s["n_episodes"] for s in all_summaries.values())
        total_success = sum(int(s["success_rate"] * s["n_episodes"]) for s in all_summaries.values())
        print(f"  Total episodes: {total_eps}, Success: {total_success} ({total_success/total_eps:.1%})")
        for name, res in all_summaries.items():
            print(f"  {name:20s}  success={res['success_rate']:.1%}  "
                  f"α={res['avg_alpha']:.3f}  Δ={res['avg_delta']:.4f}  "
                  f"(n={res['n_episodes']})")

        with open(Path(args.output_dir) / "summary.json", "w") as f:
            json.dump(json.loads(json.dumps(all_summaries, default=str)), f, indent=2)
        print(f"\nResults: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()