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
            # Match by checking if task_name is a substring of the text
            # (HDF5 text may be shorter than the full BDDL task name)
            task_short = task_name.replace("_", " ").lower()
            text_short = text.replace("_", " ").lower()
            if task_short[:30] in text_short or text_short[:30] in task_short:
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
                poses = group["noisy_poses"][:, :6]
                return {
                    "frames": frames,
                    "poses": poses,
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
    task_description: str,
    z_text: torch.Tensor,
    noise_std: float = 0.0,
    chunk_size: int = 10,
    traj_window: int = 20,
    max_steps: int = 500,
    use_bf16: bool = True,
    record_video: bool = False,
) -> dict:
    """Run one episode in MuJoCo with expert trajectory + synthetic noise.

    The expert trajectory from the dataset is replayed as the "human" input.
    Noise is injected to simulate a bad teleoperator. ALIGN corrects it.
    """
    n_expert = min(len(expert_poses), max_steps)
    rng = np.random.default_rng(42)

    # Inject noise into expert trajectory
    if noise_std > 0:
        noisy_poses = inject_noise(expert_poses[:n_expert], std=noise_std, rng=rng)
    else:
        noisy_poses = expert_poses[:n_expert].copy()

    obs = env.reset()
    step = 0
    done = False

    alpha_vals = []
    delta_norms = []
    pose_buffer = []
    chunk_cache = None
    error_before = []
    error_after = []
    frames_buffer = []

    while not done and step < max_steps and step < n_expert:
        # Get camera image from simulation
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
                from PIL import Image
                frame = np.array(Image.fromarray(frame).resize((224, 224)))
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Get current robot EEF pose from sim
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0]))
        if isinstance(eef_pos, torch.Tensor):
            eef_pos = eef_pos.cpu().numpy()
        if isinstance(eef_quat, torch.Tensor):
            eef_quat = eef_quat.cpu().numpy()
        sim_pose = np.concatenate([eef_pos, quat_to_axisangle(eef_quat)]).astype(np.float32)

        # Use the noisy expert pose as the "human teleoperation" input
        raw_pose = noisy_poses[step]
        clean_pose = expert_poses[step]

        # Fill buffer with noisy poses (simulating human input)
        pose_buffer.append(raw_pose.copy())
        if len(pose_buffer) > traj_window:
            pose_buffer.pop(0)
        while len(pose_buffer) < traj_window:
            pose_buffer.insert(0, raw_pose.copy())

        if step < 5:
            step += 1
            # Use raw noisy pose to step the sim (no ALIGN correction yet)
            action = np.zeros(7, dtype=np.float32)
            action[:6] = raw_pose
            action[6] = -1.0
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

            # Full gating signal
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
            commanded_pose = raw_pose + alpha_val * corrective
            chunk_cache = chunk_np

        alpha_vals.append(alpha_val)
        delta_norms.append(float(np.linalg.norm(chunk_np[0])))

        # Compute errors against expert trajectory
        err_before = float(np.linalg.norm(raw_pose[:3] - clean_pose[:3]))
        err_after = float(np.linalg.norm(commanded_pose[:3] - clean_pose[:3]))
        error_before.append(err_before)
        error_after.append(err_after)

        # Overlay on frame
        if record_video:
            from PIL import Image, ImageDraw, ImageFont
            display = np.array(frame)
            img = Image.fromarray(display)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
            draw.text((10, 10), f"α={alpha_val:.2f}  Δ={delta_norms[-1]:.3f}  step={step}", fill=(0, 255, 0), font=font)
            draw.text((10, 30), f"err {err_before:.3f}→{err_after:.3f}", fill=(255, 255, 0), font=font)
            frames_buffer.append(np.array(img))

        # Step sim with ALIGN's commanded pose
        action = np.zeros(7, dtype=np.float32)
        action[:6] = commanded_pose
        action[6] = -1.0
        obs, reward, done, info = env.step(action)
        step += 1

    success = info.get("success", False)

    # Summary
    avg_alpha = float(np.mean(alpha_vals)) if alpha_vals else 0.0
    avg_delta = float(np.mean(delta_norms)) if delta_norms else 0.0
    avg_err_before = float(np.mean(error_before)) if error_before else 0.0
    avg_err_after = float(np.mean(error_after)) if error_after else 0.0
    improvement = avg_err_before - avg_err_after
    improvement_pct = (improvement / avg_err_before * 100) if avg_err_before > 0 else 0.0

    return {
        "success": success,
        "steps": step,
        "mean_alpha": avg_alpha,
        "mean_delta_norm": avg_delta,
        "mean_error_before": avg_err_before,
        "mean_error_after": avg_err_after,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "frames_buffer": frames_buffer,
    }


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
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir) / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    chunk_size = ckpt.get("config", {}).get("chunk_size", 10)

    model = ALIGNModel(
        embed_dim=256, chunk_size=chunk_size, use_text=True, device=device,
    ).to(device)

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
                    camera_names=["agentview"],
                    camera_widths=224,
                    camera_heights=224,
                    reward_shaping=True,
                    control_freq=20,
                )

                result = run_episode_in_sim(
                    env=env, model=model, device=device,
                    expert_frames=expert["frames"],
                    expert_poses=expert["poses"],
                    task_description=task_name,
                    z_text=z_text,
                    noise_std=noise_std,
                    chunk_size=chunk_size,
                    max_steps=max_steps,
                    record_video=record_video,
                )

                result["task_name"] = task_name
                result["task_idx"] = task_idx
                result["episode"] = ep
                all_results.append(result)

                status = "✓" if result["improvement"] > 0 else "✗"
                print(f"    {status}  α={result['mean_alpha']:.3f}  "
                      f"Δ={result['mean_delta_norm']:.4f}  "
                      f"err {result['mean_error_before']:.4f}→{result['mean_error_after']:.4f}  "
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
        avg_err_before = float(np.mean([r["mean_error_before"] for r in all_results]))
        avg_err_after = float(np.mean([r["mean_error_after"] for r in all_results]))
        avg_improvement = float(np.mean([r["improvement"] for r in all_results]))
        n_improved = sum(1 for r in all_results if r["improvement"] > 0)

        print(f"\n  --- {suite_name} Summary ---")
        print(f"  Avg α:              {avg_alpha:.3f}")
        print(f"  Avg Δ:              {avg_delta:.4f}")
        print(f"  Error before:       {avg_err_before:.4f}")
        print(f"  Error after:        {avg_err_after:.4f}")
        print(f"  Avg improvement:    {avg_improvement:.4f} ({avg_improvement/avg_err_before*100:+.1f}%)")
        print(f"  Episodes improved:  {n_improved}/{len(all_results)} ({n_improved/len(all_results):.0%})")

        results = {
            "suite": suite_name,
            "noise_std": noise_std,
            "n_episodes": len(all_results),
            "avg_alpha": avg_alpha,
            "avg_delta": avg_delta,
            "avg_error_before": avg_err_before,
            "avg_error_after": avg_err_after,
            "avg_improvement": avg_improvement,
            "avg_improvement_pct": avg_improvement / avg_err_before * 100 if avg_err_before > 0 else 0.0,
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
                  f"err {res['avg_error_before']:.4f}→{res['avg_error_after']:.4f}  "
                  f"{res['avg_improvement_pct']:+.1f}%  "
                  f"improved={res['n_improved']}/{res['n_episodes']}")

        with open(Path(args.output_dir) / "summary.json", "w") as f:
            json.dump(json.loads(json.dumps(all_summaries, default=str)), f, indent=2)
        print(f"\nResults: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()