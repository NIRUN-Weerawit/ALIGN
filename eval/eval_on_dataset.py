#!/usr/bin/env python3
"""Evaluate ALIGN on recorded LIBERO dataset episodes.

Loads episodes from the pre-decoded HDF5, runs ALIGN inference on each step,
and measures whether the model's correction improves or degrades the trajectory.

Two modes:
  - clean:  Use the recorded poses as-is (α should be low — no correction needed)
  - noisy:  Inject synthetic noise into poses, measure if ALIGN corrects it

Usage:
    # Evaluate on clean data (baseline — α should be near 0)
    python eval/eval_on_dataset.py \
        --data ./data/libero_10.h5 \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --encoder-checkpoint ./checkpoints/pretrain/pretrain/best.pt \
        --n-episodes 10

    # Evaluate with synthetic noise (simulates bad teleoperator)
    python eval/eval_on_dataset.py \
        --data ./data/libero_10.h5 \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --encoder-checkpoint ./checkpoints/pretrain/pretrain/best.pt \
        --noise-std 0.03 --n-episodes 10

    # Record video of one episode
    python eval/eval_on_dataset.py \
        --data ./data/libero_10.h5 \
        --checkpoint ./checkpoints/heads_libero_helios/heads_best.pt \
        --encoder-checkpoint ./checkpoints/pretrain/pretrain/best.pt \
        --noise-std 0.03 --n-episodes 1 --record-video
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel


# ================================================================
# Data loading
# ================================================================

def load_episodes(h5_path: str, n_episodes: int = 10, camera: str = "wrist_image",
                  max_frames: int = 500) -> list[dict]:
    """Load episodes from pre-decoded HDF5.

    Returns list of dicts with:
        frames: (N, H, W, 3) uint8
        poses:  (N, 6) float32 — EEF position + axis-angle
        text:   str — task description
    """
    episodes = []
    with h5py.File(h5_path, "r") as h5:
        ep_keys = sorted([k for k in h5.keys() if k.startswith("ep_")])
        for ep_key in ep_keys[:n_episodes]:
            group = h5[ep_key]

            # Frames
            frames_group = group.get("frames", None)
            if frames_group is None:
                continue
            if camera in frames_group:
                frames = frames_group[camera][:]
            elif len(frames_group) > 0:
                # Use first available camera
                cam_name = list(frames_group.keys())[0]
                frames = frames_group[cam_name][:]
            else:
                continue

            # Poses
            poses = group["noisy_poses"][:, :6]  # (N, 6)

            # Text
            texts_raw = group.get("texts", None)
            if texts_raw is not None:
                import json as _json
                text = _json.loads(texts_raw[()])[0]
            else:
                text = "pick and place"

            # Trim to max_frames
            n = min(len(frames), max_frames)
            episodes.append({
                "frames": frames[:n],
                "poses": poses[:n],
                "text": text,
            })

    return episodes


# ================================================================
# Noise injection
# ================================================================

def inject_noise(poses: np.ndarray, std: float = 0.02, rng: np.random.Generator = None) -> np.ndarray:
    """Add Gaussian noise to poses to simulate a bad teleoperator.

    Args:
        poses: (N, 6) clean EEF poses [x,y,z,rx,ry,rz].
        std: Noise standard deviation in meters (position) and radians (orientation).

    Returns:
        (N, 6) noisy poses.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noisy = poses.copy()
    noisy[:, :3] += rng.normal(0, std, size=poses[:, :3].shape)
    noisy[:, 3:6] += rng.normal(0, std * 10, size=poses[:, 3:6].shape)  # orientation noise in degrees
    return noisy


# ================================================================
# Evaluation
# ================================================================

def evaluate_on_dataset(
    data_path: str,
    checkpoint_path: str,
    encoder_checkpoint: Optional[str] = None,
    output_dir: str = "./eval/dataset_results",
    device: str = None,
    n_episodes: int = 10,
    noise_std: float = 0.0,
    traj_window: int = 20,
    chunk_size: int = 10,
    max_steps: int = 500,
    record_video: bool = False,
    use_bf16: bool = True,
):
    """Evaluate ALIGN on recorded dataset episodes."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    chunk_size = ckpt.get("config", {}).get("chunk_size", chunk_size)

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

    # Load episodes
    episodes = load_episodes(data_path, n_episodes=n_episodes, max_frames=max_steps)
    print(f"\n=== ALIGN × Dataset Evaluation ===")
    print(f"  Data:         {data_path}")
    print(f"  Episodes:     {len(episodes)}")
    print(f"  Noise std:    {noise_std:.3f}")
    print(f"  Device:       {device}")
    print(f"  chunk_size:   {chunk_size}")
    print()

    all_results = []

    for ep_i, ep in enumerate(episodes):
        frames = ep["frames"]
        clean_poses = ep["poses"]
        text = ep["text"]
        n = len(frames)

        # Precompute text embedding
        z_text = model.encode_text([text])

        # Inject noise if requested
        if noise_std > 0:
            rng = np.random.default_rng(42 + ep_i)
            noisy_poses = inject_noise(clean_poses, std=noise_std, rng=rng)
        else:
            noisy_poses = clean_poses.copy()

        # Run episode
        pose_buffer = []
        chunk_cache = None
        alpha_vals = []
        delta_norms = []
        error_before = []
        error_after = []
        frames_buffer = []

        for step in range(min(n, max_steps)):
            frame = frames[step]
            raw_pose = noisy_poses[step]
            clean_pose = clean_poses[step]

            # Current action = pose diff (or zero at first step)
            if step > 0:
                current_action = raw_pose - noisy_poses[step - 1]
            else:
                current_action = np.zeros_like(raw_pose)

            # Fill buffer
            # Buffer size = model.decision_K so the future prediction head
            # receives exactly K past embeddings.
            traj_window = model.decision_K
            pose_buffer.append(raw_pose.copy())
            if len(pose_buffer) > traj_window:
                pose_buffer.pop(0)
            while len(pose_buffer) < traj_window:
                pose_buffer.insert(0, raw_pose.copy())

            if step < 5:
                continue

            # ALIGN inference: future prediction (Decision) + corrective delta (Assistant)
            with torch.no_grad():
                frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
                traj_t = torch.from_numpy(np.stack(pose_buffer)).unsqueeze(0).float().to(device)

                # Get the actual future K clean poses as prediction targets
                K = model.decision_K
                future_start = min(step + 1, len(clean_poses) - 1)
                future_end = min(step + 1 + K, len(clean_poses))
                future_poses = clean_poses[future_start:future_end]
                if len(future_poses) < K:
                    pad = np.tile(future_poses[-1], (K - len(future_poses), 1))
                    future_poses = np.concatenate([future_poses, pad], axis=0)
                traj_future_t = torch.from_numpy(future_poses).unsqueeze(0).float().to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    # Encode current (noised) trajectory
                    mixed = model.encode_mixed(frame_t, traj_t, [""])
                    z_v = mixed["z_v"]
                    z_t_tokens = mixed["z_t_tokens"]
                    z_text = mixed["z_text"]
                    # Encode the actual future (clean) trajectory
                    mixed_future = model.encode_mixed(frame_t, traj_future_t, [""])
                    z_t_future_tokens = mixed_future["z_t_tokens"]

            # Decision head: predict K future embeddings from K past
            z_v_window = z_v.unsqueeze(1).expand(-1, K, -1)
            z_v_target = z_v.unsqueeze(1).expand(-1, K, -1)
            predicted_z_v, predicted_z_t = model.decision_head(
                z_v_window, z_t_tokens, z_text
            )

            # α from prediction error
            alpha = ALIGNModel.compute_alpha_from_predictions(
                predicted_z_v, predicted_z_t,
                z_v_target, z_t_future_tokens,
                aggregation="weighted_mean", decay=0.7,
            )
            alpha_val = float(alpha.squeeze().cpu())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # Assistant head needs mean-pooled z_t
                z_t = z_t_tokens.mean(dim=1)

                # Assistant head: predicts current action from K past frames.
                # No need to pass current_action as input — the model just
                # predicts it as output.
                action_pred = model.assistant_head(z_v, z_t, z_text)
                action_pred_np = action_pred.squeeze(0).cpu().numpy()

                if chunk_cache is not None:
                    corrective = 0.7 * action_pred_np + 0.3 * chunk_cache
                else:
                    corrective = action_pred_np
                commanded_pose = raw_pose + alpha_val * corrective
                chunk_cache = action_pred_np

            alpha_vals.append(alpha_val)
            delta_norms.append(float(np.linalg.norm(action_pred_np)))

            # Compute errors
            err_before = float(np.linalg.norm(raw_pose[:3] - clean_pose[:3]))
            err_after = float(np.linalg.norm(commanded_pose[:3] - clean_pose[:3]))
            error_before.append(err_before)
            error_after.append(err_after)

            # Record video frame
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
                draw.text((10, 30), f"err_before={err_before:.3f}  err_after={err_after:.3f}", fill=(255, 255, 0), font=font)
                frames_buffer.append(np.array(img))

        # Episode summary
        avg_alpha = float(np.mean(alpha_vals)) if alpha_vals else 0.0
        avg_delta = float(np.mean(delta_norms)) if delta_norms else 0.0
        avg_err_before = float(np.mean(error_before)) if error_before else 0.0
        avg_err_after = float(np.mean(error_after)) if error_after else 0.0
        improvement = avg_err_before - avg_err_after
        improvement_pct = (improvement / avg_err_before * 100) if avg_err_before > 0 else 0.0

        result = {
            "episode": ep_i,
            "text": text[:60],
            "n_steps": len(alpha_vals),
            "mean_alpha": avg_alpha,
            "mean_delta_norm": avg_delta,
            "mean_error_before": avg_err_before,
            "mean_error_after": avg_err_after,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
        }
        all_results.append(result)

        status = "✓" if improvement > 0 else "✗"
        print(f"  Ep {ep_i:2d}: α={avg_alpha:.3f}  Δ={avg_delta:.4f}  "
              f"err {avg_err_before:.4f}→{avg_err_after:.4f}  "
              f"{status} {improvement_pct:+.1f}%  text={text[:40]}")

        # Save video
        if record_video and frames_buffer:
            try:
                import imageio
                video_path = out_dir / f"ep{ep_i}.mp4"
                writer = imageio.get_writer(str(video_path), fps=20, codec="libx264", quality=8)
                for f in frames_buffer:
                    writer.append_data(f)
                writer.close()
                print(f"    Video: {video_path}")
            except ImportError:
                pass

    # Overall summary
    if all_results:
        avg_alpha = float(np.mean([r["mean_alpha"] for r in all_results]))
        avg_delta = float(np.mean([r["mean_delta_norm"] for r in all_results]))
        avg_err_before = float(np.mean([r["mean_error_before"] for r in all_results]))
        avg_err_after = float(np.mean([r["mean_error_after"] for r in all_results]))
        avg_improvement = float(np.mean([r["improvement"] for r in all_results]))
        n_improved = sum(1 for r in all_results if r["improvement"] > 0)

        print(f"\n  --- Overall Summary ({len(all_results)} episodes) ---")
        print(f"  Avg α:              {avg_alpha:.3f}")
        print(f"  Avg Δ:              {avg_delta:.4f}")
        print(f"  Error before:       {avg_err_before:.4f}")
        print(f"  Error after:        {avg_err_after:.4f}")
        print(f"  Avg improvement:    {avg_improvement:.4f} ({avg_improvement/avg_err_before*100:+.1f}%)")
        print(f"  Episodes improved:  {n_improved}/{len(all_results)} ({n_improved/len(all_results):.0%})")

        summary = {
            "data_path": data_path,
            "checkpoint": checkpoint_path,
            "encoder_checkpoint": encoder_checkpoint,
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
            json.dump(json.loads(json.dumps(summary, default=str)), f, indent=2)
        print(f"\n  Results: {out_dir / 'results.json'}")

    return all_results


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ALIGN on recorded dataset episodes")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, help="Heads checkpoint (.pt)")
    parser.add_argument("--encoder-checkpoint", default=None, help="Phase 1 backbone checkpoint")
    parser.add_argument("--output-dir", default="./eval/dataset_results", help="Output directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes to evaluate")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Synthetic noise std (0 = clean data)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--record-video", action="store_true", help="Record MP4 videos")
    args = parser.parse_args()

    evaluate_on_dataset(
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


if __name__ == "__main__":
    main()