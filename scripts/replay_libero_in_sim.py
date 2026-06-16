#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a LIBERO LeRobot v3 episode in the actual LIBERO simulation.

The goal is to verify that the dataset can be reproduced by the simulator:
given the stored actions, can the LIBERO env replay the same trajectory?

Side-by-side video output:
  - LEFT:  original LeRobot dataset frame (wrist or agentview)
  - RIGHT: LIBERO sim's offscreen render after the same step
  - bottom: text annotation showing step / cumulative reward / done flag

No model inference — pure data-driven replay.

Usage:
    # Replay first episode of libero_10
    python scripts/replay_libero_in_sim.py \\
        --data-dir ~/.cache/huggingface/lerobot/nvidia/LIBERO_LeRobot_v3/libero_10 \\
        --bddl-root /path/to/libero/libero/bddl_files \\
        --episode 0 \\
        --max-steps 200 \\
        --output replay.mp4

    # Replay all episodes of the first task in libero_10
    python scripts/replay_libero_in_sim.py \\
        --data-dir ~/.cache/huggingface/lerobot/nvidia/LIBERO_LeRobot_v3/libero_10 \\
        --bddl-root /path/to/libero/libero/bddl_files \\
        --task-name "KITCHEN_SCENE3_turn_on_the_stove" \\
        --output-dir ./replays/

The script will:
  1. Load the LeRobot episode (frames, actions, language instruction)
  2. Find the matching BDDL file via the language instruction
  3. Create the LIBERO env (OffScreenRenderEnv) for that task
  4. Step the env with the dataset's stored actions
  5. Save a side-by-side MP4 of dataset view vs sim view
  6. Report success rate, reward, action mismatches
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

# Add ALIGN to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_bddl_file(bddl_root: Path, task_name: str) -> Optional[Path]:
    """Find the BDDL file for a given task name.

    LIBERO BDDL files are named like:
      libero_10/KITCHEN_SCENE3_turn_on_the_stove.bddl
    or sometimes nested by scene:
      libero_10/KITCHEN_SCENE3/KITCHEN_SCENE3_turn_on_the_stove.bddl
    """
    # Try direct path
    direct = bddl_root / f"{task_name}.bddl"
    if direct.exists():
        return direct

    # Try under each subdirectory
    for sub in bddl_root.rglob(f"{task_name}.bddl"):
        return sub

    return None


def parse_task_name_from_lang(lang: str) -> str:
    """Convert a natural language instruction into a BDDL file name.

    LIBERO task names follow the pattern:
      {SCENE_NAME}_{verb}_{object}_{modifier}
    but the language instruction doesn't always give us the scene.

    Heuristic: try matching by keywords in the instruction.
    """
    return lang.lower().strip()


def load_episode(
    data_dir: Path,
    episode: int,
    video_backend: str = "pyav",
) -> dict:
    """Load one episode from LeRobot LIBERO data, returning frames + actions + lang.

    Returns:
        dict with:
            frames_wrist: (T, H, W, 3) uint8 — wrist camera frames from dataset
            frames_agent: (T, H, W, 3) uint8 — agentview frames from dataset (or None)
            actions: (T, 7) float32 — action deltas (EEF delta + gripper)
            states: (T, 9) float32 — full state (ee_pos + ee_quat + gripper)
            language_instruction: str — task description
            task_index: int — task ID in the benchmark (or None)
            episode_id: int
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"  Loading LeRobot dataset from {data_dir}...")
    ds = LeRobotDataset("nvidia/LIBERO_LeRobot_v3", root=str(data_dir), video_backend=video_backend)

    # Find episode boundaries — LeRobot v3 stores episode_index in the parquet metadata
    print(f"  Indexing {len(ds.hf_dataset)} rows for episode {episode}...")
    ep_rows = []
    for row_idx in range(len(ds.hf_dataset)):
        sample = ds.hf_dataset[row_idx]
        # Try both possible key names
        raw_ep = sample.get("episode_index")
        if raw_ep is None:
            for k in sample.keys():
                if "episode" in k.lower():
                    raw_ep = sample[k]
                    break
        if raw_ep is None:
            raise ValueError(f"No episode_index in row {row_idx}; keys: {list(sample.keys())}")
        if hasattr(raw_ep, "item"):
            raw_ep = raw_ep.item()
        if int(float(raw_ep)) == episode:
            ep_rows.append(row_idx)

    if not ep_rows:
        raise ValueError(f"Episode {episode} not found in dataset")

    print(f"  Episode {episode} has {len(ep_rows)} frames")

    # Identify feature keys
    features = ds.meta.features
    wrist_key = "observation.images.wrist_image"
    agent_key = "observation.images.image"
    state_key = "observation.state"
    action_key = "action"

    if wrist_key not in features:
        # Fallback to whatever image is there
        img_keys = [k for k in features if "images" in k]
        wrist_key = img_keys[0] if img_keys else None

    # Load frames + actions for this episode
    frames_wrist = []
    frames_agent = []
    actions = []
    states = []
    lang = None
    task_index = None

    for row_idx in tqdm(ep_rows, desc="Loading episode"):
        sample = ds[row_idx]
        # Get the LAST frame in the temporal window (default behavior)
        if wrist_key and wrist_key in sample:
            f = sample[wrist_key]
            if hasattr(f, "dim") and f.dim() == 4:
                f = f[-1]  # last in window
            # Convert (C, H, W) → (H, W, C)
            if f.dim() == 3 and f.shape[0] in (1, 3):
                f = f.permute(1, 2, 0)
            # To uint8
            if f.dtype == torch.float32:
                if f.max() <= 1.0:
                    f = (f * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    f = f.clamp(0, 255).to(torch.uint8)
            elif f.dtype != torch.uint8:
                f = f.to(torch.uint8)
            frames_wrist.append(f.numpy())

        if agent_key and agent_key in sample:
            f = sample[agent_key]
            if hasattr(f, "dim") and f.dim() == 4:
                f = f[-1]
            if f.dim() == 3 and f.shape[0] in (1, 3):
                f = f.permute(1, 2, 0)
            if f.dtype == torch.float32:
                if f.max() <= 1.0:
                    f = (f * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    f = f.clamp(0, 255).to(torch.uint8)
            elif f.dtype != torch.uint8:
                f = f.to(torch.uint8)
            frames_agent.append(f.numpy())

        if action_key in sample:
            a = sample[action_key]
            if hasattr(a, "dim") and a.dim() == 2:
                a = a[0] if a.size(0) > 1 else a[-1]
            actions.append(a.float().numpy() if torch.is_tensor(a) else a)

        if state_key in sample:
            s = sample[state_key]
            if hasattr(s, "dim") and s.dim() == 2:
                s = s[0] if s.size(0) > 1 else s[-1]
            states.append(s.float().numpy() if torch.is_tensor(s) else s)

        if lang is None and "task" in sample:
            t = sample["task"]
            if isinstance(t, bytes):
                t = t.decode("utf-8")
            if isinstance(t, str) and t.startswith("{"):
                # LeRobot v3 sometimes stores language as JSON bytes
                try:
                    t = json.loads(t)
                    if isinstance(t, dict) and "task" in t:
                        t = t["task"]
                except Exception:
                    pass
            lang = str(t) if not isinstance(t, str) else t

        if task_index is None and "task_index" in sample:
            ti = sample["task_index"]
            if hasattr(ti, "item"):
                ti = ti.item()
            task_index = int(ti) if ti is not None else None

    return {
        "frames_wrist": np.stack(frames_wrist) if frames_wrist else None,
        "frames_agent": np.stack(frames_agent) if frames_agent else None,
        "actions": np.stack(actions) if actions else None,
        "states": np.stack(states) if states else None,
        "language_instruction": lang or "unknown",
        "task_index": task_index,
        "episode_id": episode,
        "n_frames": len(ep_rows),
    }


def create_sim_env(bddl_file: Path, camera_name: str = "agentview"):
    """Create a LIBERO OffScreenRenderEnv from a BDDL file.

    Args:
        bddl_file: Path to .bddl file
        camera_name: Which camera to render. LIBERO has 'agentview' and 'robot0_eye_in_hand'.
    """
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        cameras=[camera_name, "robot0_eye_in_hand"],
        camera_widths=128,
        camera_heights=128,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    return env


def replay_episode(
    bddl_file: Path,
    actions: np.ndarray,
    max_steps: Optional[int] = None,
    render_width: int = 256,
    render_height: int = 256,
    camera_name: str = "agentview",
) -> dict:
    """Replay actions in the sim and capture frames + rewards.

    Returns:
        dict with:
            sim_frames: (T, H, W, 3) uint8 — sim's camera view at each step
            rewards: (T,) float — per-step reward
            dones: (T,) bool — done flag at each step
            success: bool — task succeeded at end
            total_reward: float
    """
    env = create_sim_env(bddl_file, camera_name=camera_name)
    sim_frames = []
    rewards = []
    dones = []

    obs = env.reset()
    T = max_steps or len(actions)

    for t in range(T):
        # Render current state (BEFORE stepping, so we see the same thing the dataset recorded)
        # Some LIBERO versions return obs[cam] as the rendered frame; otherwise render explicitly
        try:
            frame = env.sim.render(
                width=render_width,
                height=render_height,
                camera_name=camera_name,
            )[::-1]  # flip vertically (mujoco is bottom-up)
            sim_frames.append(frame.copy())
        except Exception:
            sim_frames.append(np.zeros((render_height, render_width, 3), dtype=np.uint8))

        # Step the env with the stored action
        action = actions[t]
        if len(action) < 7:
            # Pad to 7D if dataset only stored 6D
            action = np.concatenate([action, np.array([0.0])])

        try:
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            dones.append(bool(done))
        except Exception as e:
            print(f"  Step {t} failed: {e}")
            break

    success = bool(env.check_success()) if hasattr(env, "check_success") else False
    env.close()

    return {
        "sim_frames": np.stack(sim_frames) if sim_frames else None,
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "success": success,
        "total_reward": float(np.sum(rewards)),
        "n_steps": T,
    }


def make_side_by_side_video(
    dataset_frames: np.ndarray,
    sim_frames: np.ndarray,
    output_path: Path,
    fps: int = 20,
    task_name: str = "",
    rewards: Optional[np.ndarray] = None,
    dones: Optional[np.ndarray] = None,
) -> None:
    """Create a side-by-side MP4: dataset (left) vs sim (right).

    Uses imageio (broadly available) with libx264. Falls back to cv2 if needed.
    """
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            str(output_path), fps=fps, codec="libx264", quality=8
        )
    except ImportError:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = dataset_frames[0].shape[:2]
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2 + 4, h))
            cv2_writer = True
        except ImportError:
            print("  ERROR: Need imageio or opencv-python to write video")
            return
    else:
        cv2_writer = False

    # Resize to match
    if dataset_frames.shape[1:3] != sim_frames.shape[1:3]:
        from PIL import Image
        target_hw = sim_frames.shape[1:3]
        resized_dataset = np.stack([
            np.array(Image.fromarray(f).resize((target_hw[1], target_hw[0])))
            for f in dataset_frames
        ])
    else:
        resized_dataset = dataset_frames

    T = min(len(resized_dataset), len(sim_frames))
    for t in range(T):
        if rewards is not None and t < len(rewards):
            r = rewards[t]
            d = dones[t] if dones is not None else False
            label = f"step={t:3d} reward={r:.2f} done={d}"
        else:
            label = f"step={t:3d}"

        # Stack side-by-side with a thin black divider
        h, w = sim_frames[t].shape[:2]
        divider = np.zeros((h, 4, 3), dtype=np.uint8)
        combined = np.hstack([resized_dataset[t], divider, sim_frames[t]])

        # Add text overlay at top
        try:
            from PIL import Image, ImageDraw, ImageFont
            img_pil = Image.fromarray(combined)
            draw = ImageDraw.Draw(img_pil)
            # Top: task name; bottom-left: dataset; bottom-right: sim
            draw.text((5, 5), f"TASK: {task_name[:60]}", fill=(255, 255, 0))
            draw.text((5, h - 20), "DATASET", fill=(255, 255, 0))
            draw.text((w + 10, h - 20), "SIM", fill=(0, 255, 0))
            draw.text((5, 25), label, fill=(255, 255, 255))
            combined = np.array(img_pil)
        except Exception:
            pass

        if cv2_writer:
            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        else:
            writer.append_data(combined)

    if cv2_writer:
        writer.release()
    else:
        writer.close()

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay LIBERO LeRobot episode in sim to verify dataset reproducibility"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to LeRobot LIBERO dataset (e.g., .../libero_10)")
    parser.add_argument("--bddl-root", required=True,
                        help="Path to LIBERO bddl_files directory")
    parser.add_argument("--episode", type=int, default=0,
                        help="Which episode index to replay (default: 0)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps to replay (default: full episode)")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: ./replays/episode_NNNN.mp4)")
    parser.add_argument("--output-dir", default="./replays",
                        help="Output directory when replaying many episodes")
    parser.add_argument("--task-name", default=None,
                        help="Override BDDL task name (default: search by language instruction)")
    parser.add_argument("--video-backend", default="pyav",
                        choices=["pyav", "torchcodec"],
                        help="Video decoding backend for LeRobot")
    parser.add_argument("--fps", type=int, default=20,
                        help="Output video FPS (default: 20, matching LIBERO's control_freq)")
    parser.add_argument("--sim-camera", default="agentview",
                        choices=["agentview", "robot0_eye_in_hand"],
                        help="Which LIBERO sim camera to use for side-by-side")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Sim render resolution (default: 256x256)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video output, just print metrics")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    bddl_root = Path(args.bddl_root).expanduser()

    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}")
        sys.exit(1)
    if not bddl_root.exists():
        print(f"ERROR: BDDL root not found: {bddl_root}")
        sys.exit(1)

    # -- Load episode --
    print(f"\n[1/4] Loading episode {args.episode} from {data_dir}...")
    ep = load_episode(data_dir, args.episode, video_backend=args.video_backend)
    print(f"  Language instruction: '{ep['language_instruction']}'")
    print(f"  Frames: {ep['n_frames']}, Actions shape: {ep['actions'].shape}, States shape: {ep['states'].shape}")

    # -- Find BDDL file --
    print(f"\n[2/4] Finding BDDL file for this task...")
    task_name = args.task_name or ep["language_instruction"]
    bddl_file = find_bddl_file(bddl_root, task_name)
    if bddl_file is None:
        # Try fuzzy match
        print(f"  Direct match failed. Searching for '{task_name[:50]}'...")
        for f in bddl_root.rglob("*.bddl"):
            if task_name.lower()[:30] in f.stem.lower():
                bddl_file = f
                break
    if bddl_file is None:
        print(f"  ERROR: No BDDL file matches '{task_name}'")
        print(f"  Available BDDL files (first 10):")
        for f in sorted(bddl_root.rglob("*.bddl"))[:10]:
            print(f"    {f.relative_to(bddl_root)}")
        sys.exit(1)
    print(f"  Using BDDL: {bddl_file.relative_to(bddl_root.parent)}")

    # -- Replay in sim --
    print(f"\n[3/4] Replaying in sim (camera: {args.sim_camera})...")
    max_steps = args.max_steps or ep["n_frames"]
    sim_result = replay_episode(
        bddl_file=bddl_file,
        actions=ep["actions"],
        max_steps=max_steps,
        render_width=args.render_size,
        render_height=args.render_size,
        camera_name=args.sim_camera,
    )
    print(f"  Replayed {sim_result['n_steps']} steps")
    print(f"  Total reward: {sim_result['total_reward']:.2f}")
    print(f"  Final success: {sim_result['success']}")
    print(f"  Mean reward/step: {sim_result['total_reward'] / max(1, sim_result['n_steps']):.3f}")

    # -- Build side-by-side video --
    if not args.no_video:
        print(f"\n[4/4] Building side-by-side video...")
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"episode_{args.episode:04d}.mp4"

        # Pick dataset camera to use for the LEFT side
        if args.sim_camera == "robot0_eye_in_hand" and ep["frames_wrist"] is not None:
            ds_frames = ep["frames_wrist"]
        elif ep["frames_agent"] is not None:
            ds_frames = ep["frames_agent"]
        else:
            ds_frames = ep["frames_wrist"]

        make_side_by_side_video(
            dataset_frames=ds_frames,
            sim_frames=sim_result["sim_frames"],
            output_path=output_path,
            fps=args.fps,
            task_name=ep["language_instruction"],
            rewards=sim_result["rewards"],
            dones=sim_result["dones"],
        )

    # -- Summary --
    print(f"\n=== Summary ===")
    print(f"  Episode:        {ep['episode_id']}")
    print(f"  Task:           {ep['language_instruction']}")
    print(f"  BDDL file:      {bddl_file.name}")
    print(f"  Frames loaded:  {ep['n_frames']}")
    print(f"  Steps replayed: {sim_result['n_steps']}")
    print(f"  Total reward:   {sim_result['total_reward']:.2f}")
    print(f"  Final success:  {sim_result['success']}")
    if not sim_result["success"]:
        print(f"\n  ⚠️  Replay did NOT achieve task success. Possible causes:")
        print(f"     1. Action format mismatch (dataset action scale != sim action scale)")
        print(f"     2. Initial state randomness (LIBERO has stochastic init noise)")
        print(f"     3. The dataset was generated from a different BDDL task than this one")
        print(f"     4. Action execution has drift that compounds over time")


if __name__ == "__main__":
    main()