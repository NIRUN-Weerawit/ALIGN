#!/usr/bin/env python3
"""ALIGN Ground Truth Generation — processes raw episodes into smooth trajectories.

Takes a recorded episode (frames + noisy_poses) and generates:
  1. Smooth trajectory via hybrid SavGol + motion interpolation
  2. α_target per timestep for Decision head training
  3. Δpose chunk targets for Assistant head training

Pipeline:
  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
  │ Load episode │───▶│ Detect      │───▶│ SavGol       │
  │ (frames,     │    │ approach    │    │ filter       │
  │  noisy_poses)│    │ phase       │    │ (transit)    │
  └──────────────┘    └──────┬──────┘    └──────┬───────┘
                             │                  │
                             └────────┬─────────┘
                                      ▼
                            ┌──────────────────┐
                            │ Motion planner   │──▶ smooth_poses
                            │ (approach phase) │
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Compute targets  │──▶ α_target
                            │ α_target + chunk │──▶ Δpose_target
                            └──────────────────┘

Usage:
    # Single episode
    python generate_ground_truth.py --episode ./align_data/ep_0001

    # Batch
    python generate_ground_truth.py --input-dir ./align_data --output-dir ./align_data

    # With visualization
    python generate_ground_truth.py --episode ./align_data/ep_0001 --visualize
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

# Optional approach planners (installed alongside this module)
try:
    from align_dmp import dmp_approach_plan as _dmp_plan
except ImportError:
    _dmp_plan = None
try:
    from align_chomp import chomp_approach_plan as _chomp_plan
except ImportError:
    _chomp_plan = None


# ================================================================
# Constants
# ================================================================
APPROACH_THRESHOLD = 0.08       # meters — hand within this distance = approach phase
APPROACH_END_BUFFER = 5         # frames to trim from end for stable grasp pose
D_MAX = 0.10                    # max tolerable deviation for α_target normalization
CHUNK_SIZE = 5                  # K — number of future Δposes in one chunk
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3
DT = 1.0 / 30.0                 # ~33ms per frame at 30Hz
POSE_EPSILON = 1e-8             # small value to avoid division by zero


# ================================================================
# Loading
# ================================================================

def load_episode(episode_dir: str) -> tuple:
    """Load a recorded episode from disk.

    Returns:
        frames: (N, H, W, 3) uint8 RGB
        noisy_poses: (N, 6) or (N, 7) float64
        gripper_states: (N,) float64
        timestamps: (N,) float64
        meta: dict
    """
    ep_path = Path(episode_dir)
    if not ep_path.exists():
        raise FileNotFoundError("Episode not found: " + str(ep_path))

    # Frames
    frames_dir = ep_path / "frames"
    camera_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
    if camera_dirs:
        label_dir = camera_dirs[0]
        frame_files = sorted(label_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    else:
        frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = []
    for f in frame_files:
        frames.append(np.array(Image.open(f)))
    frames_arr = np.stack(frames, axis=0) if frames else np.array([])

    # NPZ data
    npz_path = ep_path / "data.npz"
    if not npz_path.exists():
        raise FileNotFoundError("data.npz not found: " + str(npz_path))
    data = dict(np.load(npz_path))

    noisy_poses = data.get("noisy_poses", [])
    gripper_states = data.get("gripper_states", [])
    timestamps = data.get("timestamps", [])
    smooth_poses_existing = data.get("smooth_poses", None)

    # Metadata
    meta_path = ep_path / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return (frames_arr, noisy_poses, gripper_states, timestamps, smooth_poses_existing, meta)


# ================================================================
# Smoothing — Position
# ================================================================

def _plan_approach_quintic(
    noisy_approach_pos: np.ndarray,
    start_idx: int,
    grasp_goal: np.ndarray,
    n_approach: int,
) -> np.ndarray:
    """Quintic polynomial interpolation (default)."""
    start_pos = noisy_approach_pos[0].copy()
    t_vals = np.linspace(0, 1, n_approach)
    smooth_step = lambda t: 10*t**3 - 15*t**4 + 6*t**5
    weights = smooth_step(t_vals)
    return start_pos + np.outer(weights, grasp_goal - start_pos)


def _plan_approach_dmp(
    noisy_approach_pos: np.ndarray,
    start_idx: int,
    grasp_goal: np.ndarray,
    n_approach: int,
) -> np.ndarray:
    """DMP-based approach — encodes human approach style from demo."""
    if _dmp_plan is None:
        raise ImportError("align_dmp module not found")
    return _dmp_plan(noisy_approach_pos, grasp_goal, n_steps=n_approach)


def _plan_approach_chomp(
    noisy_approach_pos: np.ndarray,
    start_idx: int,
    grasp_goal: np.ndarray,
    n_approach: int,
) -> np.ndarray:
    """CHOMP-based approach — optimizes for smoothness from straight-line init."""
    if _chomp_plan is None:
        raise ImportError("align_chomp module not found")
    # CHOMP uses the noisy demo only to size the output — it optimizes from
    # a straight-line initialization, using the demo as a rough prior.
    return _chomp_plan(noisy_approach_pos, grasp_goal, n_steps=n_approach)


# Registry of available planners
APPROACH_PLANNERS = {
    "quintic": _plan_approach_quintic,
    "dmp": _plan_approach_dmp,
    "chomp": _plan_approach_chomp,
}


def smooth_position(
    noisy_pos: np.ndarray,
    grasp_goal: np.ndarray,
    is_approach: np.ndarray,
    approach_planner: str = "quintic",
) -> np.ndarray:
    """Generate smooth position trajectory.

    Transit phase: SavGol filter (preserves human path shape, removes noise).
    Approach phase: configurable planner (quintic / dmp / chomp).
    Blend: smooth step transition between the two.

    Args:
        noisy_pos: (N, 3) noisy EEF positions.
        grasp_goal: (3,) target grasp position (last stable pose).
        is_approach: (N,) bool mask — True for approach phase frames.
        approach_planner: One of 'quintic', 'dmp', 'chomp'.

    Returns:
        (N, 3) smooth positions.
    """
    N = len(noisy_pos)
    smooth_pos = np.zeros_like(noisy_pos)

    planner_fn = APPROACH_PLANNERS.get(approach_planner)
    if planner_fn is None:
        print(f"  [GT] WARNING: Unknown approach planner '{approach_planner}', falling back to quintic")
        planner_fn = _plan_approach_quintic
        approach_planner = "quintic"

    # ── Step 1: SavGol on full trajectory ──
    if N >= SAVGOL_WINDOW:
        window = min(SAVGOL_WINDOW, N if N % 2 == 1 else N - 1)
        polyorder = min(SAVGOL_POLYORDER, window - 1)
        if window > polyorder and window >= 3:
            smooth_transit = savgol_filter(noisy_pos, window_length=window, polyorder=polyorder, axis=0)
        else:
            smooth_transit = noisy_pos.copy()
    else:
        smooth_transit = noisy_pos.copy()

    # ── Step 2: Generate approach trajectory ──
    approach_idx = np.where(is_approach)[0]
    if len(approach_idx) > 1:
        start_idx = approach_idx[0]
        end_idx = approach_idx[-1]
        n_approach = end_idx - start_idx + 1

        # Extract approach segment positions and plan
        noisy_approach_pos = noisy_pos[start_idx:end_idx + 1]
        try:
            approach_traj = planner_fn(noisy_approach_pos, start_idx, grasp_goal, n_approach)
        except (ImportError, ValueError) as e:
            print(f"  [GT] WARNING: {approach_planner} planner failed ({e}), falling back to quintic")
            approach_traj = _plan_approach_quintic(noisy_approach_pos, start_idx, grasp_goal, n_approach)
            approach_planner = "quintic"

        # Fill ALL frames in three regions: pre-approach, approach blend, post-approach
        approach_start = max(0, start_idx - 10)   # blend zone begins a bit early
        approach_end   = min(N, end_idx)

        # Region 1: Pre-approach (pure transit)
        for i in range(0, approach_start):
            smooth_pos[i] = smooth_transit[i]

        # Region 2: Approach blend (transit → planned trajectory)
        for i in range(approach_start, approach_end):
            local_idx = i - start_idx
            blend_w = local_idx / max(n_approach - 1, 1)
            blend_w = max(0, min(1, blend_w))
            smooth_pos[i] = (1 - blend_w) * smooth_transit[i] + blend_w * approach_traj[min(local_idx, n_approach - 1)]

        # Region 3: Post-approach (hold grasp)
        for i in range(approach_end, N):
            smooth_pos[i] = grasp_goal
    else:
        # No approach detected: use transit only
        smooth_pos = smooth_transit.copy()

    return smooth_pos


# ================================================================
# Smoothing — Orientation
# ================================================================

def smooth_orientation(
    noisy_quats: np.ndarray,
    grasp_quat: np.ndarray,
    is_approach: np.ndarray,
) -> np.ndarray:
    """Generate smooth orientation trajectory.

    Transit: SLERP between extracted keyframes (decimated for stability).
    Approach: SLERP from approach start → grasp orientation.

    Args:
        noisy_quats: (N, 4) noisy EEF quaternions (xyzw).
        grasp_quat: (4,) target grasp quaternion (xyzw).
        is_approach: (N,) bool mask.

    Returns:
        (N, 4) smooth quaternions (xyzw).
    """
    N = len(noisy_quats)
    smooth_quats = np.zeros_like(noisy_quats)

    # ── Transit: SLERP through sub-sampled keyframes ──
    # Pick evenly spaced keyframes, SLERP between them
    n_keyframes = max(2, N // 20)  # ~1 keyframe per 20 frames
    key_indices = np.linspace(0, N - 1, n_keyframes, dtype=int)

    # Fix quaternion signs for consistency (ensure dot product > 0)
    quats_fixed = noisy_quats.copy()
    for i in range(1, N):
        if np.dot(quats_fixed[i], quats_fixed[i - 1]) < 0:
            quats_fixed[i] = -quats_fixed[i]

    key_times = key_indices / max(N - 1, 1)  # normalized 0→1
    key_quats = quats_fixed[key_indices]

    try:
        slerp_transit = Slerp(key_times, R.from_quat(key_quats))
        t_grid = np.linspace(0, 1, N)
        smooth_transit_quats = slerp_transit(t_grid).as_quat()
    except Exception:
        # Fallback: direct quaternion smoothing via moving average
        smooth_transit_quats = quats_fixed.copy()
        window = 5
        for i in range(window, N - window):
            # Average the 4D vectors and renormalize
            mean_q = np.mean(quats_fixed[i - window:i + window + 1], axis=0)
            smooth_transit_quats[i] = mean_q / max(np.linalg.norm(mean_q), POSE_EPSILON)

    # ── Approach: SLERP from start to grasp ──
    approach_idx = np.where(is_approach)[0]
    if len(approach_idx) > 1:
        start_idx = approach_idx[0]
        end_idx = approach_idx[-1]
        n_approach = end_idx - start_idx + 1

        approach_start_q = smooth_transit_quats[start_idx]
        approach_end_q = grasp_quat

        # Ensure consistent sign
        if np.dot(approach_start_q, approach_end_q) < 0:
            approach_end_q = -approach_end_q

        try:
            t_vals = np.linspace(0, 1, n_approach)
            # Quintic smoothstep for the interpolation parameter
            smooth_step = lambda t: 10*t**3 - 15*t**4 + 6*t**5
            slerp_approach = Slerp([0, 1], R.from_quat([approach_start_q, approach_end_q]))
            approach_quats = slerp_approach(smooth_step(t_vals)).as_quat()

            # Blend
            for i in range(start_idx, end_idx + 1):
                local_idx = i - start_idx
                blend_w = local_idx / max(n_approach - 1, 1)
                blend_w = max(0, min(1, blend_w))
                smooth_quats[i] = slerp_transit([blend_w]).as_quat() if False else approach_quats[local_idx]
            # Actually just use the approach SLERP directly
            for i in range(start_idx, end_idx + 1):
                smooth_quats[i] = approach_quats[i - start_idx]

            # Fill pre-approach and post-approach
            for i in range(0, start_idx):
                smooth_quats[i] = smooth_transit_quats[i]
            for i in range(end_idx + 1, N):
                smooth_quats[i] = smooth_transit_quats[i]
        except Exception:
            for i in range(start_idx, end_idx + 1):
                smooth_quats[i] = approach_end_q
    else:
        smooth_quats = smooth_transit_quats.copy()

    # Fill leading/trailing frames
    idx = np.where(is_approach)[0]
    if len(idx) > 1:
        smooth_quats[:idx[0]] = smooth_transit_quats[:idx[0]]
        smooth_quats[idx[-1] + 1:] = smooth_transit_quats[idx[-1] + 1:]

    return smooth_quats


# ================================================================
# Approach phase detection
# ================================================================

def detect_approach_phase(
    noisy_poses: np.ndarray,
    grasp_goal: np.ndarray,
    threshold: float = APPROACH_THRESHOLD,
    end_buffer: int = APPROACH_END_BUFFER,
) -> np.ndarray:
    """Detect which frames are in the approach phase.

    Approach starts when the hand is within `threshold` of the grasp goal
    (using a moving average of the last 5 frames to reduce noise effects).

    Args:
        noisy_poses: (N, 6) or (N, 7) noisy poses.
        grasp_goal: (6,) or (7,) grasp pose (last stable position).
        threshold: Distance threshold for approach detection (meters).
        end_buffer: Number of frames from end to always include.

    Returns:
        (N,) bool mask — True for approach frames.
    """
    N = len(noisy_poses)
    positions = noisy_poses[:, :3]
    grasp_pos = grasp_goal[:3]

    # Distance to grasp goal (smoothed with moving average)
    distances = np.linalg.norm(positions - grasp_pos, axis=1)
    if N >= 5:
        kernel = np.ones(5) / 5
        distances = np.convolve(distances, kernel, mode="same")

    is_approach = distances < threshold

    # Ensure last `end_buffer` frames are included (they're the grasp)
    if N > end_buffer:
        is_approach[-end_buffer:] = True

    # Find contiguous approach regions, keep the last one (closest to grasp)
    if np.any(is_approach):
        # Find transitions
        diffs = np.diff(is_approach.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1

        if is_approach[0]:
            starts = np.concatenate([[0], starts])
        if is_approach[-1]:
            ends = np.concatenate([ends, [N]])

        if len(starts) > 0 and len(ends) > 0:
            # Keep only the last approach region (closest to grasp)
            last_start = starts[-1]
            last_end = ends[-1]
            is_approach[:] = False
            is_approach[last_start:last_end] = True

    return is_approach


# ================================================================
# Compute targets
# ================================================================

def compute_alpha_target(
    noisy_poses: np.ndarray,
    smooth_poses: np.ndarray,
) -> np.ndarray:
    """Compute α_target for Decision head.

    α_target = need × capability

    need = clip(||noisy - smooth|| / D_MAX, 0, 1)
    capability = 1.0 (placeholder — will be replaced by cos_sim after Phase 2)

    For standalone ground truth generation, capability defaults to 1.0.
    After contrastive pretraining (Phase 2), capability is recomputed as
    min(cos_vt, cos_vl, cos_tl) using the trained encoders.

    Args:
        noisy_poses: (N, 6) or (N, 7) noisy poses.
        smooth_poses: (N, 6) or (N, 7) smooth poses.

    Returns:
        (N,) α_target values in [0, 1].
    """
    N = min(len(noisy_poses), len(smooth_poses))
    needs = np.zeros(N)

    for i in range(N):
        pos_error = np.linalg.norm(noisy_poses[i, :3] - smooth_poses[i, :3])
        need = min(pos_error / D_MAX, 1.0)
        needs[i] = need

    # capability = 1.0 (placeholder — updated after contrastive pretraining)
    # Gradient clip safety: if smooth_poses == noisy_poses, need is ~0 → α=0
    alpha_target = needs * 1.0

    return alpha_target


def compute_chunk_targets(
    noisy_poses: np.ndarray,
    smooth_poses: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Compute Δpose chunk targets for Assistant head.

    Δpose_target[t][i] = smooth_pose[t + i] - noisy_pose[t]
    for i = 1..chunk_size

    Args:
        noisy_poses: (N, 6) or (N, 7) noisy poses.
        smooth_poses: (N, 6) or (N, 7) smooth poses.
        chunk_size: K — number of future steps per chunk.

    Returns:
        (N - chunk_size, chunk_size, 6) chunk targets.
        Returns empty array if N <= chunk_size.
    """
    N = len(noisy_poses)
    if N <= chunk_size:
        return np.array([])

    # Ensure both are 6D (position + axis-angle)
    def to_6d(poses):
        if poses.shape[1] == 7:
            result = np.zeros((poses.shape[0], 6))
            result[:, :3] = poses[:, :3]
            # Convert quaternion to axis-angle
            quats = poses[:, 3:7]
            rotations = R.from_quat(quats)
            result[:, 3:6] = rotations.as_rotvec()
            return result
        return poses

    noisy_6d = to_6d(noisy_poses)
    smooth_6d = to_6d(smooth_poses)

    n_chunks = N - chunk_size
    chunks = np.zeros((n_chunks, chunk_size, 6))

    for t in range(n_chunks):
        for i in range(1, chunk_size + 1):
            # Δpose[t+i] = smooth[t+i] - noisy[t]
            delta_pos = smooth_6d[t + i, :3] - noisy_6d[t, :3]
            delta_orn = smooth_6d[t + i, 3:6] - noisy_6d[t, 3:6]
            chunks[t, i - 1, :3] = delta_pos
            chunks[t, i - 1, 3:6] = delta_orn

    return chunks


# ================================================================
# Main pipeline
# ================================================================

def process_episode(
    episode_dir: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    approach_planner: str = "quintic",
) -> dict:
    """Run full ground truth pipeline on one episode.

    Args:
        episode_dir: Path to episode directory.
        output_dir: Output path (default: same as episode_dir).
        visualize: If True, print summary statistics.
        approach_planner: 'quintic', 'dmp', or 'chomp'.

    Returns:
        dict with keys: smooth_poses, alpha_target, chunk_targets
    """
    # ── 1. Load ──
    frames, noisy_poses, gripper_states, timestamps, existing_smooth, meta = load_episode(episode_dir)

    N = len(noisy_poses)
    if N == 0:
        print("  [GT] No poses found, skipping")
        return {}

    # ── 2. Detect grasp goal (last stable position) ──
    goal = noisy_poses[-APPROACH_END_BUFFER:].mean(axis=0)

    # ── 3. Detect approach phase ──
    is_approach = detect_approach_phase(noisy_poses, goal)

    # ── 4. Smooth position ──
    smooth_pos = smooth_position(noisy_poses[:, :3], goal[:3], is_approach, approach_planner=approach_planner)

    # ── 5. Smooth orientation ──
    # Ensure quaternion format for orientation smoothing
    if noisy_poses.shape[1] == 7:
        noisy_quats = noisy_poses[:, 3:7]
    else:
        # Convert axis-angle to quaternion
        noisy_quats = R.from_rotvec(noisy_poses[:, 3:6]).as_quat()

    if noisy_poses.shape[1] == 7:
        grasp_quat = goal[3:7]
    else:
        grasp_quat = R.from_rotvec(goal[3:6]).as_quat()

    smooth_quats = smooth_orientation(noisy_quats, grasp_quat, is_approach)

    # ── 6. Reconstruct smooth pose ──
    if noisy_poses.shape[1] == 7:
        smooth_poses = np.zeros((N, 7))
        smooth_poses[:, :3] = smooth_pos
        smooth_poses[:, 3:7] = smooth_quats
    else:
        # Convert quaternion back to axis-angle
        smooth_poses = np.zeros((N, 6))
        smooth_poses[:, :3] = smooth_pos
        smooth_poses[:, 3:6] = R.from_quat(smooth_quats).as_rotvec()

    # ── 7. Compute targets ──
    alpha_target = compute_alpha_target(noisy_poses, smooth_poses)
    chunk_targets = compute_chunk_targets(noisy_poses, smooth_poses)

    result = {
        "smooth_poses": smooth_poses,
        "alpha_target": alpha_target,
        "chunk_targets": chunk_targets,
        "is_approach": is_approach,
    }

    # ── 8. Save ──
    out_path = Path(output_dir) if output_dir else Path(episode_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Update data.npz with smooth_poses
    npz_path = out_path / "data.npz"
    npz_data = {
        "noisy_poses": noisy_poses,
        "gripper_states": gripper_states,
        "timestamps": timestamps,
        "smooth_poses": smooth_poses,
        "alpha_target": alpha_target,
    }

    # Save chunk targets as separate file (large)
    if len(chunk_targets) > 0:
        chunk_path = out_path / "chunk_targets.npz"
        np.savez_compressed(chunk_path, chunk_targets=chunk_targets)
        npz_data["chunk_targets_path"] = str(chunk_path.name)

    # Merge with existing data.npz
    np.savez_compressed(npz_path, **npz_data)

    # ── 9. Statistics ──
    pos_error = np.linalg.norm(noisy_poses[:, :3] - smooth_poses[:, :3], axis=1)

    stats = {
        "episode": Path(episode_dir).name,
        "frames": N,
        "approach_frames": int(is_approach.sum()),
        "mean_pos_error": float(pos_error.mean()),
        "max_pos_error": float(pos_error.max()),
        "mean_alpha": float(alpha_target.mean()),
        "chunks": len(chunk_targets),
    }

    if visualize:
        print("\n  Episode: " + stats["episode"])
        print("  Frames:       " + str(stats["frames"]))
        print("  Approach:     " + str(stats["approach_frames"]) + " frames")
        print("  Pos error:    " + "{:.3f}m mean, {:.3f}m max".format(
            stats["mean_pos_error"], stats["max_pos_error"]))
        print("  Mean α:       " + "{:.3f}".format(stats["mean_alpha"]))
        print("  Chunks:       " + str(stats["chunks"]) + " (K=" + str(CHUNK_SIZE) + ")")
        print("  Saved to:     " + str(npz_path))

        # Detect potential issues
        if stats["mean_pos_error"] < 0.001:
            print("  ⚠ WARNING: Very low pos error — noisy pose might be too clean")
        if stats["mean_alpha"] > 0.9:
            print("  ⚠ WARNING: Mean α very high — trajectory may be excessively noisy")
        if stats["mean_alpha"] < 0.01:
            print("  ⚠ WARNING: Mean α near zero — trajectory is already smooth")

    # Write stats to meta.json
    meta_path = out_path / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta["gt_stats"] = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return result


# ================================================================
# Batch processing
# ================================================================

def process_all(input_dir: str, output_dir: Optional[str] = None, approach_planner: str = "quintic"):
    """Process all episodes in a directory."""
    in_path = Path(input_dir)
    out_path = Path(output_dir) if output_dir else in_path

    episode_dirs = sorted([
        p for p in in_path.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "data.npz").exists()
    ])

    if not episode_dirs:
        print("No episodes found in " + str(in_path))
        return

    print("Processing " + str(len(episode_dirs)) + " episodes...")
    print("Approach planner: " + approach_planner)
    print("-" * 50)

    all_stats = []
    for ep in episode_dirs:
        ep_out = out_path / ep.name
        result = process_episode(str(ep), str(ep_out), visualize=True, approach_planner=approach_planner)
        if result:
            stats = {
                "episode": ep.name,
                "frames": len(result.get("smooth_poses", [])),
                "mean_pos_error": float(np.linalg.norm(
                    np.load(str(ep / "data.npz"))["noisy_poses"][:, :3] -
                    result["smooth_poses"][:, :3], axis=1).mean()) if "smooth_poses" in result else 0,
            }
            all_stats.append(stats)
        print()

    # Summary
    if all_stats:
        total_frames = sum(s["frames"] for s in all_stats)
        mean_error = np.mean([s["mean_pos_error"] for s in all_stats])
        print("=" * 50)
        print("Batch complete: " + str(len(all_stats)) + " episodes")
        print("Total frames: " + str(total_frames))
        print("Mean position error across all: " + "{:.4f}m".format(mean_error))
        print("=" * 50)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALIGN Ground Truth Generation — smooth trajectories + training targets"
    )
    parser.add_argument("--episode", type=str, help="Single episode directory to process")
    parser.add_argument("--input-dir", type=str, help="Directory containing multiple episodes")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: same as input)")
    parser.add_argument("--visualize", action="store_true", help="Print per-episode statistics")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Chunk size K for Assistant head (default: 5)")
    parser.add_argument("--approach-planner", type=str, default="quintic",
                        choices=["quintic", "dmp", "chomp"],
                        help="Approach phase trajectory planner (default: quintic)")
    args = parser.parse_args()

    CHUNK_SIZE = args.chunk_size

    if args.episode:
        result = process_episode(args.episode, args.output_dir, visualize=True,
                                 approach_planner=args.approach_planner)
        if not result:
            print("No data generated")
        sys.exit(0)

    if args.input_dir:
        process_all(args.input_dir, args.output_dir, approach_planner=args.approach_planner)
        sys.exit(0)

    # ── No args: run a synthetic test ──
    print("=== ALIGN Ground Truth: Synthetic Test ===\n")

    # Create a synthetic episode
    test_dir = Path("/tmp/align_test_gt")
    test_dir.mkdir(parents=True, exist_ok=True)

    T = 120  # ~4 seconds at 30Hz
    np.random.seed(42)

    # Clean trajectory: move from [0.3, 0.0, 0.25] toward [0.6, 0.05, 0.25]
    clean = np.zeros((T, 6))
    clean[:, 0] = np.linspace(0.3, 0.6, T)   # x
    clean[:, 1] = 0.02 * np.sin(np.linspace(0, 3, T))  # y (slight arc)
    clean[:, 2] = np.linspace(0.25, 0.28, T)  # z

    # Add noise to simulate human teleop
    noisy = clean.copy()
    noisy[:, :3] += np.random.normal(0, 0.015, (T, 3))  # 1.5cm jitter
    noisy[:, :3] += 0.005 * np.sin(2 * np.pi * 10 * np.arange(T) / 30)[:, None]  # 10Hz tremor
    noisy[:, 3:] = 0.0  # identity orientation

    # Save synthetic episode
    np.savez_compressed(
        test_dir / "data.npz",
        noisy_poses=noisy,
        gripper_states=np.zeros(T),
        timestamps=np.arange(T) / 30.0,
    )
    with open(test_dir / "meta.json", "w") as f:
        json.dump({
            "episode_name": "test_synthetic",
            "task_description": "pick up the red mug",
            "num_frames": T,
            "duration_s": T / 30.0,
        }, f, indent=2)
    # Create empty frames dir
    (test_dir / "frames").mkdir(exist_ok=True)

    # Process
    result = process_episode(str(test_dir), str(test_dir), visualize=True)

    if result and "smooth_poses" in result:
        smooth = result["smooth_poses"]
        pos_error = np.linalg.norm(noisy[:, :3] - smooth[:, :3], axis=1)
        alpha = result["alpha_target"]

        print("\n  === Verification ===")
        print("  Smooth preserves path:       " + str(bool(np.corrcoef(noisy[:, 0], smooth[:, 0])[0, 1] > 0.99)))
        print("  Smooth has less noise:       " + str(bool(np.std(smooth[:30, 0]) < np.std(noisy[:30, 0]))))
        print("  α reasonable range:          " + str(bool(alpha.min() >= 0 and alpha.max() <= 1.0)))
        print("  Chunks correct shape:        " + str(result["chunk_targets"].shape))
        print("\n" + "=" * 50)
        print("PASSED" if pos_error.mean() < 0.03 else "CHECK MANUALLY")
        print("=" * 50)