#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDF5 dataset loader for ALIGN training.

Converts recorded episodes (frames/ + data.npz + meta.json) into HDF5 format
for efficient training. Handles multi-camera, text variants, and produces
(batch frame, trajectory, text) chunks for contrastive pretraining
and head training. (Distances were removed from the Decision head
to make ALIGN fully self-contained for real deployment.)

Usage:
    # Convert raw recordings to HDF5
    python -m data.align_dataset convert --raw-dir ./align_data --output ./align.h5

    # Load for training
    from data.align_dataset import ALIGNDataset
    ds = ALIGNDataset("align.h5", mode="pretrain", frames_per_ep=8)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ================================================================
# Constants
# ================================================================

DEFAULT_CAMERA = "wrist"
DEFAULT_SIZE = (224, 224)
DEFAULT_FRAMES_PER_EP = 8
TRAJ_WINDOW = 10  # K frames per trajectory window
POSITIVE_WINDOW = 5  # W_pos — frames within this window are positive pairs


# ================================================================
# Dataset
# ================================================================

class ALIGNDataset(Dataset):
    """HDF5 dataset for ALIGN training.

    Stores episodes as HDF5 groups:
        /meta/              — json strings with episode metadata
        /ep_XXX/frames/     — (N, H, W, 3) uint8 frames
        /ep_XXX/noisy_poses — (N, 6) or (N, 7) float32
        /ep_XXX/gripper     — (N,) float32
        /ep_XXX/texts       — list of text variant strings

    Supports two modes:
    - 'pretrain': sample (frame, trajectory_window) pairs within episodes.
      Text from the same episode, negatives from different episodes.
    - 'head': sample sequential chunks for Decision/Assistant targets.
    """

    def __init__(
        self,
        h5_path: str,
        mode: str = "pretrain",
        camera: str = DEFAULT_CAMERA,
        image_size: Tuple[int, int] = DEFAULT_SIZE,
        frames_per_ep: int = DEFAULT_FRAMES_PER_EP,
        traj_window: int = TRAJ_WINDOW,
        episodes_per_batch: int = 8,
    ):
        self.h5_path = Path(h5_path)
        self.mode = mode
        self.camera = camera
        self.image_size = image_size
        self.frames_per_ep = frames_per_ep
        self.traj_window = traj_window
        self.episodes_per_batch = episodes_per_batch

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self._h5 = h5py.File(self.h5_path, "r")
        self._episode_keys = sorted([k for k in self._h5.keys() if k.startswith("ep_")])
        if not self._episode_keys:
            # Backward compat: root-level frames/noisy_poses (single episode)
            if "frames" in self._h5 and "noisy_poses" in self._h5:
                self._episode_keys = ["ep_single"]
                self._single_episode = True
            else:
                raise ValueError(f"No episodes found in {h5_path}")
        else:
            self._single_episode = False

        # Auto-detect camera key from actual HDF5 structure
        first_ep = self._episode_keys[0]
        frames_obj = self._h5[f"{first_ep}/frames"]
        if isinstance(frames_obj, h5py.Dataset):
            # Frames is a single (N, H, W, 3) array — no camera subgroups
            self.camera = None
        else:
            # Frames is a group with camera sub-datasets (e.g., frames/wrist_image)
            available_cameras = list(frames_obj.keys())
            if self.camera in available_cameras:
                pass
            elif "wrist_image" in available_cameras and self.camera in ("wrist", ""):
                self.camera = "wrist_image"
            elif len(available_cameras) == 1:
                self.camera = available_cameras[0]
            else:
                raise ValueError(f"Camera {self.camera} not found. Available: {available_cameras}")

        # Detect if noisy_poses are cumulative across episodes (LIBERO v3 quirk)
        # and pre-compute per-episode frame lengths + pose offsets
        self._ep_frame_lengths: List[int] = []
        self._ep_pose_offsets: List[int] = []

        for ep_idx in range(len(self._episode_keys)):
            key = self._episode_keys[ep_idx]
            # Frame length (handle both framedataset structures)
            if self.camera is None:
                # frames = Dataset — single array
                n_frames = len(self._h5[f"{key}/frames"])
            else:
                try:
                    n_frames = len(self._h5[f"{key}/frames/{self.camera}"])
                except KeyError:
                    n_frames = len(self._h5[f"{key}/noisy_poses"])
            self._ep_frame_lengths.append(n_frames)

            # Pose offset: in cumulative HDF5, ep_N starts AFTER all previous episodes
            n_poses = len(self._h5[f"{key}/noisy_poses"])
            if n_poses == n_frames:
                # Non-cumulative — pose aligns with frames directly
                self._ep_pose_offsets.append(0)
            elif ep_idx > 0:
                # Cumulative — offset = sum of all previous frame lengths
                self._ep_pose_offsets.append(sum(self._ep_frame_lengths[:-1]))
            else:
                self._ep_pose_offsets.append(0)

        # Warn if cumulative detected
        if any(o > 0 for o in self._ep_pose_offsets):
            print("  NOTE: Detected cumulative noisy_poses — using per-episode offsets")
        # Build index: list of (ep_idx, start_frame, n_frames) for each valid window
        self._index: List[Tuple[int, int, int]] = []
        for ep_idx in range(len(self._episode_keys)):
            n_frames = self._get_episode_length(ep_idx)
            if n_frames >= traj_window + frames_per_ep:
                max_start = n_frames - traj_window - frames_per_ep + 1
                for start in range(0, max_start, frames_per_ep):
                    self._index.append((ep_idx, start, frames_per_ep))

    def _get_episode_length(self, ep_idx: int) -> int:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            return len(self._h5["noisy_poses"])
        return self._ep_frame_lengths[ep_idx]

    def _read_frames(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            frames = self._h5["frames"][start:start + count]
        elif self.camera is None:
            # Frames is a single Dataset (N, H, W, 3)
            frames = self._h5[f"{key}/frames"][start:start + count]
        else:
            frames = self._h5[f"{key}/frames/{self.camera}"][start:start + count]
        return frames

    def _read_poses(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            return self._h5["noisy_poses"][start:start + count]
        # Handle cumulative noisy_poses (LIBERO v3 quirk)
        offset = self._ep_pose_offsets[ep_idx]
        abs_start = offset + start
        raw = self._h5[f"{key}/noisy_poses"][abs_start:abs_start + count]
        # Pad if fewer items than requested (episode boundary)
        if len(raw) < count:
            pad = np.zeros((count - len(raw), raw.shape[1]), dtype=raw.dtype)
            raw = np.concatenate([raw, pad], axis=0)
        return raw

    def _read_text(self, ep_idx: int) -> str:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            try:
                raw = self._h5["meta"][()]
                meta = json.loads(raw) if isinstance(raw, (bytes, str)) else {}
            except (KeyError, ValueError):
                return "pick and place"
            return meta.get("task_description", "pick and place")

        # Try /ep_XXX/texts first (JSON array as bytes), then /ep_XXX/meta
        ep_group = self._h5[key]
        try:
            raw = ep_group["texts"][()]
            if isinstance(raw, bytes):
                texts = json.loads(raw)
            elif isinstance(raw, str):
                texts = json.loads(raw)
            else:
                texts = [str(t) for t in list(raw)]
            if texts and isinstance(texts, list):
                return texts[0]
        except (KeyError, ValueError):
            pass

        try:
            meta = json.loads(ep_group["meta"][()])
            return meta.get("task_description", "pick and place")
        except (KeyError, ValueError):
            pass

        return "pick and place"

    def __len__(self) -> int:
        return len(self._index)

    def close(self):
        """Close the HDF5 file handle. Call when done to release the FD."""
        if hasattr(self, "_h5") and self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Best-effort cleanup; in multi-worker DataLoaders, each worker
        # has its own serialized copy of the dataset, so the file handle
        # is per-worker. Always close() explicitly when possible.
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start, count = self._index[idx]
        frames = self._read_frames(ep_idx, start, count + self.traj_window)
        poses = self._read_poses(ep_idx, start, count + self.traj_window)
        text = self._read_text(ep_idx)

        return {
            "frames": frames,
            "poses": poses,
            "text": text,
            "ep_idx": ep_idx,
        }


# ================================================================
# Head Training Utilities
# ================================================================

NOISE_STD = 0.015 # 1.5cm positional noise for synthetic human deviation
D_MAX = 0.10      # Max drift before full assist (alpha=1)

def inject_kinematic_noise(pos: np.ndarray, rng: "np.random.Generator", std: float = NOISE_STD) -> np.ndarray:
    """Inject zero-mean Gaussian noise on position [x, y, z]."""
    pos_noisy = pos.copy()
    if len(pos) >= 3:
        pos_noisy[:3] += rng.standard_normal(3).astype(np.float32) * std
    return pos_noisy


# ================================================================
# Collate function for contrastive pretraining
# ================================================================

def pretrain_collate(batch: list, traj_window: int = TRAJ_WINDOW) -> dict:
    """Collate batch for contrastive pretraining.

    Samples positive pairs within episodes, ensures negatives come from
    different episodes.

    Returns:
        {
            "frames": (B, H, W, 3) uint8,
            "trajectories": (B, K, 6) float32,
            "texts": list of strings,
            "ep_ids": (B,) int — for verifying positives/negatives,
        }
    """
    all_frames = []
    all_trajs = []
    all_texts = []
    all_ep_ids = []

    for i, item in enumerate(batch):
        frames = item["frames"]
        poses = item["poses"]
        ep_idx = item["ep_idx"]

        N = len(frames)
        # Sample a random anchor timestep within this episode
        # The vision frame and trajectory window are both anchored at the
        # same time t — vision and trajectory are temporally aligned for
        # the positive pair. (Previously t1 and t2 were independent
        # randoms — that breaks the contrastive signal.)
        max_offset = N - traj_window
        if max_offset > 0:
            t1 = np.random.randint(0, max_offset)  # vision anchor
            t2 = t1                                # trajectory anchor — SAME time
        else:
            t1 = 0
            t2 = 0

        frame_sample = frames[t1]  # (H, W, 3)
        traj_sample = poses[t2:t2 + traj_window]  # (K, 6) or (K, 7)

        all_frames.append(frame_sample)
        all_trajs.append(traj_sample)
        all_texts.append(item["text"])
        all_ep_ids.append(ep_idx)

    return {
        "frames": np.stack(all_frames, axis=0),
        "trajectories": np.stack(all_trajs, axis=0).astype(np.float32),
        "texts": all_texts,
        "ep_ids": np.array(all_ep_ids, dtype=int),
    }


def head_collate(batch: list, chunk_size: int = 5) -> dict:
    """Collate batch for head training with on-the-fly noise injection.

    Returns sequential chunks for Decision (α = need × consistency) + Assistant supervision.

    Note: The 'consistency' part (cosine similarity of embeddings) is calculated 
    during the forward pass in train_heads.py using frozen encoders. This collate
    function provides the 'need' component via kinematic error from noisy poses.

    Returns:
        {
            "frames": (B, H, W, 3) uint8,
            "noisy_pose": (B, 6) float32 — corrupted pose for input,
            "clean_pose":  (B, 6) float32 — ground truth pose from HDF5,
            "trajectory":  (B, K, 6) float32 — past window of clean poses,
            "alpha_need":  (B,) float32    — kinematic error part of alpha_target,
            "delta_target":(B, chunk_size, 6) float32,
            "texts":       list of strings,
        }
    """
    all_frames = []
    all_noisy = []
    all_clean = []
    all_trajs = []
    all_needs = []
    all_deltas = []
    all_texts = []

    # Use a fixed seed for noise injection in the batch so it's reproducible
    rng = np.random.default_rng()

    for item in batch:
        frames = item["frames"]
        poses_clean = item["poses"][..., :6]  # Ensure 6D (drop quaternion if 7D)
        texts_raw = item["text"] if isinstance(item["text"], list) else item["text"]

        N = len(poses_clean)
        max_t = N - chunk_size
        t = rng.integers(0, min(max(max_t + 1, 1), N)) if max_t >= 0 else min(rng.integers(0, 2), N - 1)

        # --- Past Trajectory Window (Clean for encoding) — fixed to always be chunk_size ---
        start = max(0, t - chunk_size + 1)
        traj_window = poses_clean[start:t + 1]
        if len(traj_window) < chunk_size:
            pad = np.zeros((chunk_size - len(traj_window), 6), dtype=np.float32)
            traj_window = np.concatenate([pad, traj_window], axis=0)

        # --- Inject Noise for Current Pose ---
        current_clean_pose = poses_clean[t]
        noisy_pose = inject_kinematic_noise(current_clean_pose, rng)

        # --- Compute "Need" (Kinematic Error / D_MAX) ---
        pos_error = np.linalg.norm(current_clean_pose[:3] - noisy_pose[:3])
        need = min(pos_error / D_MAX, 1.0)

        # --- Option B: Recovery + Incremental Expert Trajectory Targets ---
        # Step 0: Immediate recovery from deviation toward expert path
        # Steps 1..N-1: Smooth continuation along expert motion increments
        delta = np.zeros((chunk_size, 6), dtype=np.float32)

        if t + 1 < N:
            delta[0] = poses_clean[t + 1, :6] - noisy_pose[:6]  # Recovery correction

        for i in range(2, chunk_size + 1):
            if t + i < N:
                delta[i - 1] = poses_clean[t + i, :6] - poses_clean[t + i - 1, :6]  # Expert increment

        # Text variant
        if isinstance(texts_raw, list):
            text = texts_raw[rng.integers(0, len(texts_raw))]
        else:
            text = texts_raw

        all_frames.append(frames[t])
        all_noisy.append(noisy_pose[:6])
        all_clean.append(current_clean_pose[:6])
        all_trajs.append(traj_window[:, :6])
        all_needs.append(need)
        all_deltas.append(delta)
        all_texts.append(text)

    return {
        "frames": np.stack(all_frames, axis=0),
        "noisy_pose": np.stack(all_noisy, axis=0).astype(np.float32),
        "clean_pose": np.stack(all_clean, axis=0).astype(np.float32),
        "trajectory": np.stack(all_trajs, axis=0).astype(np.float32),
        "alpha_need": np.array(all_needs, dtype=np.float32),
        "delta_target": np.stack(all_deltas, axis=0).astype(np.float32),
        "texts": all_texts,
    }


# ================================================================
# Converter: raw recordings → HDF5
# ================================================================

def convert_raw_to_hdf5(
    raw_dir: str,
    output_path: str,
    camera: str = DEFAULT_CAMERA,
    image_size: Tuple[int, int] = DEFAULT_SIZE,
    max_frames_per_ep: Optional[int] = None,
) -> str:
    """Convert recorded align_data episodes to a single HDF5 file.

    Args:
        raw_dir: Path to directory with episode subdirectories.
        output_path: Output .h5 file path.
        camera: Camera view to use.
        image_size: Target (H, W) for saved frames.
        max_frames_per_ep: Maximum frames to keep per episode.

    Returns:
        Path to the created HDF5 file.
    """
    raw_path = Path(raw_dir)
    episodes = sorted([
        p for p in raw_path.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "data.npz").exists()
    ])

    if not episodes:
        raise FileNotFoundError(f"No episodes found in {raw_dir}")

    print(f"Converting {len(episodes)} episodes to {output_path}...")

    with h5py.File(output_path, "w") as h5:
        for ep_dir in episodes:
            ep_name = f"ep_{ep_dir.name}"
            group = h5.create_group(ep_name)

            # Load data
            data = dict(np.load(ep_dir / "data.npz"))
            noisy_poses = data["noisy_poses"][:max_frames_per_ep]
            gripper = data.get("gripper_states", np.zeros(len(noisy_poses)))[:max_frames_per_ep]
            smooth = data.get("smooth_poses", None)
            alpha = data.get("alpha_target", None)
            chunk_targets = None
            if (ep_dir / "chunk_targets.npz").exists():
                chunk_targets = dict(np.load(ep_dir / "chunk_targets.npz")).get("chunk_targets", None)

            # Frames
            frames_dir = ep_dir / "frames"
            camera_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
            if camera_dirs:
                label_dir = frames_dir / camera
                if not label_dir.exists():
                    label_dir = camera_dirs[0]
                frame_files = sorted(label_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            else:
                frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))

            frames = []
            for f in frame_files[:max_frames_per_ep]:
                img = Image.open(f).resize(image_size)
                frames.append(np.array(img))
            frames_arr = np.stack(frames, axis=0).astype(np.uint8)

            # Meta with text annotations
            with open(ep_dir / "meta.json") as f:
                meta = json.load(f)
            task_desc = meta.get("task_description", "pick and place")
            # Generate text variants
            target = meta.get("target_object", "")
            objects = meta.get("objects_on_table", {})
            object_list = list(objects.keys())
            object_type = target.replace("_", " ").split()[-1] if target else "object"
            color = target.replace("_", " ").split()[0] if target else "the"

            text_variants = [
                task_desc,
                f"pick up the {color} {object_type}",
                f"grasp the {object_type}",
                "pick and place the object",
                "grasp and move",
            ]

            # Save
            h5[ep_name + "/frames/" + camera] = frames_arr
            h5[ep_name + "/noisy_poses"] = noisy_poses.astype(np.float32)
            h5[ep_name + "/gripper"] = gripper.astype(np.float32)
            h5[ep_name + "/texts"] = json.dumps(text_variants)
            h5[ep_name + "/meta"] = json.dumps(meta)

            if smooth is not None:
                h5[ep_name + "/smooth_poses"] = smooth.astype(np.float32)
            if alpha is not None:
                h5[ep_name + "/alpha_target"] = alpha.astype(np.float32)
            if chunk_targets is not None:
                h5[ep_name + "/chunk_targets"] = chunk_targets.astype(np.float32)

        # Write metadata
        h5["meta/total_episodes"] = len(episodes)
        h5["meta/camera"] = camera

    print(f"  Done: {output_path}")
    return str(Path(output_path).absolute())


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ALIGN HDF5 Dataset Converter")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert raw episodes to HDF5")
    convert.add_argument("--raw-dir", required=True, help="Raw align_data directory")
    convert.add_argument("--output", default="align.h5", help="Output HDF5 file")
    convert.add_argument("--camera", default=DEFAULT_CAMERA)
    convert.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    if args.command == "convert":
        convert_raw_to_hdf5(args.raw_dir, args.output, camera=args.camera, max_frames_per_ep=args.max_frames)


if __name__ == "__main__":
    main()
