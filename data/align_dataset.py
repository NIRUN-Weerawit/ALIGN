#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDF5 dataset loader for ALIGN training.

Converts recorded episodes (frames/ + data.npz + meta.json) into HDF5 format
for efficient training. Handles multi-camera, text variants, and produces
(batch frame, trajectory, text, distances) chunks for contrastive pretraining
and head training.

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
        return len(self._h5[f"{key}/noisy_poses"])

    def _read_frames(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            frames = self._h5["frames"][start:start + count]
        else:
            frames = self._h5[f"{key}/frames/{self.camera}"][start:start + count]
        return frames

    def _read_poses(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            return self._h5["noisy_poses"][start:start + count]
        return self._h5[f"{key}/noisy_poses"][start:start + count]

    def _read_text(self, ep_idx: int) -> str:
        if self._single_episode:
            meta = json.loads(self._h5["meta"][()])
            return meta.get("task_description", "pick and place")
        meta = json.loads(self._h5[f"{self._episode_keys[ep_idx]}/meta"][()])
        return meta.get("task_description", "pick and place")

    def __len__(self) -> int:
        return len(self._index)

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
        # Sample a random positive pair within this episode
        max_offset = N - traj_window
        if max_offset > 0:
            t1 = np.random.randint(0, max_offset)
            t2 = np.random.randint(0, max_offset)
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
    """Collate batch for head training.

    Returns sequential chunks for Decision + Assistant supervision.

    Returns:
        {
            "frames": (B, H, W, 3) uint8,
            "noisy_pose": (B, 6) float32 — current frame pose,
            "trajectory": (B, K, 6) float32 — past window,
            "alpha_target": (B,) float32,
            "delta_target": (B, chunk_size, 6) float32,
            "texts": list of strings,
            "distances": (B, 3) float32,
        }
    """
    all_frames = []
    all_noisy = []
    all_trajs = []
    all_alphas = []
    all_deltas = []
    all_texts = []
    all_dists = []

    for item in batch:
        frames = item["frames"]
        poses = item["poses"]
        texts = item["text"] if isinstance(item["text"], list) else item["text"]

        N = len(poses)
        # Pick a random valid position for chunk extraction
        max_t = N - chunk_size
        if max_t >= 0:
            t = np.random.randint(0, max_t)
        else:
            t = 0

        # Past trajectory window
        past_start = max(0, t - chunk_size)
        traj_window = poses[past_start:t + 1]  # could be < K, pad if needed
        if len(traj_window) < chunk_size:
            pad = np.zeros((chunk_size - len(traj_window), poses.shape[1]), dtype=poses.dtype)
            traj_window = np.concatenate([pad, traj_window], axis=0)

        # Future delta target
        delta = np.zeros((chunk_size, 6), dtype=np.float32)
        for i in range(1, chunk_size + 1):
            if t + i < N:
                delta[i - 1] = poses[t + i, :6] - poses[t, :6]

        # α_target placeholder — to be overwritten after GT generation
        alpha_target = 0.5  # placeholder

        # Distance placeholder
        dist = np.zeros(3, dtype=np.float32)

        # Text variant
        if isinstance(texts, list):
            text = texts[np.random.randint(len(texts))]
        else:
            text = texts

        all_frames.append(frames[t])
        all_noisy.append(poses[t, :6])
        all_trajs.append(traj_window[:, :6])
        all_alphas.append(alpha_target)
        all_deltas.append(delta)
        all_texts.append(text)
        all_dists.append(dist)

    return {
        "frames": np.stack(all_frames, axis=0),
        "noisy_pose": np.stack(all_noisy, axis=0).astype(np.float32),
        "trajectory": np.stack(all_trajs, axis=0).astype(np.float32),
        "alpha_target": np.array(all_alphas, dtype=np.float32),
        "delta_target": np.stack(all_deltas, axis=0).astype(np.float32),
        "texts": all_texts,
        "distances": np.stack(all_dists, axis=0).astype(np.float32),
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
