#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapters for open robot manipulation datasets → ALIGN format.

Supported datasets:
  - Robomimic (Lift, Can, Square, Transport, Tool Hang)
    Format: HDF5 files with obs/action keys, expert demonstrations
    Robot: Franka Panda, wrist + front camera views
    Text: None — we auto-generate task descriptions from dataset task name

  - DROID (Distributed Robot Interaction Dataset)
    Format: RLDS/TFDS, 76K episodes with language annotations
    Robot: Franka Panda, wrist camera
    Text: Language annotations provided

  - BridgeData v2
    Format: TFDS, 50K+ trajectories
    Robot: WidowX 250
    Text: Templated task descriptions

Usage:
    from data.open_dataset import OpenDatasetAdapter, create_align_dataset

    # Robomimic
    adapter = OpenDatasetAdapter("robomimic", data_dir="./robomimic_data")
    dataset = adapter.to_align_format(output_path="align_pretrain.h5")

    # DROID
    adapter = OpenDatasetAdapter("droid", data_dir="./droid_data")
    dataset = adapter.to_align_format(output_path="align_pretrain.h5")
"""

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Lazy imports — only import when needed
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("WARNING: h5py not installed. Install with: pip install h5py", file=sys.stderr)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# Constants
# ================================================================

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CAMERA = "agentview"  # Robomimic default wrist/egocentric view


# ================================================================
# Base adapter
# ================================================================

class OpenDatasetAdapter(ABC):
    """Abstract adapter for converting open datasets to ALIGN HDF5 format."""

    def __init__(
        self,
        name: str,
        data_dir: str,
        camera: str = DEFAULT_CAMERA,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_episodes: Optional[int] = None,
    ):
        self.name = name
        self.data_dir = Path(data_dir)
        self.camera = camera
        self.image_size = image_size
        self.max_episodes = max_episodes

    @abstractmethod
    def get_episode_count(self) -> int:
        """Return number of episodes in the dataset."""

    @abstractmethod
    def get_episode(self, idx: int) -> dict:
        """Return episode dict with keys: frames, poses, gripper, text.

        Returns:
            dict:
                frames: (N, H, W, 3) uint8 array
                poses: (N, 6) or (N, 7) float32 array
                gripper: (N,) float32 array
                text: str — task description
        """

    def to_align_format(self, output_path: str, max_frames_per_ep: Optional[int] = None) -> str:
        """Convert entire dataset to ALIGN HDF5 format.

        Args:
            output_path: Output HDF5 file path.
            max_frames_per_ep: Truncate episodes to this many frames.

        Returns:
            Path to the created HDF5 file.
        """
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 conversion. Install with: pip install h5py"
            )

        n_eps = self.get_episode_count()
        if self.max_episodes:
            n_eps = min(n_eps, self.max_episodes)

        print(f"[{self.name}] Converting {n_eps} episodes → {output_path}")

        with h5py.File(output_path, "w") as h5:
            for ep_idx in range(n_eps):
                ep_data = self.get_episode(ep_idx)
                frames = ep_data["frames"]
                poses = ep_data["poses"]
                gripper = ep_data.get("gripper", np.zeros(len(frames)))
                text = ep_data.get("text", "pick and place")

                # Truncate
                if max_frames_per_ep and len(frames) > max_frames_per_ep:
                    frames = frames[:max_frames_per_ep]
                    poses = poses[:max_frames_per_ep]
                    gripper = gripper[:max_frames_per_ep]

                # Generate text variants
                text_variants = _generate_text_variants(text)

                ep_name = f"ep_{ep_idx:05d}"
                group = h5.create_group(ep_name)
                group.create_dataset(f"frames/{self.camera}", data=frames.astype(np.uint8))
                group.create_dataset("noisy_poses", data=poses.astype(np.float32))
                group.create_dataset("gripper", data=gripper.astype(np.float32))
                group.create_dataset("texts", data=json.dumps(text_variants))
                group.create_dataset("meta", data=json.dumps({
                    "task_description": text,
                    "source": self.name,
                    "camera": self.camera,
                    "num_frames": len(frames),
                }))

                if (ep_idx + 1) % 100 == 0:
                    print(f"  {ep_idx + 1}/{n_eps} episodes processed")

            h5["meta/total_episodes"] = n_eps
            h5["meta/source"] = self.name
            h5["meta/camera"] = self.camera

        print(f"  Done: {output_path}")
        return str(Path(output_path).absolute())


# ================================================================
# Robomimic adapter
# ================================================================

class RobomimicAdapter(OpenDatasetAdapter):
    """Adapter for Robomimic datasets.

    Structure:
        robomimic_data/
            lift/ph/     (proficient human)
            can/ph/
            square/ph/
            transport/ph/
            tool_hang/ph/

    Each task directory contains:
        demo.hdf5 — all demonstrations
        Each demo has:
            obs/agentview_image  — (N, H, W, 3) uint8 frames
            obs/robot0_eef_pos   — (N, 3) EEF position
            obs/robot0_eef_quat  — (N, 4) EEF quaternion
            obs/robot0_gripper_qpos  — (N, 2) gripper joint positions
            actions               — (N, 7) delta actions

    Task names as text:
        Lift → "lift the cube"
        Can → "pick up the can"
        Square → "pick up the square nut"
        Transport → "pick and place the cube"
        Tool Hang → "hang the tool on the hook"
    """

    TASK_ALIASES = {
        "lift": "lift the cube",
        "can": "pick up the can",
        "square": "pick up the square nut",
        "transport": "pick and place the cube",
        "tool_hang": "hang the tool on the hook",
    }

    def __init__(
        self,
        data_dir: str,
        task: str,
        hdf5_name: str = "demo.hdf5",
        camera: str = DEFAULT_CAMERA,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_episodes: Optional[int] = None,
    ):
        super().__init__("robomimic", data_dir, camera, image_size, max_episodes)
        self.task = task
        self.hdf5_path = self.data_dir / task / "ph" / hdf5_name
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Robomimic data not found: {self.hdf5_path}")

        self._h5 = h5py.File(self.hdf5_path, "r")
        self._demo_keys = sorted([k for k in self._h5["data"].keys() if k.startswith("demo_")])

        print(f"[robomimic] {task}: {len(self._demo_keys)} demonstrations from {self.hdf5_path}")

    def get_episode_count(self) -> int:
        return len(self._demo_keys)

    def get_episode(self, idx: int) -> dict:
        key = self._demo_keys[idx]
        demo = self._h5["data"][key]

        # Extract observations
        frames_raw = demo["obs"][self.camera + "_image"][()]  # (N, H, W, 3)
        eef_pos = demo["obs"]["robot0_eef_pos"][()]
        eef_quat = demo["obs"]["robot0_eef_quat"][()]
        gripper = demo["obs"]["robot0_gripper_qpos"][()]

        # Resize frames
        N = len(frames_raw)
        frames = np.zeros((N,) + self.image_size + (3,), dtype=np.uint8)
        for i in range(N):
            img = Image.fromarray(frames_raw[i].astype(np.uint8))
            frames[i] = np.array(img.resize(self.image_size))

        # Combine pose → 7D [x, y, z, qw, qx, qy, qz] or [x, y, z, qx, qy, qz, qw]
        # Robomimic stores quaternion as (w, x, y, z) — convert to (x, y, z, w)
        poses = np.zeros((N, 7), dtype=np.float32)
        poses[:, :3] = eef_pos
        poses[:, 3] = eef_quat[:, 1]  # qx
        poses[:, 4] = eef_quat[:, 2]  # qy
        poses[:, 5] = eef_quat[:, 3]  # qz
        poses[:, 6] = eef_quat[:, 0]  # qw

        # Gripper: Robomimic has 2-finger gripper, take open/close signal
        # (0 = close, 1 = open typically)
        if gripper.ndim == 2:
            gripper_state = gripper[:, 0].astype(np.float32)
        else:
            gripper_state = gripper.astype(np.float32)

        return {
            "frames": frames,
            "poses": poses,
            "gripper": gripper_state,
            "text": self.TASK_ALIASES.get(self.task, f"complete the {self.task} task"),
        }

    def close(self):
        if hasattr(self, "_h5"):
            self._h5.close()


# ================================================================
# DROID adapter
# ================================================================

class DROIDAdapter(OpenDatasetAdapter):
    """Adapter for DROID dataset.

    Structure:
        droid_data/
            success/    — successful episodes
            <episode_id>/
                data.npz
                meta.json

    DROID stores:
        - camera images at multiple views
        - EEF poses (position + orientation)
        - Gripper state
        - Language descriptions per episode

    NOTE: DROID is a large dataset (76K episodes, ~1.7 TB).
    Use --max-episodes to limit for testing.
    """

    def __init__(
        self,
        data_dir: str,
        camera: str = "exterior_image_1_left",  # closest to wrist view
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_episodes: Optional[int] = None,
    ):
        super().__init__("droid", data_dir, camera, image_size, max_episodes)

        # Find episode directories
        if (self.data_dir / "success").exists():
            self._episode_dir = self.data_dir / "success"
        else:
            self._episode_dir = self.data_dir

        self._episode_paths = sorted([
            p for p in self._episode_dir.iterdir()
            if p.is_dir() and (p / "data.npz").exists()
        ])

        if not self._episode_paths:
            raise FileNotFoundError(f"No DROID episodes found in {data_dir}")

        print(f"[droid] {len(self._episode_paths)} episodes from {self._episode_dir}")

    def get_episode_count(self) -> int:
        return len(self._episode_paths)

    def get_episode(self, idx: int) -> dict:
        ep_path = self._episode_paths[idx]

        # Load data
        data = dict(np.load(ep_path / "data.npz"))

        # Extract observations
        frames_key = self.camera + "_image"
        if frames_key in data:
            frames = data[frames_key]
        else:
            # Try alternative keys
            alt_keys = [k for k in data.keys() if "image" in k.lower()]
            if alt_keys:
                frames = data[alt_keys[0]]
            else:
                raise KeyError(f"No image data found for camera '{self.camera}' in {ep_path}")

        eef_pos = data.get("robot0_eef_pos", data.get("eef_pos"))
        eef_rot = data.get("robot0_eef_rot", data.get("eef_rot"))
        grip = data.get("robot0_gripper_pos", data.get("gripper_pos"))

        if eef_pos is None:
            raise KeyError(f"No EEF position data in {ep_path}")

        # Build 7D pose
        N = len(frames)
        poses = np.zeros((N, 7), dtype=np.float32)
        poses[:, :3] = eef_pos
        if eef_rot is not None:
            # DROID uses 3x3 rotation matrix or quaternion
            if eef_rot.shape[-2:] == (3, 3):
                # Convert rotation matrix to quaternion
                from scipy.spatial.transform import Rotation
                quats = Rotation.from_matrix(eef_rot.reshape(-1, 3, 3)).as_quat()
                poses[:, 3:] = quats  # (x, y, z, w)
            elif eef_rot.shape[-1] == 4:
                poses[:, 3:] = eef_rot  # already quaternion
            elif eef_rot.shape[-1] == 3:
                poses[:, 3:6] = eef_rot  # axis-angle, pad with 0
        else:
            poses[:, 3:] = 0  # identity

        # Gripper
        if grip is not None:
            if grip.ndim == 2:
                gripper_state = grip[:, 0].astype(np.float32)
            else:
                gripper_state = grip.astype(np.float32)
        else:
            gripper_state = np.zeros(N, dtype=np.float32)

        # Text
        text = "pick and place"
        meta_path = ep_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            text = meta.get("language_description", text)
        elif (ep_path.parent / "language_instructions.json").exists():
            # Some versions store language separately
            try:
                with open(ep_path.parent / "language_instructions.json") as f:
                    lang_data = json.load(f)
                ep_lang = lang_data.get(ep_path.name, "")
                if ep_lang:
                    text = ep_lang
            except Exception:
                pass

        return {
            "frames": frames.astype(np.uint8),
            "poses": poses,
            "gripper": gripper_state,
            "text": str(text),
        }


# ================================================================
# BridgeData v2 adapter
# ================================================================

class BridgeDataAdapter(OpenDatasetAdapter):
    """Adapter for BridgeData v2.

    Structure: TFDS format, WidowX 250 robot.
    Note: WidowX is 6-DOF, different kinematics from Franka.
    Use with caution — vision features transfer but action space differs.

    Install: pip install tensorflow-datasets
    """

    def __init__(
        self,
        data_dir: str,
        camera: str = "image_0",  # wrist camera
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_episodes: Optional[int] = None,
    ):
        super().__init__("bridge", data_dir, camera, image_size, max_episodes)
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("tensorflow_datasets required: pip install tensorflow-datasets")

        self._builder = tfds.builder("bridge_dataset", data_dir=str(data_dir))
        self._builder.download_and_prepare()
        self._dataset = self._builder.as_dataset(split="train")
        self._episodes = list(self._dataset)
        print(f"[bridge] {len(self._episodes)} episodes")

    def get_episode_count(self) -> int:
        return len(self._episodes)

    def get_episode(self, idx: int) -> dict:
        episode = self._episodes[idx]
        steps = list(episode["steps"])

        N = len(steps)
        frames = np.zeros((N,) + self.image_size + (3,), dtype=np.uint8)
        poses = np.zeros((N, 7), dtype=np.float32)
        gripper = np.zeros(N, dtype=np.float32)

        for i, step in enumerate(steps):
            obs = step["observation"]
            img = obs[self.camera]
            if isinstance(img, bytes):
                import io
                img = Image.open(io.BytesIO(img))
            frames[i] = np.array(img.resize(self.image_size))

            # Bridge uses WidowX EEF position + orientation
            if "state" in obs:
                state = obs["state"]
                poses[i, :3] = state[:3]
                # WidowX gripper is binary open/close
                gripper[i] = float(state[-1] > 0.5)

        # Language instruction (templated)
        text = "pick and place"
        if "language_instruction" in episode:
            text = episode["language_instruction"].numpy().decode()
        elif "language_embedding" in episode:
            text = "pick and place"

        return {
            "frames": frames,
            "poses": poses,
            "gripper": gripper,
            "text": text,
        }


# ================================================================
# Helpers
# ================================================================

def _generate_text_variants(task_text: str) -> list[str]:
    """Generate multiple text variants for contrastive training."""
    task_lower = task_text.lower()
    variants = [task_text]

    # Extract key words
    if "pick up" in task_lower or "lift" in task_lower:
        # Pick-and-place variant
        obj = task_lower.replace("pick up the ", "").replace("lift the ", "")
        variants.extend([
            f"grasp the {obj}",
            "pick and place the object",
            "grasp and move",
        ])
    elif "place" in task_lower:
        obj = task_lower.replace("place the ", "").replace("place in ", "").replace("pick and place the ", "")
        variants.extend([
            f"move the {obj}",
            "pick and place the object",
        ])
    elif "push" in task_lower or "slide" in task_lower:
        variants.extend([
            "push the object",
            "move the item",
        ])
    elif "hang" in task_lower:
        variants.extend([
            "hang the object",
            "place the tool",
        ])
    else:
        variants.extend([
            "pick and place the object",
            "grasp and move",
        ])

    return list(dict.fromkeys(variants))  # deduplicate


# ================================================================
# Factory
# ================================================================

SUPPORTED_DATASETS = {
    "robomimic": RobomimicAdapter,
    "droid": DROIDAdapter,
    "bridge": BridgeDataAdapter,
}


def create_adapter(
    dataset_name: str,
    data_dir: str,
    **kwargs,
) -> OpenDatasetAdapter:
    """Factory for creating dataset adapters.

    Args:
        dataset_name: One of 'robomimic', 'droid', 'bridge'.
        data_dir: Path to dataset directory.
        **kwargs: Passed to adapter constructor.

    Returns:
        OpenDatasetAdapter instance.
    """
    cls = SUPPORTED_DATASETS.get(dataset_name.lower())
    if cls is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(SUPPORTED_DATASETS.keys())}")
    return cls(data_dir=data_dir, **kwargs)


def create_align_dataset(
    dataset_names: List[str],
    data_dirs: List[str],
    output_path: str,
    max_episodes_per_dataset: Optional[int] = None,
    **kwargs,
) -> str:
    """Convert one or more datasets to a single ALIGN-compatible HDF5 file.

    Args:
        dataset_names: List of dataset names ('robomimic', 'droid', 'bridge').
        data_dirs: List of data directories (same length as dataset_names).
        output_path: Output HDF5 file path.
        max_episodes_per_dataset: Max episodes to take from each dataset.
        **kwargs: Extra arguments for adapters.

    Returns:
        Path to the created HDF5 file.
    """
    if len(dataset_names) != len(data_dirs):
        raise ValueError("dataset_names and data_dirs must have the same length")

    if not HAS_H5PY:
        raise ImportError("h5py required: pip install h5py")

    # Collect all episodes in memory, then write once
    all_eps = []
    ep_offset = 0

    for ds_name, ds_dir in zip(dataset_names, data_dirs):
        adapter = create_adapter(ds_name, ds_dir, **kwargs)
        n_eps = adapter.get_episode_count()
        if max_episodes_per_dataset:
            n_eps = min(n_eps, max_episodes_per_dataset)

        print(f"[{ds_name}] Converting {n_eps} episodes from {ds_dir}")

        for i in range(n_eps):
            ep_data = adapter.get_episode(i)
            frames = ep_data["frames"]
            poses = ep_data["poses"]
            gripper = ep_data.get("gripper", np.zeros(len(frames)))
            text = ep_data.get("text", "pick and place")
            text_variants = _generate_text_variants(text)

            all_eps.append({
                "frames": frames,
                "poses": poses,
                "gripper": gripper,
                "text": text,
                "text_variants": text_variants,
                "source": ds_name,
            })

        # Clean up adapter (close HDF5 handles)
        if hasattr(adapter, "close"):
            adapter.close()

    print(f"\n  Total: {len(all_eps)} episodes from {len(dataset_names)} datasets")

    # Write all episodes to a single HDF5 file
    print(f"  Writing to {output_path}...")
    with h5py.File(output_path, "w") as h5:
        for i, ep in enumerate(all_eps):
            ep_name = f"ep_{i:05d}"
            group = h5.create_group(ep_name)
            group.create_dataset(f"frames/{kwargs.get('camera', DEFAULT_CAMERA)}", data=ep["frames"].astype(np.uint8))
            group.create_dataset("noisy_poses", data=ep["poses"].astype(np.float32))
            group.create_dataset("gripper", data=ep["gripper"].astype(np.float32))
            group.create_dataset("texts", data=json.dumps(ep["text_variants"]))
            group.create_dataset("meta", data=json.dumps({
                "task_description": ep["text"],
                "source": ep["source"],
                "num_frames": len(ep["frames"]),
            }))

        h5["meta/total_episodes"] = len(all_eps)
        h5["meta/sources"] = json.dumps(dataset_names)

    output_abs = str(Path(output_path).absolute())
    print(f"  Done: {output_abs}")
    return output_abs


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert open datasets to ALIGN HDF5 format")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS.keys(),
                        help="Dataset name")
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--task", help="Robomimic task name (required for robomimic)")
    parser.add_argument("--output", default="align_pretrain.h5", help="Output HDF5 file")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to convert")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Camera view to extract")

    args = parser.parse_args()

    kwargs = {"camera": args.camera}
    if args.dataset == "robomimic":
        if not args.task:
            parser.error("--task is required for robomimic")
        kwargs["task"] = args.task

    adapter = create_adapter(args.dataset, args.data_dir, **kwargs)
    adapter.to_align_format(args.output, max_frames_per_ep=None)

    # Multi-dataset merge example
    print("\n[INFO] To merge multiple datasets:")
    print(f"  python -c \"from data.open_dataset import create_align_dataset;")
    print(f"  create_align_dataset(")
    print(f"      dataset_names=['robomimic', 'droid'],")
    print(f"      data_dirs=['./robomimic_data', './droid_data'],")
    print(f"      output_path='align_pretrain.h5',")
    print(f"      max_episodes_per_dataset=5000)\"")


if __name__ == "__main__":
    main()
