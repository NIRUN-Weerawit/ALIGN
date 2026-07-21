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
POSITIVE_WINDOW = 5  # W_pos -- frames within this window are positive pairs


# ================================================================
# Dataset
# ================================================================

class ALIGNDataset(Dataset):
    """HDF5 dataset for ALIGN training.

    Stores episodes as HDF5 groups:
        /meta/              -- json strings with episode metadata
        /ep_XXX/frames/     -- (N, H, W, 3) uint8 frames
        /ep_XXX/noisy_poses -- (N, 6) or (N, 7) float32
        /ep_XXX/actions     -- (N, 6) float32 (optional)
        /ep_XXX/gripper     -- (N,) float32
        /ep_XXX/texts       -- list of text variant strings

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
        cameras: Optional[List[str]] = None,
        image_size: Tuple[int, int] = DEFAULT_SIZE,
        frames_per_ep: int = DEFAULT_FRAMES_PER_EP,
        traj_window: int = TRAJ_WINDOW,
        episodes_per_batch: int = 8,
    ):
        self.h5_path = Path(h5_path)
        self.mode = mode
        # Multi-camera support: prefer `cameras` (list) over `camera` (string).
        # If `cameras` is given, use it; otherwise fall back to single `camera` (legacy).
        if cameras is not None:
            self.cameras: List[str] = list(cameras)
        else:
            self.cameras = [camera]
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
            # Frames is a single (N, H, W, 3) array -- no camera subgroups
            self.cameras = []  # sentinel: no camera sub-dataset
        else:
            # Frames is a group with camera sub-datasets (e.g., frames/wrist_image)
            available_cameras = list(frames_obj.keys())
            if self.cameras and all(c in available_cameras for c in self.cameras):
                # User-specified multi-camera list -- all present, keep as is.
                pass
            elif len(self.cameras) == 1 and self.cameras[0] in ("wrist", ""):
                # Legacy "wrist" alias
                self.cameras = ["wrist_image"] if "wrist_image" in available_cameras else [available_cameras[0]]
            elif len(available_cameras) == 1:
                # Only one camera available; use it
                self.cameras = [available_cameras[0]]
            else:
                # Multi-camera dataset, but user didn't specify -- try common defaults
                if "wrist_image" in available_cameras:
                    self.cameras = ["wrist_image"]
                elif len(self.cameras) > 0 and self.cameras[0] in available_cameras:
                    pass  # already correct
                else:
                    raise ValueError(
                        f"Multiple cameras available ({available_cameras}); please specify --cameras"
                    )

        # Detect if noisy_poses are cumulative across episodes (LIBERO v3 quirk)
        # and pre-compute per-episode frame lengths + pose offsets
        self._ep_frame_lengths: List[int] = []
        self._ep_pose_offsets: List[int] = []

        # Detect optional actions dataset
        try:
            first_ep = self._episode_keys[0]
            if self._single_episode:
                self._has_actions = "actions" in self._h5
            else:
                self._has_actions = "actions" in self._h5[first_ep]
        except Exception:
            self._has_actions = False

        # Detect optional gripper dataset per episode.  Gripper may be:
        #   - /ep_XXX/gripper  -- (N,) float32   (preferred, dedicated field)
        #   - last column of /ep_XXX/actions    (fallback: 7D OSC_POSE)
        #   - missing                            (use 0.0 default)
        self._has_gripper = False
        self._gripper_keys: List[str] = []
        if not self._single_episode:
            for key in self._episode_keys:
                ep_group = self._h5[key]
                if "gripper" in ep_group:
                    self._gripper_keys.append("gripper")
                else:
                    # no dedicated gripper field -- fall back to actions[:, -1]
                    self._gripper_keys.append("__actions_last__")
            # Mark as available if ANY episode has a dedicated field, OR if
            # the actions array is wide enough to contain a gripper column.
            self._has_gripper = (
                any(k == "gripper" for k in self._gripper_keys)
                or self._has_actions
            )
        else:
            if "gripper" in self._h5:
                self._gripper_keys.append("gripper")
                self._has_gripper = True
            elif "actions" in self._h5:
                self._gripper_keys.append("__actions_last__")
                self._has_gripper = True
            else:
                self._gripper_keys.append("__none__")
                self._has_gripper = False

        # Determine the pose field name per episode. Older HDF5 files use
        # "noisy_poses" (misnomer -- actually contains clean poses).
        # Newer files use "poses". Store the resolved name per episode.
        self._pose_keys: List[str] = []
        for ep_idx in range(len(self._episode_keys)):
            key = self._episode_keys[ep_idx]
            ep_group = self._h5[key]
            if "poses" in ep_group:
                self._pose_keys.append("poses")
            elif "noisy_poses" in ep_group:
                self._pose_keys.append("noisy_poses")
            else:
                raise KeyError(
                    f"Episode {key} has neither 'poses' nor 'noisy_poses' field"
                )

        for ep_idx in range(len(self._episode_keys)):
            key = self._episode_keys[ep_idx]
            pose_key = self._pose_keys[ep_idx]

            # Frame length (handle both framedataset structures)
            if not self.cameras:
                # frames = Dataset -- single array (legacy single-frame dataset)
                n_frames = len(self._h5[f"{key}/frames"])
            else:
                # Multi-camera: verify all cameras have the same length
                n_frames = None
                for cam in self.cameras:
                    try:
                        cam_len = len(self._h5[f"{key}/frames/{cam}"])
                    except KeyError:
                        cam_len = len(self._h5[f"{key}/{pose_key}"])
                    if n_frames is None:
                        n_frames = cam_len
                    elif cam_len != n_frames:
                        raise ValueError(
                            f"Camera length mismatch in {key}: "
                            f"{self.cameras[0]}={n_frames}, {cam}={cam_len}"
                        )
            self._ep_frame_lengths.append(n_frames)

            # Pose offset: in cumulative HDF5, ep_N starts AFTER all previous episodes
            n_poses = len(self._h5[f"{key}/{pose_key}"])
            if n_poses == n_frames:
                # Non-cumulative -- pose aligns with frames directly
                self._ep_pose_offsets.append(0)
            elif ep_idx > 0:
                # Cumulative -- offset = sum of all previous frame lengths
                self._ep_pose_offsets.append(sum(self._ep_frame_lengths[:-1]))
            else:
                self._ep_pose_offsets.append(0)

        # Warn if cumulative detected
        if any(o > 0 for o in self._ep_pose_offsets):
            print("  NOTE: Detected cumulative noisy_poses -- using per-episode offsets")

        if self._has_actions:
            print("  NOTE: Detected 'actions' dataset -- will load actions when available")
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
        """Read frames for a given episode.

        Returns:
            - If multi-camera: (count, V, H, W, 3) where V = len(self.cameras)
            - If single camera:  (count, H, W, 3)  (legacy format)
        """
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            # Legacy single-frame dataset
            return self._h5["frames"][start:start + count]
        if not self.cameras:
            # frames = Dataset -- single array (legacy)
            return self._h5[f"{key}/frames"][start:start + count]
        if len(self.cameras) == 1:
            # Single camera -- keep the legacy 4D shape for back-compat
            return self._h5[f"{key}/frames/{self.cameras[0]}"][start:start + count]
        # Multi-camera: stack along a new axis (axis=1) → (count, V, H, W, 3)
        per_cam = [
            self._h5[f"{key}/frames/{cam}"][start:start + count]
            for cam in self.cameras
        ]
        return np.stack(per_cam, axis=1)

    def _read_poses(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            pose_key = getattr(self, "_pose_keys", ["noisy_poses"])[0]
            return self._h5[pose_key][start:start + count]
        # Handle cumulative poses (LIBERO v3 quirk)
        offset = self._ep_pose_offsets[ep_idx]
        abs_start = offset + start
        pose_key = self._pose_keys[ep_idx]
        raw = self._h5[f"{key}/{pose_key}"][abs_start:abs_start + count]
        # Pad if fewer items than requested (episode boundary)
        if len(raw) < count:
            pad = np.zeros((count - len(raw), raw.shape[1]), dtype=raw.dtype)
            raw = np.concatenate([raw, pad], axis=0)
        return raw

    def _read_actions(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        """Read actions for an episode window. Returns array shaped (count, 6).

        If actions are missing, returns zeros. Handles cumulative actions similarly
        to `noisy_poses` using the precomputed offsets.
        """
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            try:
                raw = self._h5["actions"][start:start + count]
            except Exception:
                return np.zeros((count, 7), dtype=np.float32)
            if len(raw) < count:
                pad = np.zeros((count - len(raw), raw.shape[1]), dtype=raw.dtype)
                raw = np.concatenate([raw, pad], axis=0)
        else:
            if not self._has_actions:
                return np.zeros((count, 7), dtype=np.float32)
            offset = self._ep_pose_offsets[ep_idx]
            abs_start = offset + start
            try:
                raw = self._h5[f"{key}/actions"][abs_start:abs_start + count]
            except Exception:
                return np.zeros((count, 6), dtype=np.float32)
            if len(raw) < count:
                pad = np.zeros((count - len(raw), raw.shape[1]), dtype=raw.dtype)
                raw = np.concatenate([raw, pad], axis=0)

        # Ensure 7 dims: trim or pad as needed
        if raw.shape[1] > 7:
            raw = raw[:, :7]
        elif raw.shape[1] < 7:
            pad = np.zeros((raw.shape[0], 7 - raw.shape[1]), dtype=raw.dtype)
            raw = np.concatenate([raw, pad], axis=1)
        return raw.astype(np.float32)

    def _read_gripper(self, ep_idx: int, t: int = -1) -> float:
        """Read the gripper value for a given episode and (absolute) timestep.

        Looks up `ep_XXX/gripper` first; falls back to the last column of
        `ep_XXX/actions` if the dedicated field is absent.  Returns 0.0
        when no gripper source is available.
        """
        if not getattr(self, "_has_gripper", False):
            return 0.0
        key = self._episode_keys[ep_idx]
        gk = self._gripper_keys[ep_idx]
        try:
            if gk == "gripper":
                if self._single_episode:
                    arr = self._h5["gripper"]
                else:
                    arr = self._h5[f"{key}/gripper"]
                idx = t if t >= 0 else (len(arr) - 1)
                idx = max(0, min(idx, len(arr) - 1))
                return float(arr[idx])
            elif gk == "__actions_last__":
                if self._single_episode:
                    arr = self._h5["actions"]
                else:
                    arr = self._h5[f"{key}/actions"]
                    # For the actions-array fallback the caller is expected
                    # to pass a LOCAL timestep (t = len(poses) - 1, no offset).
                idx = t if t >= 0 else (len(arr) - 1)
                idx = max(0, min(idx, len(arr) - 1))
                return float(arr[idx, -1])
        except Exception:
            return 0.0
        return 0.0

    def _read_robot_state(self, ep_idx: int, t: int) -> np.ndarray:
        """Read the one-step robot state at absolute timestep `t` within an
        episode. Returns a (7,) float32 vector:

            [pos_x, pos_y, pos_z, roll, pitch, yaw, gripper]
        """
        # Read just the single pose we need.
        key = self._episode_keys[ep_idx]
        if self._single_episode:
            pose_key = getattr(self, "_pose_keys", ["noisy_poses"])[0]
            pose = self._h5[pose_key][t, :6]
        else:
            offset = self._ep_pose_offsets[ep_idx]
            pose_key = self._pose_keys[ep_idx]
            pose = self._h5[f"{key}/{pose_key}"][offset + t, :6]
        pose = np.asarray(pose, dtype=np.float32).reshape(6)
        gripper = np.float32(self._read_gripper(ep_idx, t))
        return np.concatenate([pose, [gripper]], axis=0).astype(np.float32)

    def _read_poses_gripper(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        """Read a (count,) gripper array from a dedicated /ep_XXX/gripper field.

        Handles the same cumulative-poses offset as `_read_poses` and pads
        with zeros at episode boundaries.
        """
        key = self._episode_keys[ep_idx]
        try:
            if self._single_episode:
                abs_start = start
                arr = self._h5["gripper"]
            else:
                offset = self._ep_pose_offsets[ep_idx]
                abs_start = offset + start
                arr = self._h5[f"{key}/gripper"]
        except Exception:
            return np.zeros(count, dtype=np.float32)
        chunk = np.asarray(arr[abs_start:abs_start + count], dtype=np.float32).reshape(-1)
        if len(chunk) < count:
            chunk = np.concatenate(
                [chunk, np.zeros(count - len(chunk), dtype=np.float32)]
            )
        return chunk

    def _read_actions_gripper_col(self, ep_idx: int, start: int, count: int) -> np.ndarray:
        """Read a (count,) gripper array from the last column of /ep_XXX/actions.

        Returns zeros if the actions array has fewer than 7 columns.
        """
        key = self._episode_keys[ep_idx]
        try:
            if self._single_episode:
                arr = self._h5["actions"]
                abs_start = start
            else:
                offset = self._ep_pose_offsets[ep_idx]
                abs_start = offset + start
                arr = self._h5[f"{key}/actions"]
        except Exception:
            return np.zeros(count, dtype=np.float32)
        if arr.ndim < 2 or arr.shape[1] < 7:
            return np.zeros(count, dtype=np.float32)
        chunk = np.asarray(arr[abs_start:abs_start + count, -1], dtype=np.float32).reshape(-1)
        if len(chunk) < count:
            chunk = np.concatenate(
                [chunk, np.zeros(count - len(chunk), dtype=np.float32)]
            )
        return chunk

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
        actions = self._read_actions(ep_idx, start, count + self.traj_window) if getattr(self, '_has_actions', False) else np.zeros((count + self.traj_window, 6), dtype=np.float32)
        text = self._read_text(ep_idx)
        # Per-step gripper array (N,) sourced from the dedicated gripper
        # field when available, or the last column of the 7D actions array.
        n = count + self.traj_window
        if getattr(self, "_has_gripper", False):
            gk = self._gripper_keys[ep_idx]
            if gk == "gripper":
                grippers = self._read_poses_gripper(ep_idx, start, n)
            elif gk == "__actions_last__":
                grippers = self._read_actions_gripper_col(ep_idx, start, n)
            else:
                grippers = np.zeros(n, dtype=np.float32)
        else:
            grippers = np.zeros(n, dtype=np.float32)
        # New v2 one-step robot state: last pose in the window + gripper (7,).
        # Anchored at the END of the read window so it's a "current" state.
        last_t_local = n - 1
        robot_state = self._read_robot_state(ep_idx, start + last_t_local)
        # v2 one-step state at t+1 (next state). Only meaningful if the
        # episode has at least one more frame.  If we're at the end, the
        # state is identical to `robot_state` (no transition possible).
        if self._get_episode_length(ep_idx) > (start + last_t_local + 1):
            robot_state_next = self._read_robot_state(ep_idx, start + last_t_local + 1)
        else:
            robot_state_next = robot_state.copy()

        return {
            "frames": frames,
            "poses": poses,
            "actions": actions,
            "text": text,
            "ep_idx": ep_idx,
            "grippers": grippers,           # (N,) float32 per-step gripper
            "robot_state": robot_state,  # (7,) — [pos(3), euler(3), gripper(1)]
            "robot_state_next": robot_state_next,  # (7,) — state at t+1
        }


# ================================================================
# Head Training Utilities
# ================================================================

NOISE_STD = 0.020 # 2cm positional noise for synthetic human deviation
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
            "frames": (B, H, W, 3) or (B, V, H, W, 3) uint8,
            "trajectories": (B, K, 6) float32,
            "texts": list of strings,
            "ep_ids": (B,) int -- for verifying positives/negatives,
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
        # same time t -- vision and trajectory are temporally aligned for
        # the positive pair. (Previously t1 and t2 were independent
        # randoms -- that breaks the contrastive signal.)
        max_offset = N - traj_window
        if max_offset > 0:
            t1 = np.random.randint(0, max_offset)  # vision anchor
            t2 = t1                                # trajectory anchor -- SAME time
        else:
            t1 = 0
            t2 = 0

        # frames shape: (H, W, 3) for single camera, (V, H, W, 3) for multi
        frame_sample = frames[t1]
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


class MultiALIGNDataset(Dataset):
    """Concatenates multiple ALIGNDataset instances for multi-dataset training.

    Each underlying dataset can have a different file/suite. Samples are
    dispatched to the right dataset based on a flat global index. The
    ordering is: [dataset_0][dataset_1]...[dataset_{N-1}], so a single
    random index selects a sample from the right source.

    All datasets must be in the same mode ('pretrain', 'head', or 'world_model')
    and share the same trajectory window, frames_per_ep, image_size, and camera --
    these are passed to the constructor and applied to every underlying dataset.

    Note: 'mode' is just a label on the dataset; the collate function
    (pretrain_collate, head_collate, world_model_collate) is what shapes
    the data for a specific training stage.
    """

    def __init__(
        self,
        h5_paths: List[str],
        mode: str = "pretrain",
        camera: str = DEFAULT_CAMERA,
        cameras: Optional[List[str]] = None,
        image_size: Tuple[int, int] = DEFAULT_SIZE,
        frames_per_ep: int = DEFAULT_FRAMES_PER_EP,
        traj_window: int = TRAJ_WINDOW,
        episodes_per_batch: int = 8,
    ):
        if not h5_paths:
            raise ValueError("h5_paths must be a non-empty list")
        self.datasets = [
            ALIGNDataset(
                p, mode=mode, camera=camera, cameras=cameras,
                image_size=image_size,
                frames_per_ep=frames_per_ep, traj_window=traj_window,
                episodes_per_batch=episodes_per_batch,
            )
            for p in h5_paths
        ]
        self._cumulative = [0]
        for ds in self.datasets:
            self._cumulative.append(self._cumulative[-1] + len(ds))
        self.mode = mode
        self.camera = camera
        self.image_size = image_size
        self.frames_per_ep = frames_per_ep
        self.traj_window = traj_window
        self.episodes_per_batch = episodes_per_batch
        self.h5_paths = [str(p) for p in h5_paths]

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        # Find which sub-dataset this index belongs to
        import bisect
        ds_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[ds_idx]
        return self.datasets[ds_idx][local_idx]

    def get_episode_count_per_source(self) -> List[int]:
        """Return the number of episodes in each underlying dataset."""
        return [len(ds._episode_keys) for ds in self.datasets]

    def close(self):
        if hasattr(self, "datasets"):
            for ds in self.datasets:
                ds.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def world_model_collate(batch: list, traj_window: int = 5) -> dict:
    """Collate batch for world model training.

    For each item in the batch (which contains a full episode's frames,
    poses, actions), sample a random transition (s_t, a_t, s_{t+1}):

        s_t      = (frame_t, traj_window ending at t, text)
        a_t      = actions[t]                       (6D OSC_POSE)
        s_{t+1}  = (frame_{t+1}, traj_window ending at t+1, text)

    The transitions are batched. The trajectory window has the same
    size as the world model's input (traj_window=k). The next-state
    trajectory window is offset by 1.

    Returns:
        {
            "frame_t":      (B, H, W, 3) uint8 -- current frame
            "traj_t":       (B, traj_window, 6) float32 - current traj window
            "action":       (B, 6) float32 - the action at timestep t
            "frame_next":   (B, H, W, 3) uint8 -- next frame
            "traj_next":    (B, traj_window, 6) float32 - next traj window
            "text":         list of strings (B,) -- task description
            "ep_idx":       (B,) int64 -- episode index (for diagnostics)
        }

    The encoder (frozen) is applied at training time to produce the
    actual (z_v, z_s, z_sext) embeddings. The collate function only
    provides the raw inputs.
    """
    all_frame_t = []
    all_traj_t = []
    all_action = []
    all_frame_next = []
    all_traj_next = []
    all_state = []  # v2 one-step state (B, 7) at t
    all_state_next = []  # v2 one-step state (B, 7) at t+1
    all_text = []
    all_ep_idx = []

    for item in batch:
        frames = item["frames"]     # (N, H, W, 3)
        poses = item["poses"]       # (N, 6) or (N, 7)
        actions = item.get("actions")  # (N, 6) or (N, 7)
        text = item["text"]
        ep_idx = item["ep_idx"]
        # New v2 one-step state.  Prefer the dataset-provided `robot_state`
        # when present; fall back to the last pose + zero gripper otherwise.
        if "robot_state" in item and item["robot_state"] is not None:
            state = np.asarray(item["robot_state"], dtype=np.float32).reshape(7)
        else:
            state = np.concatenate(
                [poses[-1, :6], [0.0]], axis=0
            ).astype(np.float32)
        # New v2 one-step state at t+1. Prefer the dataset-provided
        # `robot_state_next`; fall back to the same state (no transition
        # is possible at the episode boundary).
        if "robot_state_next" in item and item["robot_state_next"] is not None:
            state_next = np.asarray(item["robot_state_next"], dtype=np.float32).reshape(7)
        else:
            state_next = state.copy()

        N = len(frames)
        # Need at least traj_window past + 1 transition
        if N < traj_window + 1:
            # Episode too short; skip by padding with copies
            t = 0
        else:
            # Sample t such that t+1 is valid AND t >= traj_window - 1
            # so we have traj_window past poses
            t_min = traj_window - 1
            t_max = N - 2  # need t+1
            if t_min > t_max:
                t = t_max
            else:
                t = int(np.random.randint(t_min, t_max + 1))

        # Current state: frame window [t-traj_window+1 .. t] + traj window
        frame_window = frames[t - traj_window + 1 : t + 1]
        if frame_window.shape[0] < traj_window:
            pad = np.zeros((traj_window - frame_window.shape[0], *frames.shape[1:]), dtype=frames.dtype)
            frame_window = np.concatenate([pad, frame_window], axis=0)
        # Trajectory window: (traj_window, 6) -- last `traj_window` poses
        traj_t = poses[t - traj_window + 1 : t + 1, :6]
        if traj_t.shape[0] < traj_window:
            # Pad with the first pose
            pad = np.zeros((traj_window - traj_t.shape[0], 6), dtype=poses.dtype)
            traj_t = np.concatenate([pad, traj_t], axis=0)

        # Action at t
        if actions is not None and t < len(actions):
            action_t = actions[t, :6].astype(np.float32)
        elif t > 0:
            # Fallback: derive from pose difference
            action_t = (poses[t, :6] - poses[t - 1, :6]).astype(np.float32)
        else:
            action_t = np.zeros(6, dtype=np.float32)

        # Next state: frame t+1 + traj window [t-traj_window+2 .. t+1]
        frame_next = frames[t + 1]
        traj_next = poses[t - traj_window + 2 : t + 2, :6]
        if traj_next.shape[0] < traj_window:
            pad = np.zeros((traj_window - traj_next.shape[0], 6), dtype=poses.dtype)
            traj_next = np.concatenate([pad, traj_next], axis=0)

        # Text variant: pick a random variant from the list
        if isinstance(text, list):
            text_pick = text[np.random.randint(0, len(text))]
        else:
            text_pick = text

        all_frame_t.append(frame_window)
        all_traj_t.append(traj_t.astype(np.float32))
        all_action.append(action_t)
        all_frame_next.append(frame_next)
        all_traj_next.append(traj_next.astype(np.float32))
        all_state.append(state)
        all_state_next.append(state_next)
        all_text.append(text_pick)
        all_ep_idx.append(ep_idx)

    return {
        "frame_t": np.stack(all_frame_t, axis=0),       # (B, K, H, W, 3)
        "traj_t": np.stack(all_traj_t, axis=0),
        "action": np.stack(all_action, axis=0),
        "frame_next": np.stack(all_frame_next, axis=0),
        "traj_next": np.stack(all_traj_next, axis=0),
        "state": np.stack(all_state, axis=0).astype(np.float32),  # (B, 7) v2
        "state_next": np.stack(all_state_next, axis=0).astype(np.float32),  # (B, 7) v2
        "text": all_text,
        "ep_idx": np.array(all_ep_idx, dtype=np.int64),
    }


def head_collate(batch: list, chunk_size: int = 5,
                 vision_window_size: int = 5) -> dict:
    """Collate batch for head training (Assistant).

    Samples one timestep per item, builds past/future trajectory windows,
    and returns both:
      - "frames":           (B, H, W, 3) -- single current frame (legacy)
      - "frames_window":    (B, K, H, W, 3) -- K past frames for the transformer
                            assistant head, where K = vision_window_size.
                            Anchored at the current timestep t; padded at the
                            episode boundary by replicating the earliest frame.

    Args:
        batch: list of items from ALIGNDataset (mode="head").
        chunk_size: K for the assistant head's K-step chunk.
        vision_window_size: number of past frames in the vision window.
            Defaults to chunk_size so the transformer sees K past + predicts
            K future. Set to 0 to disable the window (saves memory).
    Collate batch for head training with on-the-fly noise injection.

    Returns:
        {
            "frames": (B, H, W, 3) uint8,
            "noisy_pose": (B, 6) float32 - corrupted pose (for alpha target),
            "clean_pose": (B, 6) float32 - ground truth pose (for alpha target),
            "current_action": (B, 7) float32 - the human's delta-pose command at
                              this timestep (input to Assistant head). Sourced
                              from the dataset's `actions` field.
            "trajectory": (B, K, 7) float32 - past window of clean poses,
            "trajectory_future": (B, K, 7) float32 - NEXT K poses after the
                              current timestep (targets for future prediction).
            "alpha_need": (B,) float32 - kinematic error part of alpha_target,
            "delta_target":(B, chunk_size, 7) float32,
            "texts": list of strings,
        }
    """
    all_frames = []
    all_noisy = []
    all_clean = []
    all_actions = []
    all_trajs = []
    all_trajs_future = []
    all_needs = []
    all_deltas = []
    all_robot_state = []  # v2 one-step state (B, 7)
    all_robot_state_window = []  # v3: K past states (B, K, 7) for intention head
    all_actions_window = []  # v3: K past actions (B, K, 6) as head targets
    all_texts = []
    all_frames_window = []  # K past frames for the transformer assistant head

    # Use a fixed seed for noise injection in the batch so it's reproducible
    rng = np.random.default_rng()

    for item in batch:
        frames = item["frames"]
        poses_clean = item["poses"][..., :6]  # Ensure 6D (drop quaternion if 7D)
        # Stored actions (7D OSC_POSE: 6D delta + gripper). Use the first 6
        # dims for the Assistant head's explicit action input.
        item_actions = item.get("actions")
        texts_raw = item["text"] if isinstance(item["text"], list) else item["text"]

        N = len(poses_clean)
        max_t = N - chunk_size
        t = rng.integers(0, min(max(max_t + 1, 1), N)) if max_t >= 0 else min(rng.integers(0, 2), N - 1)

        # --- v2 one-step robot state at the head's anchor t ---
        # [pos(3), euler(3), gripper(1)].  Gripper sourced from the
        # dataset-provided `grippers` array (per-step, length N) when
        # available, else 0.0.  Aligned with the current_clean_pose /
        # current_action used by the rest of the head.
        item_grippers = item.get("grippers")
        if item_grippers is not None and t < len(item_grippers):
            gripper_t = float(item_grippers[t])
        elif item_actions is not None and t < len(item_actions) and item_actions.shape[1] >= 7:
            # Backward-compat fallback (older datasets without grippers field)
            gripper_t = float(item_actions[t, 6])
        else:
            gripper_t = 0.0
        robot_state_t = np.concatenate(
            [poses_clean[t, :6].astype(np.float32), [gripper_t]], axis=0
        ).astype(np.float32)

        # --- Past Trajectory Window (Clean for encoding) -- fixed to always be chunk_size ---
        start = max(0, t - chunk_size + 1)
        traj_window = poses_clean[start:t + 1]
        if len(traj_window) < chunk_size:
            pad = np.zeros((chunk_size - len(traj_window), 6), dtype=np.float32)
            traj_window = np.concatenate([pad, traj_window], axis=0)

        # --- Inject Noise for Current Pose ---
        current_clean_pose = poses_clean[t]
        noisy_pose = inject_kinematic_noise(current_clean_pose, rng)

        # --- Current action (the human's command at this timestep) ---
        if item_actions is not None and t < len(item_actions):
            current_action = item_actions[t, :7].astype(np.float32)
        elif t > 0:
            # Fallback: derive action from pose difference if dataset
            # has no actions field
            current_action = (poses_clean[t, :7] - poses_clean[t - 1, :7]).astype(np.float32)
        else:
            current_action = np.zeros(7, dtype=np.float32)

        # --- Compute "Need" (Kinematic Error / D_MAX) ---
        pos_error = np.linalg.norm(current_clean_pose[:3] - noisy_pose[:3])
        need = min(pos_error / D_MAX, 1.0)

        # --- Option B: Pose-Relative Goal Targets ---
        # The Assistant head now outputs K=chunk_size GOALS (relative to
        # current noisy pose) rather than recovery corrections.
        #
        # delta[k] = poses_clean[t + k + 1] - noisy_pose   (relative goal at step k+1)
        #
        # This is "where should I be at t+k+1?" (relative to current).
        # It exists even at zero error -- the target is the goal itself.
        # Combined with alpha via: action = (1-α)·a_human + α·goal[0]
        # (α is the trust weight between human's and model's actions.)
        delta = np.zeros((chunk_size, 6), dtype=np.float32)

        for k in range(chunk_size):
            if t + k + 1 < N:
                # Relative goal: clean_pose[t+k+1] relative to current noisy pose
                delta[k] = poses_clean[t + k + 1, :6] - poses_clean[t,:6]

        # --- Future trajectory window (K poses after current timestep) ---
        # Used as targets for the future-prediction (Decision) head.
        # Window is anchored at t+1: [t+1, t+2, ..., t+K]
        future_start = t + 1
        future_end = t + 1 + chunk_size
        traj_future = poses_clean[future_start:future_end, :6]
        if len(traj_future) < chunk_size:
            pad = np.zeros((chunk_size - len(traj_future), 6), dtype=np.float32)
            traj_future = np.concatenate([traj_future, pad], axis=0)

        # Text variant
        if isinstance(texts_raw, list):
            text = texts_raw[rng.integers(0, len(texts_raw))]
        else:
            text = texts_raw

        # Vision window: K past frames for the transformer assistant head.
        # Anchored at t; the K-1 frames before t, then frame t.
        # Padded at the episode boundary by replicating the earliest frame.
        if vision_window_size > 0:
            # Note: frames may be (N, H, W, 3) or (N, V, H, W, 3) for multi-cam
            window_start = max(0, t - vision_window_size + 1)
            window_end = t + 1  # inclusive
            window_frames = frames[window_start:window_end]  # (K_actual, ...) or (K_actual, V, H, W, 3)
            if len(window_frames) < vision_window_size:
                # Pad at the start (replicate earliest)
                if frames.ndim == 4:
                    pad = np.tile(
                        frames[0:1], (vision_window_size - len(window_frames), 1, 1, 1)
                    )
                else:  # 5D: (N, V, H, W, 3)
                    pad = np.tile(
                        frames[0:1], (vision_window_size - len(window_frames), 1, 1, 1, 1)
                    )
                window_frames = np.concatenate([pad, window_frames], axis=0)
            all_frames_window.append(window_frames)

        # --- v3: state_window (K past robot states) and actions_window (K past actions) ---
        # Both anchored at t: [t-K+1, ..., t-1, t]
        # Padded at the episode boundary by replicating the earliest state.
        item_grippers_all = item.get("grippers")
        window_size = chunk_size
        win_start = max(0, t - window_size + 1)
        win_end = t + 1  # inclusive
        # State window: collect past states
        state_window = []
        for k_t in range(win_start, win_end):
            if item_grippers_all is not None and k_t < len(item_grippers_all):
                gk_t = float(item_grippers_all[k_t])
            elif item_actions is not None and k_t < len(item_actions) and item_actions.shape[1] >= 7:
                gk_t = float(item_actions[k_t, 6])
            else:
                gk_t = 0.0
            state_t_k = np.concatenate(
                [poses_clean[k_t, :6].astype(np.float32), [gk_t]], axis=0
            ).astype(np.float32)
            state_window.append(state_t_k)
        # Pad if needed
        if len(state_window) < window_size:
            pad_count = window_size - len(state_window)
            state_window = [state_window[0]] * pad_count + state_window
        all_robot_state_window.append(np.stack(state_window, axis=0).astype(np.float32))

        # Actions window: K past actions [t-K+1, ..., t-1, t]
        actions_window = []
        for k_t in range(win_start, win_end):
            if item_actions is not None and k_t < len(item_actions):
                actions_window.append(item_actions[k_t, :7].astype(np.float32))
            elif k_t > 0:
                actions_window.append(
                    (poses_clean[k_t, :7] - poses_clean[k_t - 1, :7]).astype(np.float32)
                )
            else:
                actions_window.append(np.zeros(7, dtype=np.float32))
        if len(actions_window) < window_size:
            pad_count = window_size - len(actions_window)
            actions_window = [actions_window[0]] * pad_count + actions_window
        all_actions_window.append(np.stack(actions_window, axis=0).astype(np.float32))

        all_frames.append(frames[t])
        all_noisy.append(noisy_pose[:6])
        all_clean.append(current_clean_pose[:6])
        all_actions.append(current_action)
        all_trajs.append(traj_window[:, :6])
        all_trajs_future.append(traj_future)
        all_needs.append(need)
        all_deltas.append(delta)
        all_robot_state.append(robot_state_t)
        all_texts.append(text)

    return_dict = {
        "frames": np.stack(all_frames, axis=0),
        "noisy_pose": np.stack(all_noisy, axis=0).astype(np.float32),
        "clean_pose": np.stack(all_clean, axis=0).astype(np.float32),
        "current_action": np.stack(all_actions, axis=0).astype(np.float32),
        "trajectory": np.stack(all_trajs, axis=0).astype(np.float32),
        "trajectory_future": np.stack(all_trajs_future, axis=0).astype(np.float32),
        "alpha_need": np.array(all_needs, dtype=np.float32),
        "delta_target": np.stack(all_deltas, axis=0).astype(np.float32),
        "robot_state": np.stack(all_robot_state, axis=0).astype(np.float32),  # v2 (B, 7)
        "robot_state_window": np.stack(all_robot_state_window, axis=0).astype(np.float32),  # v3 (B, K, 7)
        "actions_window": np.stack(all_actions_window, axis=0).astype(np.float32),  # v3 (B, K, 6) head targets
        "texts": all_texts,
    }

    # Add the vision window for the transformer assistant head.
    # Only included if vision_window_size > 0. Shape:
    #   (B, K, H, W, 3) for single-camera
    #   (B, K, V, H, W, 3) for multi-camera
    if vision_window_size > 0 and all_frames_window:
        return_dict["frames_window"] = np.stack(all_frames_window, axis=0).astype(np.uint8)
    return return_dict


# ================================================================
# V4 Segment Collate — Variable-length segments with persistent bank
# ================================================================

def v4_segment_collate(batch: list, history_size: int = 20,
                       chunk_size: int = 10,
                       segment_min_mult: int = 2,
                       segment_max_mult: int = 5) -> dict:
    """Collate batch for V4 training with variable-length segments.

    Samples a contiguous segment of length segment_len from each episode,
    where segment_len is randomly chosen per sample in [2*H, min(5*H, ep_len)].
    The segment is processed step by step during training with a persistent
    memory bank.

    Args:
        batch: list of items from ALIGNDataset (mode="head").
        history_size: H — Mamba window size (past frames).
        chunk_size: C — future action prediction length.
        segment_min_mult: min segment length = H * this value.
        segment_max_mult: max segment length = H * this value.

    Returns:
        dict with keys:
            frames_segment: (B, S, V, H, W, 3) uint8 — full segment frames
            states_segment: (B, S, 7) float32 — full segment states
            actions_segment: (B, S, 7) float32 — full segment actions
            texts: list of strings (B,)
            segment_len: (B,) int — actual length of each segment
            history_size: int — H
            chunk_size: int — C
    """
    rng = np.random.default_rng()
    all_frames = []
    all_states = []
    all_actions = []
    all_texts = []
    all_lens = []

    for item in batch:
        frames = item["frames"]       # (N, V, H, W, 3) or (N, H, W, 3)
        poses = item["poses"]         # (N, 6)
        actions = item.get("actions", None)  # (N, 7) or None
        text = item["text"]
        item_grippers = item.get("grippers", None)

        N = len(frames)
        H = history_size
        C = chunk_size

        # Need at least H + C frames for one valid training step
        min_len = H + C
        if N < min_len:
            continue

        # Random segment length: [min_mult*H, max_mult*H], capped by episode
        # seg_min = min(segment_min_mult * H, N)
        seg_min = max(H + C, segment_min_mult * H)
        # seg_max = min(segment_max_mult * H, N)
        seg_max = min(max(segment_max_mult * H, H + C), N)
        if seg_min >= seg_max:
            seg_len = seg_max
        else:
            seg_len = int(rng.integers(seg_min, seg_max + 1))

        # Random start position
        max_start = N - seg_len
        seg_start = rng.integers(0, max_start + 1) if max_start > 0 else 0

        # Extract segment
        seg_frames = frames[seg_start:seg_start + seg_len]
        seg_poses = poses[seg_start:seg_start + seg_len]

        # Build states: [pos(3), euler(3), gripper(1)] for each timestep
        seg_states = []
        for k_t in range(seg_len):
            abs_t = seg_start + k_t
            if item_grippers is not None and abs_t < len(item_grippers):
                gk_t = float(item_grippers[abs_t])
            elif actions is not None and abs_t < len(actions) and actions.shape[1] >= 7:
                gk_t = float(actions[abs_t, 6])
            else:
                gk_t = 0.0
            state_t = np.concatenate([
                seg_poses[k_t, :6].astype(np.float32), [gk_t],
            ], axis=0).astype(np.float32)
            seg_states.append(state_t)
        seg_states = np.stack(seg_states, axis=0)  # (S, 7)

        # Actions segment
        if actions is not None:
            seg_actions = actions[seg_start:seg_start + seg_len].astype(np.float32)
        else:
            seg_actions = np.zeros((seg_len, 7), dtype=np.float32)

        # Text
        if isinstance(text, list):
            text_pick = text[rng.integers(0, len(text))]
        else:
            text_pick = text

        all_frames.append(seg_frames)
        all_states.append(seg_states)
        all_actions.append(seg_actions)
        all_texts.append(text_pick)
        all_lens.append(seg_len)

    if not all_frames:
        raise ValueError("No valid segments found in batch")

    # Stack to (B, S, ...) — segments may have different lengths
    # We pad to max_len in the batch
    max_len = max(all_lens)
    B = len(all_frames)

    padded_frames = []
    padded_states = []
    padded_actions = []
    for b in range(B):
        S = all_lens[b]
        # Pad frames
        f = all_frames[b]
        if f.ndim == 4:
            # (S, H, W, 3) — single camera
            pad_f = np.zeros((max_len, *f.shape[1:]), dtype=f.dtype)
            pad_f[:S] = f
            pad_f[S:] = f[-1:]  # replicate last frame
        else:
            # (S, V, H, W, 3) — multi camera
            pad_f = np.zeros((max_len, *f.shape[1:]), dtype=f.dtype)
            pad_f[:S] = f
            pad_f[S:] = f[-1:]
        padded_frames.append(pad_f)

        # Pad states
        s = all_states[b]
        pad_s = np.zeros((max_len, 7), dtype=np.float32)
        pad_s[:S] = s
        pad_s[S:] = s[-1:]  # replicate last state
        padded_states.append(pad_s)

        # Pad actions
        a = all_actions[b]
        # print(f"actions shape: {a.shape}, max_len: {max_len}, S: {S}")
        pad_a = np.zeros((max_len, 7), dtype=np.float32)
        pad_a[:S] = a
        pad_a[S:] = a[-1:]
        # print(f"padded_actions shape: {pad_a.shape}, max_len: {max_len}, S: {S}")
        padded_actions.append(pad_a)

    return {
        "frames_segment": np.stack(padded_frames, axis=0).astype(np.uint8),
        "states_segment": np.stack(padded_states, axis=0).astype(np.float32),
        "actions_segment": np.stack(padded_actions, axis=0).astype(np.float32),
        "texts": all_texts,
        "segment_len": np.array(all_lens, dtype=np.int32),
        "history_size": history_size,
        "chunk_size": chunk_size,
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
