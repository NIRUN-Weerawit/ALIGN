#!/usr/bin/env python3
"""ALIGN Data Recorder — standalone module for recording teleop episodes.

Designed to be imported into an Isaac Sim VR teleoperation script without
modifying the reference code. Records camera frames, noisy poses, gripper
state, and task metadata per episode.

Usage:
    from align_data_recorder import DataRecorder

    recorder = DataRecorder(output_dir="./data", episode_name="episode_001")
    recorder.set_task_description("pick up the red mug")
    recorder.set_object_poses({"mug": [0.5, 0.1, 0.0], "bowl": [0.2, -0.15, 0.0]})

    # In the sim loop:
    recorder.step(
        frame=wrist_rgb,         # (H, W, 3) uint8 RGB
        noisy_pose=vr_target,    # (6,) [x, y, z, rx, ry, rz] or [x, y, z, qx, qy, qz, qw]
        gripper_state=0.0,       # float, 0=open 1=closed
    )

    # End episode:
    recorder.finalize()
    recorder.save()
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


class DataRecorder:
    """Records frame-by-frame teleoperation data for one episode.

    Directory structure:
        output_dir/
            episode_name/
                frames/
                    00000.jpg
                    00001.jpg
                    ...
                data.npz          — noisy_pose, gripper_state, timestamp, smooth_pose
                meta.json         — task description, objects, config
    """

    def __init__(
        self,
        output_dir: str = "./align_data",
        episode_name: str | None = None,
        max_frames: int = 10000,
        jpeg_quality: int = 85,
        camera_label: str = "wrist",
    ):
        self.output_dir = Path(output_dir)
        if episode_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"episode_{timestamp}"
        self.episode_name = episode_name
        self.max_frames = max_frames
        self.jpeg_quality = jpeg_quality
        self.camera_label = camera_label

        # Data buffers
        self.frames: list[np.ndarray] = []       # stored as compressed JPEG
        self.noisy_poses: list[np.ndarray] = []  # (6,) each
        self.gripper_states: list[float] = []
        self.timestamps: list[float] = []
        self.smooth_poses: list[np.ndarray] = []  # filled later by GT generation
        self.absolute_poses: list[np.ndarray] = []  # world-frame EEF pose for reference

        # Metadata
        self.task_description: str = ""
        self.objects_on_table: dict[str, list[float]] = {}  # name → [x, y, z]
        self.target_object: str = ""
        self.operator_id: str = ""
        self.notes: str = ""
        self._start_time: float | None = None
        self._finalized: bool = False

        # Create directories
        self.frames_dir = self.output_dir / self.episode_name / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Metadata setters ──────────────────────────────────────────────

    def set_task_description(self, text: str):
        self.task_description = text

    def set_object_poses(self, objects: dict[str, list[float]]):
        """Set object poses in world frame. {name: [x, y, z]}"""
        self.objects_on_table = objects

    def set_target_object(self, name: str):
        self.target_object = name

    def set_operator(self, operator_id: str):
        self.operator_id = operator_id

    def set_notes(self, notes: str):
        self.notes = notes

    # ── Recording ─────────────────────────────────────────────────────

    def step(
        self,
        frame: np.ndarray,
        noisy_pose: np.ndarray,
        gripper_state: float = 0.0,
        absolute_pose: np.ndarray | None = None,
    ):
        """Record one timestep.

        Args:
            frame: (H, W, 3) uint8 RGB camera image.
            noisy_pose: (6,) or (7,) EEF pose. 6D = [x, y, z, rx, ry, rz]
                        (axis-angle) or 7D = [x, y, z, qx, qy, qz, qw].
            gripper_state: 0.0 = open, 1.0 = closed.
            absolute_pose: (6,) or (7,) world-frame EEF pose (optional).
        """
        if self._finalized:
            raise RuntimeError("Episode is finalized. Create a new DataRecorder.")

        if self._start_time is None:
            self._start_time = time.time()

        if len(self.frames) >= self.max_frames:
            print(f"[Recorder] Max frames ({self.max_frames}) reached. Call finalize().")
            return

        # Store raw frame (will compress on save)
        self.frames.append(frame)

        # Normalize pose to 6D if it's 7D quaternion
        pose = np.asarray(noisy_pose, dtype=np.float64).flatten()
        if len(pose) == 7:
            # Quaternion [x, y, z, w] — keep as-is for now, let GT handle conversion
            pass
        elif len(pose) != 6:
            raise ValueError(f"noisy_pose must be 6 or 7 elements, got {len(pose)}")

        self.noisy_poses.append(pose)
        self.gripper_states.append(float(gripper_state))
        self.timestamps.append(time.time() - self._start_time)

        if absolute_pose is not None:
            self.absolute_poses.append(np.asarray(absolute_pose, dtype=np.float64).flatten())

    def set_smooth_poses(self, smooth_poses: list[np.ndarray] | np.ndarray):
        """Set ground-truth smooth poses (computed offline by GT pipeline)."""
        self.smooth_poses = [np.asarray(p, dtype=np.float64).flatten() for p in smooth_poses]
        assert len(self.smooth_poses) == len(self.noisy_poses), (
            f"smooth_poses ({len(self.smooth_poses)}) must match noisy_poses ({len(self.noisy_poses)})"
        )

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    # ── Finalize & Save ───────────────────────────────────────────────

    def finalize(self):
        """Mark episode as complete. No more steps allowed."""
        self._finalized = True
        elapsed = self.timestamps[-1] if self.timestamps else 0.0
        print(f"[Recorder] Episode '{self.episode_name}' finalized: "
              f"{len(self.frames)} frames, {elapsed:.1f}s")

    def save(self):
        """Save all data to disk. Call after finalize()."""
        if not self._finalized:
            print("[Recorder] WARNING: Saving without finalize(). Consider calling finalize() first.")

        n = len(self.frames)
        if n == 0:
            print("[Recorder] WARNING: No frames recorded. Nothing saved.")
            return

        episode_dir = self.output_dir / self.episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Save frames as JPEG ──
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(self.frames):
            img = Image.fromarray(frame.astype(np.uint8))
            img.save(frames_dir / f"{i:05d}.jpg", quality=self.jpeg_quality)

        # ── 2. Save numpy arrays ──
        npz_dict = {
            "noisy_poses": np.array(self.noisy_poses, dtype=np.float64),     # (N, 6) or (N, 7)
            "gripper_states": np.array(self.gripper_states, dtype=np.float64),  # (N,)
            "timestamps": np.array(self.timestamps, dtype=np.float64),        # (N,)
        }
        if self.smooth_poses:
            npz_dict["smooth_poses"] = np.array(self.smooth_poses, dtype=np.float64)
        if self.absolute_poses:
            npz_dict["absolute_poses"] = np.array(self.absolute_poses, dtype=np.float64)

        npz_path = episode_dir / "data.npz"
        np.savez_compressed(npz_path, **npz_dict)

        # ── 3. Save metadata as JSON ──
        meta = {
            "episode_name": self.episode_name,
            "date": datetime.now().isoformat(),
            "num_frames": n,
            "duration_s": round(self.timestamps[-1], 3) if self.timestamps else 0.0,
            "camera_label": self.camera_label,
            "task_description": self.task_description,
            "target_object": self.target_object,
            "objects_on_table": self.objects_on_table,
            "operator_id": self.operator_id,
            "pose_format": "xyz_quat" if len(self.noisy_poses[0]) == 7 else "xyz_rpy",
            "notes": self.notes,
        }
        meta_path = episode_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Size estimate
        total_bytes = sum(
            os.path.getsize(frames_dir / f"{i:05d}.jpg") for i in range(min(n, 10))
        )
        avg_bytes = total_bytes / min(n, 10)
        estimated_mb = (avg_bytes * n) / (1024 * 1024)

        print(f"[Recorder] Episode saved to: {episode_dir}")
        print(f"[Recorder]   Frames:  {n} JPEG images ({estimated_mb:.1f} MB est.)")
        print(f"[Recorder]   Arrays:  data.npz ({npz_path})")
        print(f"[Recorder]   Metadata: meta.json")


# ── Loader Functions ──────────────────────────────────────────────────


def load_episode(episode_dir: str) -> tuple[np.ndarray, dict, dict]:
    """Load a recorded episode from disk.

    Args:
        episode_dir: Path to episode directory (containing frames/, data.npz, meta.json).

    Returns:
        (frames, data, meta)
            frames: (N, H, W, 3) uint8 RGB array
            data: dict with keys from data.npz (noisy_poses, gripper_states, etc.)
            meta: dict from meta.json
    """
    episode_path = Path(episode_dir)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    # Load frames
    frames_dir = episode_path / "frames"
    frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = []
    for f in frame_files:
        img = Image.open(f)
        frames.append(np.array(img))
    frames_arr = np.stack(frames, axis=0) if frames else np.array([])

    # Load numpy arrays
    data = {}
    npz_path = episode_path / "data.npz"
    if npz_path.exists():
        with np.load(npz_path) as npz:
            for key in npz:
                data[key] = npz[key]

    # Load metadata
    meta = {}
    meta_path = episode_path / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return frames_arr, data, meta


def list_episodes(data_dir: str) -> list[str]:
    """List all recorded episodes in a data directory."""
    data_path = Path(data_dir)
    episodes = []
    for p in sorted(data_path.iterdir()):
        if p.is_dir() and (p / "meta.json").exists():
            episodes.append(p.name)
    return episodes


# ── Quick Test ────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Generate synthetic test data
    print("=== ALIGN Data Recorder: Quick Test ===")

    recorder = DataRecorder(
        output_dir="/tmp/align_test_data",
        episode_name="test_episode_001",
    )
    recorder.set_task_description("pick up the red mug")
    recorder.set_target_object("mug_red")
    recorder.set_object_poses({
        "mug_red": [0.55, 0.10, 0.0],
        "bowl_blue": [0.20, -0.15, 0.0],
    })
    recorder.set_operator("test_operator")

    # Simulate 50 frames of teleop
    for t in range(50):
        # Synthetic frame (64×64 gradient)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        frame[:, :, 1] = np.linspace(0, 255, 64, dtype=np.uint8)[:, None]

        # Synthetic noisy pose (oscillating toward target)
        progress = t / 50
        noisy = np.array([
            0.3 + 0.2 * progress + 0.02 * np.sin(t * 0.5),  # x
            0.1 + 0.02 * np.sin(t * 0.7),                     # y
            0.25 + 0.2 * progress + 0.01 * np.sin(t),         # z
            0.0, 0.0, 0.0,                                    # orientation
        ])
        grip = 0.8 if t > 40 else 0.0

        recorder.step(frame=frame, noisy_pose=noisy, gripper_state=grip)

    recorder.finalize()
    recorder.save()

    # Verify: load it back
    print("\n=== Verifying Load ===")
    frames, data, meta = load_episode("/tmp/align_test_data/test_episode_001")
    print(f"Frames:  {frames.shape}")
    print(f"Poses:   {data['noisy_poses'].shape}")
    print(f"Grip:    {data['gripper_states'].shape}")
    print(f"Time:    {data['timestamps'].shape}")
    print(f"Meta:    {json.dumps(meta, indent=2)[:200]}...")

    # List episodes
    print(f"\nEpisodes: {list_episodes('/tmp/align_test_data')}")
    print("\n✅ Data Recorder: Quick Test Passed")