"""Canonical one-arm Piper samples for LeRobot-backed X-VLA fine-tuning.

Piper is always the active/left X-VLA arm. The inactive arm is exactly zero.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


ACTIVE_DIM = 10
XVLA_DIM = 20


def piper_state_to_xvla20(position, quaternion_xyzw, gripper) -> np.ndarray:
    """Encode one Piper EE state as X-VLA's active-arm 10D + zero pad."""
    position = np.asarray(position, dtype=np.float32)
    quat = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}")
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("quaternion_xyzw must be a finite non-zero (4,) vector")
    if not np.isfinite(position).all() or not np.isfinite(gripper):
        raise ValueError("state values must be finite")
    rot6d = R.from_quat(quat / np.linalg.norm(quat)).as_matrix()[:, :2].reshape(-1).astype(np.float32)
    active = np.concatenate([position, rot6d, np.array([gripper], dtype=np.float32)])
    return np.concatenate([active, np.zeros(ACTIVE_DIM, dtype=np.float32)])


@dataclass(frozen=True)
class PiperXVLAFrame:
    """One validated collection timestep before it is written to LeRobot."""

    global_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state20: np.ndarray
    action20: np.ndarray
    task: str

    def __post_init__(self):
        for name, image in (("global_rgb", self.global_rgb), ("wrist_rgb", self.wrist_rgb)):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 HxWx3")
        for name, value in (("state20", self.state20), ("action20", self.action20)):
            value = np.asarray(value)
            if value.shape != (XVLA_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape (20,)")
            if not np.array_equal(value[ACTIVE_DIM:], np.zeros(ACTIVE_DIM, dtype=value.dtype)):
                raise ValueError(f"{name} inactive-arm slots must be zero")
        if not self.task.strip():
            raise ValueError("task must be non-empty")
