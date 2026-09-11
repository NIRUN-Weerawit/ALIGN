"""Pure Quest-3 anchored-pose mapping reused by the physical Piper bridge."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class QuestPose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self):
        if np.asarray(self.position).shape != (3,) or np.asarray(self.quaternion_xyzw).shape != (4,):
            raise ValueError("QuestPose requires position (3,) and quaternion_xyzw (4,)")


def anchored_target_pose(controller_start: QuestPose, controller_now: QuestPose, ee_start_pos, ee_start_quat):
    """Map Quest controller displacement/rotation onto an absolute Piper EE target."""
    start_q = np.asarray(controller_start.quaternion_xyzw, dtype=float)
    now_q = np.asarray(controller_now.quaternion_xyzw, dtype=float)
    ee_q = np.asarray(ee_start_quat, dtype=float)
    if min(np.linalg.norm(start_q), np.linalg.norm(now_q), np.linalg.norm(ee_q)) < 1e-8:
        raise ValueError("quaternions must be non-zero")
    pos = np.asarray(ee_start_pos, dtype=float) + (np.asarray(controller_now.position, dtype=float) - np.asarray(controller_start.position, dtype=float))
    delta = R.from_quat(now_q / np.linalg.norm(now_q)) * R.from_quat(start_q / np.linalg.norm(start_q)).inv()
    quat = (delta * R.from_quat(ee_q / np.linalg.norm(ee_q))).as_quat()
    return pos.astype(np.float32), quat.astype(np.float32)
