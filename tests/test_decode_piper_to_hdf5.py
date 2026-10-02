"""Piper absolute EE targets must survive conversion to ALIGN action labels."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_script = Path(__file__).resolve().parents[1] / "scripts/decode_piper_to_hdf5.py"
_spec = importlib.util.spec_from_file_location("decode_piper_to_hdf5", _script)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
convert_vectors = _module.convert_vectors


def _raw(position, rotation, gripper):
    return np.concatenate((position, rotation.as_matrix()[:, :2].reshape(-1), [gripper]))


def test_absolute_target_becomes_relative_pose_action():
    current = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
    relative = Rotation.from_euler("xyz", [-0.1, 0.15, 0.25])
    target = relative * current
    states = np.stack([_raw([0.1, 0.2, 0.3], current, -0.03)]).astype(np.float32)
    targets = np.stack([_raw([0.12, 0.18, 0.34], target, 0.05)]).astype(np.float32)

    poses, actions, grippers = convert_vectors(states, targets, -0.03, 0.05)

    np.testing.assert_allclose(poses[0, :3], states[0, :3])
    np.testing.assert_allclose(Rotation.from_euler("xyz", poses[0, 3:]).as_matrix(), current.as_matrix(), atol=1e-6)
    np.testing.assert_allclose(actions[0, :3], [0.02, -0.02, 0.04], atol=1e-6)
    np.testing.assert_allclose(Rotation.from_euler("xyz", actions[0, 3:6]).as_matrix(), relative.as_matrix(), atol=1e-6)
    np.testing.assert_allclose(grippers, [0.0])
    np.testing.assert_allclose(actions[:, 6], [1.0])
