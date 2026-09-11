import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from piper_xvla.schema import PiperXVLAFrame, piper_state_to_xvla20
from piper_xvla.teleop import QuestPose, anchored_target_pose


def test_single_arm_state_is_padded_to_xvla_20d():
    state = piper_state_to_xvla20(
        position=np.array([0.2, -0.1, 0.3]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        gripper=1.0,
    )

    assert state.shape == (20,)
    np.testing.assert_allclose(state[:3], [0.2, -0.1, 0.3])
    np.testing.assert_allclose(state[3:9], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert state[9] == 1.0
    np.testing.assert_array_equal(state[10:], np.zeros(10, dtype=np.float32))


def test_frame_rejects_nonzero_inactive_arm_slots():
    with pytest.raises(ValueError, match="inactive-arm"):
        PiperXVLAFrame(
            global_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            state20=np.ones(20, dtype=np.float32),
            action20=np.zeros(20, dtype=np.float32),
            task="pick up the cube",
        )


def test_quest_delta_is_anchored_to_current_piper_ee_pose():
    controller_start = QuestPose(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    controller_now = QuestPose(np.array([1.1, 1.8, 3.3]), np.array([0.0, 0.0, 0.0, 1.0]))
    ee_start_pos = np.array([0.4, 0.0, 0.2])
    ee_start_quat = R.from_euler("z", 90, degrees=True).as_quat()

    pos, quat = anchored_target_pose(
        controller_start, controller_now, ee_start_pos, ee_start_quat
    )

    np.testing.assert_allclose(pos, [0.5, -0.2, 0.5])
    np.testing.assert_allclose(quat, ee_start_quat)
