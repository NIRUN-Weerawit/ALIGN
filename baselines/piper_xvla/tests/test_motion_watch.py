import numpy as np

from piper_xvla.motion_watch import joint_delta_deg, joint_positions_deg


class JointMsg:
    def __init__(self, values):
        self.joint_state = type("State", (), {
            f"joint_{i + 1}": value for i, value in enumerate(values)
        })()


def test_converts_sdk_millidegree_joint_feedback_to_degrees():
    joints = joint_positions_deg(JointMsg([1000, -2500, 0, 4000, 500, -10]))
    np.testing.assert_allclose(joints, [1.0, -2.5, 0.0, 4.0, 0.5, -0.01])


def test_joint_delta_reports_motion_between_two_feedback_samples():
    before = JointMsg([0, 0, 0, 0, 0, 0])
    after = JointMsg([100, 0, -300, 0, 0, 0])
    assert joint_delta_deg(before, after) == 0.3
