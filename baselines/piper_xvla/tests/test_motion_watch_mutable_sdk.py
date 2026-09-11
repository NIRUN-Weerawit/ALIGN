import numpy as np

from piper_xvla.motion_watch import joint_positions_deg, max_joint_delta_deg


class MutableJointMsg:
    def __init__(self):
        self.joint_state = type("State", (), {f"joint_{i}": 0 for i in range(1, 7)})()


def test_snapshot_arrays_detect_motion_when_sdk_reuses_one_mutable_message_object():
    msg = MutableJointMsg()
    before = joint_positions_deg(msg)
    msg.joint_state.joint_3 = -250  # Piper feedback unit: 0.001 degree
    after = joint_positions_deg(msg)

    assert max_joint_delta_deg(before, after) == 0.25
    assert before[2] == 0.0  # snapshot did not mutate with the SDK object
