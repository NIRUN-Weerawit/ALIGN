from __future__ import annotations

import numpy as np
import pytest

from piper_xvla.endpose_control import (
    EndPoseTarget,
    build_endpose_payload,
    prepare_can_cartesian_control,
    send_endpose,
)


def test_build_endpose_payload_converts_metres_and_degrees_to_piper_integer_units():
    target = EndPoseTarget(
        xyz_m=np.array([0.150, -0.050, 0.150]),
        euler_xyz_deg=np.array([-179.9, 0.0, 45.678]),
    )

    assert build_endpose_payload(target) == (150000, -50000, 150000, -179900, 0, 45678)


def test_build_endpose_payload_rejects_nonfinite_or_wrong_shape_values():
    with pytest.raises(ValueError, match="xyz_m"):
        build_endpose_payload(EndPoseTarget(np.array([0.1, 0.2]), np.zeros(3)))
    with pytest.raises(ValueError, match="finite"):
        build_endpose_payload(EndPoseTarget(np.array([0.1, np.nan, 0.2]), np.zeros(3)))


def test_send_endpose_calls_only_endpose_control_with_precomputed_payload():
    calls = []

    class FakePiper:
        def EndPoseCtrl(self, *payload):
            calls.append(payload)

    target = EndPoseTarget(np.array([0.150, -0.050, 0.150]), np.array([-179.9, 0.0, 0.0]))

    send_endpose(FakePiper(), target)

    assert calls == [(150000, -50000, 150000, -179900, 0, 0)]


def test_prepare_can_cartesian_control_enables_then_selects_slow_move_p_mode():
    calls = []

    class FakePiper:
        def EnableArm(self, motor_num):
            calls.append(("enable", motor_num))

        def GripperCtrl(self, *args):
            calls.append(("gripper", args))

        def MotionCtrl_2(self, **kwargs):
            calls.append(("motion2", kwargs))

    prepare_can_cartesian_control(FakePiper(), speed_percent=10)

    assert calls == [
        ("enable", 7),
        ("gripper", (0, 1000, 0x01, 0)),
        ("motion2", {
            "ctrl_mode": 0x01,
            "move_mode": 0x00,
            "move_spd_rate_ctrl": 10,
            "is_mit_mode": 0x00,
            "residence_time": 0,
            "installation_pos": 0x00,
        }),
    ]


def test_parse_args_uses_feedback_warmup_before_reading_sdk_rates():
    from piper_xvla.endpose_control import parse_args

    args = parse_args([
        "--x-m", "0.1", "--y-m", "0", "--z-m", "0.2",
        "--rx-deg", "0", "--ry-deg", "0", "--rz-deg", "0",
    ])

    assert args.feedback_warmup_s > 0
