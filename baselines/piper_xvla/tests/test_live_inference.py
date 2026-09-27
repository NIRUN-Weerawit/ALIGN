import numpy as np

from piper_xvla.live_inference import parse_args, raw_state20_to_live_inputs


def test_live_inference_accepts_zero_frames_for_operator_stoppable_continuous_mode():
    args = parse_args(["--guard-config", "guard.json", "--frames", "0"])

    assert args.frames == 0


def test_raw_state20_to_live_inputs_normalizes_gripper_and_builds_proprio():
    raw_state20 = np.zeros(20, dtype=np.float32)
    raw_state20[:3] = [0.1, -0.1, 0.2]
    raw_state20[3:9] = [1, 0, 0, 1, 0, 0]
    raw_state20[9] = 0.00275

    proprio8, current_active10 = raw_state20_to_live_inputs(
        raw_state20, gripper_min_m=-0.0424, gripper_max_m=0.0479,
    )

    assert proprio8.shape == (8,)
    assert current_active10.shape == (10,)
    assert np.allclose(proprio8[:3], [0.1, -0.1, 0.2])
    assert np.allclose(proprio8[3:7], [0.0, 0.0, 0.0, 1.0])
    assert np.isclose(proprio8[7], 0.5)
    assert np.isclose(current_active10[9], 0.5)


def test_raw_state20_to_live_inputs_rejects_invalid_raw_state():
    raw_state20 = np.zeros(20, dtype=np.float32)
    raw_state20[3:9] = [0, 0, 0, 0, 0, 0]

    try:
        raw_state20_to_live_inputs(raw_state20, gripper_min_m=-0.0424, gripper_max_m=0.0479)
    except ValueError as exc:
        assert "rotation" in str(exc)
    else:
        raise AssertionError("degenerate rotation must be rejected")
