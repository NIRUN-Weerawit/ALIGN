import numpy as np
import pytest
import torch
from contextlib import nullcontext

from piper_xvla.live_inference import (
    _binary_gripper_action, _predict_action_with_cudnn_fallback, _target_from_active,
    parse_args, raw_state20_to_live_inputs,
)
from piper_xvla.gripper_binary import BinaryGripperConfig


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


def test_prediction_retries_cudnn_initialization_once(monkeypatch):
    import piper_xvla.live_inference as inference

    calls = []
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(inference, "configure_cuda_attention", lambda device: calls.append(("attention", device)))

    class Policy:
        def predict_action_chunk(self, batch):
            calls.append(("predict", torch.backends.cudnn.enabled))
            if len([call for call in calls if call[0] == "predict"]) == 1:
                raise RuntimeError("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED")
            return torch.ones((1, 1, 20))

    result = _predict_action_with_cudnn_fallback(Policy(), {}, "cuda")
    assert result.shape == (20,)
    assert calls == [("predict", True), ("attention", "cuda"), ("predict", False)]


def test_prediction_does_not_retry_unrelated_cuda_errors(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    calls = []

    class Policy:
        def predict_action_chunk(self, batch):
            calls.append(1)
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        _predict_action_with_cudnn_fallback(Policy(), {}, "cuda")
    assert calls == [1]


def test_gripper_closes_below_threshold_and_passes_raw_value_at_or_above_it():
    gripper_min_m, gripper_max_m = -0.0424, 0.0479
    span = gripper_max_m - gripper_min_m
    threshold = np.float32((0.020 - gripper_min_m) / span)
    prediction = np.zeros(20, dtype=np.float32)
    prediction[3:9] = [1, 0, 0, 1, 0, 0]

    prediction[9] = np.nextafter(threshold, np.float32(0))
    closed, raw_closed_m, close_target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    _, closed_command_m = _target_from_active(closed[:10], gripper_min_m, gripper_max_m)
    assert raw_closed_m < 0.020
    assert close_target_m == 0.0
    assert round(closed_command_m * 1_000_000) == 0

    prediction[9] = threshold
    passed, raw_m, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    assert raw_m == pytest.approx(0.020, abs=1e-6)
    assert target_m == raw_m
    assert np.array_equal(prediction, passed)
    assert round(_target_from_active(passed[:10], gripper_min_m, gripper_max_m)[1] * 1_000_000) == 20_000

    prediction[9] = (0.035 - gripper_min_m) / span
    passed, raw_m, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    assert target_m == raw_m == pytest.approx(0.035, abs=1e-6)
    assert round(_target_from_active(passed[:10], gripper_min_m, gripper_max_m)[1] * 1_000_000) == 35_000


def test_gripper_uses_configured_threshold_with_zero_close_and_raw_open():
    gripper_min_m, gripper_max_m = -0.0424, 0.0479
    config = BinaryGripperConfig(threshold_mm=24)
    prediction = np.zeros(20, dtype=np.float32)
    prediction[3:9] = [1, 0, 0, 1, 0, 0]
    threshold = np.float32((0.024 - gripper_min_m) / (gripper_max_m - gripper_min_m))

    prediction[9] = np.nextafter(threshold, np.float32(0))
    closed, _, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m, config)
    assert target_m == 0.0
    assert round(_target_from_active(closed[:10], gripper_min_m, gripper_max_m)[1] * 1_000_000) == 0

    prediction[9] = threshold
    opened, _, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m, config)
    assert target_m == pytest.approx(0.024, abs=1e-6)
    assert round(_target_from_active(opened[:10], gripper_min_m, gripper_max_m)[1] * 1_000_000) == 24000


def test_gripper_loads_old_binary_config_using_only_saved_threshold():
    config = BinaryGripperConfig.from_dict({"close_mm": -10, "threshold_mm": 3, "open_mm": 45})
    assert config.as_dict() == {"threshold_mm": 3.0}


@pytest.mark.parametrize("payload", [
    {"threshold_mm": 71},
    {"threshold_mm": float("nan")},
    {"threshold_mm": 20, "extra": 1},
    {},
])
def test_gripper_rejects_invalid_threshold_config(payload):
    with pytest.raises(ValueError):
        BinaryGripperConfig.from_dict(payload)


def test_gripper_rejects_threshold_outside_model_calibration():
    with pytest.raises(ValueError, match="model calibration"):
        BinaryGripperConfig(threshold_mm=50).validate_calibration(-0.0424, 0.0479)


def test_gripper_close_target_and_raw_out_of_range_prediction_are_guarded():
    from piper_xvla.action_guard import ActionGuard, ActionGuardLimits

    gripper_min_m, gripper_max_m = -0.0424, 0.0479
    current = np.array([0.1, 0, 0.2, 1, 0, 0, 1, 0, 0, (0.045 - gripper_min_m) / (gripper_max_m - gripper_min_m)], dtype=np.float32)
    prediction = np.concatenate([current.copy(), np.zeros(10, dtype=np.float32)])
    limits = ActionGuardLimits(
        workspace_min_m=np.array([-1, -1, -1]), workspace_max_m=np.array([1, 1, 1]),
        max_position_step_m=1, max_rotation_step_deg=180, max_gripper_step_normalized=0.2,
        max_inactive_arm_l2=1, max_camera_age_s=1, max_feedback_age_s=1,
    )
    guard = ActionGuard(limits)

    prediction[9] = (0.019 - gripper_min_m) / (gripper_max_m - gripper_min_m)
    closed, _, _ = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    decision = guard.check(predicted_action20=closed, current_active10=current, camera_age_s=0, feedback_age_s=0)
    assert not decision.allowed
    assert "GRIPPER_STEP_LIMIT" in {alert.code for alert in decision.alerts}

    prediction[9] = 1.2
    passed, raw_m, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    assert raw_m > gripper_max_m and target_m == raw_m and passed[9] == prediction[9]
    decision = guard.check(predicted_action20=passed, current_active10=current, camera_age_s=0, feedback_age_s=0)
    assert "GRIPPER_OUT_OF_RANGE" in {alert.code for alert in decision.alerts}

    prediction[9] = -0.2
    closed, raw_m, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    assert raw_m < gripper_min_m and target_m == 0.0
    assert round(_target_from_active(closed[:10], gripper_min_m, gripper_max_m)[1] * 1_000_000) == 0

    prediction[9] = np.nan
    invalid, raw_m, target_m = _binary_gripper_action(prediction, gripper_min_m, gripper_max_m)
    assert raw_m is None and target_m is None and np.isnan(invalid[9])
    decision = guard.check(predicted_action20=invalid, current_active10=current, camera_age_s=0, feedback_age_s=0)
    assert "NONFINITE_ACTION" in {alert.code for alert in decision.alerts}
