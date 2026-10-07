import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piper_xvla.action_guard import ActionGuard, guard_limits_from_config
from piper_xvla.align_control import ALIGNInferenceSettings, ActionHorizonCursor, align_action_to_piper_target, align_gripper_mapping, align_state7, camera_stale_pause_allowed
from piper_xvla.endpose_control import build_endpose_payload
from piper_xvla.schema import piper_state_to_xvla20


GMIN = -0.032600000500679016
GMAX = 0.04740000143647194


def feedback(gripper_m=0.025):
    quaternion = Rotation.from_euler("xyz", [15, -10, 20], degrees=True).as_quat()
    return piper_state_to_xvla20([0.18, 0.12, 0.23], quaternion, gripper_m)


def test_align_state_and_delta_action_match_training_conventions():
    measured = feedback()
    state = align_state7(measured, GMIN, GMAX)
    assert np.allclose(state[:3], [0.18, 0.12, 0.23])
    assert np.allclose(state[3:6], np.deg2rad([15, -10, 20]), atol=1e-6)
    assert state[6] == pytest.approx((0.025 - GMIN) / (GMAX - GMIN))

    action = np.array([0.01, -0.005, 0.003, 0.02, -0.01, 0.03, 0.8])
    target, gripper_m, current, target20 = align_action_to_piper_target(
        action, measured, GMIN, GMAX, ALIGNInferenceSettings(gripper_close_threshold_mm=0)
    )
    assert np.allclose(target.xyz_m, [0.19, 0.115, 0.233])
    expected = Rotation.from_euler("xyz", action[3:6]) * Rotation.from_euler("xyz", [15, -10, 20], degrees=True)
    assert np.allclose(Rotation.from_euler("xyz", target.euler_xyz_deg, degrees=True).as_matrix(), expected.as_matrix())
    assert gripper_m == pytest.approx(GMIN + 0.8 * (GMAX - GMIN))
    assert target20.shape == (20,) and np.all(target20[10:] == 0)
    assert current[9] == pytest.approx(state[6])
    payload = build_endpose_payload(target)
    assert payload[:3] == (190000, 115000, 233000)


def test_scales_and_gripper_close_threshold():
    measured = feedback(gripper_m=0.030)
    action = np.array([0.02, 0, 0, 0, 0, 0.2, (0.002 - GMIN) / (GMAX - GMIN)])
    settings = ALIGNInferenceSettings(position_scale=0.5, rotation_scale=0, gripper_scale=1,
                                       gripper_close_threshold_mm=3)
    target, gripper_m, _, _ = align_action_to_piper_target(action, measured, GMIN, GMAX, settings)
    assert target.xyz_m[0] == pytest.approx(0.19)
    assert np.allclose(target.euler_xyz_deg, [15, -10, 20])
    assert gripper_m == 0
    settings = ALIGNInferenceSettings(gripper_scale=0, gripper_close_threshold_mm=3)
    _, gripper_m, _, _ = align_action_to_piper_target(action, measured, GMIN, GMAX, settings)
    assert gripper_m == pytest.approx(0.030)


def test_unconstrained_flow_gripper_is_capped_to_calibrated_range():
    measured = feedback()
    settings = ALIGNInferenceSettings(gripper_close_threshold_mm=3)
    high = align_gripper_mapping(1.1476, float(measured[9]), GMIN, GMAX, settings)
    assert high.clamped
    assert high.raw_mm > GMAX * 1000
    assert high.bounded_normalized == 1.0
    assert high.target_mm == pytest.approx(GMAX * 1000)
    action = np.array([0, 0, 0, 0, 0, 0, 1.1476])
    _, gripper_m, current, target20 = align_action_to_piper_target(action, measured, GMIN, GMAX, settings)
    assert gripper_m == pytest.approx(GMAX)
    assert target20[9] == pytest.approx(1.0)
    assert current[9] < 1.0

    low = align_gripper_mapping(-0.2, float(measured[9]), GMIN, GMAX, settings)
    assert low.clamped and low.bounded_normalized == 0.0
    assert low.target_mm == 0.0  # close threshold is applied after bounding


def test_guard_rejects_excessive_align_target_and_cursor_consumes_once():
    measured = feedback()
    target, _, current, predicted = align_action_to_piper_target(
        np.array([0.2, 0, 0, 0, 0, 0, 0.5]), measured, GMIN, GMAX, ALIGNInferenceSettings()
    )
    guard = ActionGuard(guard_limits_from_config({
        "workspace_min_m": [-0.05, -0.01, 0.069], "workspace_max_m": [0.37, 0.33, 0.5],
        "max_position_step_m": 0.15, "max_rotation_step_deg": 35,
        "max_gripper_step_normalized": 1, "max_inactive_arm_l2": 0.05,
        "max_camera_age_s": 0.25, "max_feedback_age_s": 0.25,
    }))
    assert target.xyz_m[0] == pytest.approx(0.38)
    decision = guard.check(predicted_action20=predicted, current_active10=current,
                           camera_age_s=0.01, feedback_age_s=0.01)
    assert not decision.allowed
    assert {item.code for item in decision.alerts} >= {"POSITION_STEP_LIMIT", "WORKSPACE_VIOLATION"}
    cursor = ActionHorizonCursor(2)
    chunk = np.zeros((10, 7), dtype=np.float32)
    assert cursor.next(1, chunk)[0] == 0
    assert cursor.next(1, chunk)[0] == 1
    assert cursor.next(1, chunk) is None
    assert cursor.next(2, chunk)[0] == 0


def test_settings_reject_out_of_bounds():
    with pytest.raises(ValueError):
        ALIGNInferenceSettings.from_dict({"action_horizon": 11})
    with pytest.raises(ValueError):
        ALIGNInferenceSettings.from_dict({"ensemble_samples": 1.5})
    with pytest.raises(ValueError):
        ALIGNInferenceSettings.from_dict({"position_scale": float("nan")})


def test_camera_pause_covers_only_brief_camera_staleness():
    assert camera_stale_pause_allowed(["STALE_CAMERA"], 0.28, 0.25)
    assert not camera_stale_pause_allowed(["STALE_CAMERA"], 0.51, 0.25)
    assert not camera_stale_pause_allowed(["STALE_CAMERA", "WORKSPACE_VIOLATION"], 0.28, 0.25)
    assert not camera_stale_pause_allowed(["POSITION_STEP_LIMIT"], 0.28, 0.25)
