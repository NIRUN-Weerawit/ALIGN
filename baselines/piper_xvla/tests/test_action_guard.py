import numpy as np

from piper_xvla.action_guard import ActionGuard, ActionGuardLimits


def limits() -> ActionGuardLimits:
    return ActionGuardLimits(
        workspace_min_m=np.array([0.0, -0.2, 0.0]),
        workspace_max_m=np.array([0.5, 0.2, 0.5]),
        max_position_step_m=0.02,
        max_rotation_step_deg=10.0,
        max_gripper_step_normalized=0.1,
        max_inactive_arm_l2=0.01,
        max_camera_age_s=0.10,
        max_feedback_age_s=0.05,
    )


def active_action(x=0.10, gripper=0.50) -> np.ndarray:
    return np.array([x, 0.0, 0.2, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, gripper], dtype=np.float32)


def test_guard_allows_safe_action_and_returns_active_piper_dimensions():
    decision = ActionGuard(limits()).check(
        predicted_action20=np.concatenate([active_action(0.11, 0.55), np.zeros(10, dtype=np.float32)]),
        current_active10=active_action(),
        camera_age_s=0.01,
        feedback_age_s=0.01,
    )

    assert decision.allowed is True
    assert decision.active_action10.shape == (10,)
    assert decision.alerts == ()


def test_guard_rejects_stale_invalid_and_out_of_bounds_action():
    prediction = np.concatenate([active_action(0.8), np.zeros(10, dtype=np.float32)])
    prediction[10] = 0.5
    decision = ActionGuard(limits()).check(
        predicted_action20=prediction,
        current_active10=active_action(),
        camera_age_s=0.20,
        feedback_age_s=0.10,
    )

    assert decision.allowed is False
    assert {alert.code for alert in decision.alerts} == {
        "STALE_CAMERA", "STALE_FEEDBACK", "WORKSPACE_VIOLATION", "INACTIVE_ARM_NONZERO", "POSITION_STEP_LIMIT",
    }


def test_guard_emits_warning_when_action_nears_a_limit():
    decision = ActionGuard(limits(), warning_fraction=0.8).check(
        predicted_action20=np.concatenate([active_action(0.117, 0.59), np.zeros(10, dtype=np.float32)]),
        current_active10=active_action(),
        camera_age_s=0.01,
        feedback_age_s=0.01,
    )

    assert decision.allowed is True
    assert {alert.code for alert in decision.alerts} == {"POSITION_STEP_WARNING", "GRIPPER_STEP_WARNING"}
