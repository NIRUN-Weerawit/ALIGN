"""Fail-closed validation of X-VLA actions before any future Piper command path.

This module never connects to CAN, opens cameras, or commands hardware.  It returns
structured alerts; a caller must explicitly decide how to surface them and must send
nothing when ``allowed`` is false.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionGuardLimits:
    """Operator-verified limits in Piper active-arm action coordinates."""

    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    max_position_step_m: float
    max_rotation_step_deg: float
    max_gripper_step_normalized: float
    max_inactive_arm_l2: float
    max_camera_age_s: float
    max_feedback_age_s: float

    def __post_init__(self) -> None:
        lower, upper = np.asarray(self.workspace_min_m, dtype=float), np.asarray(self.workspace_max_m, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
            raise ValueError("workspace limits must be finite [x, y, z] bounds with min < max")
        for name in (
            "max_position_step_m", "max_rotation_step_deg", "max_gripper_step_normalized",
            "max_inactive_arm_l2", "max_camera_age_s", "max_feedback_age_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def guard_limits_from_config(payload: dict) -> ActionGuardLimits:
    """Validate the persisted limits shared by the runner and WebUI editor."""
    required = {
        "workspace_min_m", "workspace_max_m", "max_position_step_m", "max_rotation_step_deg",
        "max_gripper_step_normalized", "max_inactive_arm_l2", "max_camera_age_s", "max_feedback_age_s",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"guard config keys must be exactly {sorted(required)}")
    try:
        return ActionGuardLimits(
            workspace_min_m=np.asarray(payload["workspace_min_m"], dtype=float),
            workspace_max_m=np.asarray(payload["workspace_max_m"], dtype=float),
            **{key: float(payload[key]) for key in required if not key.startswith("workspace_")},
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid guard limits: {exc}") from exc


@dataclass(frozen=True)
class GuardAlert:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    active_action10: np.ndarray | None
    alerts: tuple[GuardAlert, ...]

    def emit_warnings(self) -> None:
        """Log structured rejection/warning alerts without changing the decision."""
        for alert in self.alerts:
            LOGGER.warning("action_guard %s %s: %s", alert.severity, alert.code, alert.message)


def _rotation6d_matrix(rotation6d: np.ndarray) -> np.ndarray | None:
    col0 = rotation6d[[0, 2, 4]].astype(float, copy=True)
    col1 = rotation6d[[1, 3, 5]].astype(float, copy=True)
    first_norm = np.linalg.norm(col0)
    if first_norm < 1e-8:
        return None
    col0 /= first_norm
    col1 -= col0 * np.dot(col0, col1)
    second_norm = np.linalg.norm(col1)
    if second_norm < 1e-8:
        return None
    col1 /= second_norm
    return np.column_stack((col0, col1, np.cross(col0, col1)))


def _rotation_distance_degrees(first6d: np.ndarray, second6d: np.ndarray) -> float | None:
    first, second = _rotation6d_matrix(first6d), _rotation6d_matrix(second6d)
    if first is None or second is None:
        return None
    cosine = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class ActionGuard:
    """Fail closed on invalid, stale, or limit-exceeding predictions."""

    def __init__(self, limits: ActionGuardLimits, *, warning_fraction: float = 0.8) -> None:
        if not 0 < warning_fraction < 1:
            raise ValueError("warning_fraction must lie strictly between 0 and 1")
        self.limits = limits
        self.warning_fraction = warning_fraction

    def check(
        self,
        *,
        predicted_action20: np.ndarray,
        current_active10: np.ndarray,
        camera_age_s: float | None,
        feedback_age_s: float | None,
    ) -> GuardDecision:
        """Validate one candidate action using measured current active-arm state.

        Rejection is intentional: the method never clips or repairs a model action.
        """
        alerts: list[GuardAlert] = []
        prediction = np.asarray(predicted_action20, dtype=float)
        current = np.asarray(current_active10, dtype=float)
        if prediction.shape != (20,) or current.shape != (10,):
            raise ValueError("expected predicted_action20=(20,) and current_active10=(10,)")
        if camera_age_s is None or not np.isfinite(camera_age_s) or camera_age_s > self.limits.max_camera_age_s:
            alerts.append(GuardAlert("STALE_CAMERA", "REJECT", f"camera age={camera_age_s!r} s; max={self.limits.max_camera_age_s:.6g} s"))
        if feedback_age_s is None or not np.isfinite(feedback_age_s) or feedback_age_s > self.limits.max_feedback_age_s:
            alerts.append(GuardAlert("STALE_FEEDBACK", "REJECT", f"feedback age={feedback_age_s!r} s; max={self.limits.max_feedback_age_s:.6g} s"))
        if not np.isfinite(prediction).all() or not np.isfinite(current).all():
            alerts.append(GuardAlert("NONFINITE_ACTION", "REJECT", f"nonfinite prediction indices={np.flatnonzero(~np.isfinite(prediction)).tolist()}; measured indices={np.flatnonzero(~np.isfinite(current)).tolist()}"))
            return GuardDecision(False, None, tuple(alerts))

        active, inactive = prediction[:10], prediction[10:]
        inactive_l2 = float(np.linalg.norm(inactive))
        if inactive_l2 > self.limits.max_inactive_arm_l2:
            alerts.append(GuardAlert("INACTIVE_ARM_NONZERO", "REJECT", f"inactive arm L2={inactive_l2:.6g} > max={self.limits.max_inactive_arm_l2:.6g}; output={inactive.tolist()}"))
        if np.any(active[:3] < self.limits.workspace_min_m) or np.any(active[:3] > self.limits.workspace_max_m):
            exceeded = [f"{axis}={active[i]:.6g} m outside [{self.limits.workspace_min_m[i]:.6g}, {self.limits.workspace_max_m[i]:.6g}] m" for i, axis in enumerate("XYZ") if active[i] < self.limits.workspace_min_m[i] or active[i] > self.limits.workspace_max_m[i]]
            alerts.append(GuardAlert("WORKSPACE_VIOLATION", "REJECT", "; ".join(exceeded)))
        if not 0.0 <= active[9] <= 1.0:
            alerts.append(GuardAlert("GRIPPER_OUT_OF_RANGE", "REJECT", f"predicted normalized gripper={active[9]:.6g} outside [0, 1]"))

        position_step = float(np.linalg.norm(active[:3] - current[:3]))
        rotation_step = _rotation_distance_degrees(active[3:9], current[3:9])
        gripper_step = float(abs(active[9] - current[9]))
        if position_step > self.limits.max_position_step_m:
            alerts.append(GuardAlert("POSITION_STEP_LIMIT", "REJECT", f"XYZ step={position_step:.6g} m > max={self.limits.max_position_step_m:.6g} m; predicted={active[:3].tolist()}; measured={current[:3].tolist()}"))
        if rotation_step is None:
            alerts.append(GuardAlert("INVALID_ROTATION", "REJECT", f"degenerate rotation-6D; predicted={active[3:9].tolist()}; measured={current[3:9].tolist()}"))
        elif rotation_step > self.limits.max_rotation_step_deg:
            alerts.append(GuardAlert("ROTATION_STEP_LIMIT", "REJECT", f"rotation step={rotation_step:.6g} deg > max={self.limits.max_rotation_step_deg:.6g} deg"))
        if gripper_step > self.limits.max_gripper_step_normalized:
            alerts.append(GuardAlert("GRIPPER_STEP_LIMIT", "REJECT", f"normalized gripper step={gripper_step:.6g} > max={self.limits.max_gripper_step_normalized:.6g}; predicted={active[9]:.6g}; measured={current[9]:.6g}"))

        warnings = (
            ("POSITION_STEP_WARNING", position_step, self.limits.max_position_step_m, "predicted XYZ step approaches its hard limit"),
            ("ROTATION_STEP_WARNING", rotation_step, self.limits.max_rotation_step_deg, "predicted rotation step approaches its hard limit"),
            ("GRIPPER_STEP_WARNING", gripper_step, self.limits.max_gripper_step_normalized, "predicted gripper step approaches its hard limit"),
        )
        for code, value, limit, message in warnings:
            if value is not None and self.warning_fraction * limit <= value <= limit:
                unit = "m" if code == "POSITION_STEP_WARNING" else "deg" if code == "ROTATION_STEP_WARNING" else "normalized"
                alerts.append(GuardAlert(code, "WARN", f"{message}: value={value:.6g} {unit}; warning threshold={self.warning_fraction * limit:.6g}; max={limit:.6g}"))

        allowed = not any(alert.severity == "REJECT" for alert in alerts)
        return GuardDecision(allowed, active.astype(np.float32, copy=True) if allowed else None, tuple(alerts))
