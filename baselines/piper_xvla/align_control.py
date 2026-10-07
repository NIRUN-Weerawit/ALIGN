"""Physical-unit adapter from Piper-trained ALIGN actions to guarded Piper targets."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from piper_xvla.endpose_control import EndPoseTarget


@dataclass(frozen=True)
class ALIGNInferenceSettings:
    inference_hz: float = 10.0
    action_horizon: int = 2
    ensemble_samples: int = 1
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    gripper_scale: float = 1.0
    gripper_close_threshold_mm: float = 3.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.inference_hz) or not 1 <= self.inference_hz <= 20:
            raise ValueError("inference_hz must be within [1, 20]")
        if not 1 <= self.action_horizon <= 10:
            raise ValueError("action_horizon must be within [1, 10]")
        if not 1 <= self.ensemble_samples <= 8:
            raise ValueError("ensemble_samples must be within [1, 8]")
        for name in ("position_scale", "rotation_scale", "gripper_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 2:
                raise ValueError(f"{name} must be within [0, 2]")
        if not np.isfinite(self.gripper_close_threshold_mm) or not -70 <= self.gripper_close_threshold_mm <= 70:
            raise ValueError("gripper_close_threshold_mm must be within [-70, 70]")

    @classmethod
    def from_dict(cls, payload: dict) -> "ALIGNInferenceSettings":
        if not isinstance(payload, dict) or set(payload) - set(cls.__dataclass_fields__):
            raise ValueError("unknown ALIGN inference setting")
        try:
            values = {key: float(value) for key, value in payload.items() if key not in {"action_horizon", "ensemble_samples"}}
            for key in ("action_horizon", "ensemble_samples"):
                if key in payload:
                    integer = int(payload[key])
                    if integer != float(payload[key]):
                        raise ValueError(f"{key} must be an integer")
                    values[key] = integer
            return cls(**values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid ALIGN inference settings: {exc}") from exc

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ALIGNGripperMapping:
    raw_normalized: float
    bounded_normalized: float
    raw_mm: float
    target_mm: float

    @property
    def clamped(self) -> bool:
        return self.raw_normalized != self.bounded_normalized


def camera_stale_pause_allowed(rejection_codes: list[str], age_s: float | None, max_age_s: float) -> bool:
    """Allow a brief no-motion pause for camera staleness, never a stale action."""
    return (bool(rejection_codes) and set(rejection_codes) == {"STALE_CAMERA"}
            and isinstance(age_s, (int, float)) and np.isfinite(age_s)
            and max_age_s < age_s <= max_age_s + min(max_age_s, 0.25))


def align_gripper_mapping(
    predicted_normalized: float,
    measured_gripper_m: float,
    gripper_min_m: float,
    gripper_max_m: float,
    settings: ALIGNInferenceSettings,
) -> ALIGNGripperMapping:
    """Bound the unconstrained model output to its trained gripper interval."""
    span = gripper_max_m - gripper_min_m
    current_normalized = np.clip((measured_gripper_m - gripper_min_m) / span, 0, 1)
    raw_normalized = float(current_normalized + settings.gripper_scale * (predicted_normalized - current_normalized))
    bounded_normalized = float(np.clip(raw_normalized, 0, 1))
    raw_mm = 1000 * (gripper_min_m + raw_normalized * span)
    bounded_mm = 1000 * (gripper_min_m + bounded_normalized * span)
    target_mm = 0.0 if bounded_mm < settings.gripper_close_threshold_mm else bounded_mm
    return ALIGNGripperMapping(raw_normalized, bounded_normalized, raw_mm, target_mm)


def _rotation_from_active(active10: np.ndarray) -> Rotation:
    axes = np.asarray(active10[3:9], dtype=np.float64).reshape(3, 2)
    x = axes[:, 0]
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        raise ValueError("measured Piper rotation has a degenerate first axis")
    x = x / x_norm
    y = axes[:, 1] - x * np.dot(x, axes[:, 1])
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-8:
        raise ValueError("measured Piper rotation has a degenerate second axis")
    y = y / y_norm
    return Rotation.from_matrix(np.column_stack((x, y, np.cross(x, y))))


def align_state7(raw_state20: np.ndarray, gripper_min_m: float, gripper_max_m: float) -> np.ndarray:
    """Match HDF5 state: XYZ metres, Euler xyz radians, normalized gripper."""
    raw = np.asarray(raw_state20, dtype=np.float64)
    if raw.shape != (20,) or not np.isfinite(raw).all():
        raise ValueError("Piper feedback must be finite shape (20,)")
    if not np.isfinite([gripper_min_m, gripper_max_m]).all() or gripper_min_m >= gripper_max_m:
        raise ValueError("invalid Piper gripper calibration")
    rotation = _rotation_from_active(raw[:10])
    gripper = np.clip((raw[9] - gripper_min_m) / (gripper_max_m - gripper_min_m), 0, 1)
    return np.concatenate((raw[:3], rotation.as_euler("xyz"), [gripper])).astype(np.float32)


def align_action_to_piper_target(
    action7: np.ndarray,
    raw_state20: np.ndarray,
    gripper_min_m: float,
    gripper_max_m: float,
    settings: ALIGNInferenceSettings,
) -> tuple[EndPoseTarget, float, np.ndarray, np.ndarray]:
    """Anchor one predicted delta to fresh measured feedback, once per action."""
    action = np.asarray(action7, dtype=np.float64)
    raw = np.asarray(raw_state20, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("ALIGN action must be finite shape (7,)")
    if raw.shape != (20,) or not np.isfinite(raw).all():
        raise ValueError("Piper feedback must be finite shape (20,)")
    if not np.isfinite([gripper_min_m, gripper_max_m]).all() or gripper_min_m >= gripper_max_m:
        raise ValueError("invalid Piper gripper calibration")

    current_rotation = _rotation_from_active(raw[:10])
    xyz_m = raw[:3] + settings.position_scale * action[:3]
    delta_rotation = Rotation.from_euler("xyz", settings.rotation_scale * action[3:6])
    target_rotation = delta_rotation * current_rotation
    span = gripper_max_m - gripper_min_m
    current_normalized = np.clip((raw[9] - gripper_min_m) / span, 0, 1)
    gripper_mapping = align_gripper_mapping(float(action[6]), float(raw[9]), gripper_min_m, gripper_max_m, settings)
    gripper_m = gripper_mapping.target_mm / 1000
    target_gripper_normalized = (gripper_m - gripper_min_m) / span

    current10 = raw[:10].copy()
    current10[9] = current_normalized
    target10 = np.concatenate((xyz_m, target_rotation.as_matrix()[:, :2].reshape(-1), [target_gripper_normalized]))
    target20 = np.concatenate((target10, np.zeros(10, dtype=np.float64)))
    return EndPoseTarget(xyz_m, target_rotation.as_euler("xyz", degrees=True)), gripper_m, current10, target20


class ActionHorizonCursor:
    """Consume each future action once; never repeatedly integrate one delta."""

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.generation = -1
        self.index = 0

    def next(self, generation: int, chunk: np.ndarray) -> tuple[int, np.ndarray] | None:
        if generation != self.generation:
            self.generation = generation
            self.index = 0
        if self.index >= min(self.horizon, len(chunk)):
            return None
        index = self.index
        self.index += 1
        return index, np.asarray(chunk[index], dtype=np.float32)
