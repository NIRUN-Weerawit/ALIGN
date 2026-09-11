"""Read-only camera and measured-Piper snapshot adapter for replay collection.

This module only calls camera ``read`` and Piper feedback getters.  It contains
no Piper command, mode, enable, replay, or stop operations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation as R

from .replay_collector import ReplayObservation
from .schema import piper_state_to_xvla20

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = MODULE_DIR / "config" / "piper_cameras.json"


class FrameCapture(Protocol):
    """The read-only subset of OpenCV's VideoCapture used by this adapter."""

    def read(self) -> tuple[bool, np.ndarray]: ...


class PiperSnapshotAdapter:
    """Capture two RGB frames and one immediately copied measured Piper state."""

    def __init__(
        self,
        piper: Any,
        global_capture: FrameCapture,
        wrist_capture: FrameCapture,
        task: str,
    ) -> None:
        self._piper = piper
        self._global_capture = global_capture
        self._wrist_capture = wrist_capture
        self._task = task

    @classmethod
    def from_camera_config(
        cls,
        piper: Any,
        task: str,
        config_path: str | Path = DEFAULT_CAMERA_CONFIG,
    ) -> "PiperSnapshotAdapter":
        """Open the configured global and wrist cameras without touching Piper."""
        import cv2

        config = json.loads(Path(config_path).read_text())

        def open_camera(role: str) -> FrameCapture:
            settings = config.get(f"{role}_camera", {})
            device = settings.get("device")
            if not device:
                raise ValueError(f"missing {role}_camera.device in {config_path}")
            capture = cv2.VideoCapture(device)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"cannot open {role} camera {device}")
            for property_id, setting in (
                (cv2.CAP_PROP_FRAME_WIDTH, "width"),
                (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
                (cv2.CAP_PROP_FPS, "fps"),
            ):
                if setting in settings:
                    capture.set(property_id, settings[setting])
            return capture

        global_capture: FrameCapture | None = None
        try:
            global_capture = open_camera("global")
            wrist_capture = open_camera("wrist")
        except Exception:
            if global_capture is not None:
                global_capture.release()  # type: ignore[attr-defined]
            raise
        return cls(piper, global_capture, wrist_capture, task)

    @staticmethod
    def _read_rgb(capture: FrameCapture, role: str) -> np.ndarray:
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to capture {role} camera frame")
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[-1] != 3:
            raise ValueError(f"{role} camera must return a BGR HxWx3 frame")
        if bgr.dtype != np.uint8:
            raise ValueError(f"{role} camera frame must be uint8")
        return np.ascontiguousarray(bgr[..., ::-1])

    def read_state20(self) -> np.ndarray:
        """Copy mutable SDK feedback scalars and encode the active Piper arm."""
        end_pose = self._piper.GetArmEndPoseMsgs().end_pose
        # Copy each field before calling another getter: SDK feedback envelopes are mutable.
        position_m = np.array(
            [end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis], dtype=np.float64
        ) * 1e-6
        euler_degrees = np.array(
            [end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis], dtype=np.float64
        ) * 1e-3
        gripper_state = self._piper.GetArmGripperMsgs().gripper_state
        gripper_m = float(gripper_state.grippers_angle) * 1e-6
        # SDK units: xyz/stroke are 0.001 mm; Euler angles are 0.001 degrees.
        quaternion_xyzw = R.from_euler("xyz", euler_degrees, degrees=True).as_quat()
        return piper_state_to_xvla20(position_m, quaternion_xyzw, gripper_m)

    def snapshot(self) -> ReplayObservation:
        """Return one valid read-only observation from current camera and Piper data."""
        global_rgb = self._read_rgb(self._global_capture, "global")
        wrist_rgb = self._read_rgb(self._wrist_capture, "wrist")
        return ReplayObservation(global_rgb, wrist_rgb, self.read_state20(), self._task)

    def close(self) -> None:
        """Release both cameras. Idempotent; safe to call after a failed open."""
        for capture in (self._global_capture, self._wrist_capture):
            release = getattr(capture, "release", None)
            if callable(release):
                release()
        self._global_capture = None  # type: ignore[assignment]
        self._wrist_capture = None  # type: ignore[assignment]
