"""Web UI for Piper X-VLA collection and data review.

Run (live):
    cd ~/ALIGN/baselines
    PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python -m piper_xvla.webui --port 8050

Then open http://localhost:8050 in a browser.

Endpoints (all JSON unless noted):
  GET  /                       single-page HTML UI
  GET  /api/status             current Piper status + mode/teach/motion/errors/drivers
  GET  /api/joints             live joint positions (deg)
  GET  /api/cameras            one-frame preflight of both cameras (base64 JPEGs)
  GET  /api/camera-stream      MJPEG multipart stream of global|wrist (live preview)
  GET  /api/xvla/checkpoints    discover local X-VLA checkpoints for inference
  POST /api/replay             {action: start|pause|resume|stop|move_start}
  POST /api/enable             energize drivers
  POST /api/disable            de-energize drivers
  POST /api/mode               {mode: standby|can}; host-owned modes only
  POST /api/reset              clear e-stop + standby
  POST /api/estop              emergency stop (never moves arm)
  GET  /api/task               current task string
  PUT  /api/task               {task: "..."} set task
  GET  /api/collect/status     progress of the active collect session (or idle)
  POST /api/collect/start      {window_s?: float} start a background collect
  POST /api/collect/stop       stop the active collect early
  GET  /api/dataset            dataset summary (episodes, frames, fps, features)
  GET  /api/episodes           per-episode metadata + thumbnails + downsampled state/action
  GET  /api/videos             list of review videos on disk
  GET  /api/video/{name}       stream a review mp4 (range-request safe)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import io
import json
import os
import re
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from piper_xvla.replay_control import (
    DryRunPiper,
    PiperReplayController,
    connect_live_piper,
    status_fields,
)
from piper_xvla.snapshot_adapter import DEFAULT_CAMERA_CONFIG
from piper_xvla.action_guard import guard_limits_from_config
from piper_xvla.align_control import ALIGNInferenceSettings, camera_stale_pause_allowed
from piper_xvla.gripper_binary import BinaryGripperConfig, DEFAULT_CONFIG_PATH as XVLA_GRIPPER_CONFIG, load_binary_gripper_config
from piper_xvla.endpose_control import EndPoseTarget, _stream_endpose, _wait_for_can_mode, _wait_until_enabled, prepare_can_cartesian_control
from piper_xvla.review_video import ensure_h264, probe_codec
from piper_xvla.master_slave_control import (
    DEFAULT_MAX_JOINT_GAP_DEG, GRIPPER_BIAS_MM, GRIPPER_SCALE, J6_BIAS_DEG, J6_SCALE, MAX_MIRROR_SPEED_PERCENT,
    JOINT_BIASES_DEG, JOINT_SCALES, MASTER_GRIPPER_PHYSICAL_LIMITS_MM,
    MASTER_JOINT_PHYSICAL_LIMITS_DEG, MASTER_J6_PHYSICAL_LIMITS_DEG,
    SLAVE_GRIPPER_PHYSICAL_LIMITS_MM, SLAVE_JOINT_PHYSICAL_LIMITS_DEG, SLAVE_J6_PHYSICAL_LIMITS_DEG,
    hold_slave_position, run_joint_mirror, validate_joint_gap,
)

MODULE_DIR = Path(__file__).resolve().parent
FRONTEND_HTML = MODULE_DIR / "webui.html"
CAN_SYSFS_ROOT = Path("/sys/class/net")


def _connect_piper_observer(piper: Any) -> None:
    """Start receiving CAN on SDKs with or without the piper_init option."""
    connect = piper.ConnectPort
    try:
        parameters = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "piper_init" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        connect(piper_init=False)
    else:
        connect()


def _joint_temperatures(info: Any) -> dict[str, list[Optional[int]]]:
    """Copy per-joint motor and driver temperatures from low-speed feedback (°C)."""
    def values(field: str) -> list[Optional[int]]:
        result: list[Optional[int]] = []
        for index in range(1, 7):
            raw = getattr(getattr(info, f"motor_{index}", None), field, None)
            try:
                result.append(int(raw) if raw is not None else None)
            except (TypeError, ValueError):
                result.append(None)
        return result
    return {"motor_temperatures_c": values("motor_temp"),
            "driver_temperatures_c": values("foc_temp")}


DEFAULT_DATA_DIR = Path.home() / "ALIGN" / "baselines" / "data" / "piper_replay"
XVLA_GUARD_CONFIG = MODULE_DIR / "config" / "piper_action_guard.camera_only.json"
XVLA_OUTPUTS_ROOT = MODULE_DIR.parent / "outputs"
XVLA_DEFAULT_CHECKPOINT = "piper_xvla_single_task/best.pt"
ALIGN_DEFAULT_CHECKPOINT = MODULE_DIR.parents[1] / "checkpoints" / "piper_replay_new" / "run_2" / "intention_best.pt"
ALIGN_CALIBRATION_MANIFEST = MODULE_DIR.parent / "data" / "piper_replay_new" / "xvla_training_manifest.json"


def _xvla_subprocess_env() -> dict[str, str]:
    """Prefer this interpreter's packages, then allow user-installed Piper SDK."""
    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    env["PYTHONPATH"] = os.pathsep.join((sysconfig.get_paths()["purelib"], str(MODULE_DIR.parent)))
    env["PYTHONUNBUFFERED"] = "1"
    return env
# Piper gripper commands are signed strokes in metres at the WebUI boundary.
# Keep the pre-existing 70 mm positive envelope and permit the matching negative
# direction rather than silently applying abs() to an operator command.
MANUAL_GRIPPER_MIN_M = -0.07
MANUAL_GRIPPER_MAX_M = 0.07

MODE_NAMES = {
    0x00: "STANDBY", 0x01: "CAN_CTRL", 0x02: "TEACHING", 0x03: "ETHERNET",
    0x04: "WIFI", 0x05: "REMOTE", 0x06: "LINKAGE_TEACH", 0x07: "OFFLINE_TRAJ",
}
ARM_STATUS_NAMES = {
    0x00: "NORMAL", 0x01: "EMERGENCY_STOP", 0x02: "NO_SOLUTION", 0x03: "SINGULARITY",
    0x04: "TARGET_LIMIT", 0x05: "JOINT_COMMS_ERROR", 0x06: "BRAKE_NOT_RELEASED",
    0x07: "COLLISION", 0x08: "TEACH_OVERSPEED", 0x09: "JOINT_STATUS_ERROR",
    0x0A: "OTHER_ERROR", 0x0B: "TEACH_RECORDING", 0x0C: "TEACH_EXECUTING", 0x0D: "TEACH_PAUSED",
}
TEACH_NAMES = {
    0x00: "DISABLED", 0x01: "START_RECORD", 0x02: "STOP_RECORD", 0x03: "EXECUTE_TRAJ",
    0x04: "PAUSE", 0x05: "RESUME", 0x06: "TERMINATE", 0x07: "MOVE_TO_START",
}

# Manufacturer-documented JointCtrl limits in degrees. These validate input;
# they do not establish a collision-free workspace.
JOINT_LIMITS_DEG = ((-150.0, 150.0), (0.0, 180.0), (-170.0, 0.0),
                    (-100.0, 100.0), (-70.0, 70.0), (-120.0, 120.0))


def _name(table: dict, value: int) -> str:
    if value is None or value < 0:
        return "n/a"
    return table.get(value, f"UNKNOWN(0x{value & 0xFF:02x})")


def _hex(value: Optional[int]) -> str:
    if value is None or value < 0:
        return "n/a"
    return f"0x{value & 0xFF:02x}"


def _active10_pose(values: object) -> dict[str, list[float]] | None:
    """Render X-VLA/Piper active 10-D xyz+rotation6D+gripper as a browser pose."""
    try:
        active = np.asarray(values, dtype=float).reshape(-1)[:10]
        if active.shape != (10,) or not np.isfinite(active).all():
            return None
        col0, col1 = active[[3, 5, 7]], active[[4, 6, 8]]
        col0 /= np.linalg.norm(col0)
        col1 -= col0 * np.dot(col0, col1)
        col1 /= np.linalg.norm(col1)
        if not np.isfinite(col0).all() or not np.isfinite(col1).all():
            return None
        matrix = np.column_stack((col0, col1, np.cross(col0, col1)))
        euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
        return {"xyz_m": np.round(active[:3], 6).tolist(), "euler_xyz_deg": np.round(euler, 3).tolist(), "gripper_normalized": [round(float(active[9]), 4)]}
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


class PiperWebUI:
    """Owns the live (or dry-run) Piper handle and the background collect task."""

    def __init__(self, can: str = "can0", dry_run: bool = False, data_dir: Path = DEFAULT_DATA_DIR):
        self.can = can
        self.dry_run = dry_run
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.dataset_root = self.data_dir / "dataset"
        if dry_run:
            self.piper = DryRunPiper()
        else:
            self.piper = connect_live_piper(can)
            time.sleep(1.0)  # let feedback streams fill
        self.speed = 20
        self.ctrl = PiperReplayController(self.piper, replay_speed_percent=self.speed)
        self.task = ""
        # Manual Cartesian control is explicitly latched on/off by the operator.
        # Marks are snapshots only until a separately confirmed movement consumes them.
        self._manual_lock = threading.Lock()
        self._manual_motion_lock = threading.Lock()
        self._manual_stop_event = threading.Event()
        self._manual_enabled = False
        self._manual_motion_active = False
        self._manual_marks: list[dict[str, Any]] = []

        # Explicitly started master-to-slave joint mirroring. The selected master
        # is read-only; only the selected slave receives commands.
        self._master_slave_lock = threading.Lock()
        self._master_slave_command_lock = threading.Lock()
        self._master_slave_stop = threading.Event()
        self._master_slave_thread: Optional[threading.Thread] = None
        self._master_slave_slave_piper: Optional[Any] = None
        self._master_slave_lease_expires = 0.0
        self._master_slave_state: dict[str, Any] = {"status": "idle", "sent": 0, "message": ""}
        self._status_pipers: dict[str, Any] = {}
        self._status_pipers_lock = threading.Lock()

        # Serialize one-frame preview acquisition with collection camera ownership.
        # Collection holds this lock until it releases both V4L2 devices.
        self._camera_lock = threading.Lock()

        # X-VLA is deliberately diagnostic-only: its separate runner has no
        # command, enable, or controller-mode path.
        self._xvla_lock = threading.Lock()
        self._xvla_stop_event = threading.Event()
        self._xvla_process: Optional[subprocess.Popen[str]] = None
        self._xvla_log_offset = 0
        self._xvla_lease_path: Optional[Path] = None
        self._xvla_lease_lock = threading.Lock()
        self._xvla_lease_expires = 0.0
        self._xvla_control_session: Optional[str] = None
        self._xvla_revoked_sessions: set[str] = set()
        self._xvla_state: dict[str, Any] = {
            "status": "idle", "frames": 0, "output": None, "message": "",
            "summary": None, "motion_capability": "none",
        }
        self._align_lock = threading.Lock()
        self._align_stop_event = threading.Event()
        self._align_process: Optional[subprocess.Popen[str]] = None
        self._align_log_offset = 0
        self._align_state: dict[str, Any] = {"status": "idle", "output": None, "message": ""}

        # Background collect state.
        self._collect_lock = threading.Lock()
        self._collect_session: Optional[Any] = None
        self._collect_stop = threading.Event()
        self._collect_thread: Optional[threading.Thread] = None
        self._collect_state: dict = {
            "status": "idle",  # idle | opening | recording | finalizing | done | error
            "frames": 0,
            "elapsed_s": 0.0,
            "window_s": 0.0,
            "episode_index": None,
            "video_path": None,
            "dataset_path": str(self.dataset_root),
            "message": "",
        }

    # -- status -------------------------------------------------------------

    def _feedback_health(self) -> tuple[bool, str, dict[str, float]]:
        """Distinguish usable live feedback from a connected-but-silent CAN adapter."""
        if self.dry_run:
            return True, "SIMULATED (dry-run)", {"status": 0.0, "joint": 0.0, "low_speed": 0.0, "end_pose": 0.0, "gripper": 0.0}
        try:
            if not bool(self.piper.isOk()):
                return False, "SDK health check is false", {"status": 0.0, "joint": 0.0, "low_speed": 0.0}
            raw_status = self.piper.GetArmStatus()
            joint_msg = self.piper.GetArmJointMsgs()
            low_msg = self.piper.GetArmLowSpdInfoMsgs()
            end_pose_msg = getattr(self.piper, "GetArmEndPoseMsgs")()
            gripper_msg = getattr(self.piper, "GetArmGripperMsgs")()
            rates = {
                "status": float(getattr(raw_status, "Hz", 0.0) or 0.0),
                "joint": float(getattr(joint_msg, "Hz", 0.0) or 0.0),
                "low_speed": float(getattr(low_msg, "Hz", 0.0) or 0.0),
                "end_pose": float(getattr(end_pose_msg, "Hz", 0.0) or 0.0),
                "gripper": float(getattr(gripper_msg, "Hz", 0.0) or 0.0),
            }
        except Exception as exc:  # noqa: BLE001
            return False, f"feedback query failed: {type(exc).__name__}: {exc}", {"status": 0.0, "joint": 0.0, "low_speed": 0.0}
        missing = [name for name, hz in rates.items() if hz < 1.0]
        if missing:
            return False, "no live feedback on " + ", ".join(missing), rates
        # A controller target-limit response is still live CAN feedback. Keep the
        # transport state separate from whether a movement command is advisable.
        return True, "LIVE", rates

    def _command_health(self) -> tuple[bool, str]:
        """Classify whether commands may continue while feedback remains live."""
        if self.dry_run:
            return True, "SIMULATED (dry-run)"
        feedback_ok, reason, _ = self._feedback_health()
        if not feedback_ok:
            return False, f"live Piper feedback is unavailable ({reason})"
        raw_status = self.piper.GetArmStatus()
        status = getattr(raw_status, "arm_status", raw_status)
        arm_status = int(getattr(status, "arm_status", -1))
        err_code = int(getattr(status, "err_code", -1))
        if arm_status == 0x00 and err_code == 0:
            return True, "controller normal"
        if arm_status == 0x04 and err_code == 0:
            return True, "TARGET_LIMIT reported; choose a reachable target"
        return False, f"controller state {_name(ARM_STATUS_NAMES, arm_status)} / error {_hex(err_code)} requires recovery"

    def snapshot_status(self) -> dict:
        feedback_ok, feedback_reason, feedback_hz = self._feedback_health()
        command_ok, command_reason = self._command_health()
        s = status_fields(self.piper)
        try:
            from piper_xvla.motion_watch import joint_positions_deg
            joints = list(joint_positions_deg(self.piper.GetArmJointMsgs()))
        except Exception:  # noqa: BLE001
            joints = [None] * 6
        try:
            info = self.piper.GetArmLowSpdInfoMsgs()
            drivers = [int(getattr(info, f"motor_{i}").foc_status.driver_enable_status) for i in range(1, 7)]
            temperatures = _joint_temperatures(info)
        except Exception:  # noqa: BLE001
            drivers = [None] * 6
            temperatures = _joint_temperatures(None)
        try:
            pose = self._manual_pose()
        except Exception:  # noqa: BLE001
            pose = None
        manual_lock = getattr(self, "_manual_lock", None)
        if manual_lock is None:
            manual_enabled = False
        else:
            with manual_lock:
                manual_enabled = bool(getattr(self, "_manual_enabled", False))
        return {
            "can": self.can,
            "dry_run": self.dry_run,
            "feedback_ok": feedback_ok,
            "feedback_state": "SIMULATED" if self.dry_run else ("LIVE" if feedback_ok else "UNAVAILABLE"),
            "feedback_reason": feedback_reason,
            "feedback_hz": feedback_hz,
            "command_ok": command_ok,
            "command_reason": command_reason,
            "ctrl_mode": int(getattr(s, "ctrl_mode", -1)),
            "ctrl_mode_name": _name(MODE_NAMES, int(getattr(s, "ctrl_mode", -1))),
            "arm_status": int(getattr(s, "arm_status", -1)),
            "arm_status_name": _name(ARM_STATUS_NAMES, int(getattr(s, "arm_status", -1))),
            "teach_status": int(getattr(s, "teach_status", -1)),
            "teach_status_name": _name(TEACH_NAMES, int(getattr(s, "teach_status", -1))),
            "motion_status": int(getattr(s, "motion_status", -1)),
            "err_code": int(getattr(s, "err_code", -1)),
            "drivers": drivers,
            **temperatures,
            "joints_deg": joints,
            "end_pose": pose,
            "manual_control_enabled": manual_enabled,
            "speed": self.speed,
            "task": self.task,
        }

    def collection_active(self) -> bool:
        with self._collect_lock:
            return self._collect_state["status"] in {"opening", "recording", "stopping", "finalizing"}

    def get_runtime_config(self) -> dict:
        """Return the editable collection settings plus immutable launch facts."""
        try:
            cameras = json.loads(DEFAULT_CAMERA_CONFIG.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not read {DEFAULT_CAMERA_CONFIG}: {exc}") from exc
        return {
            "can": self.can,
            "dry_run": self.dry_run,
            "data_dir": str(self.data_dir),
            "camera_config_path": str(DEFAULT_CAMERA_CONFIG),
            "replay_speed": self.speed,
            "cameras": cameras,
        }

    def update_runtime_config(self, payload: dict) -> dict:
        """Persist safe runtime settings; reject camera edits during collection."""
        if not isinstance(payload, dict):
            raise ValueError("settings payload must be an object")
        if "replay_speed" in payload:
            try:
                speed = int(payload["replay_speed"])
            except (TypeError, ValueError) as exc:
                raise ValueError("replay_speed must be an integer") from exc
            if not 1 <= speed <= 100:
                raise ValueError("replay_speed must be in [1, 100]")
            self.speed = speed
            self.ctrl.speed = speed
        if "cameras" in payload:
            if self.collection_active():
                raise RuntimeError("cannot change camera configuration while collection is active")
            cameras = payload["cameras"]
            if not isinstance(cameras, dict):
                raise ValueError("cameras must be an object")
            updated = json.loads(DEFAULT_CAMERA_CONFIG.read_text())
            for role in ("global", "wrist"):
                key = f"{role}_camera"
                if key not in cameras:
                    continue
                incoming = cameras[key]
                if not isinstance(incoming, dict):
                    raise ValueError(f"{key} must be an object")
                device = str(incoming.get("device", "")).strip()
                if not device:
                    raise ValueError(f"{key}.device must be non-empty")
                try:
                    width, height, fps = int(incoming["width"]), int(incoming["height"]), int(incoming["fps"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{key} needs integer width, height, and fps") from exc
                if not (64 <= width <= 7680 and 64 <= height <= 4320 and 1 <= fps <= 240):
                    raise ValueError(f"{key} dimensions/fps are outside safe bounds")
                updated[key] = {**updated.get(key, {}), "device": device, "width": width, "height": height, "fps": fps}
            temporary = DEFAULT_CAMERA_CONFIG.with_suffix(".tmp")
            temporary.write_text(json.dumps(updated, indent=2) + "\n")
            temporary.replace(DEFAULT_CAMERA_CONFIG)
        return self.get_runtime_config()

    # -- cameras -------------------------------------------------------------

    def camera_preflight(self) -> dict:
        """Read one frame per camera without racing collection camera ownership."""
        if not self._camera_lock.acquire(timeout=2.0):
            raise RuntimeError("camera preview is busy; retry after the active camera operation")
        try:
            from piper_xvla.camera_check import check_camera
            config = json.loads(DEFAULT_CAMERA_CONFIG.read_text())
            results = {}
            for role in ("global", "wrist"):
                device = config.get(f"{role}_camera", {}).get("device")
                if not device:
                    results[role] = {"ok": False, "reason": "no device configured"}
                    continue
                frame, result = check_camera(device, role)
                payload: dict = dict(result)
                if frame is not None:
                    import cv2
                    small = cv2.resize(frame, (320, 240))
                    ok, jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        payload["frame_b64"] = base64.b64encode(jpg.tobytes()).decode()
                results[role] = payload
            return results
        finally:
            self._camera_lock.release()

    def camera_stream(self):
        """MJPEG multipart stream of global|wrist at ~100 FPS."""
        import cv2
        config = json.loads(DEFAULT_CAMERA_CONFIG.read_text())
        devices = {
            "global": config.get("global_camera", {}).get("device"),
            "wrist": config.get("wrist_camera", {}).get("device"),
        }

        def gen():
            caps = {}
            try:
                for role, device in devices.items():
                    if not device:
                        continue
                    cap = cv2.VideoCapture(device)
                    if cap.isOpened():
                        caps[role] = cap
                if not caps:
                    yield b"no cameras available\n"
                    return
                while True:
                    for role, cap in caps.items():
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            continue
                        small = cv2.resize(frame, (320, 240))
                        _, jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        payload = jpg.tobytes()
                        header = (
                            "--frame\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"X-Role: {role}\r\n"
                            f"Content-Length: {len(payload)}\r\n\r\n"
                        ).encode()
                        yield header + payload + b"\r\n"
                    time.sleep(0.01)  # ~100 FPS per camera
            finally:
                for cap in caps.values():
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # -- replay control ------------------------------------------------------

    def _verified_live_feedback(self) -> tuple[bool, str]:
        ok, reason = self._command_health()
        if ok:
            return True, reason
        return False, f"refusing command: {reason}"

    def do_replay(self, action: str) -> dict:
        self._assert_no_manual_motion("starting or changing pendant replay")
        self._reject_during_xvla_control()
        ready, reason = self._verified_live_feedback()
        if not ready:
            return {"ok": False, "error": reason}
        try:
            if action == "start":
                self.ctrl.start_replay()
            elif action == "pause":
                self.ctrl.pause_replay()
            elif action == "resume":
                self.ctrl.resume_replay()
            elif action == "stop":
                self.ctrl.stop_replay()
            elif action == "move_start":
                self.ctrl.move_to_start()
            else:
                raise ValueError(f"unknown replay action {action!r}")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # MotionCtrl_1 uses CAN ID 0x150. Expose the requested bytes so an
        # operator can distinguish a browser/API dispatch from physical motion.
        payloads = {
            "pause": "0000040000000000",
            "resume": "0000050000000000",
            "stop": "0000060000000000",
            "move_start": "0000070000000000",
        }
        result = {"ok": True, "action": action}
        if action in payloads:
            result.update(can_id="0x150", payload_hex=payloads[action])
        return result

    def do_enable(self) -> dict:
        self._assert_no_manual_motion("enabling drivers")
        self._reject_during_xvla_control()
        ready, reason = self._verified_live_feedback()
        if not ready:
            return {"ok": False, "error": reason}
        try:
            for _ in range(10):
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                time.sleep(0.5)
                info = self.piper.GetArmLowSpdInfoMsgs()
                states = [int(getattr(info, f"motor_{i}").foc_status.driver_enable_status) for i in range(1, 7)]
                if all(states):
                    return {"ok": True, "drivers": states}
            return {"ok": False, "error": "drivers did not report enabled within 5 s", "drivers": states}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def do_disable(self) -> dict:
        self._assert_no_manual_motion("disabling drivers")
        self._reject_during_xvla_control()
        ready, reason = self._verified_live_feedback()
        if not ready:
            return {"ok": False, "error": reason}
        try:
            for _ in range(10):
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
                time.sleep(0.5)
                info = self.piper.GetArmLowSpdInfoMsgs()
                states = [int(getattr(info, f"motor_{i}").foc_status.driver_enable_status) for i in range(1, 7)]
                if not any(states):
                    return {"ok": True, "drivers": states}
            return {"ok": False, "error": "drivers did not all report disabled within 5 s", "drivers": states}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def do_mode(self, mode: str) -> dict:
        self._assert_no_manual_motion("changing control mode")
        self._reject_during_xvla_control()
        """Switch a host-owned Piper control mode.

        Drag-teach replay/offline mode is pendant-owned and deliberately absent:
        the validated replay path uses grag_teach_ctrl 0x07 then 0x03 without a
        MotionCtrl_2 mode change.
        """
        targets = {"standby": 0x00, "can": 0x01}
        requested = mode.strip().lower()
        if requested in {"offline", "replay"}:
            return {
                "ok": False,
                "error": "offline/replay mode is pendant-owned; use the physical pendant replay button",
            }
        if requested not in targets:
            return {"ok": False, "error": "mode must be standby or can"}
        target = targets[requested]
        ready, reason = self._verified_live_feedback()
        if not ready:
            return {"ok": False, "error": reason}
        current = self.ctrl.current_ctrl_mode()
        if current == target:
            return {"ok": True, "mode": requested, "message": f"already in {MODE_NAMES[target]}"}
        try:
            self.piper.MotionCtrl_2(
                ctrl_mode=target,
                move_mode=0x01,
                move_spd_rate_ctrl=self.speed,
                is_mit_mode=0x00,
                residence_time=0,
                installation_pos=0x00,
            )
            actual = self.ctrl._wait_for_mode(target, timeout_s=1.5)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if actual == target:
            return {"ok": True, "mode": requested, "message": f"entered {MODE_NAMES[target]}"}
        hint = " Enable the arm first, then retry CAN control." if target == 0x01 else ""
        return {
            "ok": False,
            "error": f"controller rejected {MODE_NAMES[target]} (still {_name(MODE_NAMES, actual)}).{hint}",
        }

    def do_reset(self) -> dict:
        self._assert_no_manual_motion("resetting the arm")
        self._reject_during_xvla_control()
        ready, reason = self._verified_live_feedback()
        if not ready:
            return {"ok": False, "error": reason}
        try:
            self.piper.MotionCtrl_1(emergency_stop=0x02, track_ctrl=0x00, grag_teach_ctrl=0x00)
            time.sleep(0.3)
            self.piper.MotionCtrl_2(ctrl_mode=0x00, move_mode=0x01, move_spd_rate_ctrl=self.speed,
                                    is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
            actual = self.ctrl._wait_for_mode(0x00, timeout_s=1.5)
            if actual != 0x00:
                return {"ok": False, "error": f"reset was sent but standby was not observed (actual {_name(MODE_NAMES, actual)})"}
            return {"ok": True, "mode": "standby", "message": "e-stop clear and standby verified"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def do_estop(self) -> dict:
        if self.align_status()["status"] == "running":
            self.stop_align_inference(emergency=True)
        self.revoke_xvla_control()
        with self._manual_lock:
            self._manual_enabled = False
            self._manual_stop_event.set()
        mirror_stop = self.stop_master_slave_control(emergency=True)
        try:
            self.ctrl.emergency_stop()
            return mirror_stop if not mirror_stop["ok"] else {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- manual Cartesian control -------------------------------------------

    def _manual_pose(self) -> dict:
        """Copy current feedback into browser-safe physical units."""
        if self.dry_run:
            return {"xyz_m": [0.0, 0.0, 0.0], "euler_xyz_deg": [0.0, 0.0, 0.0], "gripper_m": 0.0}
        end = self.piper.GetArmEndPoseMsgs().end_pose
        gripper = self.piper.GetArmGripperMsgs().gripper_state
        return {
            "xyz_m": [round(float(getattr(end, axis)) * 1e-6, 6) for axis in ("X_axis", "Y_axis", "Z_axis")],
            "euler_xyz_deg": [round(float(getattr(end, axis)) * 1e-3, 3) for axis in ("RX_axis", "RY_axis", "RZ_axis")],
            "gripper_m": round(float(getattr(gripper, "grippers_angle")) * 1e-6, 6),
        }

    def set_manual_control(self, enabled: bool, confirm: str = "") -> dict:
        if enabled and self.xvla_control_active():
            return {"ok": False, "error": "X-VLA held control is active; release it before changing manual control"}
        if enabled:
            ready, reason = self._verified_live_feedback()
            if not ready:
                return {"ok": False, "error": reason}
            if confirm != "ARM":
                return {"ok": False, "error": "type ARM exactly to enable manual controls"}
        with self._manual_lock:
            if enabled and self._manual_motion_active:
                return {"ok": False, "error": "the prior manual command is still cancelling; wait for it to finish"}
            self._manual_enabled = bool(enabled)
            if enabled:
                self._manual_stop_event.clear()
            else:
                self._manual_stop_event.set()
        if not enabled:
            self.revoke_xvla_control()
            self.stop_master_slave_control()
        return {"ok": True, "enabled": bool(enabled)}

    def arm_manual_control(self, confirm: str) -> dict:
        """Compatibility alias for older clients; now latches on until explicitly disabled."""
        return self.set_manual_control(True, confirm)

    def _require_manual_arm(self, *, check_motion_owner: bool = True) -> None:
        if check_motion_owner:
            self._reject_during_xvla_control()
        ready, reason = self._verified_live_feedback()
        if not ready:
            raise PermissionError(reason)
        with self._manual_lock:
            if not self._manual_enabled:
                raise PermissionError("manual controls are off; type ARM and turn them on")

    def _begin_manual_motion(self) -> None:
        """Claim exclusive manual-command ownership and start a fresh cancellation epoch."""
        self._require_manual_arm()
        with self._manual_lock:
            if self._manual_motion_active:
                raise RuntimeError("another manual motion command is still running")
            self._manual_motion_active = True
            self._manual_stop_event.clear()

    def _assert_no_manual_motion(self, action: str) -> None:
        with self._manual_lock:
            active = self._manual_motion_active
        if active:
            raise HTTPException(409, f"cannot continue master-slave/manual motion while {action}; stop motion first")

    @staticmethod
    def discover_can_devices() -> list[dict[str, str]]:
        """Read SocketCAN interfaces from sysfs without requiring netlink or sudo."""
        try:
            interfaces = sorted(CAN_SYSFS_ROOT.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise RuntimeError(f"could not enumerate network interfaces from {CAN_SYSFS_ROOT}: {exc}") from exc
        devices: list[dict[str, str]] = []
        for interface in interfaces:
            try:
                # Linux ARPHRD_CAN is 280. IFF_UP is bit 0 of sysfs flags.
                if int((interface / "type").read_text().strip()) != 280:
                    continue
                flags = int((interface / "flags").read_text().strip(), 16)
                resolved = interface.resolve()
            except (OSError, ValueError):
                continue
            usb_candidate = resolved.parent.parent.name if resolved.parent.name == "net" else ""
            usb_port = usb_candidate if re.fullmatch(r"\d+-[\d.]+:\d+\.\d+", usb_candidate) else ""
            devices.append({"name": interface.name, "state": "UP" if flags & 0x1 else "DOWN", "usb_port": usb_port})
        return devices

    def device_status(self, can_name: str) -> dict[str, Any]:
        """Read one CAN arm without sending Piper initialization or motion commands."""
        devices = {device["name"]: device for device in self.discover_can_devices()}
        if can_name not in devices:
            raise ValueError(f"CAN interface {can_name!r} is unavailable")
        result: dict[str, Any] = {"can": can_name, "interface_state": devices[can_name]["state"],
                                  "feedback_state": "UNAVAILABLE", "feedback_reason": "", "rates_hz": {}}
        if devices[can_name]["state"] != "UP":
            result["feedback_reason"] = "CAN interface is DOWN"
            return result
        try:
            if can_name == self.can:
                piper = self.piper
            else:
                with self._status_pipers_lock:
                    piper = self._status_pipers.get(can_name)
                    if piper is None:
                        from piper_sdk import C_PiperInterface_V2
                        piper = C_PiperInterface_V2(can_name)
                        _connect_piper_observer(piper)
                        self._status_pipers[can_name] = piper
            status_msg = piper.GetArmStatus()
            joint_msg = piper.GetArmJointMsgs()
            control_msg = piper.GetArmJointCtrl()
            rates = {"status": float(getattr(status_msg, "Hz", 0) or 0),
                     "joint": float(getattr(joint_msg, "Hz", 0) or 0),
                     "joint_control": float(getattr(control_msg, "Hz", 0) or 0)}
            result["rates_hz"] = rates
            joint_source = "measured" if rates["joint"] >= 1 else "commanded" if rates["joint_control"] >= 1 else None
            if not piper.isOk() or joint_source is None:
                result["feedback_reason"] = "no live joint feedback or command stream" if joint_source is None else "SDK CAN monitor is unhealthy"
                return result
            status = getattr(status_msg, "arm_status", status_msg)
            from piper_xvla.motion_watch import joint_positions_deg
            result.update(feedback_state="LIVE", feedback_reason="live joint feedback",
                          joint_source=joint_source,
                          joints_deg=list(joint_positions_deg(joint_msg)) if joint_source == "measured" else
                          [round(float(getattr(control_msg.joint_ctrl, f"joint_{index}")) * 1e-3, 3) for index in range(1, 7)],
                          ctrl_mode_name=_name(MODE_NAMES, int(getattr(status, "ctrl_mode", -1))) if rates["status"] >= 1 else "n/a",
                          arm_status_name=_name(ARM_STATUS_NAMES, int(getattr(status, "arm_status", -1))) if rates["status"] >= 1 else "n/a")
            try:
                pose_msg = piper.GetArmEndPoseMsgs()
                gripper_msg = piper.GetArmGripperMsgs()
                gripper_ctrl_msg = piper.GetArmGripperCtrl()
                result["rates_hz"].update(end_pose=float(getattr(pose_msg, "Hz", 0) or 0),
                                          gripper=float(getattr(gripper_msg, "Hz", 0) or 0),
                                          gripper_control=float(getattr(gripper_ctrl_msg, "Hz", 0) or 0))
                if result["rates_hz"]["end_pose"] >= 1:
                    pose = pose_msg.end_pose
                    result["xyz_m"] = [round(float(getattr(pose, axis)) * 1e-6, 4) for axis in ("X_axis", "Y_axis", "Z_axis")]
                if result["rates_hz"]["gripper"] >= 1:
                    result["gripper_mm"] = round(float(gripper_msg.gripper_state.grippers_angle) * 1e-3, 1)
                    result["gripper_source"] = "measured"
                elif result["rates_hz"]["gripper_control"] >= 1:
                    result["gripper_mm"] = round(float(gripper_ctrl_msg.gripper_ctrl.grippers_angle) * 1e-3, 1)
                    result["gripper_source"] = "commanded"
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                low_msg = piper.GetArmLowSpdInfoMsgs()
                result["rates_hz"]["low_speed"] = float(getattr(low_msg, "Hz", 0) or 0)
                if result["rates_hz"]["low_speed"] >= 1:
                    result.update(_joint_temperatures(low_msg))
            except (AttributeError, TypeError, ValueError):
                pass
        except Exception as exc:  # noqa: BLE001
            result["feedback_reason"] = f"{type(exc).__name__}: {exc}"
        return result

    def master_slave_status(self) -> dict[str, Any]:
        with self._master_slave_lock:
            return dict(self._master_slave_state)

    def _master_slave_lease_valid(self) -> bool:
        with self._master_slave_lock:
            return time.monotonic() < self._master_slave_lease_expires

    def renew_master_slave_control(self, session_id: str) -> dict[str, Any]:
        with self._master_slave_lock:
            if self._master_slave_state.get("status") not in {"starting", "running"}:
                raise RuntimeError("master-slave control is not active")
            if session_id != self._master_slave_state.get("session_id"):
                raise PermissionError("master-slave session does not match")
            if time.monotonic() >= self._master_slave_lease_expires:
                raise RuntimeError("master-slave control lease has expired")
            self._master_slave_lease_expires = time.monotonic() + 2.0
            return {"ok": True, "lease_ms": 2000}

    def start_master_slave_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_manual_arm()
        if self.collection_active() or self.xvla_status()["status"] in {"running", "stopping"} or self.align_status()["status"] in {"running", "stopping"}:
            raise RuntimeError("master-slave control is blocked while collection or policy inference is active")
        master_can, slave_can = str(payload.get("master_can", "")).strip(), str(payload.get("slave_can", "")).strip()
        if not master_can or not slave_can or master_can == slave_can:
            raise ValueError("choose two different CAN interfaces for master and slave")
        devices = {item["name"]: item for item in self.discover_can_devices()}
        if master_can not in devices or slave_can not in devices:
            raise ValueError("selected CAN interface is no longer available; refresh the device list")
        down = [name for name in (master_can, slave_can) if devices[name]["state"].upper() != "UP"]
        if down:
            raise ValueError("bring CAN interface(s) UP before starting: " + ", ".join(down))
        try:
            offsets = np.asarray(payload.get("offsets_deg", [0] * 6), dtype=float)
            speed_value = float(payload.get("speed_percent", 5))
            speed = int(speed_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"provide six joint offsets and a speed from 1 to {MAX_MIRROR_SPEED_PERCENT} percent") from exc
        if offsets.shape != (6,) or not np.isfinite(offsets).all():
            raise ValueError("provide six finite joint offsets in degrees")
        if np.any(np.abs(offsets) > 180.0):
            raise ValueError("each joint offset must be between −180 and 180 degrees")
        if speed_value != speed or not 1 <= speed <= MAX_MIRROR_SPEED_PERCENT:
            raise ValueError(f"master-slave speed must be 1–{MAX_MIRROR_SPEED_PERCENT} percent")
        max_joint_gap_deg = validate_joint_gap(payload.get("max_joint_gap_deg", DEFAULT_MAX_JOINT_GAP_DEG))
        with self._master_slave_lock:
            if self._master_slave_state.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("master-slave control is already active")
            self._begin_manual_motion()
            self._master_slave_stop.clear()
            self._master_slave_lease_expires = time.monotonic() + 2.0
            self._master_slave_state = {"status": "starting", "master_can": master_can,
                                        "slave_can": slave_can, "offsets_deg": offsets.tolist(),
                                        "j6_calibrated": True,
                                        "speed_percent": speed, "max_joint_gap_deg": max_joint_gap_deg,
                                        "sent": 0, "session_id": uuid.uuid4().hex,
                                        "message": "connecting to both arms"}
            self._master_slave_thread = threading.Thread(
                target=self._master_slave_worker, args=(master_can, slave_can, offsets.tolist(), speed, max_joint_gap_deg),
                name="piper-master-slave", daemon=True)
            self._master_slave_thread.start()
            return dict(self._master_slave_state)

    def _master_slave_worker(self, master_can: str, slave_can: str, offsets: list[float], speed: int,
                             max_joint_gap_deg: float = DEFAULT_MAX_JOINT_GAP_DEG) -> None:
        handles: dict[str, Any] = {}
        original_sdk_limits: dict[str, tuple[float, float]] = {}
        sdk_limit_owner: Optional[Any] = None
        try:
            for name in (master_can, slave_can):
                if name == self.can:
                    handles[name] = self.piper
                else:
                    with self._status_pipers_lock:
                        cached = self._status_pipers.get(name)
                        if cached is not None:
                            handles[name] = cached
                        elif name == master_can:
                            from piper_sdk import C_PiperInterface_V2
                            handles[name] = C_PiperInterface_V2(name)
                            _connect_piper_observer(handles[name])
                        else:
                            handles[name] = connect_live_piper(name)
            master, slave = handles[master_can], handles[slave_can]
            if master is slave:
                raise RuntimeError("master and slave must use separate Piper connections")
            # SDK releases may silently clamp outgoing targets to their
            # generic ranges. Match all six software limits to the measured
            # slave ranges for this session, then restore them on exit.
            get_sdk_limit = getattr(slave, "GetSDKJointLimitParam", None)
            set_sdk_limit = getattr(slave, "SetSDKJointLimitParam", None)
            if callable(get_sdk_limit) and callable(set_sdk_limit):
                sdk_limit_owner = slave
                for index, limits in enumerate(SLAVE_JOINT_PHYSICAL_LIMITS_DEG, 1):
                    name = f"j{index}"
                    original_sdk_limits[name] = tuple(get_sdk_limit(name))
                    set_sdk_limit(name, *np.deg2rad(limits))
                sdk_joint_limits = "calibrated"
            else:
                # The installed legacy SDK sends raw joint values and has no
                # limit API. Keep the outgoing physical-range clamp active.
                sdk_joint_limits = "SDK limit API unavailable"
            with self._master_slave_command_lock:
                self._master_slave_slave_piper = slave
            with self._master_slave_lock:
                self._master_slave_state.update(message="checking master joints, grippers and slave alignment",
                                                sdk_joint_limits=sdk_joint_limits,
                                                sdk_j6_limit=sdk_joint_limits)

            def update(sample: dict[str, Any]) -> None:
                with self._master_slave_lock:
                    self._master_slave_state.update(sample, status="running", message="mirroring master joints and gripper at 20 Hz")

            run_joint_mirror(master, slave, offsets, speed, self._master_slave_stop, update,
                             command_lock=self._master_slave_command_lock,
                             lease_ok=self._master_slave_lease_valid, mirror_gripper=True,
                             max_joint_gap_deg=max_joint_gap_deg)
            with self._master_slave_lock:
                if self._master_slave_state.get("status") != "error":
                    sent = int(self._master_slave_state.get("sent", 0))
                    self._master_slave_state.update(
                        status="stopped",
                        message="mirror stopped; slave holds its measured pose" if sent else "mirror stopped before motion",
                    )
        except Exception as exc:  # noqa: BLE001
            if self._master_slave_stop.is_set():
                with self._master_slave_lock:
                    if self._master_slave_state.get("status") != "error":
                        self._master_slave_state.update(status="stopped", message="mirror stopped")
                return
            detail = f"{type(exc).__name__}: {exc}"
            with self._master_slave_command_lock:
                with self._master_slave_lock:
                    sent = int(self._master_slave_state.get("sent", 0))
                slave = handles.get(slave_can)
                if sent and slave is not None and not self._master_slave_stop.is_set():
                    try:
                        hold_slave_position(slave)
                    except Exception as hold_exc:  # noqa: BLE001
                        detail += f"; could not hold slave ({hold_exc}); requesting E-stop"
                        try:
                            slave.EmergencyStop(0x01)
                        except Exception as estop_exc:  # noqa: BLE001
                            detail += f"; E-stop failed: {estop_exc}"
            with self._master_slave_lock:
                self._master_slave_state.update(status="error", message=detail)
        finally:
            with self._master_slave_command_lock:
                self._master_slave_slave_piper = None
                if sdk_limit_owner is not None:
                    restore_errors = []
                    for name, limits in original_sdk_limits.items():
                        try:
                            sdk_limit_owner.SetSDKJointLimitParam(name, *limits)
                        except Exception as exc:  # noqa: BLE001
                            restore_errors.append(f"{name}: {type(exc).__name__}: {exc}")
                    if restore_errors:
                        with self._master_slave_lock:
                            self._master_slave_state.update(
                                status="error", message="could not restore SDK joint limits: " + "; ".join(restore_errors))
                with self._status_pipers_lock:
                    for name, handle in handles.items():
                        if name != self.can and name not in self._status_pipers:
                            try:
                                handle.DisconnectPort()
                            except Exception:  # noqa: BLE001
                                pass
            self._finish_manual_motion()

    def stop_master_slave_control(self, *, emergency: bool = False) -> dict[str, Any]:
        failure = None
        with self._master_slave_command_lock:
            self._master_slave_stop.set()
            with self._master_slave_lock:
                active = self._master_slave_state.get("status") in {"starting", "running"}
                sent = int(self._master_slave_state.get("sent", 0))
            slave = self._master_slave_slave_piper
            if slave is not None and (active or emergency):
                try:
                    if emergency:
                        if slave is not self.piper:
                            slave.EmergencyStop(0x01)
                    elif sent:
                        hold_slave_position(slave)
                except Exception as exc:  # noqa: BLE001
                    failure = f"could not stop slave: {type(exc).__name__}: {exc}"
                    if not emergency:
                        try:
                            slave.EmergencyStop(0x01)
                            failure += "; requested slave E-stop"
                        except Exception as estop_exc:  # noqa: BLE001
                            failure += f"; slave E-stop failed: {estop_exc}"
        with self._master_slave_lock:
            if failure:
                self._master_slave_state.update(status="error", message=failure)
            elif self._master_slave_state.get("status") in {"starting", "running"}:
                self._master_slave_state.update(status="stopping", message="stopping mirror stream")
            return {"ok": not bool(failure), **self._master_slave_state, **({"error": failure} if failure else {})}

    def xvla_control_active(self) -> bool:
        with self._xvla_lease_lock:
            return time.monotonic() < self._xvla_lease_expires

    def _reject_during_xvla_control(self) -> None:
        if self.xvla_control_active():
            raise HTTPException(409, "policy held control is active; release it before using another motion control")

    def _write_xvla_lease(self, expires: float, speed_percent: int = 1, blocked: bool = False, expected_session: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        with self._xvla_lease_lock:
            if expected_session is not None and self._xvla_control_session != expected_session:
                return False
            self._xvla_lease_expires = expires
            path = self._xvla_lease_path
            if path is None:
                return True
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps({"expires_monotonic_s": expires, "speed_percent": speed_percent, "blocked": blocked, "session_id": session_id}))
            os.replace(temp, path)
            return True

    def revoke_xvla_control(self, session_id: Optional[str] = None) -> dict:
        with self._xvla_lease_lock:
            if session_id is not None and session_id != self._xvla_control_session:
                # A stop can overtake the first heartbeat. Remember that
                # pending session so its late request cannot start control.
                self._xvla_revoked_sessions.add(session_id)
                return {"ok": True, "active": time.monotonic() < self._xvla_lease_expires}
            revoked = session_id or self._xvla_control_session
            if revoked:
                self._xvla_revoked_sessions.add(revoked)
            self._xvla_control_session = None
        self._write_xvla_lease(0.0)
        return {"ok": True, "active": False}

    def renew_xvla_control(self, speed_percent: int, session_id: str, new_session: bool = False) -> dict:
        if not 1 <= int(speed_percent) <= 10:
            raise ValueError("speed_percent must be in [1, 10]")
        if not session_id or len(session_id) > 80:
            raise ValueError("a valid held-control session ID is required")
        if self.dry_run:
            raise RuntimeError("X-VLA control requires live Piper hardware")
        # A renewal belongs to the current X-VLA owner, not a competing manual
        # command. Keep the arm/feedback checks without rejecting our own lease.
        self._require_manual_arm(check_motion_owner=False)
        with self._manual_lock:
            if self._manual_motion_active:
                raise RuntimeError("X-VLA control is blocked while a manual motion command is running")
        if self.collection_active():
            raise RuntimeError("X-VLA control is blocked while camera collection is active")
        with self._xvla_lock:
            running = self._xvla_state.get("status") == "running"
        if not running or self._xvla_lease_path is None:
            raise RuntimeError("start continuous X-VLA inference before holding to send controls")
        try:
            blocked = bool(json.loads(self._xvla_lease_path.read_text()).get("blocked", False))
        except (OSError, ValueError, json.JSONDecodeError):
            blocked = False
        if blocked and not new_session:
            raise RuntimeError("the previous hold was stopped by the action guard; release and press again")
        with self._xvla_lease_lock:
            if session_id in self._xvla_revoked_sessions:
                raise RuntimeError("this held-control session was released")
            if new_session:
                self._xvla_control_session = session_id
            elif self._xvla_control_session != session_id:
                raise RuntimeError("held-control session expired; release and press again")
        latest = self.xvla_status().get("latest")
        if not latest or not latest.get("guard_allowed") or (latest.get("prediction_age_s") or 999) > 0.5:
            raise RuntimeError("latest X-VLA prediction is stale or rejected by the action guard")
        expires = time.monotonic() + 0.35
        if not self._write_xvla_lease(expires, int(speed_percent), blocked=False, expected_session=session_id, session_id=session_id):
            raise RuntimeError("held-control session expired; release and press again")
        return {"ok": True, "active": True, "lease_ms": 350}

    def _finish_manual_motion(self) -> None:
        with self._manual_lock:
            self._manual_motion_active = False

    def mark_manual_pose(self, name: str) -> dict:
        self._require_manual_arm()
        clean = str(name).strip()
        if not clean or len(clean) > 64:
            raise ValueError("mark name must contain 1–64 characters")
        mark = {"name": clean, "pose": self._manual_pose(), "created_at": time.time()}
        with self._manual_lock:
            self._manual_marks = [item for item in self._manual_marks if item["name"] != clean]
            self._manual_marks.append(mark)
        return mark

    def manual_marks(self) -> list[dict]:
        with self._manual_lock:
            return [dict(item) for item in self._manual_marks]

    def reorder_manual_marks(self, names: list[object]) -> list[dict]:
        """Persist a strict complete permutation so operator order is unambiguous."""
        requested = [str(name) for name in names]
        with self._manual_lock:
            existing = [item["name"] for item in self._manual_marks]
            if len(requested) != len(existing) or len(set(requested)) != len(requested) or set(requested) != set(existing):
                raise ValueError("names must be one complete, duplicate-free list of the current marks")
            by_name = {item["name"]: item for item in self._manual_marks}
            self._manual_marks = [by_name[name] for name in requested]
            return [dict(item) for item in self._manual_marks]

    def delete_manual_mark(self, name: str) -> list[dict]:
        clean = str(name).strip()
        with self._manual_lock:
            original = len(self._manual_marks)
            self._manual_marks = [item for item in self._manual_marks if item["name"] != clean]
            if len(self._manual_marks) == original:
                raise ValueError(f"unknown marked pose {clean!r}")
            return [dict(item) for item in self._manual_marks]

    def send_manual_endpose(self, payload: dict, *, _operation_owned: bool = False) -> dict:
        """Bounded operator target stream; requires the manual-control latch."""
        self._require_manual_arm()
        try:
            target = EndPoseTarget(
                xyz_m=np.asarray([payload[key] for key in ("x_m", "y_m", "z_m")], dtype=float),
                euler_xyz_deg=np.asarray([payload[key] for key in ("rx_deg", "ry_deg", "rz_deg")], dtype=float),
            )
            speed = int(payload.get("speed_percent", 10))
            duration_s = float(payload.get("duration_s", 1.0))
            stream_hz = float(payload.get("stream_hz", 20.0))
            max_axis_step_m = payload.get("max_axis_step_m")
            max_axis_step_m = None if max_axis_step_m is None else float(max_axis_step_m)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("end-pose needs finite x/y/z metres and rx/ry/rz degrees") from exc
        if not np.isfinite(target.xyz_m).all() or not np.isfinite(target.euler_xyz_deg).all():
            raise ValueError("end-pose values must be finite")
        if not (1 <= speed <= 10 and 0.1 <= duration_s <= 5.0 and 1 <= stream_hz <= 20):
            raise ValueError("manual limits: speed 1–10%, duration 0.1–5 s, stream 1–20 Hz")
        if max_axis_step_m is not None:
            if not np.isfinite(max_axis_step_m) or not 0.0001 <= max_axis_step_m <= 0.02:
                raise ValueError("max_axis_step_m must be within [0.0001, 0.02] m")
            current_xyz = np.asarray(self._manual_pose()["xyz_m"], dtype=float)
            axis_delta = np.abs(target.xyz_m - current_xyz)
            if np.any(axis_delta > max_axis_step_m + 1e-12):
                raise ValueError(
                    "end-pose target exceeds per-axis XYZ step guard "
                    f"({max_axis_step_m * 1000:g} mm): deltas {np.round(axis_delta * 1000, 3).tolist()} mm"
                )
        if self.dry_run:
            return {"ok": True, "simulated": True, "target": {"xyz_m": target.xyz_m.tolist(), "euler_xyz_deg": target.euler_xyz_deg.tolist()}}
        if not _operation_owned:
            self._begin_manual_motion()
        try:
            _wait_until_enabled(self.piper, enable_gripper=False, stop_event=self._manual_stop_event)
            if self._manual_stop_event.is_set():
                raise RuntimeError("manual command was cancelled before CAN mode selection")
            prepare_can_cartesian_control(self.piper, speed_percent=speed, enable_gripper=False)
            _wait_for_can_mode(self.piper, stop_event=self._manual_stop_event)
            sent = _stream_endpose(self.piper, target, speed_percent=speed, duration_s=duration_s, stream_hz=stream_hz, stop_event=self._manual_stop_event)
        finally:
            if not _operation_owned:
                self._finish_manual_motion()
        return {"ok": True, "sent": sent, "target": {"xyz_m": target.xyz_m.tolist(), "euler_xyz_deg": target.euler_xyz_deg.tolist()}}

    def send_manual_gripper(self, payload: dict) -> dict:
        self._require_manual_arm()
        try:
            position_m = float(payload["position_m"])
            effort = int(payload.get("effort", 1000))
            duration_s = float(payload.get("duration_s", 1.0))
            stream_hz = float(payload.get("stream_hz", 10.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("gripper needs position_m and optional effort/duration/stream rate") from exc
        if not np.isfinite(position_m) or not (MANUAL_GRIPPER_MIN_M <= position_m <= MANUAL_GRIPPER_MAX_M) or not (0 <= effort <= 5000):
            raise ValueError("gripper position must be within [-0.07, 0.07] m and effort within [0, 5000]")
        if not (0.1 <= duration_s <= 5.0 and 1 <= stream_hz <= 20):
            raise ValueError("gripper duration must be 0.1–5 s and stream rate 1–20 Hz")
        target_sdk = round(position_m * 1_000_000)
        sent = 0
        if not self.dry_run:
            self._begin_manual_motion()
            try:
                # Match the installed vendor demo: disable/clear stale gripper fault,
                # then enable and stream the requested stroke (CAN ID 0x159).
                self.piper.GripperCtrl(target_sdk, effort, 0x02, 0x00)
                self.piper.GripperCtrl(target_sdk, effort, 0x01, 0x00)
                deadline = time.monotonic() + duration_s
                while time.monotonic() < deadline and not self._manual_stop_event.is_set():
                    self.piper.GripperCtrl(target_sdk, effort, 0x01, 0x00)
                    sent += 1
                    time.sleep(1.0 / stream_hz)
            finally:
                self._finish_manual_motion()
        return {"ok": True, "simulated": self.dry_run, "position_m": position_m, "sdk_target": target_sdk, "effort": effort, "sent": sent}

    def jog_manual_axis(self, payload: dict) -> dict:
        """Move one small Cartesian or Euler increment from measured feedback."""
        self._require_manual_arm()
        axis = str(payload.get("axis", "")).lower()
        if axis not in {"x", "y", "z", "rx", "ry", "rz"}:
            raise ValueError("jog axis must be x, y, z, rx, ry, or rz")
        try:
            direction = int(payload.get("direction", 0))
            step = float(payload.get("step_deg" if axis.startswith("r") else "step_m", 2.0 if axis.startswith("r") else 0.002))
            speed = int(payload.get("speed_percent", 10))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid jog direction, step, or speed") from exc
        if direction not in {-1, 1}:
            raise ValueError("jog direction must be ±1")
        if axis.startswith("r") and not (0.1 <= step <= 20.0):
            raise ValueError("rotation jog step must be 0.1–20 degrees")
        if not axis.startswith("r") and not (0.0001 <= step <= 0.02):
            raise ValueError("Cartesian jog step must be 0.1–20 mm")
        pose = self._manual_pose()
        xyz, euler = list(pose["xyz_m"]), list(pose["euler_xyz_deg"])
        if axis.startswith("r"):
            euler[{"rx": 0, "ry": 1, "rz": 2}[axis]] += direction * step
        else:
            xyz[{"x": 0, "y": 1, "z": 2}[axis]] += direction * step
        return self.send_manual_endpose({
            "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2],
            "rx_deg": euler[0], "ry_deg": euler[1], "rz_deg": euler[2],
            "speed_percent": speed, "duration_s": 0.15, "stream_hz": 20,
        })

    def send_manual_joints(self, payload: dict) -> dict:
        """Bounded operator-selected MOVE-J target using documented SDK limits."""
        self._require_manual_arm()
        try:
            joints = np.asarray(payload["joints_deg"], dtype=float)
            speed = int(payload.get("speed_percent", 10))
            duration_s = float(payload.get("duration_s", 1.0))
            stream_hz = float(payload.get("stream_hz", 20.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("joints needs six finite joint angles in degrees") from exc
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("joints needs six finite joint angles in degrees")
        for index, (value, (low, high)) in enumerate(zip(joints, JOINT_LIMITS_DEG), start=1):
            if not low <= value <= high:
                raise ValueError(f"joint {index} must be within [{low:g}, {high:g}] degrees")
        if not (1 <= speed <= 10 and 0.1 <= duration_s <= 5.0 and 1 <= stream_hz <= 20):
            raise ValueError("manual limits: speed 1–10%, duration 0.1–5 s, stream 1–20 Hz")
        target = tuple(np.rint(joints * 1000).astype(int).tolist())
        result = {"ok": True, "simulated": self.dry_run, "target": {"joints_deg": joints.tolist()}, "sdk_target": list(target)}
        if self.dry_run:
            return result
        self._begin_manual_motion()
        try:
            _wait_until_enabled(self.piper, enable_gripper=False, stop_event=self._manual_stop_event)
            deadline, sent = time.monotonic() + duration_s, 0
            while time.monotonic() < deadline and not self._manual_stop_event.is_set():
                self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=speed, is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
                self.piper.JointCtrl(*target)
                sent += 1
                time.sleep(1.0 / stream_hz)
            result["sent"] = sent
        finally:
            self._finish_manual_motion()
        return result

    def run_marked_trajectory(self, payload: dict) -> dict:
        """Run the currently marked poses in their displayed order, each bounded."""
        self._require_manual_arm()
        requested_names = payload.get("names")
        with self._manual_lock:
            all_marks = [dict(item) for item in self._manual_marks]
        if requested_names is None:
            marks = all_marks
        elif isinstance(requested_names, list) and requested_names and len(set(map(str, requested_names))) == len(requested_names):
            by_name = {item["name"]: item for item in all_marks}
            try:
                marks = [by_name[str(name)] for name in requested_names]
            except KeyError as exc:
                raise ValueError(f"unknown marked pose {exc.args[0]!r}") from exc
        else:
            raise ValueError("names must be a non-empty duplicate-free list of marked-pose names")
        if not marks:
            raise ValueError("mark at least one pose before running a trajectory")
        shared = {key: payload[key] for key in ("speed_percent", "duration_s", "stream_hz") if key in payload}
        results = []
        if not self.dry_run:
            self._begin_manual_motion()
        try:
            for mark in marks:
                if self._manual_stop_event.is_set():
                    raise RuntimeError("manual trajectory was cancelled")
                pose = mark["pose"]
                command = {
                    "x_m": pose["xyz_m"][0], "y_m": pose["xyz_m"][1], "z_m": pose["xyz_m"][2],
                    "rx_deg": pose["euler_xyz_deg"][0], "ry_deg": pose["euler_xyz_deg"][1], "rz_deg": pose["euler_xyz_deg"][2],
                    **shared,
                }
                results.append(self.send_manual_endpose(command, _operation_owned=not self.dry_run))
        finally:
            if not self.dry_run:
                self._finish_manual_motion()
        return {"ok": True, "marks": [mark["name"] for mark in marks], "results": results}

    # -- X-VLA inference and guard configuration ---------------------------

    def xvla_checkpoints(self) -> dict:
        """List local torch checkpoints without loading their multi-GB weights."""
        root = XVLA_OUTPUTS_ROOT.resolve()
        checkpoints = []
        if root.is_dir():
            # rglob(root) does not descend into symlinked output directories.
            # Inspect each top-level entry so mounted checkpoint folders appear.
            for entry in root.iterdir():
                candidates = entry.rglob("*") if entry.is_dir() else (entry,)
                linked_directory = entry.is_symlink() and entry.is_dir()
                for path in candidates:
                    if path.suffix.lower() not in {".pt", ".pth", ".ckpt"} or not path.is_file():
                        continue
                    if not linked_directory and not path.resolve().is_relative_to(root):
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    checkpoints.append({
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": stat.st_size,
                        "modified_at": stat.st_mtime,
                    })
        checkpoints.sort(key=lambda item: (-item["modified_at"], item["path"]))
        names = {item["path"] for item in checkpoints}
        default = XVLA_DEFAULT_CHECKPOINT if XVLA_DEFAULT_CHECKPOINT in names else (checkpoints[0]["path"] if checkpoints else None)
        return {"root": str(root), "checkpoints": checkpoints, "default": default}

    def _resolve_xvla_checkpoint(self, selected: Optional[str]) -> tuple[str, Path]:
        choice = selected.strip() if isinstance(selected, str) else None
        if not choice:
            choice = self.xvla_checkpoints()["default"]
        if not choice:
            raise ValueError("No X-VLA checkpoint found; enter an absolute path to a .pt, .pth or .ckpt file")
        candidate = Path(choice).expanduser()
        if not candidate.is_absolute():
            if ".." in candidate.parts:
                raise ValueError("relative checkpoint paths must stay within the outputs folder; use an absolute path")
            candidate = XVLA_OUTPUTS_ROOT / candidate
        path = candidate.resolve()
        if path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            raise ValueError(f"X-VLA checkpoint must be a .pt, .pth or .ckpt file: {choice}")
        if not path.is_file():
            raise ValueError(f"X-VLA checkpoint file not found: {choice}")
        try:
            with path.open("rb"):
                pass
        except OSError as exc:
            raise ValueError(f"X-VLA checkpoint cannot be read: {path}: {exc}") from exc
        return choice, path

    def xvla_guard_config(self) -> dict:
        payload = json.loads(XVLA_GUARD_CONFIG.read_text())
        guard_limits_from_config(payload)
        return payload

    def save_xvla_guard_config(self, payload: dict) -> dict:
        limits = guard_limits_from_config(payload)
        canonical = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in limits.__dict__.items()
        }
        # Changing limits ends the current hold, including any late heartbeats.
        # The runner reloads the atomic config before validating its next action.
        self.revoke_xvla_control()
        with self._xvla_lock:
            temp = XVLA_GUARD_CONFIG.with_suffix(".json.tmp")
            temp.write_text(json.dumps(canonical, indent=2, allow_nan=False) + "\n")
            os.replace(temp, XVLA_GUARD_CONFIG)
        return {"ok": True, "limits": canonical, "message": "Guard limits saved. Held control stopped; new limits apply on the next prediction."}

    def xvla_gripper_config(self) -> dict:
        return load_binary_gripper_config(XVLA_GRIPPER_CONFIG).as_dict()

    def xvla_gripper_calibration(self) -> dict:
        inference_config = json.loads((MODULE_DIR / "config" / "piper_xvla_single_task.json").read_text())
        manifest_path = Path(inference_config["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = MODULE_DIR.parent / manifest_path
        return json.loads(manifest_path.read_text())["gripper_normalization"]

    def save_xvla_gripper_config(self, payload: dict) -> dict:
        config = BinaryGripperConfig.from_dict(payload)
        calibration = self.xvla_gripper_calibration()
        config.validate_calibration(calibration["raw_meters_min"], calibration["raw_meters_max"])
        self.revoke_xvla_control()
        with self._xvla_lock:
            temp = XVLA_GRIPPER_CONFIG.with_suffix(".json.tmp")
            temp.write_text(json.dumps(config.as_dict(), indent=2, allow_nan=False) + "\n")
            os.replace(temp, XVLA_GRIPPER_CONFIG)
        return {"ok": True, "mapping": config.as_dict(), "message": "Gripper mapping saved. Held control stopped; new values apply on the next prediction."}

    def xvla_status(self) -> dict:
        with self._xvla_lock:
            state = dict(self._xvla_state)
        output = state.get("output")
        if not output:
            return state
        log_path = Path(str(output)).with_suffix(".log")
        state["log_lines"] = []
        with self._xvla_lock:
            offset = self._xvla_log_offset
            if log_path.is_file():
                with log_path.open("rb") as log_stream:
                    log_stream.seek(offset)
                    raw_logs = log_stream.read()
                    self._xvla_log_offset = log_stream.tell()
                state["log_lines"] = [line.decode("utf-8", "replace") for line in raw_logs.splitlines() if line.strip()]
        try:
            with Path(str(output)).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                chunk_size = 8192
                data = b""
                position = end
                while position > 0 and b"\n" not in data:
                    size = min(chunk_size, position)
                    position -= size
                    stream.seek(position)
                    data = stream.read(size) + data
                lines = data.splitlines()
            row = json.loads(lines[-1]) if lines else None
            if isinstance(row, dict):
                completed_at = row.get("inference_completed_monotonic_s")
                prediction_age = None
                if isinstance(completed_at, (int, float)) and np.isfinite(completed_at):
                    prediction_age = max(0.0, time.monotonic() - float(completed_at))
                state["latest"] = {
                    "frame": int(row.get("frame", 0)),
                    "guard_allowed": bool(row.get("guard_allowed")),
                    "prediction_age_s": prediction_age,
                    "predicted_pose": _active10_pose(row.get("guarded_action20", row.get("predicted_action20"))),
                    "predicted_gripper_mm": row.get("predicted_gripper_mm"),
                    "gripper_target_mm": row.get("gripper_target_mm"),
                    "gripper_mapping_mm": row.get("gripper_mapping_mm"),
                    "measured_pose": _active10_pose(row.get("current_active10")),
                    "alerts": [str(alert.get("code", "UNKNOWN")) for alert in row.get("alerts", []) if isinstance(alert, dict)],
                    "alert_messages": [f"{alert.get('code', 'UNKNOWN')}: {alert.get('message', '')}" for alert in row.get("alerts", []) if isinstance(alert, dict)],
                    "loop_hz": row.get("loop_hz"),
                    "inference_ms": row.get("inference_ms"),
                    "control_hz": row.get("control_hz"),
                    "control_active": row.get("control_active", False),
                    "control_sent": row.get("control_sent", False),
                    "deadline_missed": row.get("deadline_missed", 0),
                }
                state["recording"] = row.get("recording")
        except (OSError, json.JSONDecodeError, TypeError):
            # The worker may be flushing a line; preserve the prior good display.
            pass
        if state.get("recording_dir") and state["status"] not in {"running", "stopping"}:
            try:
                state["recording"] = json.loads((Path(state["recording_dir"]) / "recording.json").read_text())
            except (OSError, ValueError):
                pass
        return state

    def start_xvla_diagnostic(self, frames: int = 25, record_images: bool = False, checkpoint: Optional[str] = None) -> dict:
        if not 0 <= frames <= 100:
            raise ValueError("X-VLA diagnostic frames must be in [0, 100]; 0 means continuous until Stop")
        if self.dry_run:
            raise RuntimeError("X-VLA diagnostic requires real camera and CAN feedback; WebUI is in --dry-run mode")
        if self.collection_active():
            raise RuntimeError("X-VLA diagnostic is blocked while collection owns the cameras")
        if self.align_status()["status"] in {"running", "stopping"}:
            raise RuntimeError("stop ALIGN inference before starting X-VLA")
        checkpoint_name, checkpoint_path = self._resolve_xvla_checkpoint(checkpoint)
        with self._xvla_lock:
            if self._xvla_state["status"] in {"running", "stopping"}:
                raise RuntimeError("X-VLA diagnostic is already running")
            # Clear the previous run before starting so hold-to-control cannot
            # send its last allowed prediction while this run is initializing.
            output = MODULE_DIR.parent / "outputs" / "piper_xvla_single_task" / "webui_camera_only.jsonl"
            output.unlink(missing_ok=True)
            output.with_suffix(".log").unlink(missing_ok=True)
            self._xvla_lease_path = output.with_suffix(".lease.json")
            self._write_xvla_lease(0.0)
            self._xvla_log_offset = 0
            mode_label = "continuous" if frames == 0 else f"{frames} frames"
            recording_dir = None
            if record_images:
                recording_dir = output.parent / "inference_recordings" / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self._xvla_stop_event.clear()
            self._xvla_state = {"status": "running", "frames": frames, "output": str(output), "checkpoint": checkpoint_name, "message": f"loading {checkpoint_name}; opening cameras and reading CAN feedback ({mode_label})...", "summary": None, "motion_capability": "none", "recording_dir": str(recording_dir) if recording_dir else None}
        threading.Thread(target=self._xvla_worker, args=(frames, output, recording_dir, checkpoint_path), daemon=True).start()
        return self.xvla_status()

    def stop_xvla_diagnostic(self) -> dict:
        with self._xvla_lock:
            process = self._xvla_process
            if self._xvla_state["status"] != "running":
                return {"ok": False, "error": "no running X-VLA diagnostic"}
            self.revoke_xvla_control()
            self._xvla_stop_event.set()
            self._xvla_state.update(status="stopping", message="stopping camera-only diagnostic; waiting for camera release...")
            if process is not None:
                process.terminate()
            return dict(self._xvla_state, ok=True)

    def _xvla_worker(self, frames: int, output: Path, recording_dir: Optional[Path], checkpoint_path: Path) -> None:
        try:
            # Prevent collection/preview from opening the cameras at the same time.
            with self._camera_lock:
                command = [
                    sys.executable, "-m", "piper_xvla.live_inference",
                    "--config", "piper_xvla/config/piper_xvla_single_task.json",
                    "--checkpoint", str(checkpoint_path),
                    "--guard-config", str(XVLA_GUARD_CONFIG),
                    "--binary-gripper-config", str(XVLA_GRIPPER_CONFIG),
                    "--can", self.can, "--device", "cuda", "--frames", str(frames), "--output", str(output),
                    "--enable-held-control", "--hold-lease", str(output.with_suffix(".lease.json")),
                ]
                if recording_dir is not None:
                    command.extend(["--record-images-dir", str(recording_dir)])
                log_path = output.with_suffix(".log")
                with log_path.open("w") as log_stream:
                    process = subprocess.Popen(command, cwd=MODULE_DIR.parent, env=_xvla_subprocess_env(), text=True, stdout=log_stream, stderr=subprocess.STDOUT)
                    with self._xvla_lock:
                        self._xvla_process = process
                        if self._xvla_stop_event.is_set():
                            process.terminate()
                    process.wait()
            with self._xvla_lock:
                stopping = self._xvla_state["status"] == "stopping"
                self._xvla_process = None
            if stopping:
                self.revoke_xvla_control()
                with self._xvla_lock:
                    self._xvla_state.update(status="stopped", message="X-VLA loop stopped; held-control lease revoked")
                return
            if process.returncode != 0:
                raise RuntimeError(log_path.read_text(errors="replace")[-2000:] or "X-VLA inference failed")
            allowed = frames_seen = 0
            alerts: dict[str, int] = {}
            with output.open() as rows:
                for line in rows:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    frames_seen += 1
                    allowed += bool(row.get("guard_allowed"))
                    for alert in row.get("alerts", []):
                        code = str(alert.get("code", "UNKNOWN"))
                        alerts[code] = alerts.get(code, 0) + 1
            summary = {"allowed": allowed, "frames": frames_seen, "alert_counts": alerts}
            with self._xvla_lock:
                self._xvla_state.update(status="done", message="X-VLA inference complete; held-control lease revoked", summary=summary)
            self.revoke_xvla_control()
        except Exception as exc:  # noqa: BLE001
            with self._xvla_lock:
                self._xvla_process = None
                self._xvla_state.update(status="error", message=f"{type(exc).__name__}: {exc}")
            self.revoke_xvla_control()

    # -- ALIGN intention-model inference ----------------------------------

    def align_status(self) -> dict:
        with self._align_lock:
            state = dict(self._align_state)
            state["max_camera_age_s"] = self.xvla_guard_config()["max_camera_age_s"]
            offset = self._align_log_offset
            output = state.get("output")
            if not output:
                return state
            log_path = Path(output).with_suffix(".log")
            if log_path.is_file():
                with log_path.open("rb") as stream:
                    stream.seek(offset)
                    state["log_lines"] = [line.decode("utf-8", "replace") for line in stream.read().splitlines() if line.strip()]
                    self._align_log_offset = stream.tell()
        try:
            with Path(output).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                stream.seek(max(0, end - 16384))
                lines = stream.read().splitlines()
            if lines:
                row = json.loads(lines[-1])
                state["latest"] = row
                sampled_age = row.get("prediction_age_s")
                timestamp = row.get("timestamp_monotonic_s")
                state["latest"]["prediction_age_s"] = (
                    max(0.0, float(sampled_age) + time.monotonic() - float(timestamp))
                    if isinstance(sampled_age, (int, float)) and isinstance(timestamp, (int, float)) else None
                )
        except (OSError, ValueError, TypeError):
            pass
        return state

    def start_align_inference(self, payload: dict) -> dict:
        if self.dry_run:
            raise RuntimeError("ALIGN inference needs live Piper and cameras")
        if self.collection_active() or self.xvla_status()["status"] in {"running", "stopping"}:
            raise RuntimeError("stop collection or X-VLA inference before starting ALIGN")
        with self._master_slave_lock:
            if self._master_slave_state.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("stop master-slave mirroring before starting ALIGN")
        checkpoint_value = payload.get("checkpoint") or str(ALIGN_DEFAULT_CHECKPOINT)
        checkpoint = Path(str(checkpoint_value)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = MODULE_DIR.parents[1] / checkpoint
        checkpoint = checkpoint.resolve()
        if checkpoint.suffix != ".pt" or not checkpoint.is_file():
            raise ValueError(f"ALIGN checkpoint must be an existing .pt file: {checkpoint}")
        settings = ALIGNInferenceSettings.from_dict(payload.get("settings", {}))
        if not ALIGN_CALIBRATION_MANIFEST.is_file():
            raise ValueError(f"ALIGN gripper calibration manifest is missing: {ALIGN_CALIBRATION_MANIFEST}")
        with self._align_lock:
            if self._align_state["status"] in {"running", "stopping"}:
                raise RuntimeError("ALIGN inference is already running")
            output = MODULE_DIR.parent / "outputs" / "piper_align" / "webui_live.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
            output.with_suffix(".log").unlink(missing_ok=True)
            self.revoke_xvla_control()
            self._xvla_lease_path = output.with_suffix(".lease.json")
            self._write_xvla_lease(0.0)
            self._align_log_offset = 0
            self._align_stop_event.clear()
            self._align_state = {"status": "running", "output": str(output), "checkpoint": str(checkpoint),
                                 "settings": settings.as_dict(), "message": "loading ALIGN checkpoint and cameras"}
        threading.Thread(target=self._align_worker, args=(checkpoint, settings, output), daemon=True).start()
        return self.align_status()

    def _align_worker(self, checkpoint: Path, settings: ALIGNInferenceSettings, output: Path) -> None:
        log_path = output.with_suffix(".log")
        final_status, final_message = "error", "ALIGN inference did not start"
        try:
            align_python = Path(os.environ.get("ALIGN_PYTHON", "/home/ucluser/miniconda3/envs/align/bin/python"))
            if not align_python.is_file():
                raise RuntimeError(f"ALIGN Python interpreter not found: {align_python}; set ALIGN_PYTHON")
            env = os.environ.copy()
            env.pop("PYTHONNOUSERSITE", None)
            env["PYTHONPATH"] = os.pathsep.join((str(align_python.parent.parent / "lib" / "python3.10" / "site-packages"),
                                                  str(MODULE_DIR.parents[1].parent), str(MODULE_DIR.parents[1]), str(MODULE_DIR.parent)))
            env["PYTHONUNBUFFERED"] = "1"
            command = [str(align_python), "-m", "piper_xvla.align_live_inference",
                       "--checkpoint", str(checkpoint), "--can", self.can,
                       "--camera-config", str(DEFAULT_CAMERA_CONFIG),
                       "--calibration-manifest", str(ALIGN_CALIBRATION_MANIFEST),
                       "--guard-config", str(XVLA_GUARD_CONFIG),
                       "--settings", json.dumps(settings.as_dict()),
                       "--lease", str(output.with_suffix(".lease.json")), "--output", str(output)]
            with self._camera_lock, log_path.open("w") as log_stream:
                process = subprocess.Popen(command, cwd=MODULE_DIR.parent, env=env, text=True,
                                           stdout=log_stream, stderr=subprocess.STDOUT)
                with self._align_lock:
                    self._align_process = process
                    if self._align_stop_event.is_set():
                        process.terminate()
                process.wait()
            stopping = self._align_stop_event.is_set()
            final_status = "stopped" if stopping else "done" if process.returncode == 0 else "error"
            final_message = ("ALIGN inference stopped" if stopping else "ALIGN inference ended" if process.returncode == 0
                             else log_path.read_text(errors="replace")[-1600:])
        except Exception as exc:  # noqa: BLE001
            final_message = f"{type(exc).__name__}: {exc}"
        finally:
            if self._xvla_lease_path == output.with_suffix(".lease.json"):
                self.revoke_xvla_control()
            with self._align_lock:
                self._align_process = None
                self._align_state.update(status=final_status, message=final_message)

    def stop_align_inference(self, *, emergency: bool = False) -> dict:
        with self._align_lock:
            if self._align_state["status"] != "running":
                return {"ok": False, "error": "ALIGN inference is not running"}
            self._align_state.update(status="stopping", message="stopping ALIGN inference")
        if not emergency:
            # Wait briefly for the 20 Hz sender to replace the last future
            # target with the measured pose before terminating the process.
            self.revoke_xvla_control()
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                row = self.align_status().get("latest") or {}
                if row.get("control_release_hold_sent") or (row and not row.get("control_active") and not row.get("control_sent")):
                    break
                time.sleep(0.02)
        with self._align_lock:
            self._align_stop_event.set()
            if self._align_process is not None:
                self._align_process.terminate()
            if emergency:
                self.revoke_xvla_control()
            return {"ok": True, **self._align_state}

    def renew_align_control(self, speed_percent: int, session_id: str, new_session: bool) -> dict:
        status = self.align_status()
        if status["status"] != "running":
            raise RuntimeError("start ALIGN inference before holding control")
        if not 1 <= speed_percent <= 10 or not session_id or len(session_id) > 80:
            raise ValueError("control needs a session ID and speed from 1 to 10%")
        self._require_manual_arm(check_motion_owner=False)
        if self.collection_active():
            raise RuntimeError("collection owns the cameras")
        with self._manual_lock:
            if self._manual_motion_active:
                raise RuntimeError("a manual motion is running")
        latest = status.get("latest") or {}
        if not new_session and latest.get("control_blocked") and latest.get("control_block_session") == session_id:
            raise RuntimeError(f"the action guard stopped control: {latest.get('control_block_reason') or 'a command was rejected'}")
        max_age_s = self.xvla_guard_config()["max_camera_age_s"]
        age_s = latest.get("prediction_age_s")
        fresh_approved = (bool(latest.get("guard_allowed")) and isinstance(age_s, (int, float))
                          and np.isfinite(age_s) and age_s <= max_age_s)
        if latest.get("guard_allowed"):
            rejection_codes = ["STALE_CAMERA"] if isinstance(age_s, (int, float)) and age_s > max_age_s else []
        else:
            rejection_codes = [str(alert.get("code")) for alert in latest.get("alerts", [])
                               if alert.get("severity") == "REJECT"]
        camera_pause = (not new_session and camera_stale_pause_allowed(rejection_codes, age_s, max_age_s))
        if not fresh_approved and not camera_pause:
            raise RuntimeError(f"ALIGN prediction is stale or rejected by the action guard (age={age_s!r} s; max={max_age_s:g} s)")
        try:
            lease = json.loads(self._xvla_lease_path.read_text())
            if not isinstance(lease, dict):
                lease = {}
            blocked = bool(lease.get("blocked"))
        except (OSError, ValueError, TypeError):
            blocked = False
            lease = {}
        if blocked and not new_session:
            reason = lease.get("reason") or "the action guard rejected a command"
            raise RuntimeError(f"the action guard stopped control: {reason}")
        with self._xvla_lease_lock:
            if session_id in self._xvla_revoked_sessions:
                raise RuntimeError("held-control session was released")
            if new_session:
                self._xvla_control_session = session_id
            elif self._xvla_control_session != session_id:
                raise RuntimeError("held-control session expired; release and press again")
        if not self._write_xvla_lease(time.monotonic() + 0.35, speed_percent, expected_session=session_id, session_id=session_id):
            raise RuntimeError("held-control session expired")
        return {"ok": True, "active": True, "lease_ms": 350}

    # -- background collect --------------------------------------------------

    def start_collect(self, window_s: float = 40.0, dataset_path: str | None = None) -> dict:
        if self.align_status()["status"] in {"running", "stopping"} or self.xvla_status()["status"] in {"running", "stopping"}:
            raise HTTPException(409, "stop policy inference before starting collection")
        if not self.task.strip():
            raise HTTPException(400, "task must be set before starting collect (PUT /api/task)")
        with self._collect_lock:
            if self._collect_state["status"] in ("opening", "recording", "stopping", "finalizing"):
                raise HTTPException(409, "a collect is already running")
            if dataset_path is None:
                root = self.dataset_root
            else:
                if not isinstance(dataset_path, str) or not dataset_path.strip():
                    raise ValueError("dataset_path must be a non-empty absolute folder path")
                root = Path(dataset_path.strip()).expanduser()
                if not root.is_absolute():
                    raise ValueError("dataset_path must be an absolute folder path on the Web UI host")
                root = root.resolve()
            if root.exists():
                if not root.is_dir():
                    raise ValueError(f"dataset path is not a folder: {root}")
                if any(root.iterdir()) and not (root / "meta" / "info.json").is_file():
                    raise ValueError(f"dataset folder is not empty or a valid LeRobot dataset: {root}")
            self.dataset_root = root
            self._collect_stop.clear()
            self._collect_state = {
                "status": "opening", "frames": 0, "elapsed_s": 0.0,
                "window_s": window_s, "episode_index": None,
                "video_path": None, "dataset_path": str(root),
                "message": "opening cameras + dataset...",
            }
            self._collect_thread = threading.Thread(target=self._collect_worker, args=(window_s, root, self.task), daemon=True)
            self._collect_thread.start()
        return {"ok": True, "state": dict(self._collect_state)}

    def _collect_worker(self, window_s: float, dataset_root: Path, task: str) -> None:
        from piper_xvla.collect_session import CollectSession
        period = 0.05
        if not self._camera_lock.acquire(timeout=5.0):
            with self._collect_lock:
                self._collect_state.update(status="error", message="camera preview did not release devices within 5 s")
            return
        try:
            session = CollectSession.open(self.piper, task, self.data_dir, dataset_root=dataset_root)
        except Exception as exc:  # noqa: BLE001
            self._camera_lock.release()
            with self._collect_lock:
                self._collect_state.update(status="error", message=f"open failed: {exc}")
            return

        with self._collect_lock:
            stopping = self._collect_stop.is_set()
            self._collect_state["status"] = "stopping" if stopping else "recording"
            self._collect_state["episode_index"] = session.episode_index
            self._collect_state["video_path"] = str(session.video_path) if session.video_ok else None
            self._collect_state["message"] = (
                f"stop requested; capturing first frame of episode {session.episode_index}..."
                if stopping else f"recording episode {session.episode_index}..."
            )

        start = time.monotonic()
        observations = 0
        capture_error = None
        try:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if observations and (elapsed >= window_s or self._collect_stop.is_set()):
                    break
                try:
                    obs = session.snapshot()
                    session.write_frame(obs)
                except Exception as exc:  # noqa: BLE001
                    capture_error = f"capture failed: {exc}"
                    break
                observations += 1
                with self._collect_lock:
                    self._collect_state["frames"] = session.frames_written
                    self._collect_state["elapsed_s"] = round(elapsed, 2)
                # Pace to 20 Hz.
                target = (int((time.monotonic() - start) / period) + 1) * period
                sleep_for = target - (time.monotonic() - start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except Exception as exc:  # noqa: BLE001
            capture_error = f"unexpected: {exc}"

        if capture_error and not observations:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - preserve the capture error for the operator
                pass
            finally:
                with self._collect_lock:
                    self._collect_state.update(status="error", message=capture_error)
                self._camera_lock.release()
            return

        with self._collect_lock:
            self._collect_state["status"] = "finalizing"
            self._collect_state["message"] = "finalizing episode; this can take a moment..."
        try:
            total = session.finalize()
            reason = "stopped by user" if self._collect_stop.is_set() else "window ended"
            with self._collect_lock:
                if capture_error:
                    self._collect_state.update(status="error", frames=total, message=f"{capture_error}; partial episode saved: {total} frames")
                else:
                    self._collect_state.update(
                        status="done", frames=total,
                        message=f"episode written ({reason}): {total} frames @20 Hz",
                    )
        except Exception as exc:  # noqa: BLE001
            with self._collect_lock:
                self._collect_state.update(status="error", message=f"finalize failed: {exc}")
        finally:
            self._camera_lock.release()

    def stop_collect(self) -> dict:
        with self._collect_lock:
            status = self._collect_state["status"]
            if status in ("stopping", "finalizing"):
                return {"ok": True, "state": dict(self._collect_state)}
            if status not in ("recording", "opening"):
                return {"ok": False, "error": f"no active collect (status={self._collect_state['status']})"}
            self._collect_stop.set()
            self._collect_state.update(
                status="stopping",
                message="stop requested; waiting for dataset to open..." if status == "opening"
                else "stop requested; finishing current episode...",
            )
            return {"ok": True, "state": dict(self._collect_state)}

    def collect_status(self) -> dict:
        with self._collect_lock:
            return dict(self._collect_state)


def _load_dataset_cached(app_state: dict, root: Path):
    """Return dataset metadata without generating a full Arrow frame cache."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    from piper_xvla.collect_session import ensure_writable_datasets_cache
    ensure_writable_datasets_cache()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    key = str(root.resolve())
    cached = app_state.get("dataset")
    stamp = info_path.stat().st_mtime_ns
    if cached is not None and app_state.get("dataset_root") == key and app_state.get("dataset_stamp") == stamp:
        return cached
    try:
        meta = LeRobotDatasetMetadata("local/piper-replay", root=key)
        ds = SimpleNamespace(meta=meta, root=meta.root, num_episodes=meta.total_episodes,
                             num_frames=meta.total_frames, fps=meta.fps)
        app_state["dataset"] = ds
        app_state["dataset_root"] = key
        app_state["dataset_stamp"] = stamp
        return ds
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"could not open dataset at {root}: {type(exc).__name__}: {exc}")


def _episode_table(ds, episode_index: int, columns: list[str]):
    """Read only requested columns for an episode directly from its Parquet file."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    row = ds.meta.episodes[episode_index]
    rel_path = ds.meta.data_path.format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
    )
    table = pq.read_table(Path(ds.root) / rel_path, columns=[*columns, "episode_index"])
    return table.filter(pc.equal(table["episode_index"], episode_index))


def _episode_thumbnails(ds, episode_index: int, max_frames: int = 3) -> list[str]:
    """Return base64 JPEG thumbnails sampled across one episode."""
    import cv2
    table = _episode_table(ds, episode_index, ["observation.images.global_rgb"])
    length = table.num_rows
    if not length:
        return []
    picks = sorted({0, length // 2, length - 1})[:max_frames]
    out: list[str] = []
    for idx in picks:
        try:
            image = table["observation.images.global_rgb"][idx].as_py()
            raw = image.get("bytes")
            if raw is not None:
                g = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                image_path = Path(image["path"])
                if not image_path.is_absolute():
                    image_path = Path(ds.root) / image_path
                g = cv2.imread(str(image_path))
            if g is None:
                continue
            g = cv2.resize(g, (320, 240))
            _, jpg = cv2.imencode(".jpg", g, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            out.append(base64.b64encode(jpg.tobytes()).decode())
        except Exception:  # noqa: BLE001
            continue
    return out


def _episode_state_action(ds, episode_index: int, max_points: int = 300) -> dict:
    """Downsample state/action vectors across one episode for charting."""
    table = _episode_table(ds, episode_index, ["observation.state", "action"])
    meta_row = ds.meta.episodes[episode_index]
    from_idx = int(meta_row["dataset_from_index"])
    n = table.num_rows
    if not n:
        return {"indices": [], "states": [], "actions": []}
    step = max(1, n // max_points)
    offsets = list(range(0, n, step))
    if offsets[-1] != n - 1:
        offsets.append(n - 1)
    idxs = [from_idx + i for i in offsets]
    states = [table["observation.state"][i].as_py() for i in offsets]
    actions = [table["action"][i].as_py() for i in offsets]
    return {"indices": idxs, "states": states, "actions": actions}


def build_app(piper_ui: PiperWebUI, data_dir: Path) -> FastAPI:
    app = FastAPI(title="Piper X-VLA Web UI")
    state: dict[str, Any] = {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if not FRONTEND_HTML.is_file():
            raise HTTPException(500, f"frontend HTML missing at {FRONTEND_HTML}")
        return FRONTEND_HTML.read_text()

    @app.get("/api/status")
    def api_status() -> dict:
        return piper_ui.snapshot_status()

    @app.get("/api/xvla/status")
    def api_xvla_status() -> dict:
        return piper_ui.xvla_status()

    @app.get("/api/align/status")
    def api_align_status() -> dict:
        return piper_ui.align_status()

    @app.post("/api/align/start")
    async def api_align_start(req: Request) -> dict:
        try:
            return piper_ui.start_align_inference(await req.json())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/align/stop")
    def api_align_stop() -> dict:
        result = piper_ui.stop_align_inference()
        if not result["ok"]:
            raise HTTPException(409, result["error"])
        return result

    @app.post("/api/align/control/heartbeat")
    async def api_align_control_heartbeat(req: Request) -> dict:
        try:
            body = await req.json()
            return piper_ui.renew_align_control(int(body.get("speed_percent", 5)), str(body.get("session_id", "")), bool(body.get("new_session", False)))
        except (TypeError, ValueError, PermissionError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/align/control/stop")
    async def api_align_control_stop(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        session = body.get("session_id") if isinstance(body, dict) else None
        return piper_ui.revoke_xvla_control(session if isinstance(session, str) and session else None)

    @app.get("/api/xvla/checkpoints")
    def api_xvla_checkpoints() -> dict:
        return piper_ui.xvla_checkpoints()

    @app.get("/api/xvla/guard")
    def api_xvla_guard() -> dict:
        return {"limits": piper_ui.xvla_guard_config()}

    @app.put("/api/xvla/guard")
    async def api_xvla_guard_save(req: Request) -> dict:
        try:
            return piper_ui.save_xvla_guard_config(await req.json())
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/xvla/gripper")
    def api_xvla_gripper() -> dict:
        return {"mapping": piper_ui.xvla_gripper_config(), "calibration": piper_ui.xvla_gripper_calibration()}

    @app.put("/api/xvla/gripper")
    async def api_xvla_gripper_save(req: Request) -> dict:
        try:
            return piper_ui.save_xvla_gripper_config(await req.json())
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/xvla/diagnostic")
    async def api_xvla_diagnostic(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        try:
            return piper_ui.start_xvla_diagnostic(int(body.get("frames", 25)), bool(body.get("record_images", False)), body.get("checkpoint"))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/xvla/stop")
    def api_xvla_stop() -> dict:
        result = piper_ui.stop_xvla_diagnostic()
        if not result["ok"]:
            raise HTTPException(409, result["error"])
        return result

    @app.post("/api/xvla/control/heartbeat")
    async def api_xvla_control_heartbeat(req: Request) -> dict:
        try:
            body = await req.json()
            return piper_ui.renew_xvla_control(int(body.get("speed_percent", 5)), str(body.get("session_id", "")), bool(body.get("new_session", False)))
        except (TypeError, ValueError, PermissionError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/xvla/control/stop")
    async def api_xvla_control_stop(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        session_id = body.get("session_id") if isinstance(body, dict) else None
        return piper_ui.revoke_xvla_control(session_id if isinstance(session_id, str) and session_id else None)

    @app.get("/api/joints")
    def api_joints() -> dict:
        from piper_xvla.motion_watch import joint_positions_deg
        return {"joints_deg": list(joint_positions_deg(piper_ui.piper.GetArmJointMsgs()))}

    @app.get("/api/cameras")
    def api_cameras() -> dict:
        if piper_ui.collection_active():
            raise HTTPException(409, "camera preview is suspended while collection owns the cameras")
        if piper_ui.xvla_status()["status"] == "running":
            raise HTTPException(409, "camera preview is suspended while X-VLA diagnostic owns the cameras")
        if piper_ui.align_status()["status"] in {"running", "stopping"}:
            raise HTTPException(409, "camera preview is suspended while ALIGN inference owns the cameras")
        try:
            return piper_ui.camera_preflight()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/config")
    def api_config_get() -> dict:
        try:
            return piper_ui.get_runtime_config()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.put("/api/config")
    async def api_config_put(req: Request) -> dict:
        try:
            body = await req.json()
            return piper_ui.update_runtime_config(body)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/camera-stream")
    def api_camera_stream():
        return piper_ui.camera_stream()

    @app.post("/api/replay")
    async def api_replay(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        action = str(body.get("action", "")).strip().lower()
        return piper_ui.do_replay(action)

    @app.post("/api/enable")
    def api_enable() -> dict:
        return piper_ui.do_enable()

    @app.post("/api/disable")
    def api_disable() -> dict:
        return piper_ui.do_disable()

    @app.post("/api/mode")
    async def api_mode(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        return piper_ui.do_mode(str(body.get("mode", "")))

    @app.post("/api/reset")
    def api_reset() -> dict:
        return piper_ui.do_reset()

    @app.post("/api/estop")
    def api_estop() -> dict:
        return piper_ui.do_estop()

    @app.post("/api/manual/control")
    async def api_manual_control(req: Request) -> dict:
        body = await req.json()
        result = piper_ui.set_manual_control(bool(body.get("enabled", False)), str(body.get("confirm", "")))
        if not result["ok"]:
            raise HTTPException(403, result["error"])
        return result

    @app.post("/api/manual/arm")
    async def api_manual_arm(req: Request) -> dict:
        body = await req.json()
        result = piper_ui.arm_manual_control(str(body.get("confirm", "")))
        if not result["ok"]:
            raise HTTPException(403, result["error"])
        return result

    @app.post("/api/manual/mark")
    async def api_manual_mark(req: Request) -> dict:
        body = await req.json()
        try:
            return {"ok": True, "mark": piper_ui.mark_manual_pose(str(body.get("name", "")))}
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/manual/marks")
    def api_manual_marks() -> dict:
        return {"marks": piper_ui.manual_marks()}

    @app.post("/api/manual/marks/reorder")
    async def api_manual_marks_reorder(req: Request) -> dict:
        body = await req.json()
        try:
            names = body.get("names")
            if not isinstance(names, list):
                raise ValueError("names must be a list")
            return {"ok": True, "marks": piper_ui.reorder_manual_marks(names)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/manual/marks/delete")
    async def api_manual_marks_delete(req: Request) -> dict:
        body = await req.json()
        try:
            return {"ok": True, "marks": piper_ui.delete_manual_mark(str(body.get("name", "")))}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/manual/endpose")
    async def api_manual_endpose(req: Request) -> dict:
        body = await req.json()
        try:
            return await asyncio.to_thread(piper_ui.send_manual_endpose, body)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/manual/joints")
    async def api_manual_joints(req: Request) -> dict:
        body = await req.json()
        try:
            return await asyncio.to_thread(piper_ui.send_manual_joints, body)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/manual/gripper")
    async def api_manual_gripper(req: Request) -> dict:
        body = await req.json()
        try:
            return await asyncio.to_thread(piper_ui.send_manual_gripper, body)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/manual/jog")
    async def api_manual_jog(req: Request) -> dict:
        body = await req.json()
        try:
            return await asyncio.to_thread(piper_ui.jog_manual_axis, body)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/manual/trajectory/run")
    async def api_manual_trajectory_run(req: Request) -> dict:
        body = await req.json()
        try:
            return await asyncio.to_thread(piper_ui.run_marked_trajectory, body)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/master-slave/devices")
    def api_master_slave_devices() -> dict:
        try:
            return {"devices": piper_ui.discover_can_devices()}
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.get("/api/master-slave/device-status")
    def api_master_slave_device_status(can: str) -> dict:
        try:
            return piper_ui.device_status(can)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/master-slave/status")
    def api_master_slave_status() -> dict:
        return piper_ui.master_slave_status()

    @app.get("/api/master-slave/calibration")
    def api_master_slave_calibration() -> dict:
        return {"joint_master_limits_deg": MASTER_JOINT_PHYSICAL_LIMITS_DEG,
                "joint_slave_limits_deg": SLAVE_JOINT_PHYSICAL_LIMITS_DEG,
                "joint_scales": JOINT_SCALES, "joint_biases_deg": JOINT_BIASES_DEG,
                "j6_master_limits_deg": MASTER_J6_PHYSICAL_LIMITS_DEG,
                "j6_slave_limits_deg": SLAVE_J6_PHYSICAL_LIMITS_DEG,
                "j6_scale": J6_SCALE, "j6_bias_deg": J6_BIAS_DEG,
                "gripper_master_limits_mm": MASTER_GRIPPER_PHYSICAL_LIMITS_MM,
                "gripper_slave_limits_mm": SLAVE_GRIPPER_PHYSICAL_LIMITS_MM,
                "gripper_scale": GRIPPER_SCALE, "gripper_bias_mm": GRIPPER_BIAS_MM}

    @app.post("/api/master-slave/heartbeat")
    async def api_master_slave_heartbeat(req: Request) -> dict:
        try:
            body = await req.json()
            return piper_ui.renew_master_slave_control(str(body.get("session_id", "")))
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (AttributeError, ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/master-slave/start")
    async def api_master_slave_start(req: Request) -> dict:
        try:
            body = await req.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            return piper_ui.start_master_slave_control(body)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409 if "active" in str(exc) or "blocked" in str(exc) else 400, str(exc)) from exc

    @app.post("/api/master-slave/stop")
    def api_master_slave_stop() -> dict:
        return piper_ui.stop_master_slave_control()

    @app.get("/api/task")
    def api_task_get() -> dict:
        return {"task": piper_ui.task}

    @app.put("/api/task")
    async def api_task_put(req: Request) -> dict:
        body = await req.json()
        task = str(body.get("task", "")).strip()
        if not task:
            raise HTTPException(400, "task must be non-empty")
        piper_ui.task = task
        return {"ok": True, "task": piper_ui.task}

    @app.get("/api/collect/status")
    def api_collect_status() -> dict:
        return piper_ui.collect_status()

    @app.post("/api/collect/start")
    async def api_collect_start(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001 - tolerate a missing/empty JSON body
            body = {}
        if not isinstance(body, dict):
            body = {}
        try:
            window_s = float(body.get("window_s", 40.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "window_s must be a number")
        if not 5.0 <= window_s <= 300.0:
            raise HTTPException(400, "window_s must be in [5, 300]")
        try:
            result = piper_ui.start_collect(window_s, body.get("dataset_path"))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        state.pop("dataset", None)
        return result

    @app.post("/api/collect/stop")
    def api_collect_stop() -> dict:
        return piper_ui.stop_collect()

    @app.get("/api/dataset")
    def api_dataset() -> dict:
        ds = _load_dataset_cached(state, piper_ui.dataset_root)
        if ds is None:
            return {"exists": False, "root": str(piper_ui.dataset_root)}
        try:
            info = json.loads((Path(str(ds.root)) / "meta" / "info.json").read_text())
        except Exception:  # noqa: BLE001
            info = {}
        return {
            "exists": True,
            "root": str(Path(str(ds.root))),
            "num_episodes": ds.num_episodes,
            "num_frames": ds.num_frames,
            "fps": ds.fps,
            "codebase_version": info.get("codebase_version"),
            "features": {k: {"dtype": v.get("dtype"), "shape": list(v.get("shape", []))} for k, v in info.get("features", {}).items()},
        }

    @app.get("/api/episodes")
    def api_episodes() -> dict:
        ds = _load_dataset_cached(state, piper_ui.dataset_root)
        if ds is None:
            return {"exists": False, "episodes": []}
        episodes = []
        for i in range(ds.num_episodes):
            row = ds.meta.episodes[i]
            ep_index = int(row["episode_index"])
            length = int(row["length"])
            tasks = list(row.get("tasks", [])) if isinstance(row.get("tasks"), (list, tuple)) else [row.get("tasks")]
            video_path = Path(str(ds.root)) / "images" / "review" / f"episode-{ep_index:06d}_review.mp4"
            episodes.append({
                "index": ep_index,
                "length": length,
                "duration_s": round(length / float(ds.fps), 2) if ds.fps else None,
                "tasks": tasks,
                "has_video": video_path.is_file(),
                "video_url": f"/api/video/{video_path.name}" if video_path.is_file() else None,
            })
        return {"exists": True, "episodes": episodes}

    @app.get("/api/episode/{index}")
    def api_episode_detail(index: int) -> dict:
        ds = _load_dataset_cached(state, piper_ui.dataset_root)
        if ds is None or index < 0 or index >= ds.num_episodes:
            raise HTTPException(404, "episode not found")
        thumbnails = _episode_thumbnails(ds, index)
        traj = _episode_state_action(ds, index)
        return {"index": index, "thumbnails_b64": thumbnails, **traj}

    @app.get("/api/videos")
    def api_videos() -> dict:
        review_dir = piper_ui.dataset_root / "images" / "review"
        if not review_dir.is_dir():
            return {"videos": []}
        videos = []
        for p in sorted(review_dir.glob("episode-*.mp4")):
            codec = probe_codec(p)
            videos.append({
                "name": p.name,
                "url": f"/api/video/{p.name}",
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "mtime": int(p.stat().st_mtime),
                "codec": codec,
                "browser_ready": codec == "h264",
            })
        return {"videos": videos}

    @app.post("/api/videos/migrate")
    def api_videos_migrate() -> dict:
        """Convert legacy mp4v review files to browser-compatible H.264."""
        review_dir = piper_ui.dataset_root / "images" / "review"
        results = []
        for p in sorted(review_dir.glob("episode-*.mp4")) if review_dir.is_dir() else []:
            before = probe_codec(p)
            ok, message = ensure_h264(p)
            results.append({"name": p.name, "before": before, "ok": ok, "codec": probe_codec(p), "message": message})
        return {"results": results}

    @app.get("/api/video/{name}")
    def api_video(name: str) -> FileResponse:
        review_dir = piper_ui.dataset_root / "images" / "review"
        path = (review_dir / name).resolve()
        if not str(path).startswith(str(review_dir.resolve())) or not path.is_file():
            raise HTTPException(404, "video not found")
        codec = probe_codec(path)
        if codec != "h264":
            raise HTTPException(409, f"review video codec is {codec!r}, not browser-compatible H.264; use POST /api/videos/migrate")
        return FileResponse(path, media_type="video/mp4", filename=name)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--can", default="auto", help="primary Piper CAN interface (default: auto-detect)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dry-run", action="store_true", help="no hardware; uses DryRunPiper")
    args = parser.parse_args()

    import uvicorn
    if args.can == "auto" and not args.dry_run:
        devices = [item["name"] for item in PiperWebUI.discover_can_devices() if item["state"] == "UP"]
        if not devices:
            raise RuntimeError("no UP SocketCAN interface found; bring a Piper CAN interface up or use --dry-run")
        args.can = "can0" if "can0" in devices else "can_slave" if "can_slave" in devices else devices[0]
    piper_ui = PiperWebUI(can=args.can, dry_run=args.dry_run, data_dir=Path(args.data_dir))
    app = build_app(piper_ui, Path(args.data_dir))
    print(f"Piper X-VLA Web UI: http://{args.host}:{args.port}  (dry_run={args.dry_run}, can={args.can})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
