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
import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from piper_xvla.replay_control import (
    DryRunPiper,
    PiperReplayController,
    connect_live_piper,
    status_fields,
)
from piper_xvla.snapshot_adapter import DEFAULT_CAMERA_CONFIG
from piper_xvla.review_video import ensure_h264, probe_codec

MODULE_DIR = Path(__file__).resolve().parent
FRONTEND_HTML = MODULE_DIR / "webui.html"
DEFAULT_DATA_DIR = Path.home() / "ALIGN" / "baselines" / "data" / "piper_replay"

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


def _name(table: dict, value: int) -> str:
    if value is None or value < 0:
        return "n/a"
    return table.get(value, f"UNKNOWN(0x{value & 0xFF:02x})")


def _hex(value: Optional[int]) -> str:
    if value is None or value < 0:
        return "n/a"
    return f"0x{value & 0xFF:02x}"


class PiperWebUI:
    """Owns the live (or dry-run) Piper handle and the background collect task."""

    def __init__(self, can: str = "can0", dry_run: bool = False, data_dir: Path = DEFAULT_DATA_DIR):
        self.can = can
        self.dry_run = dry_run
        self.data_dir = Path(data_dir)
        if dry_run:
            self.piper = DryRunPiper()
        else:
            self.piper = connect_live_piper(can)
            time.sleep(1.0)  # let feedback streams fill
        self.speed = 20
        self.ctrl = PiperReplayController(self.piper, replay_speed_percent=self.speed)
        self.task = ""

        # Serialize one-frame preview acquisition with collection camera ownership.
        # Collection holds this lock until it releases both V4L2 devices.
        self._camera_lock = threading.Lock()

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
            "message": "",
        }

    # -- status -------------------------------------------------------------

    def _feedback_health(self) -> tuple[bool, str, dict[str, float]]:
        """Distinguish usable live feedback from a connected-but-silent CAN adapter."""
        if self.dry_run:
            return True, "SIMULATED (dry-run)", {"status": 0.0, "joint": 0.0, "low_speed": 0.0}
        try:
            if not bool(self.piper.isOk()):
                return False, "SDK health check is false", {"status": 0.0, "joint": 0.0, "low_speed": 0.0}
            raw_status = self.piper.GetArmStatus()
            joint_msg = self.piper.GetArmJointMsgs()
            low_msg = self.piper.GetArmLowSpdInfoMsgs()
            rates = {
                "status": float(getattr(raw_status, "Hz", 0.0) or 0.0),
                "joint": float(getattr(joint_msg, "Hz", 0.0) or 0.0),
                "low_speed": float(getattr(low_msg, "Hz", 0.0) or 0.0),
            }
        except Exception as exc:  # noqa: BLE001
            return False, f"feedback query failed: {type(exc).__name__}: {exc}", {"status": 0.0, "joint": 0.0, "low_speed": 0.0}
        missing = [name for name, hz in rates.items() if hz < 1.0]
        if missing:
            return False, "no live feedback on " + ", ".join(missing), rates
        return True, "LIVE", rates

    def snapshot_status(self) -> dict:
        feedback_ok, feedback_reason, feedback_hz = self._feedback_health()
        s = status_fields(self.piper)
        try:
            from piper_xvla.motion_watch import joint_positions_deg
            joints = list(joint_positions_deg(self.piper.GetArmJointMsgs()))
        except Exception:  # noqa: BLE001
            joints = [None] * 6
        try:
            info = self.piper.GetArmLowSpdInfoMsgs()
            drivers = [int(getattr(info, f"motor_{i}").foc_status.driver_enable_status) for i in range(1, 7)]
        except Exception:  # noqa: BLE001
            drivers = [None] * 6
        return {
            "can": self.can,
            "dry_run": self.dry_run,
            "feedback_ok": feedback_ok,
            "feedback_state": "SIMULATED" if self.dry_run else ("LIVE" if feedback_ok else "UNAVAILABLE"),
            "feedback_reason": feedback_reason,
            "feedback_hz": feedback_hz,
            "ctrl_mode": int(getattr(s, "ctrl_mode", -1)),
            "ctrl_mode_name": _name(MODE_NAMES, int(getattr(s, "ctrl_mode", -1))),
            "arm_status": int(getattr(s, "arm_status", -1)),
            "arm_status_name": _name(ARM_STATUS_NAMES, int(getattr(s, "arm_status", -1))),
            "teach_status": int(getattr(s, "teach_status", -1)),
            "teach_status_name": _name(TEACH_NAMES, int(getattr(s, "teach_status", -1))),
            "motion_status": int(getattr(s, "motion_status", -1)),
            "err_code": int(getattr(s, "err_code", -1)),
            "drivers": drivers,
            "joints_deg": joints,
            "speed": self.speed,
            "task": self.task,
        }

    def collection_active(self) -> bool:
        with self._collect_lock:
            return self._collect_state["status"] in {"opening", "recording", "finalizing"}

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
        """MJPEG multipart stream of global|wrist at ~5 FPS."""
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
                    time.sleep(0.2)  # ~5 FPS per camera
            finally:
                for cap in caps.values():
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # -- replay control ------------------------------------------------------

    def _verified_live_feedback(self) -> tuple[bool, str]:
        ok, reason, _ = self._feedback_health()
        if ok:
            return True, reason
        return False, f"refusing unverified command: live Piper feedback is unavailable ({reason})"

    def do_replay(self, action: str) -> dict:
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
        try:
            self.ctrl.emergency_stop()
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- background collect --------------------------------------------------

    def start_collect(self, window_s: float = 40.0) -> dict:
        if not self.task.strip():
            raise HTTPException(400, "task must be set before starting collect (PUT /api/task)")
        with self._collect_lock:
            if self._collect_state["status"] in ("opening", "recording", "finalizing"):
                raise HTTPException(409, "a collect is already running")
            self._collect_stop.clear()
            self._collect_state = {
                "status": "opening", "frames": 0, "elapsed_s": 0.0,
                "window_s": window_s, "episode_index": None,
                "video_path": None, "message": "opening cameras + dataset...",
            }
            self._collect_thread = threading.Thread(target=self._collect_worker, args=(window_s,), daemon=True)
            self._collect_thread.start()
        return {"ok": True, "state": dict(self._collect_state)}

    def _collect_worker(self, window_s: float) -> None:
        from piper_xvla.collect_session import CollectSession
        period = 0.05
        if not self._camera_lock.acquire(timeout=5.0):
            with self._collect_lock:
                self._collect_state.update(status="error", message="camera preview did not release devices within 5 s")
            return
        try:
            session = CollectSession.open(self.piper, self.task, self.data_dir)
        except Exception as exc:  # noqa: BLE001
            self._camera_lock.release()
            with self._collect_lock:
                self._collect_state.update(status="error", message=f"open failed: {exc}")
            return

        with self._collect_lock:
            self._collect_state["status"] = "recording"
            self._collect_state["episode_index"] = session.episode_index
            self._collect_state["video_path"] = str(session.video_path) if session.video_ok else None
            self._collect_state["message"] = f"recording episode {session.episode_index}..."

        start = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= window_s or self._collect_stop.is_set():
                    break
                try:
                    obs = session.snapshot()
                    session.write_frame(obs)
                except Exception as exc:  # noqa: BLE001
                    with self._collect_lock:
                        self._collect_state.update(status="error", message=f"capture failed: {exc}")
                    break
                with self._collect_lock:
                    self._collect_state["frames"] = session.frames_written
                    self._collect_state["elapsed_s"] = round(elapsed, 2)
                # Pace to 20 Hz.
                target = (int((time.monotonic() - start) / period) + 1) * period
                sleep_for = target - (time.monotonic() - start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except Exception as exc:  # noqa: BLE001
            with self._collect_lock:
                self._collect_state.update(status="error", message=f"unexpected: {exc}")

        with self._collect_lock:
            self._collect_state["status"] = "finalizing"
            self._collect_state["message"] = "finalizing episode..."
        try:
            total = session.finalize()
            reason = "stopped by user" if self._collect_stop.is_set() else "window ended"
            with self._collect_lock:
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
            if self._collect_state["status"] not in ("recording", "opening"):
                return {"ok": False, "error": f"no active collect (status={self._collect_state['status']})"}
        self._collect_stop.set()
        return {"ok": True}

    def collect_status(self) -> dict:
        with self._collect_lock:
            return dict(self._collect_state)


def _load_dataset_cached(app_state: dict, data_dir: Path):
    """Return a LeRobotDataset for the given root, or None if absent/corrupt."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(data_dir) / "dataset"
    if not (root / "meta" / "info.json").is_file():
        return None
    key = str(root.resolve())
    cached = app_state.get("dataset")
    if cached is not None and app_state.get("dataset_root") == key:
        return cached
    try:
        ds = LeRobotDataset(key)
        app_state["dataset"] = ds
        app_state["dataset_root"] = key
        return ds
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"could not open dataset at {root}: {type(exc).__name__}: {exc}")


def _episode_thumbnails(ds, episode_index: int, max_frames: int = 3) -> list[str]:
    """Return base64 JPEG thumbnails sampled across one episode."""
    import cv2
    meta_row = ds.meta.episodes[episode_index]
    from_idx = int(meta_row["dataset_from_index"])
    to_idx = int(meta_row["dataset_to_index"]) - 1
    length = max(1, to_idx - from_idx + 1)
    picks = sorted({from_idx, from_idx + length // 2, to_idx})[:max_frames]
    out: list[str] = []
    for idx in picks:
        try:
            sample = ds[idx]
        except Exception:  # noqa: BLE001
            continue
        g = np.asarray(sample["observation.images.global_rgb"])
        if g.ndim == 3 and g.shape[0] == 3:  # LeRobot returns CHW float [0,1]
            g = (g * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        g = cv2.resize(g, (320, 240))
        _, jpg = cv2.imencode(".jpg", g[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        out.append(base64.b64encode(jpg.tobytes()).decode())
    return out


def _episode_state_action(ds, episode_index: int, max_points: int = 300) -> dict:
    """Downsample state/action vectors across one episode for charting."""
    meta_row = ds.meta.episodes[episode_index]
    from_idx = int(meta_row["dataset_from_index"])
    to_idx = int(meta_row["dataset_to_index"]) - 1
    n = max(1, to_idx - from_idx + 1)
    step = max(1, n // max_points)
    idxs = list(range(from_idx, to_idx + 1, step))
    if idxs[-1] != to_idx:
        idxs.append(to_idx)
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for i in idxs:
        s = ds[i]
        states.append([float(x) for x in np.asarray(s["observation.state"]).ravel()])
        actions.append([float(x) for x in np.asarray(s["action"]).ravel()])
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

    @app.get("/api/joints")
    def api_joints() -> dict:
        from piper_xvla.motion_watch import joint_positions_deg
        return {"joints_deg": list(joint_positions_deg(piper_ui.piper.GetArmJointMsgs()))}

    @app.get("/api/cameras")
    def api_cameras() -> dict:
        if piper_ui.collection_active():
            raise HTTPException(409, "camera preview is suspended while collection owns the cameras")
        return piper_ui.camera_preflight()

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
        return piper_ui.start_collect(window_s)

    @app.post("/api/collect/stop")
    def api_collect_stop() -> dict:
        return piper_ui.stop_collect()

    @app.get("/api/dataset")
    def api_dataset() -> dict:
        ds = _load_dataset_cached(state, data_dir)
        if ds is None:
            return {"exists": False, "root": str(data_dir / "dataset")}
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
        ds = _load_dataset_cached(state, data_dir)
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
        ds = _load_dataset_cached(state, data_dir)
        if ds is None or index < 0 or index >= ds.num_episodes:
            raise HTTPException(404, "episode not found")
        thumbnails = _episode_thumbnails(ds, index)
        traj = _episode_state_action(ds, index)
        return {"index": index, "thumbnails_b64": thumbnails, **traj}

    @app.get("/api/videos")
    def api_videos() -> dict:
        review_dir = Path(data_dir) / "dataset" / "images" / "review"
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
        review_dir = Path(data_dir) / "dataset" / "images" / "review"
        results = []
        for p in sorted(review_dir.glob("episode-*.mp4")) if review_dir.is_dir() else []:
            before = probe_codec(p)
            ok, message = ensure_h264(p)
            results.append({"name": p.name, "before": before, "ok": ok, "codec": probe_codec(p), "message": message})
        return {"results": results}

    @app.get("/api/video/{name}")
    def api_video(name: str) -> FileResponse:
        review_dir = Path(data_dir) / "dataset" / "images" / "review"
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
    parser.add_argument("--can", default="can0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dry-run", action="store_true", help="no hardware; uses DryRunPiper")
    args = parser.parse_args()

    import uvicorn
    piper_ui = PiperWebUI(can=args.can, dry_run=args.dry_run, data_dir=Path(args.data_dir))
    app = build_app(piper_ui, Path(args.data_dir))
    print(f"Piper X-VLA Web UI: http://{args.host}:{args.port}  (dry_run={args.dry_run}, can={args.can})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
