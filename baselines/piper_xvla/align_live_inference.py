"""Live ALIGN diagnostics with a lease-gated, guarded Piper command loop."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.align_intention import ALIGNIntentionModel  # noqa: E402
from piper_xvla.action_guard import ActionGuard, guard_limits_from_config  # noqa: E402
from piper_xvla.align_control import ALIGNInferenceSettings, ActionHorizonCursor, align_action_to_piper_target, align_gripper_mapping, align_state7, camera_stale_pause_allowed  # noqa: E402
from piper_xvla.endpose_control import send_endpose  # noqa: E402
from piper_xvla.live_inference import _prepare_held_control, _read_lease, _target_from_active  # noqa: E402
from piper_xvla.snapshot_adapter import DEFAULT_CAMERA_CONFIG, PiperSnapshotAdapter  # noqa: E402


def load_model(path: Path, device: torch.device):
    torch.backends.cudnn.enabled = False
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["config"]
    expected = {"num_cameras": 2, "action_dim": 7, "history_size": 1, "head_type": "flow_matching", "use_intent_tokens": False, "use_text": False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"checkpoint {key}={config.get(key)!r}; this Piper adapter requires {value!r}")
    keys = ("state_dim", "action_dim", "chunk_size", "history_size", "num_cameras", "use_patch_tokens",
            "mamba_output_dim", "mamba_d_state", "mamba_d_conv", "mamba_expand", "head_d_model",
            "head_nhead", "head_num_layers", "head_dim_ff", "head_type", "use_text", "text_dim",
            "compressed_dim", "use_intent_tokens", "num_intent_tokens", "intent_dim", "use_memory_bank", "memory_bank_len")
    model = ALIGNIntentionModel(**{key: config[key] for key in keys if key in config})
    model._build_head_and_bank(int(config["num_cameras"]) * 256 * int(config["compressed_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    print(f"Loaded ALIGN checkpoint {path} (epoch {checkpoint.get('epoch')}, chunk {config['chunk_size']})", flush=True)
    return model, int(config["chunk_size"])


def predict_chunk(model, global_rgb: np.ndarray, wrist_rgb: np.ndarray, state7: np.ndarray,
                  device: torch.device, samples: int) -> np.ndarray:
    def resize(frame: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(frame).resize((224, 224), Image.Resampling.BILINEAR))
    frames = torch.from_numpy(np.stack([resize(global_rgb), resize(wrist_rgb)])[None, None].copy()).to(device)
    state = torch.from_numpy(state7[None, None].copy()).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        encoded = model(frames, state)
        chunks = [model.sample_actions(encoded["z_v_pooled_seq"], encoded["z_s_seq"],
                                       intent_emb=encoded.get("intent_emb"))[0].float() for _ in range(samples)]
        result = torch.stack(chunks).mean(0).cpu().numpy()
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 7 or not np.isfinite(result).all():
        raise ValueError(f"ALIGN prediction is invalid: shape={result.shape}")
    return result


def _guard(path: Path) -> ActionGuard:
    return ActionGuard(guard_limits_from_config(json.loads(path.read_text())))


def _connect_observer(can_device: str):
    """Receive feedback without PiperInit's startup CAN transmissions."""
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise RuntimeError("piper_sdk/python-can is unavailable in the ALIGN interpreter") from exc
    piper = C_PiperInterface_V2(can_device)
    try:
        piper.ConnectPort(piper_init=False)
    except TypeError as exc:
        raise RuntimeError("Piper SDK does not support observer-only ConnectPort(piper_init=False)") from exc
    if not piper.isOk():
        raise RuntimeError(f"Piper connection on {can_device!r} is not healthy")
    return piper


def _revoke_lease(path: Path, session: str | None, reason: str = "") -> None:
    temp = path.with_suffix(path.suffix + ".runner.tmp")
    temp.write_text(json.dumps({"expires_monotonic_s": 0, "blocked": True, "session_id": session, "reason": reason}))
    os.replace(temp, path)


def _rejection_reason(decision) -> str:
    return "; ".join(f"{alert.code}: {alert.message}" for alert in decision.alerts if alert.severity == "REJECT")[:500]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--settings", type=json.loads, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = ALIGNInferenceSettings.from_dict(args.settings)
    if not torch.cuda.is_available():
        raise RuntimeError("ALIGN live inference requires CUDA")
    manifest = json.loads(args.calibration_manifest.read_text())
    calibration = manifest["gripper_normalization"]
    gmin, gmax = float(calibration["raw_meters_min"]), float(calibration["raw_meters_max"])
    if not gmin < 0 < gmax:
        raise ValueError("gripper calibration must include the 0 mm closed position")
    model, chunk_size = load_model(args.checkpoint, torch.device("cuda"))
    if settings.action_horizon > chunk_size:
        raise ValueError(f"action_horizon exceeds checkpoint chunk size {chunk_size}")
    guard = _guard(args.guard_config)
    guard_stat = None
    piper = _connect_observer(args.can)
    adapter = PiperSnapshotAdapter.from_camera_config(piper, "ALIGN inference", args.camera_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal_received = threading.Event()
    prediction_lock = threading.Lock()
    feedback_lock = threading.Lock()
    latest: dict = {"generation": 0, "chunk": None, "sampled_at": 0.0, "completed_at": 0.0,
                    "inference_ms": None, "inference_hz_actual": None, "error": None}
    def on_signal(*_):
        signal_received.set()
        stop.set()

    old_handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}

    def infer() -> None:
        period = 1.0 / settings.inference_hz
        next_tick = time.monotonic()
        try:
            while not stop.is_set():
                started = time.monotonic()
                with feedback_lock:
                    observation = adapter.snapshot()
                sampled_at = time.monotonic()
                state7 = align_state7(observation.state20, gmin, gmax)
                chunk = predict_chunk(model, observation.global_rgb, observation.wrist_rgb, state7,
                                      torch.device("cuda"), settings.ensemble_samples)
                completed_at = time.monotonic()
                with prediction_lock:
                    previous = latest["completed_at"]
                    latest.update(generation=latest["generation"] + 1, chunk=chunk,
                                  sampled_at=sampled_at, completed_at=completed_at,
                                  inference_ms=1000 * (completed_at - sampled_at),
                                  inference_hz_actual=1 / (completed_at - previous) if previous else None)
                next_tick = max(next_tick + period, completed_at)
                stop.wait(max(0, next_tick - time.monotonic()))
        except Exception as exc:  # surfaced to the Web UI process log
            with prediction_lock:
                latest["error"] = f"{type(exc).__name__}: {exc}"
            stop.set()

    thread = threading.Thread(target=infer, daemon=True)
    cursor = ActionHorizonCursor(settings.action_horizon)
    command_ready = False
    camera_pause_hold_active = False
    blocked_session: str | None = None
    blocked_reason: str | None = None
    active_session: str | None = None
    last_target = None
    last_target20 = None
    last_gripper_m = None
    last_gripper_mapping = None
    last_generation = -1
    control_ticks: list[float] = []
    frame = 0
    next_tick = time.monotonic()
    thread.start()
    try:
        with args.output.open("w") as stream:
            while not stop.is_set():
                held, speed, session = _read_lease(args.lease)
                if blocked_session is not None and session == blocked_session:
                    held = False
                if session != active_session:
                    cursor = ActionHorizonCursor(settings.action_horizon)
                    active_session = session
                    last_target = None
                    last_target20 = None
                    last_gripper_mapping = None
                with prediction_lock:
                    prediction = dict(latest)
                if prediction["error"]:
                    raise RuntimeError(prediction["error"])
                with feedback_lock:
                    measured = adapter.read_state20()
                feedback_at = time.monotonic()
                stat = args.guard_config.stat()
                signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if signature != guard_stat:
                    guard = _guard(args.guard_config)
                    guard_stat = signature
                    print("ALIGN action guard limits reloaded", flush=True)
                age = feedback_at - prediction["sampled_at"] if prediction["chunk"] is not None else None
                preview = None
                target = None
                target20 = None
                gripper_m = None
                gripper_mapping = None
                current = None
                action_index = None
                alerts = []
                allowed = False
                decision = None
                if prediction["chunk"] is not None:
                    chunk = prediction["chunk"]
                    # Display/lease preflight validates the next unused action.
                    horizon_end = min(settings.action_horizon, len(chunk))
                    next_index = cursor.index if cursor.generation == prediction["generation"] else 0
                    reuse_absolute = (next_index >= horizon_end and last_target20 is not None
                                      and last_generation == prediction["generation"])
                    next_index = min(next_index, horizon_end - 1)
                    preview = chunk[next_index]
                    if reuse_absolute:
                        target, gripper_m, target20 = last_target, last_gripper_m, last_target20
                        gripper_mapping = last_gripper_mapping
                        current = measured[:10].copy()
                        current[9] = np.clip((measured[9] - gmin) / (gmax - gmin), 0, 1)
                    else:
                        target, gripper_m, current, target20 = align_action_to_piper_target(
                            preview, measured, gmin, gmax, settings)
                        gripper_mapping = align_gripper_mapping(float(preview[6]), float(measured[9]), gmin, gmax, settings)
                    decision = guard.check(predicted_action20=target20, current_active10=current,
                                           camera_age_s=age, feedback_age_s=time.monotonic() - feedback_at)
                    alerts = [alert.__dict__ for alert in decision.alerts]
                    allowed = decision.allowed
                    for alert in decision.alerts:
                        if held and alert.severity == "REJECT":
                            print(f"ALIGN guard {alert.code}: {alert.message}", flush=True)
                setup_this_tick = False
                camera_paused = False
                if held and not allowed:
                    rejection_codes = [alert.code for alert in decision.alerts if alert.severity == "REJECT"] if decision else []
                    camera_paused = camera_stale_pause_allowed(rejection_codes, age, guard.limits.max_camera_age_s)
                    if not camera_paused:
                        blocked_reason = _rejection_reason(decision) if decision else "no prediction available"
                        _revoke_lease(args.lease, session, blocked_reason)
                        blocked_session = session
                        held = False
                if not camera_paused:
                    camera_pause_hold_active = False
                if held and allowed and not command_ready:
                    setup_this_tick = True
                    command_ready = _prepare_held_control(piper, args.lease, session, speed, stop)
                control_sent = False
                if held and allowed and command_ready and not setup_this_tick:
                    live, speed, live_session = _read_lease(args.lease)
                    if live and live_session == session and not stop.is_set():
                        if not bool(piper.isOk()):
                            raise RuntimeError("Piper feedback is no longer healthy; stopping ALIGN control")
                        item = cursor.next(prediction["generation"], prediction["chunk"])
                        if item is not None:
                            action_index, action = item
                            target, gripper_m, current, target20 = align_action_to_piper_target(
                                action, measured, gmin, gmax, settings)
                            gripper_mapping = align_gripper_mapping(float(action[6]), float(measured[9]), gmin, gmax, settings)
                            decision = guard.check(predicted_action20=target20, current_active10=current,
                                                   camera_age_s=age, feedback_age_s=time.monotonic() - feedback_at)
                            if not decision.allowed:
                                blocked_reason = _rejection_reason(decision)
                                _revoke_lease(args.lease, session, blocked_reason)
                                blocked_session = session
                                held = False
                            else:
                                last_target, last_gripper_m = target, gripper_m
                                last_target20 = target20
                                last_gripper_mapping = gripper_mapping
                                last_generation = prediction["generation"]
                        if held and last_target is not None and last_generation == prediction["generation"]:
                            checked = guard.check(predicted_action20=last_target20, current_active10=current,
                                                  camera_age_s=age, feedback_age_s=time.monotonic() - feedback_at)
                            if not checked.allowed:
                                for alert in checked.alerts:
                                    if alert.severity == "REJECT":
                                        print(f"ALIGN command guard {alert.code}: {alert.message}", flush=True)
                                blocked_reason = _rejection_reason(checked)
                                _revoke_lease(args.lease, session, blocked_reason)
                                blocked_session = session
                                held = False
                            else:
                                still_held, speed, still_session = _read_lease(args.lease)
                                if not still_held or still_session != session or stop.is_set():
                                    held = False
                                else:
                                    piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=speed,
                                                       is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
                                    send_endpose(piper, last_target)
                                    piper.GripperCtrl(int(round(last_gripper_m * 1_000_000)), 1000, 0x01, 0)
                                    control_sent = True
                                    control_ticks.append(time.monotonic())
                hold_sent = False
                if (not held or (camera_paused and not camera_pause_hold_active)) and command_ready and not stop.is_set():
                    # One measured hold on release or camera pause; never send a stale future target.
                    current10 = measured[:10].copy()
                    current10[9] = np.clip((measured[9] - gmin) / (gmax - gmin), 0, 1)
                    hold_target, hold_gripper_m = _target_from_active(current10, gmin, gmax)
                    piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=1,
                                       is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
                    send_endpose(piper, hold_target)
                    piper.GripperCtrl(int(round(hold_gripper_m * 1_000_000)), 1000, 0x01, 0)
                    if camera_paused:
                        camera_pause_hold_active = True
                    else:
                        command_ready = False
                    last_target = None
                    last_target20 = None
                    last_gripper_mapping = None
                    hold_sent = True
                now = time.monotonic()
                control_ticks = [value for value in control_ticks if value >= now - 1]
                if target20 is not None:
                    final_decision = guard.check(predicted_action20=target20, current_active10=current,
                                                 camera_age_s=now - prediction["sampled_at"],
                                                 feedback_age_s=now - feedback_at)
                    allowed = final_decision.allowed
                    alerts = [alert.__dict__ for alert in final_decision.alerts]
                    if gripper_mapping is not None and gripper_mapping.clamped:
                        alerts.append({"code": "GRIPPER_TARGET_CLAMPED", "severity": "WARN",
                                       "message": f"scaled model gripper={gripper_mapping.raw_normalized:.4f} ({gripper_mapping.raw_mm:.1f} mm) "
                                                  f"outside calibrated [{gmin * 1000:.1f}, {gmax * 1000:.1f}] mm; "
                                                  f"bounded to {gripper_mapping.bounded_normalized:.4f}, command {gripper_mapping.target_mm:.1f} mm"})
                frame += 1
                row = {"frame": frame, "generation": prediction["generation"],
                       "timestamp_monotonic_s": now, "inference_completed_monotonic_s": prediction["completed_at"],
                       "prediction_age_s": now - prediction["sampled_at"] if prediction["chunk"] is not None else None,
                       "inference_ms": prediction["inference_ms"],
                       "inference_hz_target": settings.inference_hz,
                       "inference_hz_actual": prediction["inference_hz_actual"], "control_hz": len(control_ticks),
                       "target_hz": 20, "control_active": held and command_ready and not camera_paused,
                       "control_paused": camera_paused,
                       "control_blocked": session is not None and session == blocked_session,
                       "control_block_session": blocked_session if session is not None and session == blocked_session else None,
                       "control_block_reason": blocked_reason if session is not None and session == blocked_session else None,
                       "control_sent": control_sent, "control_release_hold_sent": hold_sent and not camera_paused,
                       "control_pause_hold_sent": hold_sent and camera_paused,
                       "action_index": action_index, "raw_action7": preview.tolist() if preview is not None else None,
                       "target_pose": {"xyz_m": target.xyz_m.tolist(), "euler_xyz_deg": target.euler_xyz_deg.tolist()} if target is not None else None,
                       "gripper_target_mm": gripper_m * 1000 if gripper_m is not None else None,
                       "gripper_raw_mm": gripper_mapping.raw_mm if gripper_mapping is not None else None,
                       "gripper_clamped": gripper_mapping.clamped if gripper_mapping is not None else False,
                       "guard_allowed": allowed, "alerts": alerts}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                next_tick = max(next_tick + 0.05, time.monotonic())
                stop.wait(max(0, next_tick - time.monotonic()))
        if latest["error"]:
            _revoke_lease(args.lease, active_session, latest["error"])
            if command_ready and not signal_received.is_set() and bool(piper.isOk()):
                with feedback_lock:
                    measured = adapter.read_state20()
                current10 = measured[:10].copy()
                current10[9] = np.clip((measured[9] - gmin) / (gmax - gmin), 0, 1)
                hold_target, hold_gripper_m = _target_from_active(current10, gmin, gmax)
                piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=1,
                                   is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
                send_endpose(piper, hold_target)
                piper.GripperCtrl(int(round(hold_gripper_m * 1_000_000)), 1000, 0x01, 0)
            raise RuntimeError(latest["error"])
    finally:
        stop.set()
        thread.join(timeout=2)
        adapter.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
