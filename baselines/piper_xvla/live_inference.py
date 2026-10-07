"""Piper X-VLA inference, diagnostic by default, with optional UI-deadman control."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from piper_xvla.endpose_control import EndPoseTarget, _wait_for_can_mode, _wait_until_enabled, prepare_can_cartesian_control, send_endpose

from piper_xvla.action_guard import ActionGuard, guard_limits_from_config
from piper_xvla.replay_control import connect_live_piper
from piper_xvla.snapshot_adapter import DEFAULT_CAMERA_CONFIG, PiperSnapshotAdapter
from piper_xvla.inference_recorder import InferenceImageRecorder
from piper_xvla.gripper_binary import BinaryGripperConfig, DEFAULT_CONFIG_PATH, load_binary_gripper_config
from piper_xvla.train_xvla_piper import configure_cuda_attention, load_config

TASK = "grab an object and put in a cup"


def raw_state20_to_live_inputs(
    raw_state20: np.ndarray,
    *,
    gripper_min_m: float,
    gripper_max_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert copied raw Piper state to X-VLA proprio8 and guarded active action10."""
    raw = np.asarray(raw_state20, dtype=np.float32)
    if raw.shape != (20,) or not np.isfinite(raw).all():
        raise ValueError("raw Piper state must be finite shape (20,)")
    active = raw[:10].copy()
    rotation6d = active[3:9]
    col0, col1 = rotation6d[[0, 2, 4]], rotation6d[[1, 3, 5]]
    if np.linalg.norm(col0) < 1e-8:
        raise ValueError("raw Piper state has degenerate rotation-6D first column")
    col0 = col0 / np.linalg.norm(col0)
    col1 = col1 - col0 * np.dot(col0, col1)
    if np.linalg.norm(col1) < 1e-8:
        raise ValueError("raw Piper state has degenerate rotation-6D second column")
    col1 = col1 / np.linalg.norm(col1)
    matrix = np.column_stack((col0, col1, np.cross(col0, col1)))
    gripper = float(np.clip((active[9] - gripper_min_m) / (gripper_max_m - gripper_min_m), 0.0, 1.0))
    active[9] = gripper
    proprio8 = np.concatenate((active[:3], Rotation.from_matrix(matrix).as_quat().astype(np.float32), [gripper])).astype(np.float32)
    return proprio8, active


def _make_batch(global_rgb: np.ndarray, wrist_rgb: np.ndarray, proprio8: np.ndarray, tokens: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    def image(rgb: np.ndarray) -> torch.Tensor:
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError("live RGB image must be uint8 shape (480, 640, 3)")
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)

    return {
        "observation.images.image": image(global_rgb).to(device),
        "observation.images.image2": image(wrist_rgb).to(device),
        "observation.images.empty_camera_0": torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device),
        "observation.state": torch.from_numpy(proprio8).unsqueeze(0).to(device),
        "observation.language.tokens": tokens.to(device),
    }


def _load_policy(checkpoint_path: str | Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    payload = dict(checkpoint["xvla_config"])
    for key in ("input_features", "output_features"):
        payload[key] = {
            name: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
            for name, value in payload.get(key, {}).items()
        }
    config = XVLAConfig(**payload)
    config.device = device
    policy = XVLAPolicy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    return policy.to(device).eval()


def _load_guard(path: str | Path) -> ActionGuard:
    payload = json.loads(Path(path).read_text())
    return ActionGuard(guard_limits_from_config(payload))


def _predict_action_with_cudnn_fallback(policy: Any, batch: dict[str, torch.Tensor], device: str) -> np.ndarray:
    """Retry once without cuDNN if this CUDA stack cannot initialize it."""
    def predict() -> np.ndarray:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            return policy.predict_action_chunk(batch)[0, 0].detach().float().cpu().numpy()

    try:
        return predict()
    except RuntimeError as exc:
        if not device.startswith("cuda") or not torch.backends.cudnn.enabled or "CUDNN_STATUS_NOT_INITIALIZED" not in str(exc):
            raise
        torch.backends.cudnn.enabled = False
        configure_cuda_attention(device)
        torch.cuda.empty_cache()
        print("cuDNN failed to initialize; retrying X-VLA prediction without cuDNN (inference may be slower)", flush=True)
        return predict()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="piper_xvla/config/piper_xvla_single_task.json")
    parser.add_argument("--checkpoint", default="outputs/piper_xvla_single_task/best.pt")
    parser.add_argument("--guard-config", required=True)
    parser.add_argument("--binary-gripper-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--can", default="can0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=100, help="positive frame count; 0 runs continuously until terminated")
    parser.add_argument("--output", default="outputs/piper_xvla_single_task/live_camera_only.jsonl")
    parser.add_argument("--enable-held-control", action="store_true", help="allow guarded commands only while the WebUI lease remains active")
    parser.add_argument("--hold-lease", help="JSON lease file written by WebUI; required with --enable-held-control")
    parser.add_argument("--record-images-dir", type=Path, help="record both inference camera images as JPEGs using a bounded background writer")
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative; 0 means continuous")
    if args.enable_held_control != bool(args.hold_lease):
        parser.error("--enable-held-control and --hold-lease must be supplied together")
    return args


def _read_lease(path: Path | None) -> tuple[bool, int, str | None]:
    if path is None:
        return False, 1, None
    try:
        lease = json.loads(path.read_text())
        return float(lease.get("expires_monotonic_s", 0)) > time.monotonic(), max(1, min(10, int(lease.get("speed_percent", 5)))), lease.get("session_id")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False, 1, None


def _target_from_active(active: np.ndarray, gripper_min_m: float, gripper_max_m: float) -> tuple[EndPoseTarget, float]:
    col0, col1 = active[[3, 5, 7]].astype(float), active[[4, 6, 8]].astype(float)
    col0 /= np.linalg.norm(col0)
    col1 -= col0 * np.dot(col0, col1)
    col1 /= np.linalg.norm(col1)
    matrix = np.column_stack((col0, col1, np.cross(col0, col1)))
    # Invert snapshot_adapter's from_euler("xyz") using the same extrinsic
    # convention. Uppercase "XYZ" would command a different physical rotation.
    euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
    gripper_m = gripper_min_m + float(active[9]) * (gripper_max_m - gripper_min_m)
    return EndPoseTarget(np.asarray(active[:3], dtype=float), euler), gripper_m


def _binary_gripper_action(
    predicted_action20: np.ndarray, gripper_min_m: float, gripper_max_m: float,
    config: BinaryGripperConfig = BinaryGripperConfig(),
) -> tuple[np.ndarray, float | None, float | None]:
    """Force predictions below the switch to 0 mm; pass larger values through."""
    action = np.asarray(predicted_action20, dtype=np.float32).copy()
    if action.shape != (20,):
        raise ValueError("X-VLA prediction must have 20 action values")
    config.validate_calibration(gripper_min_m, gripper_max_m)

    normalized = float(action[9])
    if not np.isfinite(normalized):
        # Keep NaN/inf intact so ActionGuard rejects an invalid prediction.
        return action, None, None
    span = gripper_max_m - gripper_min_m
    predicted_m = gripper_min_m + normalized * span
    threshold_normalized = np.float32((config.threshold_mm / 1000 - gripper_min_m) / span)
    if action[9] < threshold_normalized:
        target_m = 0.0
        action[9] = (target_m - gripper_min_m) / span
    else:
        target_m = predicted_m
    return action, predicted_m, target_m


def _prepare_held_control(piper: Any, lease_path: Path, session_id: str, speed: int, stop_event: threading.Event | None = None) -> bool:
    """Cancel normal lease loss without terminating camera/policy inference."""
    class LeaseStop:
        def is_set(self) -> bool:
            active, _, live_session = _read_lease(lease_path)
            return not active or live_session != session_id or (stop_event is not None and stop_event.is_set())

    stop = LeaseStop()
    try:
        if stop.is_set():
            return False
        _wait_until_enabled(piper, enable_gripper=False, stop_event=stop)
        if stop.is_set():
            return False
        prepare_can_cartesian_control(piper, speed_percent=speed, enable_gripper=False)
        _wait_for_can_mode(piper, stop_event=stop)
        return not stop.is_set()
    except RuntimeError:
        if stop.is_set():
            return False
        raise


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    values = load_config(args.config)
    manifest = json.loads(Path(values["manifest"]).read_text())
    calibration = manifest["gripper_normalization"]
    guard = _load_guard(args.guard_config)
    guard_signature = None
    gripper_config = load_binary_gripper_config(args.binary_gripper_config)
    gripper_config.validate_calibration(calibration["raw_meters_min"], calibration["raw_meters_max"])
    gripper_signature = None
    policy = _load_policy(args.checkpoint, args.device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    tokens = tokenizer([TASK], max_length=64, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]

    piper = connect_live_piper(args.can)
    adapter = PiperSnapshotAdapter.from_camera_config(piper, TASK, args.camera_config)
    output = Path(args.output)
    lease_path = Path(args.hold_lease) if args.hold_lease else None
    period_s = 0.05
    command_ready = False
    blocked_session_id: str | None = None
    started_at = time.monotonic()
    stats_frames = 0
    loop_ticks: list[float] = []
    control_ticks: list[float] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    recorder = None
    stop_event = threading.Event()
    previous_handlers = {}
    try:
        if args.record_images_dir:
            recorder = InferenceImageRecorder(args.record_images_dir)
            print(f"Recording inference images to {args.record_images_dir}", flush=True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, lambda *_: stop_event.set())
        with output.open("w") as stream, torch.no_grad():
            frame_iterator = itertools.count() if args.frames == 0 else range(args.frames)
            for frame_index in frame_iterator:
                if stop_event.is_set():
                    break
                started = time.monotonic()
                held, speed, session_id = _read_lease(lease_path) if args.enable_held_control else (False, 1, None)
                if blocked_session_id is not None:
                    if held and session_id and session_id != blocked_session_id:
                        blocked_session_id = None
                    else:
                        held = False
                observation = adapter.snapshot()
                sampled_at = time.monotonic()
                proprio8, current_active10 = raw_state20_to_live_inputs(
                    observation.state20,
                    gripper_min_m=calibration["raw_meters_min"],
                    gripper_max_m=calibration["raw_meters_max"],
                )
                batch = _make_batch(observation.global_rgb, observation.wrist_rgb, proprio8, tokens, args.device)
                predicted = _predict_action_with_cudnn_fallback(policy, batch, args.device)
                gripper_stat = args.binary_gripper_config.stat()
                current_gripper_signature = (gripper_stat.st_ino, gripper_stat.st_mtime_ns, gripper_stat.st_size)
                if current_gripper_signature != gripper_signature:
                    gripper_config = load_binary_gripper_config(args.binary_gripper_config)
                    gripper_config.validate_calibration(calibration["raw_meters_min"], calibration["raw_meters_max"])
                    gripper_signature = current_gripper_signature
                    print("Gripper close threshold loaded: " + json.dumps(gripper_config.as_dict()), flush=True)
                guarded_action, predicted_gripper_m, gripper_target_m = _binary_gripper_action(
                    predicted, calibration["raw_meters_min"], calibration["raw_meters_max"], gripper_config
                )
                inference_completed_at = time.monotonic()
                guard_stat = Path(args.guard_config).stat()
                signature = (guard_stat.st_ino, guard_stat.st_mtime_ns, guard_stat.st_size)
                if signature != guard_signature:
                    guard = _load_guard(args.guard_config)
                    guard_signature = signature
                    print("Action guard limits loaded: " + json.dumps(json.loads(Path(args.guard_config).read_text())), flush=True)
                decision = guard.check(
                    predicted_action20=guarded_action,
                    current_active10=current_active10,
                    # These are local acquisition ages, not camera-driver/SDK source timestamps.
                    camera_age_s=sampled_at - started,
                    feedback_age_s=sampled_at - started,
                )
                decision.emit_warnings()
                control_sent = False
                if args.enable_held_control and held and not decision.allowed:
                    # A rejected sample ends this hold session; it cannot silently
                    # resume on a later good prediction without a fresh UI press.
                    lease_temp = lease_path.with_suffix(lease_path.suffix + ".runner.tmp")
                    blocked_session_id = session_id
                    lease_temp.write_text(json.dumps({"expires_monotonic_s": 0.0, "speed_percent": speed, "blocked": True, "session_id": session_id}))
                    os.replace(lease_temp, lease_path)
                    held = False
                if args.enable_held_control and held and decision.allowed:
                    # Setup occurs only after an explicit UI lease. Each setup wait is
                    # cancellable by lease expiry; no command is sent without a fresh lease.
                    setup_this_cycle = not command_ready
                    if setup_this_cycle:
                        command_ready = _prepare_held_control(piper, lease_path, session_id, speed, stop_event)
                    # Recheck just before transmission; setup may have consumed the lease.
                    expected_session = session_id
                    held, speed, session_id = _read_lease(lease_path)
                    held = held and session_id == expected_session and not stop_event.is_set()
                    if blocked_session_id is not None and session_id == blocked_session_id:
                        held = False
                    # Setup can wait for seconds. Acquire a new observation and
                    # prediction next cycle instead of sending the pre-setup action.
                    if held and command_ready and not setup_this_cycle:
                        target, gripper_m = _target_from_active(decision.active_action10, calibration["raw_meters_min"], calibration["raw_meters_max"])
                        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=speed,
                                           is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
                        send_endpose(piper, target)
                        piper.GripperCtrl(int(round(gripper_m * 1_000_000)), 1000, 0x01, 0)
                        control_sent = True
                control_hold_sent = False
                if args.enable_held_control and not held and command_ready:
                    # On release/lease loss, replace the last future target once
                    # with the just-measured pose so motion settles at current state.
                    target, gripper_m = _target_from_active(current_active10, calibration["raw_meters_min"], calibration["raw_meters_max"])
                    piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=1,
                                       is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
                    send_endpose(piper, target)
                    piper.GripperCtrl(int(round(gripper_m * 1_000_000)), 1000, 0x01, 0)
                    control_hold_sent = True
                    command_ready = False
                now = time.monotonic()
                stats_frames += 1
                loop_ticks.append(now)
                while loop_ticks and loop_ticks[0] < now - 1.0:
                    loop_ticks.pop(0)
                if control_sent:
                    control_ticks.append(now)
                while control_ticks and control_ticks[0] < now - 1.0:
                    control_ticks.pop(0)
                loop_window_s = min(1.0, max(period_s, now - started_at))
                control_window_s = min(1.0, max(period_s, now - control_ticks[0])) if control_ticks else 1.0
                control_active, _, active_session = _read_lease(lease_path) if args.enable_held_control else (False, 1, None)
                if blocked_session_id is not None and active_session == blocked_session_id:
                    control_active = False
                row = {
                    "frame": frame_index + 1,
                    "timestamp_monotonic_s": sampled_at,
                    "inference_completed_monotonic_s": inference_completed_at,
                    "camera_feedback_acquisition_s": sampled_at - started,
                    "inference_ms": (inference_completed_at - sampled_at) * 1000,
                    "cycle_ms": (now - started) * 1000,
                    "loop_hz": len(loop_ticks) / loop_window_s,
                    "control_hz": len(control_ticks) / control_window_s,
                    "control_active": control_active,
                    "control_sent": control_sent,
                    "control_release_hold_sent": control_hold_sent,
                    "target_hz": 20.0,
                    "deadline_missed": 0,
                    "guard_allowed": decision.allowed,
                    "alerts": [alert.__dict__ for alert in decision.alerts],
                    "predicted_action20": predicted.tolist(),
                    "guarded_action20": guarded_action.tolist(),
                    "predicted_gripper_mm": predicted_gripper_m * 1000 if predicted_gripper_m is not None else None,
                    "gripper_target_mm": gripper_target_m * 1000 if gripper_target_m is not None else None,
                    "gripper_mapping_mm": gripper_config.as_dict(),
                    "current_active10": current_active10.tolist(),
                }
                # Absolute 20 Hz schedule, skipping missed slots instead of bursting.
                next_tick = started + period_s
                now = time.monotonic()
                if now >= next_tick:
                    skipped = int((now - next_tick) // period_s) + 1
                    row["deadline_missed"] = skipped
                    next_tick += skipped * period_s
                else:
                    time.sleep(next_tick - now)
                if recorder is not None:
                    metadata = dict(row, raw_state20=observation.state20.tolist())
                    recorder.submit(observation.global_rgb, observation.wrist_rgb, metadata)
                    row["recording"] = recorder.status()
                stream.write(json.dumps(row) + "\n")
                stream.flush()
    finally:
        adapter.close()
        if recorder is not None:
            recorder.close()
            print("Image recording finished: " + json.dumps(recorder.status()), flush=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
