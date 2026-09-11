"""Read-only readiness check for a real Piper V2 arm.

This tool never enables, moves, homes, replays, or stops the arm. It only opens
the SDK connection and reads CAN/status/end-pose/joint/gripper feedback rates.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PiperReadinessReport:
    ready: bool
    can_hz: float
    streams_hz: dict[str, float]
    arm_status: int
    ctrl_mode: int
    teach_status: int
    motion_status: int
    err_code: int
    errors: list[str]
    warnings: list[str]


def _rate(message: Any) -> float:
    return float(getattr(message, "Hz", 0.0) or 0.0)


def assess_piper_readiness(piper: Any, min_stream_hz: float = 10.0) -> PiperReadinessReport:
    """Read status only and decide whether collection can safely be considered ready."""
    if min_stream_hz <= 0:
        raise ValueError("min_stream_hz must be positive")
    errors: list[str] = []
    warnings: list[str] = []

    try:
        connected = bool(piper.isOk())
    except Exception as exc:  # noqa: BLE001
        connected = False
        errors.append(f"SDK health query failed: {exc}")
    if not connected:
        errors.append("Piper SDK connection is not healthy")

    try:
        can_hz = float(piper.GetCanFps() or 0.0)
    except Exception as exc:  # noqa: BLE001
        can_hz = 0.0
        errors.append(f"CAN rate query failed: {exc}")
    if can_hz < min_stream_hz:
        errors.append(f"CAN stream is too slow or absent: {can_hz:.1f} Hz < {min_stream_hz:.1f} Hz")

    try:
        status = piper.GetArmStatus()
        streams = {
            "status": _rate(status),
            "end_pose": _rate(piper.GetArmEndPoseMsgs()),
            "joint": _rate(piper.GetArmJointMsgs()),
            "gripper": _rate(piper.GetArmGripperMsgs()),
        }
    except Exception as exc:  # noqa: BLE001
        status = None
        streams = {"status": 0.0, "end_pose": 0.0, "joint": 0.0, "gripper": 0.0}
        errors.append(f"Piper sensor query failed: {exc}")

    for name, hz in streams.items():
        if hz < min_stream_hz:
            errors.append(f"{name} feedback stream is too slow or absent: {hz:.1f} Hz < {min_stream_hz:.1f} Hz")

    # Piper SDK V2 wraps ArmMsgFeedbackStatus in a timestamp/Hz envelope.
    # Older SDK variants return the status object directly.
    status_fields = getattr(status, "arm_status", status)
    if not isinstance(status_fields, (int, float)) and hasattr(status_fields, "ctrl_mode"):
        status = status_fields

    arm_status = int(getattr(status, "arm_status", -1))
    ctrl_mode = int(getattr(status, "ctrl_mode", -1))
    teach_status = int(getattr(status, "teach_status", -1))
    motion_status = int(getattr(status, "motion_status", -1))
    err_code = int(getattr(status, "err_code", -1))
    if arm_status != 0:
        errors.append(f"Piper arm_status is not normal: {arm_status}")
    if err_code != 0:
        errors.append(f"Piper reports non-zero err_code: 0x{err_code:04x}")
    if ctrl_mode == 0x02:
        warnings.append("Piper is in teaching control mode (ctrl_mode=0x02); drag-teach status may still be DISABLED")
    if ctrl_mode == 0x07:
        warnings.append("Piper is currently in offline trajectory mode; ensure it is idle before a new collection run")
    if teach_status not in (0, -1):
        warnings.append(f"Piper teaching/replay state is active: {teach_status}")

    return PiperReadinessReport(
        ready=not errors, can_hz=can_hz, streams_hz=streams,
        arm_status=arm_status, ctrl_mode=ctrl_mode, teach_status=teach_status,
        motion_status=motion_status, err_code=err_code, errors=errors, warnings=warnings,
    )


def connect_readonly(can_device: str):
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise RuntimeError("piper_sdk/python-can is unavailable in this interpreter") from exc
    piper = C_PiperInterface_V2(can_device)
    piper.ConnectPort(True)
    return piper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--min-stream-hz", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=1.0,
                        help="seconds to receive Piper feedback after connecting")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.warmup_s < 0:
            parser.error("--warmup-s must be non-negative")
        piper = connect_readonly(args.can)
        # SDK feedback rates are initially zero until a few CAN frames arrive.
        time.sleep(args.warmup_s)
        report = assess_piper_readiness(piper, args.min_stream_hz)
    except Exception as exc:  # noqa: BLE001
        report = PiperReadinessReport(False, 0.0, {}, -1, -1, -1, -1, -1, [str(exc)], [])

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(f"READY: {'YES' if report.ready else 'NO'}")
        print(f"CAN: {report.can_hz:.1f} Hz")
        for name, hz in report.streams_hz.items(): print(f"{name}: {hz:.1f} Hz")
        print(f"status: arm={report.arm_status} mode={report.ctrl_mode} teach={report.teach_status} motion={report.motion_status} err=0x{report.err_code:04x}")
        for item in report.warnings: print(f"WARNING: {item}")
        for item in report.errors: print(f"ERROR: {item}")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
