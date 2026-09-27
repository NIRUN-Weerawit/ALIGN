"""Run one bounded, low-speed Piper V2 Cartesian EndPoseCtrl test.

Default operation is a dry run: it does not import the Piper SDK, open CAN, change
control mode, enable motors, or send a CAN frame. Live transmission requires both
``--live`` and ``--i-understand-this-moves-arm``. Live mode follows the vendor demo:
enable motors, verify driver feedback, select CAN/MOVE-P control at <=10%, then
stream one explicit Cartesian target for a bounded duration. It intentionally does
not invoke the model ActionGuard because this is an operator-specified SDK smoke
 test, not a model-policy action.

Piper EndPoseCtrl format:
  xyz: metres -> integer units of 0.001 mm (1e-6 m)
  Euler XYZ: degrees -> integer units of 0.001 degree
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EndPoseTarget:
    """Cartesian target in model-independent physical units."""

    xyz_m: np.ndarray
    euler_xyz_deg: np.ndarray


class EndPosePiper(Protocol):
    def EndPoseCtrl(self, X: int, Y: int, Z: int, RX: int, RY: int, RZ: int) -> None: ...


def build_endpose_payload(target: EndPoseTarget) -> tuple[int, int, int, int, int, int]:
    """Convert a physical Cartesian target to the six Piper SDK integers."""
    xyz = np.asarray(target.xyz_m, dtype=float)
    euler = np.asarray(target.euler_xyz_deg, dtype=float)
    if xyz.shape != (3,):
        raise ValueError("xyz_m must have shape (3,)")
    if euler.shape != (3,):
        raise ValueError("euler_xyz_deg must have shape (3,)")
    if not np.isfinite(xyz).all() or not np.isfinite(euler).all():
        raise ValueError("target values must be finite")
    return tuple(np.rint(np.concatenate((xyz * 1_000_000, euler * 1_000))).astype(int).tolist())  # type: ignore[return-value]


def prepare_can_cartesian_control(piper: object, *, speed_percent: int, enable_gripper: bool = True) -> None:
    """Issue the vendor demo's motor/CAN setup, optionally preserving gripper target."""
    if not 1 <= speed_percent <= 100:
        raise ValueError("speed_percent must be in [1, 100]")
    getattr(piper, "EnableArm")(7)
    if enable_gripper:
        getattr(piper, "GripperCtrl")(0, 1000, 0x01, 0)
    getattr(piper, "MotionCtrl_2")(
        ctrl_mode=0x01,
        move_mode=0x00,
        move_spd_rate_ctrl=speed_percent,
        is_mit_mode=0x00,
        residence_time=0,
        installation_pos=0x00,
    )


def send_endpose(piper: EndPosePiper, target: EndPoseTarget) -> tuple[int, int, int, int, int, int]:
    """Send a prevalidated target through EndPoseCtrl; no mode/enable commands."""
    payload = build_endpose_payload(target)
    piper.EndPoseCtrl(*payload)
    return payload


def _status_fields(piper: object) -> object:
    status = getattr(piper, "GetArmStatus")()
    return getattr(status, "arm_status", status)


def _verify_live_feedback(piper: object) -> None:
    """Require healthy, live feedback; do not treat a successful send as acceptance."""
    if not getattr(piper, "isOk")():
        raise RuntimeError("Piper SDK connection is unhealthy")
    status = _status_fields(piper)
    if int(getattr(status, "err_code", 0) or 0) != 0:
        raise RuntimeError("Piper reports an arm error; refusing target")
    end_pose = getattr(piper, "GetArmEndPoseMsgs")()
    if float(getattr(end_pose, "Hz", 0.0) or 0.0) <= 0.0:
        raise RuntimeError("end-pose feedback is not live; refusing target")


def _all_drivers_enabled(piper: object) -> bool:
    info = getattr(piper, "GetArmLowSpdInfoMsgs")()
    return all(bool(getattr(getattr(info, f"motor_{index}").foc_status, "driver_enable_status")) for index in range(1, 7))


def _wait_until_enabled(
    piper: object,
    *,
    timeout_s: float = 5.0,
    enable_gripper: bool = True,
    stop_event: object | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while not _all_drivers_enabled(piper):
        if stop_event is not None and bool(getattr(stop_event, "is_set")()):
            raise RuntimeError("manual command was cancelled before drivers enabled")
        getattr(piper, "EnableArm")(7)
        if enable_gripper:
            getattr(piper, "GripperCtrl")(0, 1000, 0x01, 0)
        if time.monotonic() >= deadline:
            raise RuntimeError("all six joint drivers did not report enabled within 5 seconds")
        time.sleep(0.25)


def _wait_for_can_mode(piper: object, *, timeout_s: float = 2.0, stop_event: object | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    while int(getattr(_status_fields(piper), "ctrl_mode", -1)) != 0x01:
        if stop_event is not None and bool(getattr(stop_event, "is_set")()):
            raise RuntimeError("manual command was cancelled before CAN control mode")
        if time.monotonic() >= deadline:
            raise RuntimeError("Piper did not enter CAN command control mode (ctrl_mode=0x01)")
        time.sleep(0.05)


def _stream_endpose(
    piper: object,
    target: EndPoseTarget,
    *,
    speed_percent: int,
    duration_s: float,
    stream_hz: float,
    stop_event: object | None = None,
) -> int:
    """Repeat CAN/MOVE-P + EndPoseCtrl for a bounded, cancellable interval."""
    if duration_s <= 0 or stream_hz <= 0:
        raise ValueError("duration_s and stream_hz must be positive")
    period_s = 1.0 / stream_hz
    deadline = time.monotonic() + duration_s
    sent = 0
    while time.monotonic() < deadline and not (stop_event is not None and bool(getattr(stop_event, "is_set")())):
        tick_start = time.monotonic()
        getattr(piper, "MotionCtrl_2")(
            ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=speed_percent,
            is_mit_mode=0x00, residence_time=0, installation_pos=0x00,
        )
        send_endpose(piper, target)
        sent += 1
        time.sleep(max(0.0, period_s - (time.monotonic() - tick_start)))
    return sent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-m", required=True, type=float, help="target X in metres")
    parser.add_argument("--y-m", required=True, type=float, help="target Y in metres")
    parser.add_argument("--z-m", required=True, type=float, help="target Z in metres")
    parser.add_argument("--rx-deg", required=True, type=float, help="target Euler X in degrees")
    parser.add_argument("--ry-deg", required=True, type=float, help="target Euler Y in degrees")
    parser.add_argument("--rz-deg", required=True, type=float, help="target Euler Z in degrees")
    parser.add_argument("--can", default="can0", help="SocketCAN interface used only with --live")
    parser.add_argument("--speed-percent", type=int, default=10, choices=range(1, 11),
                        metavar="1..10", help="Piper controller speed percentage (default: 10)")
    parser.add_argument("--duration-s", type=float, default=1.0,
                        help="bounded target-stream duration in seconds (default: 1.0)")
    parser.add_argument("--stream-hz", type=float, default=20.0,
                        help="bounded EndPoseCtrl stream rate (default: 20 Hz)")
    parser.add_argument("--feedback-warmup-s", type=float, default=1.0,
                        help="wait for fresh SDK feedback after connecting (default: 1.0 s)")
    parser.add_argument("--live", action="store_true", help="enable, enter CAN/MOVE-P mode, and transmit target")
    parser.add_argument("--i-understand-this-moves-arm", action="store_true", help="required alongside --live")
    args = parser.parse_args(argv)
    if args.live and not args.i_understand_this_moves_arm:
        parser.error("--live requires --i-understand-this-moves-arm")
    if args.duration_s <= 0 or args.stream_hz <= 0 or args.feedback_warmup_s <= 0:
        parser.error("--duration-s, --stream-hz, and --feedback-warmup-s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = EndPoseTarget(
        xyz_m=np.array([args.x_m, args.y_m, args.z_m]),
        euler_xyz_deg=np.array([args.rx_deg, args.ry_deg, args.rz_deg]),
    )
    payload = build_endpose_payload(target)
    print(f"EndPoseCtrl payload: {payload}")
    print("units: XYZ=0.001 mm; Euler XYZ=0.001 degree")

    if not args.live:
        print(
            "DRY RUN: no SDK import, CAN connection, mode change, motor enable, or command transmission.\n"
            f"Live plan: EnableArm(7), verify drivers, CAN/MOVE-P at {args.speed_percent}%, "
            f"then stream at {args.stream_hz:g} Hz for {args.duration_s:g} s."
        )
        return 0

    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    # Fresh SDK envelopes start at Hz=0. Wait for actual CAN feedback before judging readiness.
    time.sleep(args.feedback_warmup_s)
    _verify_live_feedback(piper)
    _wait_until_enabled(piper)
    prepare_can_cartesian_control(piper, speed_percent=args.speed_percent)
    _wait_for_can_mode(piper)
    sent = _stream_endpose(
        piper, target, speed_percent=args.speed_percent,
        duration_s=args.duration_s, stream_hz=args.stream_hz,
    )
    print(f"Transmitted {sent} EndPoseCtrl updates. This is a send count, not proof of controller acceptance or motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
