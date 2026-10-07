"""Validated joint and gripper position mirroring for two Piper V2 arms."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

MASTER_JOINT_PHYSICAL_LIMITS_DEG = ((-153.0, 153.0), (-1.0, 193.0), (-174.0, 0.0),
                                    (-102.0, 102.0), (-74.0, 74.0), (-162.0, 81.0))
SLAVE_JOINT_PHYSICAL_LIMITS_DEG = ((-153.0, 153.0), (-1.0, 193.0), (-177.0, -2.0),
                                   (-102.0, 102.0), (-78.0, 70.0), (-108.0, 134.0))
JOINT_SCALES = tuple((slave_hi - slave_lo) / (master_hi - master_lo)
                     for (master_lo, master_hi), (slave_lo, slave_hi) in
                     zip(MASTER_JOINT_PHYSICAL_LIMITS_DEG, SLAVE_JOINT_PHYSICAL_LIMITS_DEG))
JOINT_BIASES_DEG = tuple(slave_lo - scale * master_lo
                         for scale, (master_lo, _), (slave_lo, _) in
                         zip(JOINT_SCALES, MASTER_JOINT_PHYSICAL_LIMITS_DEG, SLAVE_JOINT_PHYSICAL_LIMITS_DEG))
MASTER_J6_PHYSICAL_LIMITS_DEG = MASTER_JOINT_PHYSICAL_LIMITS_DEG[5]
SLAVE_J6_PHYSICAL_LIMITS_DEG = SLAVE_JOINT_PHYSICAL_LIMITS_DEG[5]
J6_SCALE = JOINT_SCALES[5]
J6_BIAS_DEG = JOINT_BIASES_DEG[5]
JOINT_LIMITS_DEG = SLAVE_JOINT_PHYSICAL_LIMITS_DEG
# Feedback may differ slightly from an in-range command at a physical stop.
# This tolerance applies only to measured feedback; sent targets remain inside
# the exact slave ranges above.
JOINT_FEEDBACK_TOLERANCE_DEG = 2.0
MASTER_FEEDBACK_MIN_HZ = 5.0
CONTROL_HZ = 20.0
MAX_MIRROR_SPEED_PERCENT = 30
MAX_FEEDBACK_AGE_S = 0.35
DEFAULT_MAX_JOINT_GAP_DEG = 20.0
MIN_CONFIGURABLE_JOINT_GAP_DEG = 5.0
MAX_CONFIGURABLE_JOINT_GAP_DEG = 60.0
MAX_TARGET_STEP_DEG = 10.0
J6_MAX_LEAD_DEG = 5.0
MASTER_GRIPPER_PHYSICAL_LIMITS_MM = (-8.1, 64.5)
SLAVE_GRIPPER_PHYSICAL_LIMITS_MM = (-41.5, 48.0)
GRIPPER_SCALE = ((SLAVE_GRIPPER_PHYSICAL_LIMITS_MM[1] - SLAVE_GRIPPER_PHYSICAL_LIMITS_MM[0]) /
                 (MASTER_GRIPPER_PHYSICAL_LIMITS_MM[1] - MASTER_GRIPPER_PHYSICAL_LIMITS_MM[0]))
GRIPPER_BIAS_MM = SLAVE_GRIPPER_PHYSICAL_LIMITS_MM[0] - GRIPPER_SCALE * MASTER_GRIPPER_PHYSICAL_LIMITS_MM[0]
# Master command frames can overshoot the arm's physical gripper endpoints.
# Accept a modest overrun, then saturate to the measured physical range.
MASTER_GRIPPER_COMMAND_OVERRUN_MM = 15.0
SLAVE_GRIPPER_FEEDBACK_SLACK_MM = 2.0
GRIPPER_MAX_LEAD_MM = 5.0
GRIPPER_EFFORT = 1000


def _feedback_age(message: Any) -> float:
    """The installed Piper SDK uses nanoseconds for gripper timestamps."""
    timestamp = float(getattr(message, "time_stamp", 0.0) or 0.0)
    if not np.isfinite(timestamp) or timestamp <= 0:
        return float("inf")
    if timestamp > 1e12:
        timestamp /= 1e9
    return time.time() - timestamp


def _stream_age_label(age: float) -> str:
    return f"{age:.2f} s old" if np.isfinite(age) else "no timestamp"


def _fresh_joints(message: Any, field: str, label: str) -> np.ndarray:
    """Copy all six SDK values before its mutable feedback object changes."""
    hz = float(getattr(message, "Hz", 0.0) or 0.0)
    age = _feedback_age(message)
    if not np.isfinite(hz) or hz < MASTER_FEEDBACK_MIN_HZ or not 0 <= age <= MAX_FEEDBACK_AGE_S:
        raise RuntimeError(f"{label} is stale/slow ({hz:.1f} Hz, {_stream_age_label(age)})")
    state = getattr(message, field)
    values = np.asarray([getattr(state, f"joint_{index}") / 1000.0 for index in range(1, 7)], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite values")
    return values


def _fresh_gripper_mm(piper: Any, label: str, *, command: bool = False) -> float:
    message = piper.GetArmGripperCtrl() if command else piper.GetArmGripperMsgs()
    hz = float(getattr(message, "Hz", 0.0) or 0.0)
    age = _feedback_age(message)
    if not np.isfinite(hz) or hz < MASTER_FEEDBACK_MIN_HZ or not 0 <= age <= MAX_FEEDBACK_AGE_S:
        raise RuntimeError(f"{label} is stale/slow ({hz:.1f} Hz, {_stream_age_label(age)})")
    state = message.gripper_ctrl if command else message.gripper_state
    if command:
        code = int(state.status_code)
        if code not in (0x01, 0x03):
            raise RuntimeError(f"{label} is disabled (status_code=0x{code:02x})")
        if int(state.set_zero) != 0:
            raise RuntimeError(f"{label} requests a zero-setting operation")
    value = float(state.grippers_angle) / 1000.0
    lo, hi = MASTER_GRIPPER_PHYSICAL_LIMITS_MM if command else SLAVE_GRIPPER_PHYSICAL_LIMITS_MM
    slack = MASTER_GRIPPER_COMMAND_OVERRUN_MM if command else SLAVE_GRIPPER_FEEDBACK_SLACK_MM
    if not np.isfinite(value) or not lo - slack <= value <= hi + slack:
        raise RuntimeError(f"{label}={value:.2f} mm outside [{lo - slack:g}, {hi + slack:g}] mm guard")
    return value


def mapped_gripper_target(master_command_mm: float) -> tuple[float, float]:
    """Saturate master command to its physical range, then map to slave range."""
    master_lo, master_hi = MASTER_GRIPPER_PHYSICAL_LIMITS_MM
    bounded_master_mm = float(np.clip(master_command_mm, master_lo, master_hi))
    slave_target_mm = GRIPPER_SCALE * bounded_master_mm + GRIPPER_BIAS_MM
    slave_lo, slave_hi = SLAVE_GRIPPER_PHYSICAL_LIMITS_MM
    return bounded_master_mm, float(np.clip(slave_target_mm, slave_lo, slave_hi))


def validate_joint_gap(value: Any) -> float:
    try:
        gap = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max joint gap must be 5–60 degrees") from exc
    if not np.isfinite(gap) or not MIN_CONFIGURABLE_JOINT_GAP_DEG <= gap <= MAX_CONFIGURABLE_JOINT_GAP_DEG:
        raise ValueError("max joint gap must be 5–60 degrees")
    return gap


def slave_measured_joints(slave: Any) -> np.ndarray:
    joints = _fresh_joints(slave.GetArmJointMsgs(), "joint_state", "slave joint feedback")
    outside = [f"J{i}={value:.2f}° outside feedback [{lo - JOINT_FEEDBACK_TOLERANCE_DEG:g}, "
               f"{hi + JOINT_FEEDBACK_TOLERANCE_DEG:g}]°"
               for i, (value, (lo, hi)) in enumerate(zip(joints, JOINT_LIMITS_DEG), 1)
               if not lo - JOINT_FEEDBACK_TOLERANCE_DEG <= value <= hi + JOINT_FEEDBACK_TOLERANCE_DEG]
    if outside:
        raise RuntimeError("slave joint feedback exceeds configured limits: " + ", ".join(outside))
    return joints


def _check_slave_health(slave: Any, *, require_can_mode: bool = False) -> None:
    if not slave.isOk():
        raise RuntimeError("slave CAN connection is unhealthy")
    status_msg = slave.GetArmStatus()
    status = getattr(status_msg, "arm_status", status_msg)
    if float(getattr(status_msg, "Hz", 0.0) or 0.0) < 1.0:
        raise RuntimeError("slave status feedback is unavailable")
    arm_status = int(getattr(status, "arm_status", -1))
    err_code = int(getattr(status, "err_code", -1))
    if arm_status != 0 or err_code != 0:
        raise RuntimeError(f"slave controller reports arm_status={arm_status}, err_code=0x{err_code & 0xffff:04x}")
    if require_can_mode and int(getattr(status, "ctrl_mode", -1)) != 0x01:
        raise RuntimeError("slave left CAN control mode")


def hold_slave_position(slave: Any) -> list[float]:
    """Replace the last moving target with the nearest in-range measured pose."""
    _check_slave_health(slave, require_can_mode=True)
    joints = slave_measured_joints(slave)
    limits = np.asarray(SLAVE_JOINT_PHYSICAL_LIMITS_DEG, dtype=float)
    held = np.clip(joints, limits[:, 0], limits[:, 1])
    raw = tuple(np.rint(held * 1000.0).astype(int).tolist())
    slave.JointCtrl(*raw)
    return held.tolist()


def map_joint_positions(master_joints_deg: Sequence[float], offsets_deg: Sequence[float]
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map all six physical ranges and bound targets after applying offsets."""
    source = np.asarray(master_joints_deg, dtype=float)
    offsets = np.asarray(offsets_deg, dtype=float)
    if source.shape != (6,) or not np.isfinite(source).all():
        raise ValueError("master joints must contain six finite degree values")
    if offsets.shape != (6,) or not np.isfinite(offsets).all():
        raise ValueError("joint offsets must contain six finite degree values")
    master_limits = np.asarray(MASTER_JOINT_PHYSICAL_LIMITS_DEG, dtype=float)
    slave_limits = np.asarray(SLAVE_JOINT_PHYSICAL_LIMITS_DEG, dtype=float)
    bounded_master = np.clip(source, master_limits[:, 0], master_limits[:, 1])
    mapped = bounded_master * np.asarray(JOINT_SCALES) + np.asarray(JOINT_BIASES_DEG)
    target = np.clip(mapped + offsets, slave_limits[:, 0], slave_limits[:, 1])
    return bounded_master, mapped, target


def _mapped_joint_command(master: Any, offsets_deg: Sequence[float]
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not master.isOk():
        raise RuntimeError("master CAN connection is unhealthy")
    source = _fresh_joints(master.GetArmJointCtrl(), "joint_ctrl", "master joint command stream")
    return (source, *map_joint_positions(source, offsets_deg))


def mapped_joint_target(master: Any, offsets_deg: Sequence[float]) -> np.ndarray:
    """Read the master's outgoing 0x155–0x157 stream and map to the slave."""
    return _mapped_joint_command(master, offsets_deg)[3]


def run_joint_mirror(master: Any, slave: Any, offsets_deg: Sequence[float], speed_percent: int,
                     stop_event: threading.Event, on_update=lambda _state: None,
                     command_lock: Any = None, lease_ok: Callable[[], bool] = lambda: True,
                     mirror_gripper: bool = False,
                     max_joint_gap_deg: float = DEFAULT_MAX_JOINT_GAP_DEG) -> None:
    """Stream master joint and gripper commands to the measured slave pose."""
    if not 1 <= int(speed_percent) <= MAX_MIRROR_SPEED_PERCENT:
        raise ValueError(f"master-slave speed must be 1–{MAX_MIRROR_SPEED_PERCENT} percent")
    max_joint_gap_deg = validate_joint_gap(max_joint_gap_deg)
    from piper_xvla.endpose_control import _all_drivers_enabled, _wait_for_can_mode
    lock = command_lock or threading.Lock()

    # New SDK connections need time to populate their streams. Do this before
    # enabling the slave or switching its mode.
    ready_by = time.monotonic() + 2.0
    while not stop_event.is_set():
        if not lease_ok():
            raise RuntimeError("Web UI master-slave heartbeat expired")
        try:
            target_deg = mapped_joint_target(master, offsets_deg)
            measured_deg = slave_measured_joints(slave)
            if mirror_gripper:
                _fresh_gripper_mm(master, "master gripper command stream", command=True)
                _fresh_gripper_mm(slave, "slave gripper feedback")
            _check_slave_health(slave)
            break
        except RuntimeError:
            if time.monotonic() >= ready_by:
                raise
            stop_event.wait(0.05)
    else:
        return

    # The hardware master may already be sending to a slave on the same CAN
    # bus. Refuse to introduce a second sender on that bus.
    existing_ctrl_hz = float(getattr(slave.GetArmJointCtrl(), "Hz", 0.0) or 0.0)
    if existing_ctrl_hz >= MASTER_FEEDBACK_MIN_HZ:
        raise RuntimeError(f"slave CAN already has joint commands ({existing_ctrl_hz:.1f} Hz); isolate the buses before mirroring")
    gap = np.abs(target_deg[:5] - measured_deg[:5])
    if np.any(gap > max_joint_gap_deg):
        joints = ", ".join(f"J{i}={value:.1f}°" for i, value in enumerate(gap, 1) if value > max_joint_gap_deg)
        raise RuntimeError(f"master/slave start poses differ by more than {max_joint_gap_deg:g}°: {joints}")

    enable_by = time.monotonic() + 5.0
    while not _all_drivers_enabled(slave):
        with lock:
            if stop_event.is_set():
                return
            if not lease_ok():
                raise RuntimeError("Web UI master-slave heartbeat expired")
            slave.EnableArm(7)
        if time.monotonic() >= enable_by:
            raise RuntimeError("slave joint drivers did not enable within 5 seconds")
        stop_event.wait(0.25)
    if stop_event.is_set():
        return
    with lock:
        if stop_event.is_set():
            return
        if not lease_ok():
            raise RuntimeError("Web UI master-slave heartbeat expired")
        slave.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=int(speed_percent),
                           is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
    _wait_for_can_mode(slave, stop_event=stop_event)

    period = 1.0 / CONTROL_HZ
    deadline = time.monotonic()
    sent = 0
    previous_target = None
    while not stop_event.is_set():
        if not lease_ok():
            raise RuntimeError("Web UI master-slave heartbeat expired")
        source_deg, bounded_master_deg, mapped_deg, desired_deg = _mapped_joint_command(master, offsets_deg)
        measured_deg = slave_measured_joints(slave)
        gripper_update = {}
        if mirror_gripper:
            master_gripper_mm = _fresh_gripper_mm(master, "master gripper command stream", command=True)
            measured_gripper_mm = _fresh_gripper_mm(slave, "slave gripper feedback")
            bounded_master_mm, desired_gripper_mm = mapped_gripper_target(master_gripper_mm)
            gripper_target_mm = float(np.clip(desired_gripper_mm,
                                              measured_gripper_mm - GRIPPER_MAX_LEAD_MM,
                                              measured_gripper_mm + GRIPPER_MAX_LEAD_MM))
            gripper_target_mm = float(np.clip(gripper_target_mm, *SLAVE_GRIPPER_PHYSICAL_LIMITS_MM))
            gripper_update = {"master_gripper_mm": round(master_gripper_mm, 2),
                              "bounded_master_gripper_mm": round(bounded_master_mm, 2),
                              "mapped_gripper_mm": round(desired_gripper_mm, 2),
                              "gripper_input_clamped": bounded_master_mm != master_gripper_mm,
                              "slave_gripper_mm": round(measured_gripper_mm, 2),
                              "gripper_target_mm": round(gripper_target_mm, 2),
                              "gripper_remaining_mm": round(desired_gripper_mm - gripper_target_mm, 2)}
        target_deg = desired_deg.copy()
        # J6 can begin far from the inverse-mapped master pose. Advance only
        # a small amount ahead of measured feedback instead of commanding the
        # full gap at startup. The regular tracking check still applies to
        # this bounded command target.
        target_deg[5] = np.clip(desired_deg[5], measured_deg[5] - J6_MAX_LEAD_DEG,
                                measured_deg[5] + J6_MAX_LEAD_DEG)
        target_deg[5] = np.clip(target_deg[5], *SLAVE_J6_PHYSICAL_LIMITS_DEG)
        _check_slave_health(slave, require_can_mode=True)
        if not _all_drivers_enabled(slave):
            raise RuntimeError("slave joint drivers are no longer enabled")
        if previous_target is not None and np.any(np.abs(target_deg - previous_target) > MAX_TARGET_STEP_DEG):
            raise RuntimeError(f"master target changed by more than {MAX_TARGET_STEP_DEG:g}° in one control tick")
        if np.any(np.abs(target_deg - measured_deg) > max_joint_gap_deg):
            raise RuntimeError(f"slave tracking error exceeds {max_joint_gap_deg:g}°")
        target_raw = tuple(np.rint(target_deg * 1000.0).astype(int).tolist())
        with lock:
            if stop_event.is_set():
                break
            if not lease_ok():
                raise RuntimeError("Web UI master-slave heartbeat expired")
            slave.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=int(speed_percent),
                               is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
            if mirror_gripper:
                # Only mirror opening. Never forward a master disable,
                # fault-clear or zero-setting command to the slave.
                slave.GripperCtrl(int(round(gripper_target_mm * 1000)), GRIPPER_EFFORT, 0x01, 0x00)
            slave.JointCtrl(*target_raw)
            sent += 1
            previous_target = target_deg
            on_update({"sent": sent, "target_joints_deg": target_deg.tolist(), "control_hz": CONTROL_HZ,
                       "master_joints_deg": source_deg.tolist(),
                       "bounded_master_joints_deg": bounded_master_deg.tolist(),
                       "mapped_joints_deg": mapped_deg.tolist(),
                       "clamped_master_joints": [f"J{i}" for i in range(1, 7)
                                                 if source_deg[i - 1] != bounded_master_deg[i - 1]],
                       "desired_j6_deg": float(desired_deg[5]),
                       "j6_remaining_deg": round(float(desired_deg[5] - target_deg[5]), 2),
                       **gripper_update})
        deadline += period
        stop_event.wait(max(0.0, deadline - time.monotonic()))
