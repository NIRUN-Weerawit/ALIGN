"""Safe command surface for Piper drag-teach/replay collection.

Defaults to dry-run. Add --live only after confirming the correct CAN device,
clear workspace, enabled arm, and a human emergency-stop observer.

An emergency stop deliberately does NOT move the arm. Use recover-to-start as a
separate deliberate command after the emergency condition is cleared.

Replay lifecycle (drag-teach reproduction):
  The pendant replay button does TWO things in sequence:
    grag_teach_ctrl=0x07  move to the trajectory's START pose
    grag_teach_ctrl=0x03  execute / reproduce the taught trajectory
  A bare 0x03 only works if the arm is already at the start pose - which is why
  manually "priming" with the pendant made a CAN-only 0x03 work. `start_replay`
  therefore sends 0x07, waits for the arm to settle at the start, then sends 0x03.
  No control-mode (MotionCtrl_2) frame is sent: drag-teach reproduction runs from
  the controller's current mode.

The collector must use copied joint/EE motion—not ctrl_mode or teach_status—to
observe that a replay started.
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from typing import Protocol


class PiperReplaySDK(Protocol):
    def MotionCtrl_1(self, emergency_stop: int = 0, track_ctrl: int = 0, grag_teach_ctrl: int = 0) -> None: ...
    def MotionCtrl_2(self, ctrl_mode: int = 0, move_mode: int = 0, move_spd_rate_ctrl: int = 0,
                     is_mit_mode: int = 0, residence_time: int = 0, installation_pos: int = 0) -> None: ...
    def EmergencyStop(self, command: int) -> None: ...
    def GetArmStatus(self) -> object: ...
    def GetArmJointMsgs(self) -> object: ...
    def GetArmLowSpdInfoMsgs(self) -> object: ...
    def isOk(self) -> bool: ...


OFFLINE_MODE = 0x07
TEACHING_MODE = 0x02


def status_fields(piper: PiperReplaySDK):
    """Return flat status fields from either SDK envelope layout.

    Newer V2 SDK wraps ArmMsgFeedbackStatus in a timestamp/Hz envelope; older
    variants return the status object directly.
    """
    status = piper.GetArmStatus()
    return getattr(status, "arm_status", status)


class PiperReplayController:
    """Small, auditable mapping of collection lifecycle verbs to Piper SDK calls."""

    def __init__(self, piper: PiperReplaySDK, replay_speed_percent: int = 30,
                 installation_pos: int = 0x00, settle_s: float = 0.5):
        if not 1 <= replay_speed_percent <= 100:
            raise ValueError("replay_speed_percent must be in [1, 100]")
        if installation_pos not in (0x00, 0x01, 0x02, 0x03):
            raise ValueError("installation_pos must be 0x00 (keep) or 0x01/0x02/0x03")
        if settle_s < 0:
            raise ValueError("settle_s must be non-negative")
        self.piper = piper
        self.speed = replay_speed_percent
        self.installation_pos = installation_pos
        self.settle_s = settle_s

    # -- status helpers -----------------------------------------------------

    def current_ctrl_mode(self) -> int:
        return int(getattr(status_fields(self.piper), "ctrl_mode", -1))

    def _wait_for_mode(self, target: int, timeout_s: float = 3.0) -> int:
        """Poll until ctrl_mode == target; return the mode actually observed."""
        deadline = time.time() + timeout_s
        last = self.current_ctrl_mode()
        while last != target:
            if time.time() > deadline:
                return last
            time.sleep(0.1)
            last = self.current_ctrl_mode()
        return last

    def _wait_for_offline_mode(self, timeout_s: float = 3.0) -> None:
        last = self._wait_for_mode(OFFLINE_MODE, timeout_s=timeout_s)
        if last != OFFLINE_MODE:
            raise RuntimeError(
                f"Piper did not enter offline trajectory mode (ctrl_mode=0x{OFFLINE_MODE:02x}); "
                f"last seen ctrl_mode=0x{last:02x}. Check installation_pos / firmware, or press "
                "the pendant replay-mode button and retry."
            )

    # -- drag-teach recording (SDK-driven; optional) ------------------------

    def start_recording(self) -> None:
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)

    def stop_recording(self) -> None:
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)

    def discard_current_recording(self) -> None:
        # 0x03 clears the current trajectory; never clear all trajectories here.
        self.piper.MotionCtrl_1(track_ctrl=0x03)

    # -- replay --------------------------------------------------------------

    def _wait_for_settle(self, timeout_s: float = 5.0, threshold_deg: float = 0.05) -> None:
        """Wait until joint motion stops (arm reached its target), or timeout."""
        from piper_xvla.motion_watch import joint_positions_deg, max_joint_delta_deg
        previous = joint_positions_deg(self.piper.GetArmJointMsgs())
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(0.1)
            current = joint_positions_deg(self.piper.GetArmJointMsgs())
            if max_joint_delta_deg(previous, current) < threshold_deg:
                return
            previous = current

    def start_replay(self) -> None:
        """Replicate the pendant replay button: move to start, then execute.

        A bare 0x03 only works when the arm is already at the trajectory's start
        pose (which is why manual pendant-priming made CAN replay work). The
        pendant does 0x07 (move to start) THEN 0x03 (execute); we do the same so
        no priming is required. No control-mode change is sent - drag-teach
        reproduction runs from the controller's current mode.
        """
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x07)  # move to trajectory start
        self._wait_for_settle(timeout_s=max(5.0, self.settle_s * 10))  # wait until it arrives
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x03)  # execute / reproduce

    def stop_replay(self) -> None:
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x06)

    def pause_replay(self) -> None:
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x04)

    def resume_replay(self) -> None:
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x05)

    def move_to_start(self) -> None:
        self.piper.MotionCtrl_1(grag_teach_ctrl=0x07)

    # -- safety --------------------------------------------------------------

    def emergency_stop(self) -> None:
        # Never combine E-stop and motion: a stopped arm must remain stopped.
        self.piper.EmergencyStop(0x01)

    def recover_and_move_to_start(self) -> None:
        self.piper.EmergencyStop(0x02)
        self.move_to_start()


class _FakeJointState:
    def __init__(self):
        # 0.001-degree units; all zero (arm stationary in dry-run).
        for i in range(1, 7):
            setattr(self, f"joint_{i}", 0)


class _FakeFocStatus:
    def __init__(self, enabled: bool):
        self.driver_enable_status = enabled


class _FakeMotorLowSpd:
    def __init__(self, enabled: bool):
        self.foc_status = _FakeFocStatus(enabled)


class _FakeLowSpdInfo:
    def __init__(self, enabled: bool):
        self.Hz = 200.0 if enabled else 0.0
        for i in range(1, 7):
            setattr(self, f"motor_{i}", _FakeMotorLowSpd(enabled))


class DryRunPiper:
    """Prints SDK calls and simulates mode transitions; used unless --live."""

    def __init__(self, start_ctrl_mode: int = TEACHING_MODE):
        self.ctrl_mode = start_ctrl_mode
        self.calls: list[tuple] = []
        self._joint = _FakeJointState()
        self._enabled = False

    def isOk(self):
        return True

    def GetArmJointMsgs(self):
        return SimpleNamespace(joint_state=self._joint)

    def MotionCtrl_1(self, emergency_stop=0, track_ctrl=0, grag_teach_ctrl=0):
        print(f"DRY-RUN MotionCtrl_1(emergency_stop=0x{emergency_stop:02x}, track_ctrl=0x{track_ctrl:02x}, grag_teach_ctrl=0x{grag_teach_ctrl:02x})")
        self.calls.append(("motion1", emergency_stop, track_ctrl, grag_teach_ctrl))

    def MotionCtrl_2(self, ctrl_mode=0, move_mode=0, move_spd_rate_ctrl=0, is_mit_mode=0, residence_time=0, installation_pos=0):
        print(f"DRY-RUN MotionCtrl_2(ctrl_mode=0x{ctrl_mode:02x}, move_mode=0x{move_mode:02x}, speed={move_spd_rate_ctrl}%, installation_pos=0x{installation_pos:02x})")
        self.calls.append(("motion2", ctrl_mode, move_mode, move_spd_rate_ctrl, is_mit_mode, residence_time, installation_pos))
        # Simulate the controller accepting the mode change. CAN command control
        # additionally requires energized drivers (see cmd_enable).
        if ctrl_mode in (0x00, 0x03, 0x04, 0x07):
            self.ctrl_mode = ctrl_mode
        elif ctrl_mode == 0x01:
            if self._enabled:
                self.ctrl_mode = 0x01

    def EmergencyStop(self, command):
        print(f"DRY-RUN EmergencyStop(0x{command:02x})")
        self.calls.append(("estop", command))

    def GetArmStatus(self):
        return SimpleNamespace(ctrl_mode=self.ctrl_mode)

    # -- driver enable/disable (mirrors piper_enable.py / piper_disable.py) ----

    def EnableArm(self, motor_num=7, enable_flag=0x02):
        print(f"DRY-RUN EnableArm(motor_num={motor_num})")
        self.calls.append(("enable_arm", motor_num))
        self._enabled = True

    def DisableArm(self, motor_num=7, enable_flag=0x01):
        print(f"DRY-RUN DisableArm(motor_num={motor_num})")
        self.calls.append(("disable_arm", motor_num))
        self._enabled = False

    def GripperCtrl(self, gripper_angle=0, gripper_effort=0, gripper_code=0x00, set_zero=0):
        print(f"DRY-RUN GripperCtrl(angle={gripper_angle}, effort={gripper_effort}, code=0x{gripper_code:02x})")
        self.calls.append(("gripper_ctrl", gripper_angle, gripper_effort, gripper_code))

    def GetArmLowSpdInfoMsgs(self):
        return _FakeLowSpdInfo(self._enabled)


def connect_live_piper(can_device: str):
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise RuntimeError("piper_sdk/python-can is unavailable in this interpreter; use the Piper SDK environment.") from exc
    piper = C_PiperInterface_V2(can_device)
    piper.ConnectPort()
    if not piper.isOk():
        raise RuntimeError(f"Piper connection on {can_device!r} is not healthy")
    return piper


INSTALLATION_CHOICES = {"keep": 0x00, "horizontal": 0x01, "side-left": 0x02, "side-right": 0x03}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "record-start", "record-stop", "discard",
        "replay", "stop", "pause", "resume", "move-to-start",
        "estop", "recover-to-start", "status",
    ])
    parser.add_argument("--can", default="can0", help="SocketCAN interface (default: can0)")
    parser.add_argument("--live", action="store_true", help="actually send CAN commands; default is dry-run")
    parser.add_argument("--i-understand-this-moves-arm", action="store_true",
                        help="required for live replay, resume, move-to-start, and recovery")
    args = parser.parse_args()

    moves_arm = args.command in {"replay", "resume", "move-to-start", "recover-to-start"}
    if args.live and moves_arm and not args.i_understand_this_moves_arm:
        parser.error("--live motion requires --i-understand-this-moves-arm")

    piper = connect_live_piper(args.can) if args.live else DryRunPiper()
    control = PiperReplayController(piper)
    actions = {
        "record-start": control.start_recording,
        "record-stop": control.stop_recording,
        "discard": control.discard_current_recording,
        "replay": control.start_replay,
        "stop": control.stop_replay,
        "pause": control.pause_replay,
        "resume": control.resume_replay,
        "move-to-start": control.move_to_start,
        "estop": control.emergency_stop,
        "recover-to-start": control.recover_and_move_to_start,
    }
    if args.command == "status":
        fields = status_fields(piper)
        print(f"ctrl_mode=0x{int(getattr(fields, 'ctrl_mode', -1)):02x} "
              f"teach_status=0x{int(getattr(fields, 'teach_status', -1)):02x} "
              f"motion_status=0x{int(getattr(fields, 'motion_status', -1)):02x} "
              f"err_code=0x{int(getattr(fields, 'err_code', -1)):04x}")
        return 0
    actions[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
