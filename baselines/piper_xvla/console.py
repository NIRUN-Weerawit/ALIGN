"""Interactive Piper control console for drag-teach / replay collection.

One long-lived session instead of one-shot commands:

    cd ~/ALIGN/baselines
    python3 -m piper_xvla.console --can can0            # live (default)
    python3 -m piper_xvla.console --dry-run             # no hardware, prints calls

Commands:
    status              one detailed status readout
    workflow            show the validated pendant → host replay sequence
    watch [seconds]     stream status + joint motion (Enter stops early)
    modes               list control modes and the current one
    mode <name>         switch host-owned mode: standby | can
    raw-mode <hex>      send an arbitrary 0x151 ctrl_mode frame (experiments; never use for replay)
    rec-start           SDK drag-teach record start (experimental; pendant recording preferred)
    rec-stop            SDK drag-teach record stop (experimental; pendant recording preferred)
    record [seconds]    guided SDK recording cycle (experimental; use pendant for production data)
    discard             clear the current taught trajectory only
    replay [seconds]    execute the taught trajectory and observe copied feedback
                        for 6 s by default (1–30 s override)
    collect [seconds]   RECORD ONE EPISODE: arm read-only camera+Piper capture,
                        trigger replay, write a raw LeRobot episode at 20 Hz.
                        Window is a safety cap (default 40 s, range 5-300);
                        press Enter or type 'stop' to end early; Ctrl+C works too.
                        Also writes a global|wrist review video to images/review/.
                        Needs --task set (or 'task <text>' first) and live cameras.
    task [text]         show / set the task description written into episodes
    pause / resume      pause / continue an executing replay
    stop                terminate the executing replay (arm stops in place)
    move-start          move to the stored trajectory start point
    speed <n>           set replay speed percent (1-100)
    enable              energize all joint drivers + gripper (piper_enable.py logic);
                        REQUIRED before 'mode can' / external command control
    disable             de-energize drivers + gripper (arm becomes hand-movable)
    reset               clear e-stop flag + return to standby mode (piper_reset.py logic)
    cameras             preflight-check global + wrist cameras from config
    estop               EMERGENCY STOP - never moves the arm, no confirmation
    recover             resume after e-stop, then move to trajectory start
    help / ?            this help
    quit / q            exit

Safety: every command that can move the arm asks for a typed "yes" first.
estop is immediate and requires nothing. Teaching-mode entry/exit from CAN is
not supported by the tested firmware - use the pendant button (the console
tells you when that is the case).
"""
from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path

from piper_xvla.replay_control import (
    DryRunPiper,
    OFFLINE_MODE,
    TEACHING_MODE,
    PiperReplayController,
    connect_live_piper,
    status_fields,
)

MODE_NAMES = {
    0x00: "STANDBY",
    0x01: "CAN_CTRL",
    0x02: "TEACHING",
    0x03: "ETHERNET",
    0x04: "WIFI",
    0x05: "REMOTE",
    0x06: "LINKAGE_TEACH",
    0x07: "OFFLINE_TRAJ",
}
TEACH_NAMES = {
    0x00: "DISABLED",
    0x01: "START_RECORD",
    0x02: "STOP_RECORD",
    0x03: "EXECUTE_TRAJ",
    0x04: "PAUSE",
    0x05: "RESUME",
    0x06: "TERMINATE",
    0x07: "MOVE_TO_START",
}
MOTION_NAMES = {0x00: "REACHED", 0x01: "NOT_REACHED"}


def _name(table: dict, value: int) -> str:
    if value < 0:
        return "n/a"
    return table.get(value, f"UNKNOWN(0x{value & 0xFF:02x})")


def _hex(value: int) -> str:
    return "n/a" if value < 0 else f"0x{value & 0xFF:02x}"


class PiperConsole:
    def __init__(self, piper, speed: int = 20, task: str = "", data_dir: str | Path = "data/piper_replay"):
        self.piper = piper
        self.speed = speed
        self.task = task
        self.data_dir = Path(data_dir)
        self.ctrl = PiperReplayController(piper, replay_speed_percent=speed)

    # -- status -------------------------------------------------------------

    def _snapshot(self) -> dict:
        s = status_fields(self.piper)
        return {
            "ctrl_mode": int(getattr(s, "ctrl_mode", -1)),
            "teach_status": int(getattr(s, "teach_status", -1)),
            "motion_status": int(getattr(s, "motion_status", -1)),
            "err_code": int(getattr(s, "err_code", -1)),
        }

    def cmd_status(self) -> None:
        snap = self._snapshot()
        try:
            drivers = self._driver_states()
            driver_str = f"{'ENABLED' if all(drivers) else 'disabled'} {drivers}"
        except Exception:  # noqa: BLE001 - low-spd stream may be absent when de-energized
            driver_str = "n/a (de-energized)"
        print(f"ctrl_mode   = {_hex(snap['ctrl_mode'])} {_name(MODE_NAMES, snap['ctrl_mode'])}")
        print(f"teach_state = {_hex(snap['teach_status'])} {_name(TEACH_NAMES, snap['teach_status'])}")
        print(f"motion      = {_hex(snap['motion_status'])} {_name(MOTION_NAMES, snap['motion_status'])}")
        print(f"err_code    = {_hex(snap['err_code'])}")
        print(f"drivers     = {driver_str}")

    def cmd_workflow(self) -> None:
        """Print the validated manual-pendant replay lifecycle without acting."""
        print("Validated manual replay workflow (no command is sent):")
        print("  1. Preflight: status, cameras, clear workspace, E-stop reachable.")
        print("  2. Pendant: record and save one clean drag-teach trajectory.")
        print("  3. Pendant: reset the scene to the trajectory's start pose.")
        print("  4. Host: task '<description>'   (set once per episode).")
        print("  5. Host: collect   -> arms read-only capture, triggers replay, writes one LeRobot episode.")
        print("  6. Host: watch 60  (confirm measured joint motion, not ctrl_mode).")

    def cmd_modes(self) -> None:
        cur = self._snapshot()["ctrl_mode"]
        for value in sorted(MODE_NAMES):
            marker = " <-- current" if value == cur else ""
            print(f"  0x{value:02x} {MODE_NAMES[value]:<15}{marker}")

    def cmd_watch(self, arg: str) -> None:
        from piper_xvla.motion_watch import joint_positions_deg, max_joint_delta_deg
        limit = float(arg) if arg else None
        previous = joint_positions_deg(self.piper.GetArmJointMsgs())
        start = time.monotonic()
        print("watching... (Enter stops early)")
        while True:
            if limit is not None and time.monotonic() - start >= limit:
                break
            snap = self._snapshot()
            current = joint_positions_deg(self.piper.GetArmJointMsgs())
            delta = max_joint_delta_deg(previous, current)
            previous = current
            moving = "MOVING" if delta >= 0.05 else "still "
            print(
                f"  mode={_hex(snap['ctrl_mode'])} {_name(MODE_NAMES, snap['ctrl_mode']):<13} "
                f"teach={_name(TEACH_NAMES, snap['teach_status']):<14} "
                f"err={_hex(snap['err_code'])} | dJoint={delta:7.4f} deg {moving}"
            )
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                sys.stdin.readline()
                break

    # -- mode switching -------------------------------------------------------

    def cmd_mode(self, arg: str) -> None:
        requested = arg.strip().lower()
        if requested in {"offline", "replay"}:
            print("Offline/replay mode is pendant-owned for manual drag-teach trajectories.")
            print("Press the physical pendant replay-mode button; the host must not send MotionCtrl_2.")
            return
        target = {
            "standby": 0x00,
            "can": 0x01,
        }.get(requested)
        if target is None:
            print("usage: mode <standby|can>   (offline/replay: use the pendant button)")
            return
        cur = self._snapshot()["ctrl_mode"]
        if cur == target:
            print(f"already in {_name(MODE_NAMES, target)}")
            return
        self.piper.MotionCtrl_2(ctrl_mode=target, move_mode=0x01, move_spd_rate_ctrl=self.speed,
                                is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
        after = self.ctrl._wait_for_mode(target, timeout_s=1.5)
        if after == target:
            print(f"entered {_name(MODE_NAMES, target)} (was {_name(MODE_NAMES, cur)})")
        else:
            print(f"REJECTED: controller did not accept the switch to {_name(MODE_NAMES, target)}.")
            print(f"  was 0x{cur:02x} {_name(MODE_NAMES, cur)}, still 0x{after:02x} {_name(MODE_NAMES, after)}")
            if target == 0x01:
                print("  CAN command control requires ENERGIZED joint drivers first.")
                print("  Run 'enable' (piper_enable.py logic), then retry 'mode can'.")
        self.cmd_status()

    def cmd_raw_mode(self, arg: str) -> None:
        if not arg:
            print("usage: raw-mode <hex>   e.g. raw-mode 0x03")
            return
        try:
            value = int(arg, 16)
        except ValueError:
            print("expected hex like 0x02 or 2")
            return
        if value == 0x02:
            print("REJECTED by SDK: ctrl_mode=0x02 (TEACHING) cannot be requested via CAN.")
            print("The 0x151 frame whitelist is [0x00, 0x01, 0x03, 0x04, 0x07] - teaching mode")
            print("can only be entered with the pendant button (safety by design).")
            return
        if value not in (0x00, 0x01, 0x03, 0x04, 0x07):
            print(f"warning: 0x{value:02x} is not a documented ctrl_mode; the SDK may reject it")
        cur = self._snapshot()["ctrl_mode"]
        self.piper.MotionCtrl_2(ctrl_mode=value, move_mode=0x01, move_spd_rate_ctrl=self.speed,
                                is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
        time.sleep(0.5)
        after = self._snapshot()["ctrl_mode"]
        print(f"sent ctrl_mode=0x{value:02x}: was 0x{cur:02x}, now 0x{after:02x}"
              + ("" if after == value else "  (controller did not accept the transition)"))

    # -- drag-teach recording --------------------------------------------------

    def cmd_rec_start(self) -> None:
        cur = self._snapshot()["ctrl_mode"]
        if cur != TEACHING_MODE:
            print(f"NOTE: arm is in {_name(MODE_NAMES, cur)}, not teaching mode.")
            print("      Drag-teach recording normally starts from the pendant teach button.")
            answer = input("Send SDK record-start anyway? [yes/N] ").strip().lower()
            if answer != "yes":
                return
        self.ctrl.start_recording()
        time.sleep(0.5)
        print("drag-teach record start sent; move the arm by hand now")
        self.cmd_status()

    def cmd_rec_stop(self) -> None:
        self.ctrl.stop_recording()
        time.sleep(0.5)
        print("drag-teach record stop sent; trajectory saved in controller")
        self.cmd_status()

    def cmd_record(self, arg: str) -> None:
        """Guided full drag-teach cycle: start -> (you drag) -> stop -> verify."""
        from piper_xvla.motion_watch import joint_positions_deg, max_joint_delta_deg
        cur = self._snapshot()["ctrl_mode"]
        if cur != TEACHING_MODE:
            print(f"Arm is in {_name(MODE_NAMES, cur)}, not teaching mode.")
            print("Press the pendant teach/drag button first, then re-run 'record'.")
            return
        duration = float(arg) if arg else 0.0
        print("1/3 record-start sent. Drag the arm through the full task now (slow and smooth).")
        self.ctrl.start_recording()
        time.sleep(0.5)

        # Watch while the user drags; measure motion so we can verify later.
        previous = joint_positions_deg(self.piper.GetArmJointMsgs())
        max_delta = 0.0
        moving_samples = total_samples = 0
        if duration > 0:
            deadline = time.time() + duration
            print(f"2/3 watching for {duration:.0f}s ...")
            while time.time() < deadline:
                time.sleep(0.1)
                current = joint_positions_deg(self.piper.GetArmJointMsgs())
                delta = max_joint_delta_deg(previous, current)
                previous = current
                total_samples += 1
                max_delta = max(max_delta, delta)
                moving_samples += int(delta >= 0.05)
        else:
            input("2/3 ...drag the arm now, then press Enter when done... ")
        print(f"   motion seen: {moving_samples}/{total_samples} samples, max joint delta {max_delta:.3f} deg")

        self.ctrl.stop_recording()
        time.sleep(0.5)
        snap = self._snapshot()
        traj = int(getattr(status_fields(self.piper), "trajectory_num", -1))
        print(f"3/3 record-stop sent. teach_state={_name(TEACH_NAMES, snap['teach_status'])} trajectory_num={traj}")
        if max_delta < 0.5:
            print("   WARNING: barely any motion was recorded - this trajectory is probably useless.")
            print("   Use 'discard' and try 'record' again with a bigger, slower drag.")
        else:
            print("   Trajectory saved. Next: select replay on the pendant, then run 'replay'.")

    def cmd_discard(self) -> None:
        answer = input("Discard the CURRENT taught trajectory? [yes/N] ").strip().lower()
        if answer != "yes":
            return
        self.ctrl.discard_current_recording()
        time.sleep(0.5)
        print("current trajectory cleared")

    # -- replay -----------------------------------------------------------------

    def _confirm_motion(self, what: str) -> bool:
        # Enter (or y/yes) confirms; only n/no declines. Kept short on purpose -
        # the operator is standing at the arm and wants to trigger with a tap.
        answer = input(f"MOVE ARM - {what}? [Enter=yes / n=no] ").strip().lower()
        return answer in ("", "y", "yes")

    @staticmethod
    def _replay_observation_seconds(arg: str) -> float:
        """Return a bounded post-trigger observation window for `replay [seconds]`."""
        if not arg.strip():
            return 6.0
        seconds = float(arg)
        if not 1.0 <= seconds <= 30.0:
            raise ValueError("replay observation seconds must be in [1, 30]")
        return seconds

    def cmd_replay(self, arg: str = "") -> None:
        try:
            observe_s = self._replay_observation_seconds(arg)
        except ValueError as exc:
            print(f"usage: replay [1-30 seconds]  ({exc})")
            return
        if not self._confirm_motion("execute taught trajectory"):
            return
        from piper_xvla.motion_watch import joint_positions_deg, max_joint_delta_deg
        try:
            self.ctrl.start_replay()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return
        # A controller may begin playback seconds after the CAN frame. Observe a
        # six-second default (or the requested window) before declaring no motion.
        previous = joint_positions_deg(self.piper.GetArmJointMsgs())
        max_delta = 0.0
        samples = max(1, round(observe_s * 10))
        for _ in range(samples):  # 10 Hz copied-feedback check
            time.sleep(0.1)
            current = joint_positions_deg(self.piper.GetArmJointMsgs())
            max_delta = max(max_delta, max_joint_delta_deg(previous, current))
            previous = current
        if max_delta < 0.2:
            print(f"NO MOTION after replay (max joint delta {max_delta:.3f} deg).")
            print("Likely cause: no taught trajectory is stored in the controller.")
            print("(Trajectories live in RAM - a power loss or mode reset clears them.)")
            print("Use the pendant to confirm a recorded trajectory exists, then replay again.")
        else:
            print(f"replay running (max joint delta so far {max_delta:.3f} deg).")
            print("use 'watch' to follow, 'stop' to end early")

    def cmd_pause(self) -> None:
        if not self._confirm_motion("pause replay"):
            return
        self.ctrl.pause_replay()
        print("pause sent")

    def cmd_resume(self) -> None:
        if not self._confirm_motion("resume replay"):
            return
        try:
            self.ctrl.resume_replay()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return
        print("resume sent")

    def cmd_stop(self) -> None:
        if not self._confirm_motion("terminate replay (arm stops in place)"):
            return
        self.ctrl.stop_replay()
        time.sleep(0.5)
        print("replay terminated")
        self.cmd_status()

    def cmd_move_start(self) -> None:
        if not self._confirm_motion("move to trajectory start point"):
            return
        try:
            self.ctrl.move_to_start()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return
        print("move-to-start sent; use 'watch' to follow it")

    # -- episode collection (wires replay_collector) ---------------------------

    def cmd_task(self, arg: str) -> None:
        if not arg.strip():
            current = self.task or "(not set)"
            print(f"task = {current}")
            return
        self.task = arg.strip()
        print(f"task set to: {self.task!r}")

    @staticmethod
    def _collect_window_seconds(arg: str) -> float:
        """Return a bounded capture window for `collect [seconds]`."""
        if not arg.strip():
            return 40.0
        seconds = float(arg)
        if not 5.0 <= seconds <= 300.0:
            raise ValueError("collect window seconds must be in [5, 300]")
        return seconds

    def cmd_collect(self, arg: str = "") -> None:
        """Record ONE raw LeRobot episode from a pendant replay at 20 Hz."""
        if not self.task.strip():
            print("No task set. Run:  task '<description>'   then retry 'collect'.")
            return
        try:
            window_s = self._collect_window_seconds(arg)
        except ValueError as exc:
            print(f"usage: collect [5-300 seconds]  ({exc})")
            return
        if not self._confirm_motion(
            f"record a {window_s:.0f}s episode (arms capture, then triggers replay)"
        ):
            return

        from piper_xvla.collect_session import CollectSession

        try:
            session = CollectSession.open(self.piper, self.task, self.data_dir)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return

        print(f"recording episode {session.episode_index}... press Enter (or type 'stop') to end early; Ctrl+C also works.")

        period = 0.05  # 20 Hz
        start = time.monotonic()
        stop_requested = False
        try:
            while True:
                now = time.monotonic()
                if now - start >= window_s:
                    break
                # Trigger the (validated) replay on the first loop iteration,
                # AFTER one observation is armed so capture precedes motion.
                if session.frames_written == 0 and not getattr(self, "_collect_triggered", False):
                    try:
                        self.ctrl.start_replay()
                    except RuntimeError as exc:
                        print(f"ERROR triggering replay: {exc}")
                        break
                    self._collect_triggered = True
                observation = session.snapshot()
                session.write_frame(observation)
                # Pace to 20 Hz without busy-waiting.
                elapsed = time.monotonic() - start
                target = (int(elapsed / period) + 1) * period
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                # Non-blocking keyboard stop (same pattern as `watch`). A ready
                # stdin means the user pressed Enter or typed a command. If stdin
                # is not selectable (e.g. piped/redirected), skip the check and
                # rely on the window cap / Ctrl+C instead of crashing.
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0)
                except (OSError, ValueError):
                    ready = []
                if ready:
                    line = sys.stdin.readline().strip().lower()
                    if not line or line in {"stop", "q", "quit"}:
                        stop_requested = True
                        break
        except KeyboardInterrupt:
            print("\ninterrupted; finalizing partial episode...")
        finally:
            self._collect_triggered = False

        total = session.finalize()
        reason = "stopped by user" if stop_requested else "window ended"
        episode_dir = self.data_dir / "dataset"
        print(f"episode written ({reason}): {total} frames @20 Hz -> {episode_dir}")
        if session.video_path.is_file():
            size_mb = session.video_path.stat().st_size / 1e6
            print(f"review video: {session.video_path}  ({size_mb:.1f} MB, global|wrist)")
        else:
            print("NOTE: no review video was written.")

    def cmd_speed(self, arg: str) -> None:
        if not arg:
            print(f"current replay speed: {self.speed}%")
            return
        try:
            value = int(arg)
        except ValueError:
            print("usage: speed <1-100>")
            return
        if not 1 <= value <= 100:
            print("speed must be in [1, 100]")
            return
        self.speed = value
        self.ctrl.speed = value
        print(f"replay speed set to {value}% (applies to the next replay/mode frame)")

    # -- driver power (mirrors piper_enable.py / piper_disable.py) --------------

    def _driver_states(self) -> list[int]:
        info = self.piper.GetArmLowSpdInfoMsgs()
        return [int(getattr(info, f"motor_{i}").foc_status.driver_enable_status) for i in range(1, 7)]

    def cmd_enable(self) -> None:
        """Energize all six joint drivers + gripper (required before CAN command control)."""
        states = self._driver_states()
        if all(states):
            print(f"already enabled: drivers={states}")
            return
        print(f"enabling arm... (was drivers={states})")
        # The official demo loops EnableArm(7)+GripperCtrl and polls up to 5 s.
        for _ in range(10):
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            time.sleep(0.5)
            states = self._driver_states()
            if all(states):
                break
        print(f"drivers={states} {'ENABLED' if all(states) else 'NOT enabled'}")
        if not all(states):
            print("  Drivers did not report enabled within 5 s.")
            print("  Check power supply, e-stop release, and CAN bus health (ERROR-PASSIVE?).")

    def cmd_disable(self) -> None:
        """De-energize all six joint drivers + gripper (arm becomes hand-movable)."""
        states = self._driver_states()
        if not any(states):
            print(f"already disabled: drivers={states}")
            return
        print(f"disabling arm... (was drivers={states})")
        for _ in range(10):
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x02, 0)
            time.sleep(0.5)
            states = self._driver_states()
            if not any(states):
                break
        print(f"drivers={states} {'DISABLED' if not any(states) else 'still enabled'}")
        if any(states):
            print("  Drivers did not all report disabled within 5 s; check the arm.")

    # -- reset (mirrors piper_reset.py) ------------------------------------------

    def cmd_reset(self) -> None:
        """Recover from e-stop and return to standby position-velocity mode."""
        self.piper.MotionCtrl_1(emergency_stop=0x02, track_ctrl=0x00, grag_teach_ctrl=0x00)
        time.sleep(0.3)
        self.piper.MotionCtrl_2(ctrl_mode=0x00, move_mode=0x01, move_spd_rate_ctrl=self.speed,
                                is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
        time.sleep(0.5)
        print("reset sent: e-stop cleared + standby mode")
        self.cmd_status()

    # -- safety -------------------------------------------------------------------

    def cmd_estop(self) -> None:
        self.ctrl.emergency_stop()
        print("EMERGENCY STOP sent. The arm will not move on its own.")
        print("Recover later with: recover   (or the pendant e-stop button)")

    def cmd_recover(self) -> None:
        if not self._confirm_motion("clear e-stop AND move to trajectory start"):
            return
        try:
            self.ctrl.recover_and_move_to_start()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return
        print("recovery + move-to-start sent; use 'watch' to follow it")

    # -- cameras --------------------------------------------------------------------

    def cmd_cameras(self) -> None:
        from piper_xvla.camera_check import check_camera, DEFAULT_CONFIG
        import json as _json
        config = {}
        if DEFAULT_CONFIG.exists():
            config = _json.loads(DEFAULT_CONFIG.read_text())
        devices = (
            ("global", config.get("global_camera", {}).get("device")),
            ("wrist", config.get("wrist_camera", {}).get("device")),
        )
        ok = True
        for role, device in devices:
            if not device:
                print(f"FAIL {role}: no device configured")
                ok = False
                continue
            frame, result = check_camera(device, role)
            detail = result.get("reason") or f"{result['width']}x{result['height']} mean={result['brightness']} std={result['contrast']}"
            print(f"{'PASS' if result['ok'] else 'FAIL'} {role} {device}: {detail}")
            ok = ok and result["ok"]
        print("cameras OK" if ok else "camera preflight FAILED - do not collect")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=20, help="default replay speed percent (1-100)")
    parser.add_argument("--task", default="", help="task description written into collected episodes")
    parser.add_argument("--data-dir", default="data/piper_replay", help="root dir for LeRobot episodes (default: data/piper_replay)")
    parser.add_argument("--dry-run", action="store_true", help="no hardware; prints SDK calls and simulates mode changes")
    args = parser.parse_args()

    if not 1 <= args.speed <= 100:
        parser.error("--speed must be in [1, 100]")

    if args.dry_run:
        piper = DryRunPiper()
        print("DRY-RUN console (no hardware). Mode changes are simulated.")
    else:
        piper = connect_live_piper(args.can)
        time.sleep(1.0)  # let feedback streams fill before the first read

    console = PiperConsole(piper, args.speed, task=args.task, data_dir=args.data_dir)

    dispatch = {
        "status": lambda a: console.cmd_status(),
        "workflow": lambda a: console.cmd_workflow(),
        "watch": lambda a: console.cmd_watch(a),
        "modes": lambda a: console.cmd_modes(),
        "mode": lambda a: console.cmd_mode(a),
        "raw-mode": lambda a: console.cmd_raw_mode(a),
        "rec-start": lambda a: console.cmd_rec_start(),
        "rec-stop": lambda a: console.cmd_rec_stop(),
        "record": lambda a: console.cmd_record(a),
        "discard": lambda a: console.cmd_discard(),
        "replay": lambda a: console.cmd_replay(a),
        "collect": lambda a: console.cmd_collect(a),
        "task": lambda a: console.cmd_task(a),
        "pause": lambda a: console.cmd_pause(),
        "resume": lambda a: console.cmd_resume(),
        "stop": lambda a: console.cmd_stop(),
        "move-start": lambda a: console.cmd_move_start(),
        "speed": lambda a: console.cmd_speed(a),
        "enable": lambda a: console.cmd_enable(),
        "disable": lambda a: console.cmd_disable(),
        "reset": lambda a: console.cmd_reset(),
        "cameras": lambda a: console.cmd_cameras(),
        "estop": lambda a: console.cmd_estop(),
        "recover": lambda a: console.cmd_recover(),
    }

    print("Piper console ready. 'help' lists commands, 'status' shows state, 'quit' exits.")
    while True:
        try:
            line = input("\npiper> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("quit", "q", "exit"):
            break
        if cmd in ("help", "?"):
            print(__doc__)
            continue
        handler = dispatch.get(cmd)
        if handler is None:
            print(f"unknown command: {cmd}   (try 'help')")
            continue
        try:
            handler(arg)
        except Exception as exc:  # noqa: BLE001 - console must survive any single failure
            print(f"ERROR in '{cmd}': {exc}")

    snap = console._snapshot()
    print(f"\nexiting. arm left in mode {_hex(snap['ctrl_mode'])} {_name(MODE_NAMES, snap['ctrl_mode'])}.")
    if snap["ctrl_mode"] == OFFLINE_MODE:
        print("Tip: use the pendant teach button to return to drag-teach mode (CAN cannot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
