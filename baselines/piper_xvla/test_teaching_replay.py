"""Test hypothesis: drag-teach replay = grag_teach_ctrl=0x03 sent FROM teaching mode.

The user confirmed the pendant replay works but our console (teaching->standby->offline
->execute) does not. This script, run while the arm is ALREADY in teaching mode, sends
ONLY MotionCtrl_1(grag_teach_ctrl=0x03) and watches joint motion to see if the stored
drag-teach trajectory replays WITHOUT any offline-mode switch.

SAFETY: this plays back a trajectory the user already recorded by hand (known-safe path).
It sends no mode-change frames, so it cannot put the arm into a new control context.
"""
from __future__ import annotations
import time
from piper_sdk import C_PiperInterface_V2
from piper_xvla.motion_watch import joint_positions_deg, max_joint_delta_deg


def flat(piper):
    s = piper.GetArmStatus()
    return getattr(s, "arm_status", s)


def main():
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort(True)
    time.sleep(1.5)

    s = flat(piper)
    cur = int(getattr(s, "ctrl_mode", -1))
    print(f"current ctrl_mode=0x{cur:02x}")

    if cur != 0x02:
        print("Arm is NOT in teaching mode (0x02). Press the pendant teach button first,")
        print("then re-run this script. Aborting (no commands sent).")
        return 2

    # Send ONLY the drag-teach execute command, no mode change.
    print("sending MotionCtrl_1(grag_teach_ctrl=0x03) from teaching mode...")
    piper.MotionCtrl_1(emergency_stop=0x00, track_ctrl=0x00, grag_teach_ctrl=0x03)

    previous = joint_positions_deg(piper.GetArmJointMsgs())
    max_delta = 0.0
    moving = total = 0
    print("watching joints for 6 s ...")
    for _ in range(60):
        time.sleep(0.1)
        current = joint_positions_deg(piper.GetArmJointMsgs())
        delta = max_joint_delta_deg(previous, current)
        previous = current
        total += 1
        max_delta = max(max_delta, delta)
        moving += int(delta >= 0.05)

    print(f"RESULT: moving_samples={moving}/{total}  max_joint_delta={max_delta:.3f} deg")
    if max_delta >= 0.5:
        print("SUCCESS: the drag-teach trajectory replayed from TEACHING mode with no offline switch.")
        print("=> The fix is to send grag_teach_ctrl=0x03 directly in teaching mode.")
        return 0
    print("No significant motion - hypothesis not confirmed (or trajectory empty).")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
