"""Probe + enter CAN command control mode on a real Piper V2.

Sequence (mirrors piper_enable.py / piper_vr_connection.py):
  1. EnableArm(7)                       -- energize all six joints + gripper
  2. poll GetArmLowSpdInfoMsgs() up to ~6 s until all driver_enable_status True
  3. MotionCtrl_2(ctrl_mode=0x01)       -- request CAN command control
  4. verify ctrl_mode == 0x01

Use --enable-only to stop after step 2 (just energize, no mode switch).
WARNING: this makes the arm ACTIVE (motors energized, holds pose). It does not
move the arm to a new location.
"""
from __future__ import annotations

import argparse
import time


def driver_states(piper):
    info = piper.GetArmLowSpdInfoMsgs()
    hz = float(getattr(info, "Hz", 0.0) or 0.0)
    states = [int(getattr(info, f"motor_{i}").foc_status.driver_enable_status) for i in range(1, 7)]
    return hz, states


def ctrl_mode_of(piper) -> int:
    status = piper.GetArmStatus()
    fields = getattr(status, "arm_status", status)
    return int(getattr(fields, "ctrl_mode", -1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--enable-only", action="store_true", help="energize arm but do not switch to CAN mode")
    args = parser.parse_args()

    from piper_sdk import C_PiperInterface_V2
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort(True)
    if not piper.isOk():
        print("Piper connection unhealthy; aborting")
        return 2
    time.sleep(1.0)

    hz, states = driver_states(piper)
    print(f"before: ctrl_mode=0x{ctrl_mode_of(piper):02x} lowspd_Hz={hz:.1f} drivers={states}")

    # Step 1: energize.
    piper.EnableArm(7)
    print("EnableArm(7) sent; polling driver enable status...")

    # Step 2: poll up to ~6 s (the official demo polls up to 5 s at 0.5 s).
    all_on = False
    for _ in range(12):
        time.sleep(0.5)
        hz, states = driver_states(piper)
        print(f"  lowspd_Hz={hz:6.1f} drivers={states}")
        if all(states):
            all_on = True
            break

    if not all_on:
        print("ERROR: joint drivers did not all report enabled within 6 s; not switching mode.")
        return 2
    print("all six joint drivers enabled")

    if args.enable_only:
        print("--enable-only set; leaving arm energized in current mode.")
        return 0

    # Step 3: request CAN command control.
    piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50,
                       is_mit_mode=0x00, residence_time=0, installation_pos=0x00)

    # Step 4: verify.
    deadline = time.time() + 3.0
    mode = ctrl_mode_of(piper)
    while mode != 0x01 and time.time() < deadline:
        time.sleep(0.1)
        mode = ctrl_mode_of(piper)

    print(f"result: ctrl_mode=0x{mode:02x} {'CAN_CTRL' if mode == 0x01 else 'NOT CAN_CTRL'}")
    if mode == 0x01:
        print("SUCCESS: arm is in CAN command control mode, motors active.")
        print("It will HOLD its current pose. In this mode you stream joint/EE commands;")
        print("to stop commanding, DisableArm(7) or use the pendant to change mode.")
    else:
        print(f"REJECTED even with drivers enabled; last ctrl_mode=0x{mode:02x}")
    return 0 if mode == 0x01 else 3


if __name__ == "__main__":
    raise SystemExit(main())
