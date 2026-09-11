"""Live probe for the Piper teach-mode -> offline-trajectory-mode transition.

SAFETY: this script only sends mode-switch frames (0x151) and drag-teach
session close (grag_teach_ctrl=0x02). It NEVER sends execute/pause/resume/
move-to-start commands, so the arm cannot start a trajectory from here.

It tries candidate sequences in order and stops at the first one that lands
ctrl_mode == 0x07 (OFFLINE_TRAJECTORY_MODE), printing which sequence worked.
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace


def flat_status(piper):
    status = piper.GetArmStatus()
    return getattr(status, "arm_status", status)


def snapshot(piper) -> dict:
    s = flat_status(piper)
    return {
        "ctrl_mode": int(getattr(s, "ctrl_mode", -1)),
        "teach_status": int(getattr(s, "teach_status", -1)),
        "motion_status": int(getattr(s, "motion_status", -1)),
        "err_code": int(getattr(s, "err_code", -1)),
    }


def wait_for_mode(piper, target: int, timeout_s: float = 2.0) -> dict | None:
    deadline = time.time() + timeout_s
    last = snapshot(piper)
    while time.time() < deadline:
        last = snapshot(piper)
        if last["ctrl_mode"] == target:
            return last
        time.sleep(0.15)
    return None


def build_candidates():
    """Each candidate is (name, [steps]); a step is (label, callable)."""
    def mode(inst):
        def _step(piper):
            piper.MotionCtrl_2(ctrl_mode=0x07, move_mode=0x01, move_spd_rate_ctrl=30,
                               is_mit_mode=0x00, residence_time=0, installation_pos=inst)
        return _step

    def end_teach(piper):
        piper.MotionCtrl_1(grag_teach_ctrl=0x02)

    def standby(piper):
        piper.MotionCtrl_2(ctrl_mode=0x00, move_mode=0x01, move_spd_rate_ctrl=30,
                           is_mit_mode=0x00, residence_time=0, installation_pos=0x00)

    def can_ctrl(piper):
        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30,
                           is_mit_mode=0x00, residence_time=0, installation_pos=0x00)

    def mode_burst(piper):
        for _ in range(20):  # ~2 s of repeated frames
            piper.MotionCtrl_2(ctrl_mode=0x07, move_mode=0x01, move_spd_rate_ctrl=30,
                               is_mit_mode=0x00, residence_time=0, installation_pos=0x00)
            time.sleep(0.1)

    return [
        ("A: direct 0x151(0x07)", [("mode07", mode(0x00))]),
        ("B: end-teach then 0x151(0x07)", [("end_teach", end_teach), ("mode07", mode(0x00))]),
        ("C: standby then 0x151(0x07)", [("standby", standby), ("mode07", mode(0x00))]),
        ("D: end-teach, standby, 0x151(0x07)", [("end_teach", end_teach), ("standby", standby), ("mode07", mode(0x00))]),
        ("E: CAN-ctrl then 0x151(0x07)", [("can_ctrl", can_ctrl), ("mode07", mode(0x00))]),
        ("F: repeated 0x151(0x07) burst", [("burst", mode_burst)]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--skip", action="store_true", help="skip candidates already known to fail (A)")
    args = parser.parse_args()

    from piper_sdk import C_PiperInterface_V2
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort(True)
    if not piper.isOk():
        print("Piper connection unhealthy; aborting probe")
        return 2

    initial = snapshot(piper)
    print(f"initial: {initial}")
    if initial["ctrl_mode"] == 0x07:
        print("already in offline trajectory mode; nothing to do")
        return 0
    if initial["err_code"] != 0 or initial["motion_status"] not in (0,):
        print(f"WARNING: arm reports err_code=0x{initial['err_code']:04x} motion_status={initial['motion_status']}; aborting probe")
        return 2

    candidates = build_candidates()
    if args.skip:
        candidates = [c for c in candidates if not c[0].startswith("A:")]

    for name, steps in candidates:
        print(f"\n=== trying {name} ===")
        landed = None
        for label, fn in steps:
            print(f"  step: {label}")
            fn(piper)
            time.sleep(0.5)
            landed = wait_for_mode(piper, 0x07, timeout_s=1.5)
            if landed is not None:
                break
        final = snapshot(piper)
        print(f"  result after candidate: {final}")
        if final["ctrl_mode"] == 0x07:
            print(f"\nSUCCESS: sequence '{name}' entered offline trajectory mode")
            print("NEXT: run `replay` (it will now skip the mode frame and send execute)")
            return 0
        # continue to next candidate; a failed step may have changed state,
        # so re-print status for context
    print("\nNO candidate entered offline trajectory mode.")
    print("The controller likely only accepts this transition from the teach pendant.")
    print("If you need teaching mode again: use the pendant button (CAN cannot force it back).")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
