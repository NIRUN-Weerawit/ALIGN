"""Read-only motion watcher for Piper teach/replay validation.

Use this instead of teach_status to prove that a manual replay is executing.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def joint_positions_deg(joint_msg) -> np.ndarray:
    """Convert Piper SDK feedback (0.001 degree units) into six degrees."""
    state = joint_msg.joint_state
    return np.asarray([getattr(state, f"joint_{i}") / 1000.0 for i in range(1, 7)], dtype=float)


def max_joint_delta_deg(before_deg: np.ndarray, after_deg: np.ndarray) -> float:
    """Maximum absolute joint movement between immutable degree snapshots."""
    return float(np.max(np.abs(np.asarray(after_deg) - np.asarray(before_deg))))


def joint_delta_deg(before, after) -> float:
    """Compatibility helper for two independent feedback messages."""
    return max_joint_delta_deg(joint_positions_deg(before), joint_positions_deg(after))


def connect_readonly(can_device: str):
    from piper_sdk import C_PiperInterface_V2
    piper = C_PiperInterface_V2(can_device)
    piper.ConnectPort(True)
    return piper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--poll-hz", type=float, default=20.0)
    parser.add_argument("--motion-threshold-deg", type=float, default=0.05,
                        help="max joint change between polls that counts as movement")
    args = parser.parse_args()
    if args.seconds <= 0 or args.poll_hz <= 0 or args.motion_threshold_deg <= 0:
        parser.error("seconds, poll-hz, and motion-threshold-deg must be positive")

    piper = connect_readonly(args.can)
    time.sleep(1.0)
    # The SDK returns the same mutable feedback object each call. Snapshot its
    # numeric values now; retaining the message object makes every delta zero.
    previous = joint_positions_deg(piper.GetArmJointMsgs())
    start = time.monotonic()
    moving_samples = 0
    total_samples = 0
    print("Watching Piper joint feedback. Start replay now; this process sends no arm commands.")
    while time.monotonic() - start < args.seconds:
        time.sleep(1.0 / args.poll_hz)
        current = joint_positions_deg(piper.GetArmJointMsgs())
        delta = max_joint_delta_deg(previous, current)
        total_samples += 1
        moving = delta >= args.motion_threshold_deg
        moving_samples += int(moving)
        print(f"t={time.monotonic() - start:5.2f}s max_joint_delta={delta:7.4f} deg {'MOVING' if moving else 'still'}")
        previous = current
    print(f"SUMMARY moving_samples={moving_samples}/{total_samples}")
    return 0 if moving_samples else 2


if __name__ == "__main__":
    raise SystemExit(main())
