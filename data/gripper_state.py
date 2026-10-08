"""Causal gripper-state fallback when measured observations are unavailable."""

import numpy as np


def previous_gripper_commands(actions, start=0, count=None,
                              initial_gripper=0.0, episode_start=0):
    """Return the last executed command, never the command being predicted.

    Supports numpy arrays and HDF5 datasets without reading the full episode.
    At the first observation there is no previous action: use initial_gripper.
    episode_start prevents reading the previous episode in cumulative arrays.
    """
    if start < 0:
        raise ValueError("start must be non-negative")
    if count is None:
        count = max(0, len(actions) - episode_start - start)
    result = np.full(count, initial_gripper, dtype=np.float32)
    if count == 0 or actions.ndim != 2 or actions.shape[1] < 7:
        return result
    previous = episode_start + np.arange(start, start + count) - 1
    valid = previous >= episode_start
    if not valid.any() or len(actions) <= episode_start:
        return result
    previous = np.minimum(previous, len(actions) - 1)
    lo, hi = int(previous[valid].min()), int(previous[valid].max())
    commands = np.asarray(actions[lo:hi + 1, 6], dtype=np.float32)
    result[valid] = commands[previous[valid] - lo]
    return result


def carry_gripper_state(last_state, current_pose):
    """Update pose while carrying the last available gripper observation."""
    state = np.asarray(last_state, dtype=np.float32).copy()
    state[:6] = current_pose
    return state


def executed_binary_gripper(prediction, threshold=0.5):
    """Dataset command (0=open, 1=close) and matching simulator polarity.

    Feed the binary command actually executed back into the next observation,
    rather than the continuous head score or the simulator's +/-1 polarity.
    """
    command = float(prediction > threshold)
    return command, 1.0 - 2.0 * command
