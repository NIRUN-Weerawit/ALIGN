"""Read-only building blocks for 20 Hz pendant-replay collection.

The collector records measured Piper state and camera frames. It derives a
one-step future measured state as the behavior-cloning action, because manual
pendant replay has no host-side operator command stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import PiperXVLAFrame


@dataclass(frozen=True)
class ReplayObservation:
    """One immutable, synchronized measured observation at collection time."""

    global_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state20: np.ndarray
    task: str

    def __post_init__(self) -> None:
        # Validate using the canonical frame contract without inventing an action.
        PiperXVLAFrame(
            global_rgb=self.global_rgb,
            wrist_rgb=self.wrist_rgb,
            state20=self.state20,
            action20=self.state20,
            task=self.task,
        )


class FutureStateLabeler:
    """Turn consecutive measured observations into state→next-state examples.

    `push` delays one observation so its action can be labeled with the next
    measured Piper state. `flush` retains the final observation using its own
    state as a terminal action; callers can identify it from matching state and
    action if they choose to exclude it during later X-VLA conversion.
    """

    def __init__(self) -> None:
        self._pending: ReplayObservation | None = None

    def push(self, current: ReplayObservation) -> PiperXVLAFrame | None:
        previous, self._pending = self._pending, current
        if previous is None:
            return None
        return PiperXVLAFrame(
            global_rgb=previous.global_rgb,
            wrist_rgb=previous.wrist_rgb,
            state20=previous.state20.copy(),
            action20=current.state20.copy(),
            task=previous.task,
        )

    def flush(self) -> PiperXVLAFrame | None:
        previous, self._pending = self._pending, None
        if previous is None:
            return None
        return PiperXVLAFrame(
            global_rgb=previous.global_rgb,
            wrist_rgb=previous.wrist_rgb,
            state20=previous.state20.copy(),
            action20=previous.state20.copy(),
            task=previous.task,
        )


class ReplayEpisodeRecorder:
    """Append measured replay observations to one raw LeRobot episode.

    This class is intentionally command-free: a caller controls capture timing
    and may trigger Piper replay only after its first observation is armed.
    """

    def __init__(self, episode, labeler: FutureStateLabeler | None = None) -> None:
        self._episode = episode
        self._labeler = labeler or FutureStateLabeler()
        self._written = 0
        self._finalized = False

    def add_observation(self, observation: ReplayObservation) -> int:
        if self._finalized:
            raise RuntimeError("episode recorder is finalized")
        frame = self._labeler.push(observation)
        if frame is not None:
            self._episode.add(frame)
            self._written += 1
        return self._written

    def finalize(self) -> int:
        if self._finalized:
            raise RuntimeError("episode recorder is already finalized")
        frame = self._labeler.flush()
        if frame is not None:
            self._episode.add(frame)
            self._written += 1
        self._episode.finalize()
        self._finalized = True
        return self._written
