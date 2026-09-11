"""Write canonical one-arm Piper demonstrations through a LeRobotDataset-like writer.

This deliberately stores raw 10-D Piper state/action in LeRobot. X-VLA's 20-D
padding and third black view belong in the later X-VLA dataset adapter, not in
raw teleoperation data.
"""
from typing import Protocol

from .schema import PiperXVLAFrame


class LeRobotWritableDataset(Protocol):
    def add_frame(self, frame: dict) -> None: ...
    def save_episode(self) -> None: ...


class PiperLeRobotEpisode:
    """One explicit episode boundary around LeRobotDataset.add_frame/save_episode."""

    def __init__(self, dataset: LeRobotWritableDataset):
        self._dataset = dataset
        self._finalized = False
        self.frame_count = 0

    def add(self, frame: PiperXVLAFrame) -> None:
        if self._finalized:
            raise RuntimeError("episode is finalized; create a new PiperLeRobotEpisode")
        self._dataset.add_frame(
            {
                "observation.images.global_rgb": frame.global_rgb,
                "observation.images.wrist_rgb": frame.wrist_rgb,
                "observation.state": frame.state20[:10].copy(),
                "action": frame.action20[:10].copy(),
                "task": frame.task,
            }
        )
        self.frame_count += 1

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("episode is already finalized")
        if self.frame_count == 0:
            raise RuntimeError("cannot save an empty episode")
        self._dataset.save_episode()
        self._finalized = True
