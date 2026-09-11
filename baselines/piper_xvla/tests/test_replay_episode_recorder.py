import numpy as np

from piper_xvla.lerobot_adapter import PiperLeRobotEpisode
from piper_xvla.replay_collector import FutureStateLabeler, ReplayEpisodeRecorder, ReplayObservation


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def observation(x):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.zeros(20, dtype=np.float32)
    state[0] = x
    return ReplayObservation(image, image.copy(), state, "put cube in basket")


def test_recorder_writes_delayed_future_action_and_terminal_frame():
    dataset = FakeDataset()
    recorder = ReplayEpisodeRecorder(PiperLeRobotEpisode(dataset), FutureStateLabeler())
    assert recorder.add_observation(observation(1.0)) == 0
    assert recorder.add_observation(observation(2.0)) == 1
    assert recorder.finalize() == 2
    assert dataset.saved == 1
    assert [frame["observation.state"][0] for frame in dataset.frames] == [1.0, 2.0]
    assert [frame["action"][0] for frame in dataset.frames] == [2.0, 2.0]
