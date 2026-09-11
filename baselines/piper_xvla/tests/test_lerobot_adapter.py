import numpy as np

from piper_xvla.lerobot_adapter import PiperLeRobotEpisode
from piper_xvla.schema import PiperXVLAFrame


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def make_frame():
    return PiperXVLAFrame(
        global_rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        wrist_rgb=np.ones((4, 6, 3), dtype=np.uint8),
        state20=np.array([*range(10), *([0] * 10)], dtype=np.float32),
        action20=np.array([*range(10, 20), *([0] * 10)], dtype=np.float32),
        task="put cube in basket",
    )


def test_recorder_writes_raw_piper_10d_to_lerobot_dataset():
    dataset = FakeDataset()
    episode = PiperLeRobotEpisode(dataset)
    episode.add(make_frame())
    episode.finalize()

    assert dataset.saved == 1
    assert len(dataset.frames) == 1
    raw = dataset.frames[0]
    assert set(raw) == {
        "observation.images.global_rgb",
        "observation.images.wrist_rgb",
        "observation.state",
        "action",
        "task",
    }
    np.testing.assert_array_equal(raw["observation.state"], np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(raw["action"], np.arange(10, 20, dtype=np.float32))


def test_recorder_cannot_append_after_finalization():
    episode = PiperLeRobotEpisode(FakeDataset())
    episode.add(make_frame())
    episode.finalize()
    try:
        episode.add(make_frame())
    except RuntimeError as error:
        assert "finalized" in str(error)
    else:
        raise AssertionError("expected finalized episode to reject frames")
