import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from piper_xvla.lerobot_adapter import PiperLeRobotEpisode
from piper_xvla.lerobot_factory import piper_lerobot_features
from piper_xvla.replay_collector import FutureStateLabeler, ReplayObservation


def test_raw_piper_frame_writes_to_a_real_local_lerobot_episode(tmp_path):
    dataset = LeRobotDataset.create(
        repo_id="local/piper-replay-test",
        root=tmp_path / "dataset",
        fps=20,
        robot_type="piper",
        features=piper_lerobot_features(height=4, width=6),
        use_videos=False,
    )
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    state = np.zeros(20, dtype=np.float32)
    observation = ReplayObservation(image, image.copy(), state, "put cube in basket")
    labeler = FutureStateLabeler()
    assert labeler.push(observation) is None
    frame = labeler.flush()
    episode = PiperLeRobotEpisode(dataset)
    episode.add(frame)
    episode.finalize()
    dataset.finalize()

    assert (tmp_path / "dataset" / "meta" / "info.json").is_file()
    assert (tmp_path / "dataset" / "data").exists()
