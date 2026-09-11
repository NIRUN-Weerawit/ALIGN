import numpy as np
import pytest
import torch

from piper_xvla.xvla_dataset import (
    CachedPiperXVLALiberoDataset,
    PiperXVLAConversion,
    build_prepared_numeric_cache,
)


def test_prepared_numeric_cache_converts_state_and_action_once_by_global_index():
    states = np.array([
        [0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0, -0.04],
        [0.4, 0.5, 0.6, 1, 0, 0, 1, 0, 0, 0.04],
    ], dtype=np.float32)
    actions = states.copy()
    cache = build_prepared_numeric_cache(
        states, actions, np.array([3, 7]), np.array([12, 13]), PiperXVLAConversion(-0.04, 0.04),
    )

    assert cache["format"] == "piper-xvla-prepared-cache-v1"
    assert tuple(cache["state"].shape) == (8, 8)
    assert tuple(cache["action"].shape) == (8, 20)
    assert cache["available"].tolist() == [False, False, False, True, False, False, False, True]
    torch.testing.assert_close(cache["state"][3, :3], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(cache["state"][3, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert cache["state"][3, 7].item() == pytest.approx(0.0)
    assert cache["action"][7, 9].item() == pytest.approx(1.0)
    assert torch.count_nonzero(cache["action"][7, 10:]) == 0


class _Source:
    def __init__(self):
        self.items = [{
            "index": torch.tensor(3),
            "task": "grab an object and put in a cup",
            "observation.images.global_rgb": torch.full((3, 480, 640), 0.5),
            "observation.images.wrist_rgb": torch.full((3, 480, 640), 0.25),
        }]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_cached_dataset_uses_prepared_numeric_values_without_raw_conversion():
    raw = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0, -0.04]], dtype=np.float32)
    cache = build_prepared_numeric_cache(
        raw, raw, np.array([3]), np.array([12]), PiperXVLAConversion(-0.04, 0.04),
    )
    sample = CachedPiperXVLALiberoDataset(_Source(), cache)[0]

    assert tuple(sample["observation.state"].shape) == (8,)
    assert tuple(sample["action"].shape) == (20,)
    assert tuple(sample["observation.images.image"].shape) == (3, 480, 640)
    assert tuple(sample["observation.images.image2"].shape) == (3, 480, 640)
    assert tuple(sample["observation.images.empty_camera_0"].shape) == (3, 224, 224)
