"""Frozen DINOv2 cache preserves the training loop's camera ordering."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from data.align_dataset import ALIGNDataset


_script = Path(__file__).resolve().parents[1] / "scripts/precompute_dinov2.py"
_spec = importlib.util.spec_from_file_location("precompute_dinov2", _script)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class MarkerEncoder:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, images):
        self.batch_sizes.append(len(images))
        marker = images[:, 0, 0, 0].float().view(-1, 1, 1)
        return marker.expand(-1, 257, 768).clone()

    def parameters(self):
        return iter(())


def test_encode_frames_keeps_time_camera_patch_order():
    images = np.zeros((2, 2, 2, 2, 3), dtype=np.uint8)
    images[:, :, 0, 0, 0] = [[1, 2], [3, 4]]
    encoder = MarkerEncoder()

    features = _module.encode_frames(encoder, images, torch.device("cpu"), batch_size=3)

    assert features.shape == (2, 514, 768)
    assert encoder.batch_sizes == [3, 1]
    np.testing.assert_array_equal(features[:, 0, 0], [1, 3])
    np.testing.assert_array_equal(features[:, 256, 0], [1, 3])
    np.testing.assert_array_equal(features[:, 257, 0], [2, 4])
    np.testing.assert_array_equal(features[:, 513, 0], [2, 4])


def test_main_reaches_disk_check_and_writes_cache(tmp_path, monkeypatch):
    source = tmp_path / "sample.h5"
    with h5py.File(source, "w") as h5:
        ep = h5.create_group("ep_000000")
        ep.create_dataset("frames/image", data=np.ones((1, 2, 2, 3), dtype=np.uint8))

    output = tmp_path / "cache"
    monkeypatch.setattr(_module, "build_encoder", lambda _: MarkerEncoder())
    monkeypatch.setattr(_module.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(_module.sys, "argv", [
        "precompute_dinov2.py", "--data", str(source), "--cameras", "image",
        "--output", str(output), "--device", "cpu",
    ])

    _module.main()

    assert (output / "ep_000000.npy").exists()
    assert json.loads((output / "index.json").read_text())["ep_000000"]["length"] == 1


def test_dataset_rejects_legacy_cache_that_could_change_features(tmp_path):
    source = tmp_path / "sample.h5"
    with h5py.File(source, "w") as h5:
        ep = h5.create_group("ep_000000")
        ep.create_dataset("frames/wrist_image", data=np.zeros((10, 2, 2, 3), dtype=np.uint8))
        ep.create_dataset("poses", data=np.zeros((10, 6), dtype=np.float32))
        ep.create_dataset("actions", data=np.zeros((10, 7), dtype=np.float32))
        ep.create_dataset("texts", data=json.dumps(["task"]))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "index.json").write_text(json.dumps({"ep_000000": {"length": 10}}))

    dataset = ALIGNDataset(source, mode="head", traj_window=2,
                           cameras=["wrist_image"], dinov2_path=cache)
    with pytest.raises(ValueError, match="Legacy DINOv2 cache"):
        dataset[0]
    dataset.close()
