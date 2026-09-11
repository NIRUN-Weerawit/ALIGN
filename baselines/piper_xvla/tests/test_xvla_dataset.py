import numpy as np
import pytest
import torch

from piper_xvla.xvla_dataset import (
    PiperXVLAAdapterDataset,
    PiperXVLALiberoAdapterDataset,
    PiperXVLAConversion,
    discover_intact_replay_episodes,
    open_piper_xvla_dataset,
    write_xvla_training_manifest,
)


class SourceDataset:
    def __init__(self):
        self.item = {
            "observation.state": torch.tensor([0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0, -0.04]),
            "action": torch.tensor([0.2, 0.3, 0.4, 1, 0, 0, 1, 0, 0, 0.04]),
            "task": "grab an object and put in a cup",
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.item


def test_adapter_pads_raw_piper_vectors_and_normalizes_gripper_without_mutating_source():
    source = SourceDataset()
    dataset = PiperXVLAAdapterDataset(source, PiperXVLAConversion(-0.04, 0.04))

    sample = dataset[0]

    assert sample["observation.state"].shape == (20,)
    assert sample["action"].shape == (20,)
    torch.testing.assert_close(sample["observation.state"][:9], source.item["observation.state"][:9])
    torch.testing.assert_close(sample["action"][:9], source.item["action"][:9])
    assert sample["observation.state"][9].item() == pytest.approx(0.0)
    assert sample["action"][9].item() == pytest.approx(1.0)
    torch.testing.assert_close(sample["observation.state"][10:], torch.zeros(10))
    torch.testing.assert_close(sample["action"][10:], torch.zeros(10))
    assert source.item["observation.state"].shape == (10,)
    assert source.item["observation.state"][9].item() == pytest.approx(-0.04)



def test_libero_checkpoint_adapter_emits_8d_proprio_two_named_views_and_black_third_view():
    source = SourceDataset()
    source.item["observation.images.global_rgb"] = torch.full((3, 480, 640), 0.5)
    source.item["observation.images.wrist_rgb"] = torch.full((3, 480, 640), 0.25)
    dataset = PiperXVLALiberoAdapterDataset(source, PiperXVLAConversion(-0.04, 0.04))

    sample = dataset[0]

    assert tuple(sample["observation.state"].shape) == (8,)
    torch.testing.assert_close(sample["observation.state"][:3], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(sample["observation.state"][3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert sample["observation.state"][7].item() == pytest.approx(0.0)
    assert tuple(sample["action"].shape) == (20,)
    assert tuple(sample["observation.images.image"].shape) == (3, 480, 640)
    assert tuple(sample["observation.images.image2"].shape) == (3, 480, 640)
    assert tuple(sample["observation.images.empty_camera_0"].shape) == (3, 224, 224)
    assert torch.count_nonzero(sample["observation.images.empty_camera_0"]) == 0

    with pytest.raises(ValueError, match="strictly greater"):
        PiperXVLAConversion(0.1, 0.1)
    conversion = PiperXVLAConversion(-0.04, 0.04)
    with pytest.raises(ValueError, match="shape"):
        conversion.to_xvla20(np.zeros(9, dtype=np.float32))


def test_discover_intact_episodes_uses_existing_data_files_not_stale_metadata():
    root = "/home/ucluser/ALIGN/baselines/data/piper_replay/dataset"

    episodes = discover_intact_replay_episodes(root)

    assert len(episodes) == 101
    assert episodes[0] == 12
    assert episodes[-1] == 124
    assert 0 not in episodes
    assert 116 not in episodes


def test_real_retained_dataset_opens_as_20d_xvla_view_without_copying_images():
    dataset, conversion = open_piper_xvla_dataset(
        "/home/ucluser/ALIGN/baselines/data/piper_replay/dataset"
    )

    assert len(dataset) == 24352
    assert conversion.gripper_min_m == pytest.approx(-0.0424)
    assert conversion.gripper_max_m == pytest.approx(0.0479)
    sample = dataset[0]
    assert tuple(sample["observation.state"].shape) == (20,)
    assert tuple(sample["action"].shape) == (20,)
    torch.testing.assert_close(sample["observation.state"][10:], torch.zeros(10))
    torch.testing.assert_close(sample["action"][10:], torch.zeros(10))
    assert 0.0 <= sample["observation.state"][9].item() <= 1.0


def test_manifest_records_explicit_kept_episode_ids_and_xvla_contract(tmp_path):
    output = tmp_path / "training_manifest.json"

    manifest = write_xvla_training_manifest(
        "/source/dataset", output, episode_ids=[12, 13], frame_count=400,
        gripper_min_m=-0.04, gripper_max_m=0.04,
    )

    assert manifest["source_episode_ids"] == [12, 13]
    assert manifest["contiguous_training_episode_count"] == 2
    assert manifest["frame_count"] == 400
    assert manifest["features"]["action"]["shape"] == [20]
    assert manifest["features"]["observation.state"]["shape"] == [8]
    assert manifest["xvla"]["action_mode"] == "ee6d"
    assert manifest["xvla"]["empty_cameras"] == 1
    assert output.is_file()
