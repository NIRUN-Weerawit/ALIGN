"""Exercise real HDF5 readers through both ablation loader paths."""
import h5py
import numpy as np
from data.align_dataset import ALIGNDataset
from scripts.run_intention_ablation import CachedEpisodes
from training.plain_history_training import SparseHistoryEpisodes


def test_missing_measured_gripper_is_causal_in_dense_and_sparse(tmp_path):
    with h5py.File(tmp_path / 'episodes.h5', 'w') as f:
        group = f.create_group('ep_000000')
        actions = np.zeros((20, 7), dtype=np.float32)
        actions[4:10, 6] = 1
        group['actions'] = actions
        group.attrs['initial_gripper'] = 1.
        dataset = object.__new__(ALIGNDataset)
        dataset._h5 = f
        dataset._single_episode = False
        dataset._has_actions = True
        dataset._episode_keys = ['ep_000000']
        dataset._ep_pose_offsets = [0]
        dataset._get_episode_length = lambda ep: 20
        dataset._read_frames_dinov2 = lambda ep, start, count, **kwargs: np.zeros((count, 2, 3), np.float32)
        dataset._read_poses = lambda ep, start, count: np.zeros((count, 6), np.float32)
        dataset._read_text = lambda ep: 'move'
        dense = CachedEpisodes(dataset, [0], 20, 42, True, 'episode', 8, 4,
                               gripper_transition_fraction=.5)
        item = dense[0]
        expected = np.r_[1., actions[:-1, 6]]
        np.testing.assert_array_equal(item['states_segment'][:, 6], expected)
        assert item['states_segment'][4, 6] != actions[4, 6]
        assert item['loss_anchor_mask'][[4, 10]].all()
        assert item['loss_anchor_mask'].sum() == 8
        sparse = SparseHistoryEpisodes(dense, 3)[0]
        times = np.maximum(0, sparse['anchors'][:, None] + np.arange(-2, 1))
        np.testing.assert_array_equal(sparse['states'][sparse['history_indices'], 6], expected[times])
        np.testing.assert_array_equal(dataset._read_poses_gripper(0, 5, 3), expected[5:8])
        group['gripper'] = np.full(20, .3, np.float32)
        np.testing.assert_allclose(dataset._read_poses_gripper(0, 5, 3), .3)
        np.testing.assert_allclose(dataset._read_poses_gripper(0, 20, 3), .3)
        dense.gripper_state_source = 'legacy_zero'
        assert not dense[0]['states_segment'][:, 6].any()
        dataset._h5 = None


def test_validation_anchor_distribution_unchanged():
    from tests.test_intention_ablation_setup import FakeDataset
    a = CachedEpisodes(FakeDataset(), [0], 100, 42, False, 'episode', 8, 4)
    b = CachedEpisodes(FakeDataset(), [0], 100, 42, False, 'episode', 8, 4,
                       gripper_transition_fraction=1.)
    np.testing.assert_array_equal(a[0]['loss_anchor_mask'], b[0]['loss_anchor_mask'])


def test_transition_sampling_cli_rejects_invalid_or_inactive_settings():
    import pytest
    from scripts.run_intention_ablation import parse_args
    for value in ['-0.1', '1.1', 'nan']:
        with pytest.raises(SystemExit):
            parse_args(['--output', 'example', '--gripper-transition-fraction', value])
    with pytest.raises(SystemExit):
        parse_args(['--output', 'example', '--temporal-sampling', 'crop',
                        '--gripper-transition-fraction', '.25'])
    args = parse_args(['--output', 'example', '--temporal-sampling', 'crop',
                       '--gripper-transition-fraction', '0'])
    assert args.gripper_state_source == 'causal'
