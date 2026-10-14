from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

import models.align_intention as align
from scripts.run_intention_ablation import CachedEpisodes, collate_segments
from training.plain_history_training import (SparseHistoryEpisodes,
    collate_sparse_histories, history_conditions, plain_history_epoch)
from training.train_intention import train_v4_epoch, validate
from tests.test_intention_training_contracts import FakeVision


class Features:
    cameras = ['image']
    def __init__(self):
        rng = np.random.default_rng(5)
        self.frames = [rng.normal(size=(n, 257, 768)).astype('float32') for n in (17, 11)]
    def _get_episode_length(self, ep): return len(self.frames[ep])
    def _read_frames_dinov2(self, ep, start, count): return self.frames[ep][start:start+count]
    def _read_poses(self, ep, start, count):
        return np.tile(np.arange(start, start+count)[:, None] / 20, (1, 6)).astype('float32')
    def _read_poses_gripper(self, ep, start, count): return np.arange(start, start+count) % 2
    def _read_actions(self, ep, start, count):
        return np.concatenate((self._read_poses(ep, start, count),
                               self._read_poses_gripper(ep, start, count)[:, None]), 1).astype('float32')
    def _read_text(self, ep): return 'move object'


@pytest.mark.parametrize('history', [1, 4, 20])
@pytest.mark.parametrize('copy_mode', ['direct', 'materialized'])
def test_sparse_preserves_anchors_targets_and_dropout_histories(history, copy_mode):
    parent = CachedEpisodes(Features(), [0, 1], 20, 42, True, 'episode', 4, 4, .8, True)
    sparse = SparseHistoryEpisodes(parent, history, profile=True, copy_mode=copy_mode)
    items = [sparse[i] for i in range(2)]
    batch = collate_sparse_histories(items)
    assert set(batch['data_stage_cpu_s']) == {
        'episode_metadata_and_mmap_wall_s', 'history_selection_wall_s',
        'feature_materialization_wall_s', 'target_and_mask_gather_wall_s', 'batch_pack_wall_s'}
    assert all(value >= 0 for value in batch['data_stage_cpu_s'].values())
    expected = []
    for ep in range(2):
        dense = parent[ep]
        for anchor in np.flatnonzero(dense['loss_anchor_mask']):
            expected.append((anchor, ep, dense))
    for i, (anchor, ep, dense) in enumerate(sorted(expected, key=lambda x: (x[0], x[1]))):
        times = np.maximum(0, np.arange(anchor-history+1, anchor+1))
        index = batch['history_indices'][i]
        np.testing.assert_array_equal(batch['frames'][index], dense['frames_segment'][times])
        np.testing.assert_array_equal(batch['states'][index], dense['states_segment'][times])
        np.testing.assert_array_equal(batch['camera_mask'][index], dense['observation_camera_mask'][times])
        np.testing.assert_array_equal(batch['state_mask'][index], dense['observation_state_mask'][times])
        np.testing.assert_array_equal(batch['targets'][i], dense['actions_segment'][anchor:anchor+4])
    assert len(batch['frames']) <= sum(len(parent[i]['frames_segment']) for i in range(2))


@pytest.mark.parametrize('history,head_mode', [(1, 'mean'), (1, 'full'), (4, 'full')])
@pytest.mark.parametrize('accumulate', [False, True])
@pytest.mark.parametrize('copy_mode', ['direct', 'materialized'])
def test_sparse_matches_dense_updates_and_validation(monkeypatch, history, accumulate, copy_mode, head_mode):
    monkeypatch.setattr(align, 'VisionEncoder', FakeVision)
    parent = CachedEpisodes(Features(), [0, 1], 20, 42, True, 'episode', 100, 4, .8, True)
    items = [SparseHistoryEpisodes(parent, history, copy_mode=copy_mode)[i] for i in range(2)]
    sparse = collate_sparse_histories(items)
    dense = collate_segments([parent[i] for i in range(2)])
    torch.manual_seed(4)
    reference = align.ALIGNIntentionModel(state_dim=4, compressed_dim=4,
        mamba_output_dim=0, use_intent_tokens=False, use_memory_bank=False,
        history_size=history, head_history_mode=head_mode, head_type='diffusion',
        head_d_model=8, chunk_size=4)
    reference._build_head_and_bank(1024)
    for module in reference.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    optimized = copy.deepcopy(reference)
    # Deterministic objective isolates conditions and gradients from diffusion RNG regrouping.
    def loss(target, cond, sample_mask=None, **kwargs):
        if sample_mask is not None: target, cond = target[sample_mask], cond[sample_mask]
        return cond.square().mean() + target.square().mean()
    for model in (reference, optimized):
        model.intention_head.loss = loss
        model.intention_head.sample = lambda cond: cond.new_zeros((len(cond), 4, 7))
    args = SimpleNamespace(history_size=history, chunk_size=4, action_dim=7,
        head_type='diffusion', skip_nan=False, grad_clip=1., no_sample_during_train=True,
        head_window_batch_size=2, plain_head_batch_size=3, gripper_threshold=.5,
        gradient_accum_steps=2 if accumulate else 1, profile_data_stages=True)
    dense_loss, _ = train_v4_epoch(reference, [dense], torch.optim.SGD(reference.parameters(), lr=.001), torch.device('cpu'), args)
    sparse_batches = [collate_sparse_histories([item]) for item in items] if accumulate else [sparse]
    sparse_loss, _ = plain_history_epoch(optimized, sparse_batches, torch.device('cpu'), args,
        optimizer=torch.optim.SGD(optimized.parameters(), lr=.001))
    assert dense_loss == pytest.approx(sparse_loss, rel=1e-6)
    for left, right in zip(reference.parameters(), optimized.parameters()):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    dense_val = validate(reference, [dense], torch.device('cpu'), args)
    sparse_val = plain_history_epoch(optimized, [sparse], torch.device('cpu'), args)
    assert dense_val[0] == pytest.approx(sparse_val[0], rel=1e-6)
    for key in ('pos_mse', 'rot_mse', 'grip_mse', 'grip_acc'):
        assert dense_val[2][key] == pytest.approx(sparse_val[2][key], rel=1e-6)


def test_direct_worker_batches_match_main_process_and_share_epoch():
    from torch.utils.data import DataLoader
    parent = CachedEpisodes(Features(), [0, 1], 20, 42, True, 'episode', 4, 4)
    data = SparseHistoryEpisodes(parent, 4)
    loader = DataLoader(data, batch_size=2, num_workers=1,
        multiprocessing_context='spawn', persistent_workers=True,
        prefetch_factor=1, collate_fn=collate_sparse_histories)
    try:
        for epoch in (1, 2):
            parent.epoch = epoch
            expected = collate_sparse_histories([data[i] for i in range(2)])
            actual = next(iter(loader))
            for key, value in expected.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
            assert actual['frames'].is_shared()
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
