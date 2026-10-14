import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import models.align_intention as align
from scripts.run_intention_ablation import CachedEpisodes, collate_segments
from training.cached_loading import collate_cached_segments
from training.train_intention import train_v4_epoch
from tests.test_sparse_plain_history import Features
from tests.test_intention_training_contracts import FakeVision


class CausalEncoder(torch.nn.Module):
    def __init__(self, state_dim, intent_dim, **kwargs):
        super().__init__()
        self.project = torch.nn.Linear(768 + state_dim, intent_dim)
        self.seen = []
    def forward_sequence(self, visual, states, readout_start=0):
        self.seen.append(visual.shape[1])
        return self.project(torch.cat((visual.flatten(2), states), -1)).cumsum(1).unsqueeze(2)


def test_full_segment_collation_preserves_every_frame_and_padding():
    parent = CachedEpisodes(Features(), [0,1], 20, 42, True, 'episode', 4, 4, .8, True, profile=True)
    items = [parent[i] for i in range(2)]
    expected = collate_segments(items)
    actual = collate_cached_segments(items)
    for key, value in expected.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            np.testing.assert_array_equal(actual[key], value)
        else:
            assert actual[key] == value
    assert actual['frames_segment'].shape[1] == 17
    assert torch.count_nonzero(actual['frames_segment'][1,11:]) == 0
    assert all(value >= 0 for value in actual['data_stage_cpu_s'].values())


@pytest.mark.parametrize('intent,memory', [(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('head_type', ['diffusion','flow_matching'])
def test_full_loader_preserves_updates_and_causal_bank(monkeypatch, intent, memory, head_type):
    monkeypatch.setattr(align, 'VisionEncoder', FakeVision)
    monkeypatch.setattr(align, 'IntentionEncoder', CausalEncoder)
    parent = CachedEpisodes(Features(), [0,1], 20, 42, True, 'episode', 4, 4, .8, True)
    items = [parent[i] for i in range(2)]
    reference = align.ALIGNIntentionModel(state_dim=4, compressed_dim=4,
        mamba_output_dim=4 if intent else 0, use_intent_tokens=intent, intent_dim=4,
        num_intent_tokens=1, use_memory_bank=memory, memory_mode='episodic',
        memory_detach_writes=True, memory_bank_len=3, history_size=1,
        head_type=head_type, head_d_model=8, chunk_size=4)
    reference._build_head_and_bank(1024)
    optimized = copy.deepcopy(reference)
    def loss(target, cond, sample_mask=None, **kwargs):
        if sample_mask is not None: target, cond = target[sample_mask], cond[sample_mask]
        return cond.square().mean() + target.square().mean()
    for model in (reference,optimized): model.intention_head.loss = loss
    args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7, head_type=head_type,
        skip_nan=False, grad_clip=1., no_sample_during_train=True, head_window_batch_size=2,
        profile_data_stages=True)
    losses = []
    for model,collator in ((reference,collate_segments),(optimized,collate_cached_segments)):
        torch.manual_seed(17)
        value,_ = train_v4_epoch(model,[collator(items)],torch.optim.SGD(model.parameters(),lr=.001),torch.device('cpu'),args)
        losses.append(value)
    assert losses[0] == pytest.approx(losses[1],rel=1e-6)
    for left,right in zip(reference.parameters(),optimized.parameters()):
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)
    if intent:
        assert optimized.intention_encoder.seen == reference.intention_encoder.seen == [14]
    if memory:
        for key in ('perceptual_bank','state_bank','_count'):
            torch.testing.assert_close(getattr(reference.memory_module,key),getattr(optimized.memory_module,key))


def test_full_direct_loader_spawn_worker_and_epoch_updates():
    parent = CachedEpisodes(Features(),[0,1],20,42,True,'episode',4,4)
    loader = DataLoader(parent,batch_size=2,num_workers=1,persistent_workers=True,
        multiprocessing_context='spawn',prefetch_factor=1,collate_fn=collate_cached_segments)
    try:
        for epoch in (1,2):
            parent.epoch = epoch
            expected = collate_segments([parent[i] for i in range(2)])
            actual = next(iter(loader))
            for key,value in expected.items():
                if isinstance(value,(np.ndarray,torch.Tensor)):
                    np.testing.assert_array_equal(actual[key],value)
            assert actual['frames_segment'].is_shared()
    finally:
        if loader._iterator is not None: loader._iterator._shutdown_workers()
