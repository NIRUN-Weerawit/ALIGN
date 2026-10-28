"""Experimental controls must preserve the starting actor and fixed decoder."""
import copy

import pytest
import torch

from models.task_context_policy import ContextDiffusionPolicyHead
from tools.experiment_context_fusion import make_head, paired_interval, payload


def checkpoint():
    torch.manual_seed(8)
    head = ContextDiffusionPolicyHead(cond_dim=12, action_dim=7, hidden_dim=8,
        time_dim=8, chunk_size=8, num_train_timesteps=20,
        num_inference_steps=10, context_tokens=4, context_dim=4)
    # A trained decoder has nonzero output weights; gradients must reach fusion.
    torch.nn.init.normal_(head.unet.output_proj.weight, std=.01)
    for name, value in head.unet.named_parameters():
        if name.endswith('film.1.weight'):
            torch.nn.init.normal_(value, std=.01)
    return dict(config=dict(action_dim=7, head_d_model=8, chunk_size=8, history_size=1),
                model_state_dict={'intention_head.'+k: v for k, v in head.state_dict().items()})


@pytest.mark.parametrize('mode', ContextDiffusionPolicyHead.FUSION_MODES)
def test_experiment_copies_identical_actor_and_freezes_decoder(mode):
    torch.set_num_threads(2)
    source = checkpoint()
    base = make_head(source, 'additive', 8, torch.device('cpu'))
    head = make_head(source, mode, 8, torch.device('cpu'))
    visual, state, tokens = torch.randn(3, 1, 8), torch.randn(3, 1, 4), torch.randn(3, 5, 4)
    assert torch.equal(base(visual, state, tokens), head(visual, state, tokens))
    assert all(p.requires_grad == name.startswith('context_') for name, p in head.named_parameters())
    before = copy.deepcopy(head.unet.state_dict())
    loss = head.loss(torch.randn(3, 8, 7), head(visual, state, tokens))
    loss.backward()
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in head.unet.state_dict().items())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters() if p.requires_grad)


def test_snapshot_changes_only_task_and_bootstrap_respects_pairing():
    data = dict(task=torch.ones(2, 4, 3), frozen_task40=torch.zeros(2, 4, 3), execution=torch.full((2, 3), 5.))
    dynamic = payload(data, slice(None), 'dynamic', 'cpu')
    frozen = payload(data, slice(None), 'freeze40', 'cpu')
    assert torch.equal(dynamic[:, -1], frozen[:, -1])
    assert not torch.equal(dynamic[:, :-1], frozen[:, :-1])
    a = [dict(episode='a', anchor=4, position_mse=3.), dict(episode='b', anchor=5, position_mse=4.)]
    b = [dict(episode='a', anchor=4, position_mse=2.), dict(episode='b', anchor=5, position_mse=3.)]
    result = paired_interval(a, b)
    assert result['mean'] == 1. and result['episode_bootstrap_95pct'] == [1., 1.]
    with pytest.raises(AssertionError):
        paired_interval(a, b[::-1])


def test_assessment_pairs_seed_averages_at_episode_level():
    from tools.assess_context_fusion_experiment import FIELDS, mean_rows, paired
    def rows(offset):
        return [dict(episode=episode, task='task', anchor=anchor,
                     **{field: value + offset for field in FIELDS})
                for episode, anchor, value in [('a', 4, 1.), ('a', 5, 3.), ('b', 6, 8.)]]
    averaged = mean_rows([dict(rows=rows(0)), dict(rows=rows(2))])
    assert [r['position_mse'] for r in averaged] == [2., 4., 9.]
    result = paired(averaged, rows(0))
    assert result['position_mse']['mean'] == 1.
    assert result['position_mse']['episode_bootstrap_95pct'] == [1., 1.]
    with pytest.raises(ValueError, match='anchor order'):
        mean_rows([dict(rows=rows(0)), dict(rows=rows(0)[::-1])])
