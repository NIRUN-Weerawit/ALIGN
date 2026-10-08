"""Check DDIM endpoint inversion and full-grid subsampling independently of horizon."""
import math
import pytest
import torch
from models.intention_head import DiffusionPolicyHead
from data.gripper_state import executed_binary_gripper

@pytest.mark.parametrize('steps', [1, 4, 8, 10])
def test_ddim_oracle_recovers_clean_actions_at_every_step_budget(steps):
    head = DiffusionPolicyHead(cond_dim=4, hidden_dim=8, time_dim=8, chunk_size=8)
    target = torch.linspace(-1, 1, 56).reshape(1, 8, 7)
    visited = []
    def oracle(x, t, cond):
        visited.append(int(t[0]))
        alpha = head.alpha_bar[t][:, None, None]
        return (x - alpha.sqrt() * target) / head.sigma[t][:, None, None]
    head.predict_noise = oracle
    actual = head.sample(torch.zeros(1, 1, 4), num_steps=steps)
    assert visited[0] == 10
    assert len(visited) == steps
    assert visited == sorted(set(visited), reverse=True)
    torch.testing.assert_close(actual, target, atol=2e-5, rtol=2e-5)
    assert head.alpha_bar[0] == 1 and head.alpha_bar[-1] > 0


def test_legacy_checkpoint_zero_endpoint_is_not_inverted():
    head = DiffusionPolicyHead(cond_dim=4, hidden_dim=8, time_dim=8, chunk_size=8)
    alpha = torch.cos((torch.arange(11).double()/10 + .008)/1.008 * math.pi/2).square().float()
    legacy = head.state_dict()
    legacy['alpha_bar'] = alpha
    legacy['sigma'] = (1-alpha).sqrt()
    head.load_state_dict(legacy, strict=True)
    assert head.sampling_timesteps().tolist() == list(range(9, -1, -1))
    assert head.sampling_timesteps(8)[0] == 9
    target = torch.ones(1, 8, 7) * .3
    head.predict_noise = lambda x,t,c: (x - head.alpha_bar[t][:,None,None].sqrt()*target) / head.sigma[t][:,None,None]
    torch.testing.assert_close(head.sample(torch.zeros(1,1,4)), target, atol=2e-5, rtol=2e-5)

@pytest.mark.parametrize('steps', [0, -1, 11, 1.5])
def test_invalid_step_budget_rejected(steps):
    head = DiffusionPolicyHead(cond_dim=4, hidden_dim=8, time_dim=8)
    with pytest.raises(ValueError):
        head.sample(torch.zeros(1,1,4), steps)

@pytest.mark.parametrize('score,command,sim', [(0.2,0,1),(.5,0,1),(.51,1,-1),(.9,1,-1)])
def test_gripper_feedback_matches_executed_binary_action(score,command,sim):
    assert executed_binary_gripper(score) == (command,sim)


def test_sampling_disables_autocast_for_epsilon_inversion():
    head = DiffusionPolicyHead(cond_dim=4, hidden_dim=8, time_dim=8, chunk_size=8)
    called = []
    def noise(x,t,cond):
        assert not torch.is_autocast_enabled('cpu')
        assert x.dtype == cond.dtype == torch.float32
        called.append(int(t[0]))
        return x
    head.predict_noise = noise
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output = head.sample(torch.zeros(1,1,4,dtype=torch.bfloat16))
    assert len(called)==10 and torch.isfinite(output).all()
