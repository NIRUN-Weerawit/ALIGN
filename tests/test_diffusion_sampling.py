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


def test_clipped_ddim_update_uses_noise_consistent_with_clipped_clean_actions():
    head=DiffusionPolicyHead(cond_dim=4,hidden_dim=8,chunk_size=4,num_train_timesteps=100,clip_denoised=True)
    calls=[]
    def zero_noise(x,t,cond):
        calls.append(x.clone())
        return torch.zeros_like(x)
    head.predict_noise=zero_noise
    torch.manual_seed(17)
    head.sample(torch.zeros(2,1,4))
    steps=head.sampling_timesteps()
    alpha=head.alpha_bar[steps[0]];sigma=head.sigma[steps[0]]
    clean=(calls[0]/alpha.sqrt()).clamp(-1,1)
    clean[:,:,6]=clean[:,:,6].clamp(0,1)
    consistent_noise=(calls[0]-alpha.sqrt()*clean)/sigma
    previous=head.alpha_bar[steps[1]]
    expected=previous.sqrt()*clean+(1-previous).sqrt()*consistent_noise
    torch.testing.assert_close(calls[1],expected,rtol=0,atol=1e-7)


def test_full_history_diffusion_conditions_on_ordered_visual_and_state_steps():
    head = DiffusionPolicyHead(cond_dim=6, hidden_dim=8, time_dim=8,
                               chunk_size=8, history_size=3)
    visual = torch.randn(2, 3, 4)
    states = torch.randn(2, 3, 2)
    cond = head(visual, states)
    assert cond.shape == (2, 3, 6)
    torch.testing.assert_close(cond, torch.cat((visual, states), dim=-1))
    with torch.no_grad():
        head.unet.output_proj.weight.normal_(std=0.1)
    noisy = torch.randn(2, 8, 7)
    timestep = torch.full((2,), 5, dtype=torch.long)
    original = head.predict_noise(noisy, timestep, cond)
    reordered = cond[:, [1, 0, 2]]
    changed = head.predict_noise(noisy, timestep, reordered)
    assert not torch.allclose(original, changed)


def test_full_history_start_padding_matches_streaming_window():
    from training.train_intention import _pad_full_history
    visual = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    states = torch.arange(2 * 2 * 5, dtype=torch.float32).reshape(2, 2, 5)
    v, s = _pad_full_history(visual, states, 4)
    assert v.shape == (2, 4, 3, 4) and s.shape == (2, 4, 5)
    torch.testing.assert_close(v[:, :3], visual[:, :1].expand(-1, 3, -1, -1))
    torch.testing.assert_close(s[:, :3], states[:, :1].expand(-1, 3, -1))
    torch.testing.assert_close(v[:, -1], visual[:, -1])


def test_plain_history_stream_encodes_each_observation_once(monkeypatch):
    import models.align_intention as module
    from models.align_intention import ALIGNIntentionModel
    from models.intention_stream import IntentionStream
    from tests.test_intention_training_contracts import FakeVision

    monkeypatch.setattr(module, "VisionEncoder", FakeVision)
    model = ALIGNIntentionModel(state_dim=8, mamba_output_dim=0, action_dim=7,
        chunk_size=4, history_size=3, num_cameras=1, compressed_dim=4,
        head_type="diffusion", head_history_mode="full", head_d_model=8)
    model.eval()
    calls = []
    original = model._vision_forward
    def counted(frames):
        calls.append(frames.shape[0])
        return original(frames)
    monkeypatch.setattr(model, "_vision_forward", counted)
    stream = IntentionStream(model, 3)
    for step in range(3):
        output = stream.observe(
            torch.full((1, 1, 4, 4, 3), step, dtype=torch.uint8),
            torch.full((1, 7), float(step)))
        assert output["z_v_pooled_seq"].shape == (1, 3, 8)
    assert calls == [1, 1, 1]
    visual, states, intent = model.condition_actions(
        output["z_v_pooled_seq"], output["z_s_seq"])
    assert intent is None
    assert model.action_condition(visual, states).shape == (1, 3, 16)
    assert model.sample_actions(visual, states).shape == (1, 4, 7)
