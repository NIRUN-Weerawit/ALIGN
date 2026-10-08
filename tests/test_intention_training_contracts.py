"""Regression coverage for causal state, padded loss, and optional memory."""

import json

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from data.align_dataset import ALIGNDataset, v4_segment_collate
from data.gripper_state import previous_gripper_commands, carry_gripper_state
from models.align_intention import ALIGNIntentionModel
from models.intention_head import DiffusionPolicyHead, FlowMatchingPolicyHead
from models.memory_bank import MemoryRetrieval, PerceptualCognitiveMemoryModule


def write_dataset(path, measured=False):
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:, 6] = np.arange(12) % 2
    with h5py.File(path, "w") as f:
        ep = f.create_group("ep_000")
        ep.create_dataset("frames/wrist_image", data=np.zeros((12, 8, 8, 3), np.uint8))
        ep.create_dataset("poses", data=np.zeros((12, 6), np.float32))
        ep.create_dataset("actions", data=actions)
        ep.create_dataset("texts", data=json.dumps(["task"]))
        ep.attrs["initial_gripper"] = -0.25
        if measured:
            ep.create_dataset("gripper", data=np.full(12, 0.3, np.float32))
    return actions


def test_gripper_fallback_is_causal_across_cropped_windows(tmp_path):
    path = tmp_path / "sample.h5"
    actions = write_dataset(path)
    with ALIGNDataset(str(path), cameras=["wrist_image"], frames_per_ep=2, traj_window=2) as ds:
        np.testing.assert_array_equal(ds[0]["grippers"], [-0.25, 0, 1, 0])
        np.testing.assert_array_equal(ds[1]["grippers"], actions[1:5, 6])
        assert ds._read_robot_state(0, 3)[6] == actions[2, 6]
        batch = v4_segment_collate([ds[0]], history_size=1, chunk_size=2,
                                   segment_min_mult=4, segment_max_mult=4)
        np.testing.assert_array_equal(batch["states_segment"][0, :, 6], [-0.25, 0, 1, 0])
        np.testing.assert_array_equal(batch["actions_segment"][0], actions[:4])


def test_measured_gripper_takes_precedence_and_padding_carries_state(tmp_path):
    path = tmp_path / "sample.h5"
    write_dataset(path, measured=True)
    with ALIGNDataset(str(path), cameras=["wrist_image"], frames_per_ep=2, traj_window=2) as ds:
        np.testing.assert_allclose(ds[0]["grippers"], 0.3)
        np.testing.assert_allclose(ds._read_poses_gripper(0, 10, 4), 0.3)


def test_causal_gripper_never_reads_previous_episode_or_mutates_last_state():
    actions = np.zeros((8, 7), np.float32)
    actions[:, 6] = np.arange(8)
    np.testing.assert_array_equal(previous_gripper_commands(actions, count=3, episode_start=4), [0, 4, 5])
    last = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    next_state = carry_gripper_state(last, np.ones(6))
    np.testing.assert_array_equal(next_state, np.ones(7))
    np.testing.assert_array_equal(last[:6], np.zeros(6))


@pytest.mark.parametrize("head_cls", [DiffusionPolicyHead, FlowMatchingPolicyHead])
def test_invalid_generative_samples_do_not_change_loss_or_gradients(head_cls):
    head = head_cls(cond_dim=4, hidden_dim=8, chunk_size=4)
    # A simple predictor exposes conditioning gradients without U-Net zero init.
    predictor = lambda x, t, cond: cond[:, :, :1].expand_as(x) + 0.1 * x
    if head_cls is DiffusionPolicyHead:
        head.predict_noise = predictor
    else:
        head.predict_velocity = predictor
    target = torch.randn(2, 4, 7, requires_grad=True)
    cond = torch.randn(2, 1, 4, requires_grad=True)
    with torch.no_grad():
        target[1] = float("nan")
        cond[1] = float("nan")
    weights = torch.tensor([1., 1., 1., 1., 1., 1., .01])
    torch.manual_seed(17)
    masked = head.loss(target, cond, dim_weights=weights, sample_mask=torch.tensor([True, False]))
    torch.manual_seed(17)
    reference = head.loss(target[:1], cond[:1], dim_weights=weights)
    torch.testing.assert_close(masked, reference)
    masked.backward()
    assert torch.isfinite(masked)
    assert torch.count_nonzero(cond.grad[1]) == 0
    assert torch.count_nonzero(target.grad[1]) == 0
    assert torch.count_nonzero(cond.grad[0]) > 0


def test_mixed_empty_memory_rows_are_finite_and_preserve_empty_query():
    retrieval = MemoryRetrieval(4, num_heads=2)
    query = torch.randn(2, 4, requires_grad=True)
    output = retrieval(query, torch.randn(2, 3, 4),
                       torch.tensor([[True, False, False], [False, False, False]]))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[1], query[1])
    output.square().sum().backward()
    assert torch.isfinite(query.grad).all()


class FakeVision(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fusion_type = kwargs.get("fusion_type", "transformer")
        self.backbone = nn.Linear(1, 1)

    def forward(self, frames):
        tokens = frames.float().reshape(frames.shape[0], -1).mean(-1)
        return tokens[:, None, None].expand(-1, 3, 768)


def test_streaming_camera_features_match_batched_training(monkeypatch):
    import models.align_intention as module
    monkeypatch.setattr(module, "VisionEncoder", FakeVision)
    model = ALIGNIntentionModel(state_dim=4, compressed_dim=4, num_cameras=2,
                                use_intent_tokens=False, head_type="flow_matching",
                                head_d_model=8).eval()
    frames = torch.randint(0, 256, (2, 3, 2, 4, 4, 3), dtype=torch.uint8)
    states = torch.randn(2, 3, 7)
    with torch.no_grad():
        batched = model(frames, states)
        for t in range(3):
            visual, state, _, _ = model.encode_step(frames[:, t], states[:, t])
            assert visual.shape == (2, 16)  # 2 cameras * 2 patches * 4 compressed dims
            torch.testing.assert_close(visual, batched["z_v_pooled_seq"][:, t])
            torch.testing.assert_close(state, batched["z_s_seq"][:, t])


def test_disabled_intent_ablation_omits_encoder_and_retrieves_memory(monkeypatch):
    import models.align_intention as module
    monkeypatch.setattr(module, "VisionEncoder", FakeVision)
    monkeypatch.setattr(module, "IntentionEncoder", lambda **kwargs: pytest.fail("Ablation constructed Mamba"))
    model = ALIGNIntentionModel(state_dim=4, compressed_dim=4,
                                use_intent_tokens=False, use_memory_bank=True,
                                head_type="flow_matching", head_d_model=8,
                                memory_bank_len=2)
    assert model.use_history  # Observation history is still available.
    assert model.intention_encoder is None
    assert not any(name.startswith("intention_encoder.") for name in model.state_dict())
    model._build_head_and_bank(8)
    model.memory_module.reset(2, torch.device("cpu"))
    history = model.forward_intent(torch.randn(2, 3, 1, 768), torch.randn(2, 3, 4))
    assert history["intent_emb"] is None
    calls = []
    handle = model.memory_module.perceptual_retrieval.register_forward_hook(lambda *args: calls.append(True))
    losses = []
    for _ in range(4):
        v, s, intent = model.condition_actions(torch.randn(2, 3, 8), torch.randn(2, 3, 4),
                                              observed_mask=torch.tensor([True, False]))
        assert v.shape == (2, 1, 8) and s.shape == (2, 1, 4)
        assert intent is None
        losses.append(v.square().sum() + s.square().sum())
    handle.remove()
    assert len(calls) == 4
    assert model.memory_module._count.tolist() == [2, 0]
    sum(losses).backward()
    grad = model.memory_module.perceptual_retrieval.retrieval_attn.in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_memory_with_intent_still_fuses_all_three_streams():
    memory = PerceptualCognitiveMemoryModule(8, 4, 4, bank_len=2, num_heads=2)
    memory.reset(1, torch.device("cpu"))
    p, s, c = memory(torch.randn(1, 8), torch.randn(1, 4), torch.randn(1, 2, 2))
    p, s, c = memory(torch.randn(1, 8), torch.randn(1, 4), torch.randn(1, 2, 2))
    assert (p.shape, s.shape, c.shape) == (torch.Size([1, 8]), torch.Size([1, 4]), torch.Size([1, 2, 2]))
    assert torch.isfinite(c).all()
    (p.square().sum() + s.square().sum() + c.square().sum()).backward()
    assert memory.cognitive_retrieval.retrieval_attn.in_proj_weight.grad is not None


def test_memory_observed_mask_preserves_real_samples_only():
    memory = PerceptualCognitiveMemoryModule(8, 0, 4, bank_len=2, num_heads=2)
    memory.reset(2, torch.device("cpu"))
    memory(torch.randn(2, 8), torch.randn(2, 4), observed_mask=torch.tensor([False, True]))
    assert memory._count.tolist() == [0, 1]


@pytest.mark.parametrize("use_memory", [False, True])
def test_real_trainer_and_validation_ignore_padded_targets(monkeypatch, use_memory):
    from types import SimpleNamespace
    from training.train_intention import train_v4_epoch, train_v4_batched_epoch, validate
    import models.align_intention as module
    monkeypatch.setattr(module, "VisionEncoder", FakeVision)
    model = ALIGNIntentionModel(state_dim=4, compressed_dim=4,
                                mamba_output_dim=0, use_memory_bank=use_memory,
                                head_type="flow_matching", head_d_model=8,
                                chunk_size=2, memory_bank_len=2)
    model._build_head_and_bank(8)
    # Fast differentiable head objective isolates the trainer's masking/reduction.
    calls = []
    def loss(target, cond, dim_weights=None, sample_mask=None):
        calls.append(sample_mask.tolist())
        torch.testing.assert_close(dim_weights, torch.ones(7))
        valid_cond = cond[sample_mask]
        return target[sample_mask].square().mean() + valid_cond.square().mean() * .001
    model.intention_head.loss = loss
    def sample(cond, num_steps=None):
        assert num_steps is None  # Action chunk length must not set denoising steps.
        return cond.new_zeros(cond.shape[0], 2, 7)
    model.intention_head.sample = sample
    actions = np.ones((2, 5, 7), np.float32)
    actions[1, 3:] = 1000.0  # Replicated padding must not enter loss or metrics.
    batch = {"frames_segment": np.zeros((2, 5, 1, 4, 4, 3), np.uint8),
             "states_segment": np.zeros((2, 5, 7), np.float32),
             "actions_segment": actions, "segment_len": np.array([5, 3])}
    args = SimpleNamespace(history_size=2, chunk_size=2, action_dim=7,
                           head_type="flow_matching", no_sample_during_train=True,
                           skip_nan=True, grad_clip=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_fn = train_v4_epoch if use_memory else train_v4_batched_epoch
    train_loss, _ = train_fn(model, [batch], optimizer, torch.device("cpu"), args)
    assert 1.0 <= train_loss < 1.1
    assert calls == [[True, True], [True, True], [True, False], [True, False]]
    _, _, metrics = validate(model, [batch], torch.device("cpu"), args)
    assert metrics["pos_mse"] == 1.0
    assert metrics["rot_mse"] == 1.0
    assert metrics["grip_mse"] == 1.0
    assert metrics["gripper_genuine_batches"] == 4


@pytest.mark.parametrize("commands", [[1., 0., 1.], [0.8, 0.2, 0.9], [0.5, 0.51, 0.1]])
@pytest.mark.parametrize("action_scale", [1.0, 10.0])
def test_rollout_carries_executed_gripper_without_rewriting_history(monkeypatch, commands, action_scale):
    import eval.eval_libero_v4_trajectory as evaluator
    monkeypatch.setattr(evaluator, "get_sim_eef_pose", lambda obs: np.zeros(6, np.float32))
    monkeypatch.setattr(evaluator, "get_sim_frame", lambda *args, **kwargs: np.zeros((4, 4, 3), np.uint8))
    class Env:
        def reset(self):
            return {}
        def step(self, action):
            executed.append(float(action[6]))
            return {}, 0.0, False, {}
    class Model:
        head_type = "flow_matching"
        use_memory_bank = False
        def __init__(self):
            self.states = []
        def __call__(self, frames, states):
            self.states.append(states.clone())
            return {"h_seq": states.new_zeros(1, 2, 1), "intent_emb": None,
                    "z_v_pooled_seq": states.new_zeros(1, 2, 4), "z_s_seq": states}
        def condition_actions(self, vision, states, intent):
            return vision, states, intent
        def sample_actions(self, *args):
            action = torch.zeros(1, 2, 7)
            action[:, :, 6] = commands[len(self.states) - 1]
            return action
    model = Model()
    executed = []
    evaluator.run_model_in_sim(Env(), model, torch.device("cpu"), np.zeros((3, 7), np.float32),
                               chunk_size=2, max_steps=3, switch_at=0.0,
                               use_camera=["wrist_image"], initial_gripper=0.25, action_scale=action_scale)
    np.testing.assert_array_equal([x[0, :, 6].numpy() for x in model.states],
                                  [[0.25, 0.25], [0.25, float(commands[0] > .5)],
                                   [float(commands[0] > .5), float(commands[1] > .5)]])

    np.testing.assert_array_equal(executed, [1.0 - 2.0 * (command > .5) for command in commands])


@pytest.mark.parametrize("head_type", ["transformer", "mamba", "hybrid"])
def test_future_heads_fail_before_model_download(head_type):
    from types import SimpleNamespace
    from training.train_intention import build_model
    with pytest.raises(ValueError, match="future work"):
        build_model(SimpleNamespace(head_type=head_type), 1, torch.device("cpu"))


def test_cli_defaults_to_supported_diffusion_head(monkeypatch):
    from training.train_intention import parse_args
    monkeypatch.setattr("sys.argv", ["train_intention.py", "--data", "sample.h5", "--output-dir", "unused"])
    args = parse_args()
    assert args.head_type == "diffusion"
    assert args.gripper_loss_weight == 1.0
    assert args.gripper_threshold == 0.5


def test_training_packet_outage_hides_current_features_and_does_not_store_them(monkeypatch):
    from types import SimpleNamespace
    import models.align_intention as module
    from training.train_intention import train_v4_epoch
    monkeypatch.setattr(module,'VisionEncoder',FakeVision)
    model=ALIGNIntentionModel(state_dim=4,compressed_dim=4,num_cameras=2,
        use_intent_tokens=False,use_memory_bank=True,head_type='flow_matching',head_d_model=8,history_size=1,action_dim=7,chunk_size=4)
    model._build_head_and_bank(2048)
    batch=dict(frames_segment=torch.randn(1,8,514,768),states_segment=torch.randn(1,8,7),
               actions_segment=torch.randn(1,8,7),segment_len=np.array([8]),
               loss_anchor_mask=torch.ones(1,8,dtype=torch.bool),observation_timesteps=torch.arange(8)[None],
               observation_camera_mask=torch.ones(1,8,2,dtype=torch.bool),observation_state_mask=torch.ones(1,8,dtype=torch.bool))
    batch['observation_camera_mask'][:,2]=False;batch['observation_state_mask'][:,2]=False
    calls=[]
    def capture(module,inputs,kwargs):calls.append((inputs[0].detach().clone(),inputs[1].detach().clone(),kwargs['observed_mask'].clone()))
    hook=model.memory_module.register_forward_pre_hook(capture,with_kwargs=True)
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type='flow_matching',skip_nan=True,grad_clip=1.,gripper_loss_weight=1.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
    loss,_=train_v4_epoch(model,[batch],optimizer,torch.device('cpu'),args,max_steps=1)
    hook.remove()
    assert np.isfinite(loss)
    assert not calls[2][0].any() and not calls[2][1].any() and not calls[2][2].any()
    assert model.memory_module._count.tolist()==[4]
    assert model.memory_module.timestamps[0,:4].tolist()==[0,1,3,4]


def test_streaming_visibility_masks_match_training_and_remove_hidden_input_leakage(monkeypatch):
    import models.align_intention as module
    from models.intention_stream import IntentionStream
    monkeypatch.setattr(module,'VisionEncoder',FakeVision)
    model=ALIGNIntentionModel(state_dim=4,compressed_dim=4,num_cameras=2,use_intent_tokens=False,
        use_memory_bank=True,memory_patch_retrieval=True,head_type='flow_matching',head_d_model=8,history_size=1)
    frames=torch.ones(1,2,4,4,3);states=torch.randn(1,7)
    camera_mask=torch.tensor([[True,False]]);state_mask=torch.tensor([False])
    first=model.encode_step(frames,states,camera_mask=camera_mask,state_mask=state_mask)
    frames[:,1]=100;states*=100
    second=model.encode_step(frames,states,camera_mask=camera_mask,state_mask=state_mask)
    torch.testing.assert_close(first[0],second[0]);assert not first[1].any()
    assert not second[0][:,8:].any() # two patches * four channels in hidden camera
    model.memory_module.reset(1,torch.device('cpu'))
    stream=IntentionStream(model,1)
    stream.observe(frames,states,store_memory=True)
    missing=stream.observe(frames,states,store_memory=True,camera_mask=torch.zeros(1,2,dtype=torch.bool),state_mask=state_mask)
    assert not missing['z_v_pooled_seq'].any() and not missing['z_s_seq'].any()
    assert not missing['observed_mask'].any() and missing['timestamp'].item()==1
    stream.observe(frames,states,store_memory=True)
    assert model.memory_module._count.item()==2
    assert model.memory_module.timestamps[0,:2].tolist()==[0,2]
