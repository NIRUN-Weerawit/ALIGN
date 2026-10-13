"""Check acceleration paths against padding, causal memory, and CUDA precision."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import models.align_intention as align
from tests.test_intention_training_contracts import FakeVision
from training.train_intention import train_v4_epoch, validate


def segment():
    actions = np.ones((2, 6, 7), dtype=np.float32)
    actions[1, 4:] = np.nan  # Padding must be excluded from loss and metrics.
    return dict(frames_segment=np.zeros((2, 6, 1, 4, 4, 3), np.uint8),
                states_segment=np.ones((2, 6, 7), np.float32),
                actions_segment=actions, segment_len=np.array([6, 4]),
                observation_timesteps=torch.arange(6).expand(2, -1).float())


def test_gradient_accumulation_matches_one_logical_batch(monkeypatch):
    monkeypatch.setattr(align, "VisionEncoder", FakeVision)
    batch = segment()
    batch["states_segment"][1] *= 2
    results = []
    for split in (False, True):
        torch.manual_seed(23)
        model = align.ALIGNIntentionModel(state_dim=4, compressed_dim=4,
            action_dim=7, use_intent_tokens=False, mamba_output_dim=0,
            use_memory_bank=False, head_type="flow_matching", head_d_model=8,
            chunk_size=4, history_size=1)
        model._build_head_and_bank(8)

        def loss(target, cond, dim_weights=None, sample_mask=None):
            if sample_mask is not None:
                target, cond = target[sample_mask], cond[sample_mask]
            return cond.square().mean() + target.square().mean()

        model.intention_head.loss = loss
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7,
            head_type="flow_matching", skip_nan=False, grad_clip=1.0,
            no_sample_during_train=True, head_window_batch_size=2,
            gradient_accum_steps=2 if split else 1)
        if split:
            batches = [{key: value[index:index + 1] for key, value in batch.items()}
                       for index in range(2)]
        else:
            batches = [batch]
        train_v4_epoch(model, batches, optimizer, torch.device("cpu"), args)
        results.append([p.detach().clone() for p in model.parameters()])
    for full, accumulated in zip(*results):
        torch.testing.assert_close(full, accumulated, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_type", ["diffusion", "flow_matching"])
@pytest.mark.parametrize("memory", [False, True])
def test_batched_validation_keeps_cuda_sampling_context(monkeypatch, head_type, memory):
    monkeypatch.setattr(align, "VisionEncoder", FakeVision)
    model = align.ALIGNIntentionModel(state_dim=4, compressed_dim=4, action_dim=7,
        use_intent_tokens=False, mamba_output_dim=0, use_memory_bank=memory,
        memory_mode="episodic", memory_bank_len=3, memory_patch_retrieval=True,
        memory_patch_temporal=True, memory_value_preserving=True,
        memory_pre_state_visual=True, visual_token_attention=True,
        head_type=head_type, head_d_model=8, chunk_size=4, history_size=1)
    model._build_head_and_bank(8)
    model.to("cuda")
    real_sample = model.intention_head.sample
    calls = []

    def sample(cond, num_steps=None):
        assert torch.is_autocast_enabled("cuda")
        assert num_steps is None  # Retain the configured sampler step count.
        calls.append(cond.shape[0])
        prediction = real_sample(cond, num_steps=num_steps)
        assert torch.isfinite(prediction).all()
        return prediction

    monkeypatch.setattr(model.intention_head, "sample", sample)
    args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7,
                           head_type=head_type, skip_nan=False)
    loss, _, metrics = validate(model, [segment()], torch.device("cuda"), args)
    assert np.isfinite(loss)
    assert calls == [4]  # Three anchors: 2 + 1 + 1 valid episode rows.
    assert metrics["gripper_genuine_batches"] == 3
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("memory", [False, True])
def test_grouped_losses_preserve_weighted_updates_and_memory_order(monkeypatch, memory):
    monkeypatch.setattr(align, "VisionEncoder", FakeVision)
    runs = []
    for group_size in (1, 2):
        torch.manual_seed(23)
        model = align.ALIGNIntentionModel(state_dim=4, compressed_dim=4, action_dim=7,
            use_intent_tokens=False, mamba_output_dim=0, use_memory_bank=memory,
            memory_mode="episodic", memory_bank_len=3,
            head_type="flow_matching", head_d_model=8, chunk_size=4, history_size=1)
        model._build_head_and_bank(8)

        # A deterministic per-example objective isolates grouping and weighting
        # from the different random-noise order of the stochastic real objective.
        def loss(target, cond, dim_weights=None, sample_mask=None):
            if sample_mask is not None:
                target, cond = target[sample_mask], cond[sample_mask]
            return cond.square().mean() + target.square().mean()

        model.intention_head.loss = loss
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7,
            head_type="flow_matching", skip_nan=False, grad_clip=1.0,
            no_sample_during_train=True, head_window_batch_size=group_size)
        train_loss, _ = train_v4_epoch(model, [segment()], optimizer, torch.device("cpu"), args)
        runs.append((model, train_loss))
    assert runs[0][1] == pytest.approx(runs[1][1], rel=1e-6)
    for left, right in zip(runs[0][0].parameters(), runs[1][0].parameters()):
        torch.testing.assert_close(left, right)
    if memory:
        for field in ("perceptual_bank", "state_bank", "_count"):
            torch.testing.assert_close(getattr(runs[0][0].memory_module, field),
                                       getattr(runs[1][0].memory_module, field))
