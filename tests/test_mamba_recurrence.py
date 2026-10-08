"""Causality, state isolation and gradient parity for real Mamba1 weights."""
import copy

import pytest
import torch

from models.intention_encoder import HAS_MAMBA, IntentionEncoder
from models.intention_stream import IntentionStream


pytestmark = pytest.mark.skipif(not HAS_MAMBA or not torch.cuda.is_available(),
                                reason="Requires mamba_ssm and CUDA")


@pytest.fixture(autouse=True)
def exact_matmul():
    # Trainer imports enable TF32 globally. Different batched GEMM shapes can
    # then round differently; test mathematical parity in full FP32 here.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_encoder(tokens=2):
    torch.manual_seed(42)
    return IntentionEncoder(state_dim=4, mamba_output_dim=4, num_cameras=1,
                            compressed_dim=4, raw_dim=4, mamba_d_state=4,
                            use_intent_tokens=True, num_intent_tokens=tokens,
                            intent_dim=4).cuda()


@pytest.mark.parametrize("tokens", [1, 2])
def test_causal_readouts_match_prefixes_and_streaming(tokens):
    encoder = make_encoder(tokens).eval()
    visual = torch.randn(2, 9, 1, 4, device="cuda")
    state = torch.randn(2, 9, 4, device="cuda")
    with torch.no_grad():
        intents = encoder.forward_sequence(visual, state, readout_start=2, chunk_size=3)
        cache = encoder.allocate_state(2, torch.device("cuda"))
        control_cache = encoder.allocate_state(2, torch.device("cuda"))
        for t in range(9):
            _, cache, streaming = encoder.forward_step(visual[:, t], state[:, t], cache, True)
            _, control_cache = encoder.forward_step(visual[:, t], state[:, t], control_cache, False)
            for actual, expected in zip(cache, control_cache):
                torch.testing.assert_close(actual, expected)
            if t >= 2:
                prefix = encoder(visual[:, :t + 1], state[:, :t + 1])[1]
                torch.testing.assert_close(intents[:, t], prefix, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(streaming, prefix, atol=2e-6, rtol=2e-5)
        altered = visual.clone()
        altered[:, 5:] += 100
        changed = encoder.forward_sequence(altered, state, chunk_size=4)
        torch.testing.assert_close(changed[:, 2:5], intents[:, 2:5])


def test_checkpointed_recurrence_matches_prefix_gradients():
    encoder = make_encoder().train()
    reference = copy.deepcopy(encoder)
    visual = torch.randn(2, 7, 1, 4, device="cuda", requires_grad=True)
    state = torch.randn(2, 7, 4, device="cuda", requires_grad=True)
    ref_visual = visual.detach().clone().requires_grad_()
    ref_state = state.detach().clone().requires_grad_()
    encoder.forward_sequence(visual, state, readout_start=2, chunk_size=2)[:, 2:].square().sum().backward()
    expected = torch.stack([reference(ref_visual[:, :t + 1], ref_state[:, :t + 1])[1]
                            for t in range(2, 7)], dim=1)
    expected.square().sum().backward()
    torch.testing.assert_close(visual.grad, ref_visual.grad, atol=1e-7, rtol=2e-4)
    torch.testing.assert_close(state.grad, ref_state.grad, atol=1e-7, rtol=2e-4)
    assert visual.grad[:, 0].abs().sum() > 0
    for name in ("intent_tokens", "mamba.in_proj.weight", "mamba.A_log", "intent_proj.weight"):
        actual = dict(encoder.named_parameters())[name].grad
        expected_grad = dict(reference.named_parameters())[name].grad
        torch.testing.assert_close(actual, expected_grad, atol=1e-7, rtol=2e-4)


def test_history_one_still_retains_recurrent_context():
    encoder = make_encoder().eval()
    visual = torch.randn(1, 6, 1, 4, device="cuda")
    state = torch.randn(1, 6, 4, device="cuda")
    with torch.no_grad():
        full = encoder.forward_sequence(visual, state)
        singleton = encoder(visual[:, -1:], state[:, -1:])[1]
    assert not torch.equal(full[:, -1], singleton)


def test_bfloat16_readouts_are_finite_and_match_native_prefixes():
    encoder = make_encoder(1).train()
    visual = torch.randn(2, 25, 1, 4, device="cuda", requires_grad=True)
    state = torch.randn(2, 25, 4, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        intents = encoder.forward_sequence(visual, state, readout_start=4)
        for t in (4, 15, 24):
            reference = encoder(visual[:, :t + 1], state[:, :t + 1])[1]
            torch.testing.assert_close(intents[:, t], reference, atol=2e-3, rtol=2e-2)
        loss = intents[:, 4:].float().square().sum()
    loss.backward()
    assert torch.isfinite(intents).all()
    assert torch.isfinite(visual.grad).all()
    assert visual.grad[:, 0].abs().sum() > 0


def test_stream_warmup_does_not_repeat_observations_into_mamba():
    class Model:
        def __init__(self):
            self.calls = []

        def encode_step(self, frames, state, cache, produce_intent):
            self.calls.append(cache)
            new_cache = len(self.calls)
            result = (frames, state, state, new_cache)
            return result + (state.unsqueeze(1),) if produce_intent else result

    model = Model()
    stream = IntentionStream(model, 3)
    state = torch.ones(1, 4)
    first = stream.observe(state, state, True)
    stream.observe(state * 2, state * 2)
    assert model.calls == [None, 1]
    assert first["z_v_pooled_seq"].shape == (1, 3, 4)
    assert stream.cache == 2
