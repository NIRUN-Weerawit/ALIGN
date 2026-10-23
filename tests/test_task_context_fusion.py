"""Controlled context fusion starts from the same policy and remains trainable."""
import io

import pytest
import torch

from models.task_context_policy import ContextDiffusionPolicyHead


MODES = ContextDiffusionPolicyHead.FUSION_MODES


def make_head(mode='additive', **kwargs):
    return ContextDiffusionPolicyHead(cond_dim=12, action_dim=7, hidden_dim=16,
        time_dim=8, chunk_size=8, context_tokens=4, context_dim=8,
        context_fusion=mode, context_fusion_dim=16, **kwargs)


def inputs(history=1):
    generator = torch.Generator().manual_seed(19)
    return (torch.randn(3, history, 8, generator=generator),
            torch.randn(3, history, 4, generator=generator),
            torch.randn(3, 5, 8, generator=generator))


def activate_interaction(head):
    layer = {'gated_residual': 'context_gate', 'film': 'context_film'}.get(head.context_fusion)
    output = getattr(head, layer)[-1] if layer else head.context_attention_output
    with torch.no_grad():
        output.weight.normal_(std=.05)
        output.bias.normal_(std=.01)


def test_additive_keeps_original_parameter_names_and_computation():
    head = make_head()
    visual, state, payload = inputs()
    current = torch.cat((visual, state), -1)
    task = head.context_task_scale * head.context_task(payload[:, :4].flatten(1))[:, None]
    execution = head.context_execution_scale * head.context_execution(payload[:, -1])[:, None]
    assert torch.equal(head(visual, state, payload), current + task + execution)
    context_keys = {key for key in head.state_dict() if key.startswith('context_')}
    assert context_keys == {
        'context_task_scale', 'context_execution_scale',
        'context_task.0.weight', 'context_task.0.bias',
        'context_task.1.weight', 'context_task.1.bias',
        'context_execution.0.weight', 'context_execution.0.bias',
        'context_execution.1.weight', 'context_execution.1.bias',
    }


@pytest.mark.parametrize('mode', MODES)
def test_fusion_neutral_start_and_strict_self_contained_reload(mode):
    torch.manual_seed(3)
    legacy = make_head()
    head = make_head(mode)
    incompatible = head.load_state_dict(legacy.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith(('context_gate.', 'context_film.', 'context_query.',
        'context_key_value.', 'context_attention.', 'context_attention_output.'))
        for key in incompatible.missing_keys)
    visual, state, payload = inputs()
    assert torch.equal(head(visual, state, payload), legacy(visual, state, payload))

    # These architecture attributes are sufficient to reconstruct the optional
    # interaction modules before loading a complete checkpoint strictly.
    buffer = io.BytesIO()
    torch.save(dict(config=dict(context_fusion=head.context_fusion,
        context_fusion_dim=head.context_fusion_dim), state=head.state_dict()), buffer)
    buffer.seek(0)
    record = torch.load(buffer, weights_only=True)
    clone = make_head(record['config']['context_fusion'])
    assert clone.context_fusion_dim == record['config']['context_fusion_dim']
    clone.load_state_dict(record['state'], strict=True)
    assert torch.equal(clone(visual, state, payload), head(visual, state, payload))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('history', [1, 3])
def test_components_have_condition_shape_and_execution_remains_separate(mode, history):
    head = make_head(mode, history_size=history)
    if mode != 'additive':
        activate_interaction(head)
    visual, state, payload = inputs(history)
    current, task, execution = head.conditioning_components(visual, state, payload)
    assert current.shape == task.shape == execution.shape == (3, history, 12)
    assert torch.equal(head(visual, state, payload), current + task + execution)
    changed = payload.clone()
    changed[:, -1] += torch.arange(8.)
    current2, task2, execution2 = head.conditioning_components(visual, state, changed)
    assert torch.equal(current, current2)
    assert torch.equal(task, task2)
    assert not torch.equal(execution, execution2)


@pytest.mark.parametrize('mode', MODES[1:])
def test_interaction_uses_current_state_and_native_task_tokens(mode):
    torch.manual_seed(13)
    head = make_head(mode)
    activate_interaction(head)
    visual, state, payload = inputs()
    _, task, _ = head.conditioning_components(visual, state, payload)
    changed_state = state.clone()
    changed_state[..., 0] += 2.
    _, with_state, _ = head.conditioning_components(visual, changed_state, payload)
    assert (task - with_state).abs().max() > 1e-8
    changed_task = payload.clone()
    changed_task[:, 0, 0] += 2.
    _, with_task, _ = head.conditioning_components(visual, state, changed_task)
    assert (task - with_task).abs().max() > 1e-8


@pytest.mark.parametrize('mode', MODES[1:])
def test_zero_initialized_interaction_opens_gradient_path_after_first_update(mode):
    torch.manual_seed(11)
    head = make_head(mode)
    visual, state, payload = inputs()
    payload.requires_grad_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    probe = torch.randn(3, 1, 12)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (head(visual, state, payload) * probe).sum()
        loss.backward()
        assert torch.isfinite(payload.grad).all()
        assert head.context_task[1].weight.grad.abs().sum() > 0
        assert head.context_execution[1].weight.grad.abs().sum() > 0
        optimizer.step()
    upstream = {
        'gated_residual': ['context_gate.1.weight'],
        'film': ['context_film.1.weight'],
        'cross_attention': ['context_query.1.weight', 'context_key_value.1.weight',
                            'context_attention.in_proj_weight'],
    }[mode]
    parameters = dict(head.named_parameters())
    for name in upstream:
        assert torch.isfinite(parameters[name].grad).all()
        assert parameters[name].grad.abs().sum() > 0, name


def test_invalid_fusion_configuration_and_payload_fail_early():
    with pytest.raises(ValueError, match='Unknown'):
        make_head('unknown')
    with pytest.raises(ValueError, match='positive integer'):
        ContextDiffusionPolicyHead(context_fusion_dim=0)
    with pytest.raises(ValueError, match='divisible by four'):
        ContextDiffusionPolicyHead(context_fusion='cross_attention', context_fusion_dim=15)
    head = make_head()
    visual, state, payload = inputs()
    with pytest.raises(ValueError, match='requires task tokens'):
        head(visual, state, payload[:, :-1])
