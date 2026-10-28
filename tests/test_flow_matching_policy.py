"""Flow math, current conditioning, and production configuration round trips."""
import pytest
import torch

from models.flow_policy_config import flow_policy_options
from models.intention_head import FlowMatchingPolicyHead
from models.task_context_policy import ContextDiffusionPolicyHead, ContextFlowMatchingPolicyHead
from tests.test_task_context_policy import FakeVision, bundle


@pytest.fixture(autouse=True)
def small_cpu_workload():
    torch.set_num_threads(2)


@pytest.mark.parametrize('solver', ['euler', 'heun'])
@pytest.mark.parametrize('steps', [1, 4, 10])
def test_linear_transport_recovers_data_and_uses_fp32_under_amp(solver, steps):
    head = FlowMatchingPolicyHead(cond_dim=4, hidden_dim=8, chunk_size=4, solver=solver)
    target = torch.linspace(-.7, .8, 28).reshape(1, 4, 7)
    initial_noise, calls = [], []

    def oracle(x, t, cond):
        assert not torch.is_autocast_enabled('cpu')
        assert x.dtype == cond.dtype == torch.float32
        if not initial_noise:
            initial_noise.append(x.clone())
        calls.append(float(t[0]))
        # Constant exact velocity of the straight line joining noise and data.
        return initial_noise[0] - target

    head.predict_velocity = oracle
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = head.sample(torch.zeros(1, 1, 4, dtype=torch.bfloat16), steps)
    torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-6)
    assert calls[0] == 1.
    assert len(calls) == steps * (2 if solver == 'heun' else 1)
    if solver == 'heun':
        assert calls[-1] == 0.


@pytest.mark.parametrize('solver,factor', [('euler', .5), ('heun', .625)])
def test_one_step_solver_on_known_ode(solver, factor):
    head = FlowMatchingPolicyHead(cond_dim=4, hidden_dim=8, chunk_size=4, solver=solver)
    head.predict_velocity = lambda x, t, c: .5 * x
    torch.manual_seed(15)
    noise = torch.randn(2, 4, 7)
    torch.manual_seed(15)
    actual = head.sample(torch.zeros(2, 1, 4), 1)
    torch.testing.assert_close(actual, noise * factor)


@pytest.mark.parametrize('sigma_min', [0., .1])
def test_loss_interpolant_velocity_repeats_mask_and_visual_gradients(sigma_min):
    head = FlowMatchingPolicyHead(cond_dim=4, hidden_dim=8, chunk_size=4,
        loss_repeats=3, sigma_min=sigma_min)
    head.unet.configure_visual_context(4, 1)
    actions = torch.randn(2, 4, 7)
    actions[1] = float('nan')
    cond = torch.randn(2, 1, 4, requires_grad=True)
    visual = torch.randn(2, 256, 4, requires_grad=True)
    weights = torch.arange(1., 8.)
    captured = {}
    prepared = []
    original = head.unet.prepare_visual_context

    def prepare(context):
        prepared.append(context.shape[0])
        return original(context)

    def predict(x, t, c, visual_kv):
        captured.update(x=x, t=t, cond=c, kv=visual_kv)
        return c[:, :, :1].expand_as(x) + sum(v.mean() for v in visual_kv)

    head.unet.prepare_visual_context = prepare
    head.predict_velocity = predict
    torch.manual_seed(30)
    times = torch.rand(3) * (1 - 1e-3) + 5e-4
    noise = torch.randn(3, 4, 7)
    torch.manual_seed(30)
    loss = head.loss(actions, cond, dim_weights=weights,
        sample_mask=torch.tensor([True, False]), visual_context=visual)
    expected_x = (1 - times[:, None, None]) * actions[:1] + (
        sigma_min + (1 - sigma_min) * times[:, None, None]) * noise
    expected_v = (1 - sigma_min) * noise - actions[:1]
    torch.testing.assert_close(captured['x'], expected_x)
    torch.testing.assert_close(captured['t'], times)
    torch.testing.assert_close(captured['cond'], cond[:1].repeat(3, 1, 1))
    assert all(v.shape[0] == 3 for v in captured['kv'])
    assert prepared == [1]
    pred = cond[:1, :, :1].expand_as(expected_v) + sum(v.mean() for v in captured['kv'])
    torch.testing.assert_close(loss, ((pred - expected_v).square() * weights).mean())
    loss.backward()
    assert torch.isfinite(cond.grad).all() and torch.isfinite(visual.grad).all()
    assert cond.grad[0].abs().sum() > 0 and visual.grad[0].abs().sum() > 0
    assert cond.grad[1].abs().sum() == visual.grad[1].abs().sum() == 0


def test_visual_kv_projected_once_for_sampling_and_terminal_clipping():
    head = FlowMatchingPolicyHead(cond_dim=4, hidden_dim=8, chunk_size=4,
        solver='heun', clip_denoised=True)
    head.unet.configure_visual_context(4, 1)
    calls, states = [], []
    original = head.unet.prepare_visual_context

    def prepare(visual):
        assert not torch.is_autocast_enabled('cpu')
        assert visual.dtype == torch.float32
        calls.append(visual)
        return original(visual)

    def velocity(x, t, c, visual_kv):
        states.append(x.clone())
        return torch.zeros_like(x)

    head.unet.prepare_visual_context = prepare
    head.predict_velocity = velocity
    torch.manual_seed(37)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = head.sample(torch.zeros(2, 1, 4), 3,
            visual_context=torch.zeros(2, 256, 4, dtype=torch.bfloat16))
    assert len(calls) == 1
    assert states[0].abs().max() > 1
    assert all(torch.equal(states[0], x) for x in states)
    expected = states[0].clamp(-1, 1)
    expected[..., 6].clamp_(0, 1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('steps', [0, -1, 1.5, True])
def test_invalid_sample_steps(steps):
    head = FlowMatchingPolicyHead(cond_dim=4, hidden_dim=8)
    with pytest.raises(ValueError, match='positive integer'):
        head.sample(torch.zeros(1, 1, 4), steps)


@pytest.mark.parametrize('option', [dict(loss_repeats=0), dict(history_size=0),
    dict(num_inference_steps=0), dict(sigma_min=-.1), dict(sigma_min=1.),
    dict(sigma_min=float('nan')), dict(solver='invalid')])
def test_invalid_constructor_options(option):
    with pytest.raises(ValueError):
        FlowMatchingPolicyHead(**option)


@pytest.mark.parametrize('tokens', [1, 2, 4, 8])
@pytest.mark.parametrize('fusion', ['additive', 'gated_residual', 'film', 'cross_attention'])
def test_flow_and_diffusion_share_ordered_token_conditioning(tokens, fusion):
    options = dict(cond_dim=6, hidden_dim=8, chunk_size=4, history_size=3,
        context_tokens=tokens, context_dim=4, context_fusion=fusion, context_fusion_dim=8)
    diffusion = ContextDiffusionPolicyHead(**options)
    flow = ContextFlowMatchingPolicyHead(**options)
    state = {k: v for k, v in diffusion.state_dict().items() if k not in ('alpha_bar', 'sigma')}
    flow.load_state_dict(state, strict=True)
    visual, states, context = torch.randn(2, 3, 4), torch.randn(2, 3, 2), torch.randn(2, tokens+1, 4)
    torch.testing.assert_close(flow(visual, states, context), diffusion(visual, states, context))
    with torch.no_grad():
        flow.unet.output_proj.weight.normal_(std=.1)
    cond = flow(visual, states, context)
    x, t = torch.randn(2, 4, 7), torch.full((2,), .4)
    flipped = flow(visual.flip(1), states.flip(1), context)
    assert not torch.allclose(flow.predict_velocity(x, t, cond), flow.predict_velocity(x, t, flipped))
    flow.loss(torch.randn_like(x), cond).backward()
    for parameter in (flow.context_task[1].weight, flow.context_execution[1].weight,
                      flow.unet.history_attention.in_proj_weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('variant', ['plain_history', 'context_history', 'context_memory', 'intent_memory'])
def test_ablation_cli_accepts_flow_current_variants(variant):
    from scripts.run_intention_ablation import parse_args
    argv = ['--output', 'unused', '--head-type', 'flow_matching', '--variants', variant]
    if variant.startswith('context_'):
        argv += ['--task-context-checkpoint', 'observer.pt', '--task-context-split-manifest', 'split.json']
    args = parse_args(argv)
    assert args.flow_loss_repeats == 4 and args.flow_clip_sample
    assert args.flow_inference_steps == 10 and args.flow_solver == 'euler'


@pytest.mark.parametrize('flag,value', [('--flow-loss-repeats', '0'),
    ('--flow-inference-steps', '0'), ('--flow-sigma-min', 'nan')])
def test_cli_rejects_invalid_flow_options(flag, value):
    from scripts.run_intention_ablation import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--output', 'unused', '--head-type', 'flow_matching', flag, value])


@pytest.mark.parametrize('legacy', [False, True])
def test_production_loader_restores_flow_options(monkeypatch, tmp_path, legacy):
    import models.align_intention as align
    from eval.eval_intention import load_intention_model
    monkeypatch.setattr(align, 'VisionEncoder', FakeVision)
    options = {} if legacy else dict(flow_inference_steps=3, flow_solver='heun',
        flow_loss_repeats=4, flow_clip_sample=True, flow_sigma_min=.05)
    config = dict(state_dim=4, mamba_output_dim=0, action_dim=7, chunk_size=4,
        history_size=1, num_cameras=1, compressed_dim=4, head_type='flow_matching',
        head_d_model=8, **options)
    model = align.ALIGNIntentionModel(**config).eval()
    model._build_head_and_bank(1024)
    config.update(pool_out_dim=1024, frozen_vision_omitted=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    path = tmp_path / 'flow.pt'
    torch.save(dict(config=config, model_state_dict=state), path)
    loaded, _ = load_intention_model(str(path), torch.device('cpu'))
    loaded.eval()
    for key, value in flow_policy_options(options).items():
        assert getattr(loaded, key) == value
    assert loaded.intention_head.solver == ('euler' if legacy else 'heun')
    assert loaded.intention_head.clip_denoised == (not legacy)
    cond = torch.randn(2, 1, 1028)
    torch.manual_seed(50)
    expected = model.intention_head.sample(cond)
    torch.manual_seed(50)
    torch.testing.assert_close(loaded.intention_head.sample(cond), expected, atol=0, rtol=0)


@pytest.mark.parametrize('memory', [False, True])
def test_main_trainer_flow_context_factory_and_real_training(monkeypatch, bundle, memory):
    import sys
    from types import SimpleNamespace
    import models.align_intention as align
    from training.train_intention import parse_args, build_model, train_v4_epoch, validate
    from scripts.run_intention_ablation import CachedEpisodes, collate_segments
    from tests.test_sparse_plain_history import Features
    argv = ['train_intention.py', '--data', 'fake.h5', '--output-dir', 'unused',
        '--head-type', 'flow_matching', '--head-d-model', '8', '--state-dim', '8',
        '--compressed-dim', '4', '--chunk-size', '4', '--history-size', '1',
        '--vision-backbone', 'dinov2_vitb14', '--task-context-checkpoint', str(bundle),
        '--task-context-split-manifest', 'split.json', '--flow-loss-repeats', '2',
        '--flow-inference-steps', '2', '--flow-solver', 'heun']
    if memory:
        argv.append('--use-memory-bank')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(align, 'VisionEncoder', FakeVision)
    parsed = parse_args()
    model = build_model(parsed, 1, torch.device('cpu'))
    model._build_head_and_bank(1024)
    assert model.intention_head.loss_repeats == 2
    assert model.intention_head.solver == 'heun'
    assert model.intention_head.clip_denoised
    assert (model.memory_module is not None) == memory
    with torch.no_grad():
        model.intention_head.unet.output_proj.weight.normal_(std=.1)
    parent = CachedEpisodes(Features(), [0], 20, 42, False, 'episode', 100, 4)
    batch = collate_segments([parent[0]])
    batch['loss_anchor_mask'][:] = False
    batch['loss_anchor_mask'][:, [0, 6]] = True
    args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7,
        head_type='flow_matching', skip_nan=False, grad_clip=1.,
        no_sample_during_train=True, head_window_batch_size=2, gripper_threshold=.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    # FiLM starts at identity with zero weights. The first step opens that
    # reader; upstream context adapters receive gradients on the next step.
    for _ in range(2):
        train_loss, _ = train_v4_epoch(model, [batch], optimizer, torch.device('cpu'), args)
    val_loss, _, _ = validate(model, [batch], torch.device('cpu'), args)
    assert torch.isfinite(torch.tensor([train_loss, val_loss])).all()
    assert all(p.grad is None for p in model.task_context.parameters())
    for p in (model.intention_head.context_task[1].weight,
              model.intention_head.context_execution[1].weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_amp_full_history_visual_context_training_and_sampling():
    head = FlowMatchingPolicyHead(cond_dim=6, hidden_dim=8, chunk_size=4,
        history_size=3, loss_repeats=4, solver='heun', clip_denoised=True)
    head.unet.configure_visual_context(4, 1)
    head.cuda()
    with torch.no_grad():
        head.unet.output_proj.weight.normal_(std=.1)
        # Expose a trained visual reader rather than its zero residual init.
        for block in head.unet.visual_context_blocks.values():
            block.output.weight.normal_(std=.01)
    cond = torch.randn(2, 3, 6, device='cuda', requires_grad=True)
    visual = torch.randn(2, 16, 4, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = head.loss(torch.randn(2, 4, 7, device='cuda'), cond, visual_context=visual)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(cond.grad).all()
    assert visual.grad.abs().sum() > 0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actions = head.sample(cond.to(torch.bfloat16), 2, visual_context=visual.to(torch.bfloat16))
    assert actions.dtype == torch.float32 and torch.isfinite(actions).all()
    assert actions.abs().max() <= 1. and actions[..., 6].min() >= 0.
