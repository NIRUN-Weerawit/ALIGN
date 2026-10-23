"""Native context tokens plus an ordered action-head observation window."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

import models.align_intention as align
from data.task_context_policy_dataset import context_policy_options
from models.intention_stream import IntentionStream
from scripts.run_intention_ablation import CachedEpisodes, collate_segments, parse_args
from tests.test_sparse_plain_history import Features
from tests.test_task_context_policy import FakeVision, bundle
from training.train_intention import train_v4_epoch, validate


def make_model(monkeypatch, bundle, history=4, mode='full'):
    monkeypatch.setattr(align, 'VisionEncoder', FakeVision)
    model = align.ALIGNIntentionModel(state_dim=8, compressed_dim=4,
        mamba_output_dim=0, use_intent_tokens=False, use_memory_bank=False,
        history_size=history, head_history_mode=mode, head_type='diffusion',
        head_d_model=8, chunk_size=4, action_dim=7, diffusion_loss_repeats=1,
        task_context_checkpoint=str(bundle))
    model._build_head_and_bank(1024)
    return model


@pytest.mark.parametrize('history', [1, 4, 16])
def test_context_history_cli_and_worker(history):
    args = parse_args(['--output', 'unused', '--variants', 'context_no_memory',
        'context_history', '--worker-variant', 'context_history',
        '--plain-history-size', str(history),
        '--task-context-checkpoint', 'observer.pt',
        '--task-context-split-manifest', 'manifest.json'])
    assert args.worker_variant == 'context_history'
    assert args.history_size == 1 and args.plain_history_size == history
    with pytest.raises(SystemExit):
        parse_args(['--output', 'unused', '--variants', 'context_history'])


def test_context_history_preserves_frame_order_and_context(monkeypatch, bundle):
    model = make_model(monkeypatch, bundle).eval()
    assert model.intention_encoder is None and model.memory_module is None
    assert all(not p.requires_grad for p in model.task_context.parameters())
    visual, states, context = torch.randn(2, 4, 1024), torch.randn(2, 4, 8), torch.randn(2, 5, 4)
    cond = model.action_condition(visual, states, context)
    assert cond.shape == (2, 4, 1032)
    head = model.intention_head
    live = torch.cat((visual, states), dim=-1)
    task = head.context_task_scale * head.context_task(context[:, :4].flatten(1))[:, None]
    execution = head.context_execution_scale * head.context_execution(context[:, -1])[:, None]
    torch.testing.assert_close(cond, live + task + execution)
    with torch.no_grad():
        # The decoder's final projection starts at zero. Expose conditioning
        # dependence rather than mistaking that initialization for invariance.
        model.intention_head.unet.output_proj.weight.normal_(std=.1)
    noisy, timestep = torch.randn(2, 4, 7), torch.full((2,), 5, dtype=torch.long)
    predict = model.intention_head.predict_noise
    expected = predict(noisy, timestep, cond)
    reordered = model.action_condition(visual[:, [2, 1, 0, 3]], states[:, [2, 1, 0, 3]], context)
    assert not torch.allclose(expected, predict(noisy, timestep, reordered))
    changed = model.action_condition(visual, states, torch.randn_like(context))
    assert not torch.allclose(expected, predict(noisy, timestep, changed))
    model.intention_head.loss(torch.randn(2, 4, 7), cond).backward()
    attention = model.intention_head.unet.history_attention.in_proj_weight
    assert attention.grad is not None and attention.grad.abs().sum() > 0
    assert model.intention_head.context_task[1].weight.grad.abs().sum() > 0
    assert model.intention_head.context_execution[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('history', [1, 4, 20])
def test_dense_training_and_validation_match_causal_stream(monkeypatch, bundle, history):
    model = make_model(monkeypatch, bundle, history=history)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    parent = CachedEpisodes(Features(), [0], 20, 42, False, 'episode', 100, 4)
    batch = collate_segments([parent[0]])
    # Select only two anchors. The encoder must still scan every observation
    # between them; the direct head window must include unsupervised frames.
    batch['loss_anchor_mask'][:] = False
    batch['loss_anchor_mask'][:, [0, 6]] = True
    expected = []
    model.eval()
    stream = IntentionStream(model, history)
    for t in range(7):
        output = stream.observe(torch.as_tensor(batch['frames_segment'][:, t]),
            torch.as_tensor(batch['states_segment'][:, t]))
        if t in (0, 6):
            expected.append(tuple(output[k].clone() for k in
                ('z_v_pooled_seq', 'z_s_seq', 'intent_emb')))
    assert stream.cache.recurrence.steps == 7
    calls = []
    def capture(module, arguments):
        calls.append(tuple(value.detach().clone() for value in arguments))
    handle = model.intention_head.register_forward_pre_hook(capture)
    # Deterministic differentiable objective isolates train/validation window
    # construction from diffusion noise. Other tests exercise the real decoder.
    model.intention_head.loss = lambda target, cond, **kwargs: cond.square().mean()
    model.intention_head.sample = lambda cond: cond.new_zeros((len(cond), 4, 7))
    args = SimpleNamespace(history_size=history, chunk_size=4, action_dim=7,
        head_type='diffusion', skip_nan=False, grad_clip=1.,
        no_sample_during_train=True, head_window_batch_size=2, gripper_threshold=.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    train_loss, _ = train_v4_epoch(model, [batch], optimizer, torch.device('cpu'), args)
    val_loss, _, _ = validate(model, [batch], torch.device('cpu'), args)
    handle.remove()
    assert np.isfinite(train_loss) and val_loss == pytest.approx(train_loss, rel=1e-6)
    assert len(calls) == 4
    for actual, reference in zip(calls, expected * 2):
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    assert all(p.grad is None for p in model.task_context.parameters())
    assert model.intention_head.context_task[1].weight.grad.abs().sum() > 0


def test_context_history_production_checkpoint_reload(monkeypatch, bundle, tmp_path):
    from eval.eval_intention import load_intention_model
    model = make_model(monkeypatch, bundle).eval()
    config = dict(state_dim=8, compressed_dim=4, mamba_output_dim=0,
        action_dim=7, chunk_size=4, history_size=4, head_history_mode='full',
        head_type='diffusion', head_d_model=8, num_cameras=1,
        diffusion_train_steps=100, diffusion_loss_repeats=1, pool_out_dim=1024,
        frozen_vision_omitted=True, **context_policy_options(model))
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    path = tmp_path / 'policy.pt'
    torch.save(dict(config=config, model_state_dict=state), path)
    bundle.unlink()
    loaded, _ = load_intention_model(str(path), torch.device('cpu'))
    loaded.eval()
    assert loaded.history_size == loaded.intention_head.history_size == 4
    assert loaded.head_history_mode == 'full'
    visual, states, context = torch.randn(1, 4, 1024), torch.randn(1, 4, 8), torch.randn(1, 5, 4)
    torch.testing.assert_close(loaded.action_condition(visual, states, context),
        model.action_condition(visual, states, context), atol=0, rtol=0)
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    state.pop('intention_head.unet.history_attention.in_proj_weight')
    torch.save(dict(config=config, model_state_dict=state), path)
    with pytest.raises(ValueError, match='missing trained'):
        load_intention_model(str(path), torch.device('cpu'))


@pytest.mark.parametrize('history,mode', [(4, 'full'), (1, 'mean')])
def test_ablation_reevaluation_keeps_context_on_dense_episode_path(monkeypatch, bundle, tmp_path, history, mode):
    import scripts.run_intention_ablation as script
    model = make_model(monkeypatch, bundle, history=history, mode=mode)
    config = dict(state_dim=8, compressed_dim=4, mamba_output_dim=0,
        chunk_size=4, history_size=history, head_history_mode=mode,
        head_type='diffusion', head_d_model=8, intent_dim=128,
        use_intent_tokens=False, use_memory_bank=False, memory_bank_len=16,
        pool_out_dim=1024, **context_policy_options(model))
    run = tmp_path / 'context_history'
    run.mkdir()
    torch.save(dict(config=config, model_state_dict=model.state_dict(), epoch=1),
        run / 'intention_best.pt')
    manifest = dict(data='fake.h5', cache='fake.cache', cameras=['image'],
        segment_length=20, seed=42, temporal_sampling='episode',
        chunk_size=4, batch_size=1, history_size=1, head_type='diffusion',
        val_episodes=['ep_0'], epochs=1, max_steps=1)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'summary.json').write_text(json.dumps({'context_history': {'best': {'epoch': 1}}}))
    dataset = Features()
    dataset._episode_keys = ['ep_0', 'ep_1']
    dataset.close = lambda: None
    monkeypatch.setattr(script, 'ALIGNDataset', lambda *a, **kw: dataset)
    # Only the script's device selection is redirected; the real model and
    # checkpoint reconstruction are exercised on CPU without touching CUDA.
    cuda = SimpleNamespace(set_per_process_memory_fraction=lambda *a: None,
        empty_cache=lambda: None, manual_seed_all=lambda *a: None)
    monkeypatch.setattr(script, 'torch', SimpleNamespace(**{
        **vars(torch), 'device': lambda *a: torch.device('cpu'), 'cuda': cuda}))
    calls = []
    def dense_validate(loaded, loader, device, args):
        assert loaded.task_context is not None
        assert loaded.history_size == args.history_size == history
        batch = next(iter(loader))
        assert batch['frames_segment'].shape[1] == 17
        calls.append(True)
        return 1., 0., dict(pos_mse=0., rot_mse=0., grip_mse=0., grip_acc=1.)
    monkeypatch.setattr(script, 'validate', dense_validate)
    def reject_sparse(*a, **kw):
        pytest.fail('Sparse histories discard the context encoder observation stream')
    monkeypatch.setattr(script, 'SparseHistoryEpisodes', reject_sparse)
    script.evaluate_existing(tmp_path)
    assert calls == [True]
    assert (tmp_path / 'EVALUATION_COMPLETE').exists()
