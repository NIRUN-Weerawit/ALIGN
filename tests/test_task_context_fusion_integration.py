"""Persist fusion choices through the production CLI and checkpoint loader."""
from types import SimpleNamespace

import pytest
import torch

from data.task_context_policy_dataset import context_policy_options
from models.task_context_observer import TaskContextObserver


class FakeVision(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)


@pytest.fixture
def observer_bundle(tmp_path):
    torch.set_num_threads(2)
    summary = dict(cameras=1, raw_dim=768, width=16)
    encoder = dict(width=16, tokens=4, token_dim=4, modes=4, layers=1,
        d_state=4, d_conv=3, expand=2, visual_width=768, horizons=[2, 5], execution_dim=4)
    observer = TaskContextObserver(summary, encoder)
    path = tmp_path / 'observer.pt'
    torch.save(dict(schema='align-causal-task-observer-v1', summary_config=summary,
        encoder_config=encoder, summary=observer.summary.state_dict(),
        encoder=observer.encoder.state_dict(), state_mean=observer.state_mean,
        state_std=observer.state_std,
        metadata=dict(fps=20., target_manifest_sha256='test', cameras=['image'])), path)
    return path


def _model(monkeypatch, observer_bundle, fusion):
    import models.align_intention as module
    monkeypatch.setattr(module, 'VisionEncoder', FakeVision)
    model = module.ALIGNIntentionModel(state_dim=8, mamba_output_dim=16,
        num_cameras=1, compressed_dim=4, action_dim=7, chunk_size=4,
        history_size=1, head_d_model=8, diffusion_loss_repeats=1,
        task_context_checkpoint=str(observer_bundle), task_context_fusion=fusion,
        task_context_fusion_dim=16)
    model._build_head_and_bank(1024)
    return model


@pytest.mark.parametrize('fusion', ['additive', 'gated_residual', 'film', 'cross_attention'])
def test_fusion_checkpoint_reload_is_self_contained_and_strict(monkeypatch, observer_bundle, tmp_path, fusion):
    from eval.eval_intention import load_intention_model
    model = _model(monkeypatch, observer_bundle, fusion).eval()
    cfg = dict(state_dim=8, mamba_output_dim=16, action_dim=7, chunk_size=4,
        history_size=1, num_cameras=1, compressed_dim=4, head_type='diffusion',
        head_d_model=8, diffusion_train_steps=100, diffusion_loss_repeats=1,
        pool_out_dim=1024, frozen_vision_omitted=True, **context_policy_options(model))
    assert cfg['task_context_fusion'] == fusion
    assert cfg['task_context_fusion_dim'] == 16
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    path = tmp_path / 'policy.pt'
    torch.save(dict(config=cfg, model_state_dict=state), path)
    observer_bundle.unlink()
    loaded, loaded_cfg = load_intention_model(str(path), torch.device('cpu'))
    assert loaded_cfg['task_context_fusion'] == loaded.task_context_fusion == fusion
    assert loaded.intention_head.context_fusion == fusion
    assert loaded.task_context_fusion_dim == 16
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    # Missing adapter weights must abort rather than silently initialize a new reader.
    adapter_prefix = {'additive': 'context_task.', 'gated_residual': 'context_gate.',
        'film': 'context_film.', 'cross_attention': 'context_attention.'}[fusion]
    adapter_key = next(k for k in state if k.startswith('intention_head.' + adapter_prefix))
    state.pop(adapter_key)
    torch.save(dict(config=cfg, model_state_dict=state), path)
    with pytest.raises(ValueError, match='missing trained'):
        load_intention_model(str(path), torch.device('cpu'))


def test_legacy_checkpoint_defaults_to_additive(monkeypatch, observer_bundle, tmp_path):
    from eval.eval_intention import load_intention_model
    model = _model(monkeypatch, observer_bundle, 'additive')
    cfg = dict(state_dim=8, mamba_output_dim=16, action_dim=7, chunk_size=4,
        history_size=1, num_cameras=1, compressed_dim=4, head_type='diffusion',
        head_d_model=8, diffusion_train_steps=100, diffusion_loss_repeats=1,
        pool_out_dim=1024, frozen_vision_omitted=True, **context_policy_options(model))
    cfg.pop('task_context_fusion')
    cfg.pop('task_context_fusion_dim')
    path = tmp_path / 'legacy.pt'
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    torch.save(dict(config=cfg, model_state_dict=state), path)
    loaded, _ = load_intention_model(str(path), torch.device('cpu'))
    assert loaded.task_context_fusion == 'additive'
    assert loaded.task_context_fusion_dim == 128


@pytest.mark.parametrize('fusion', ['gated_residual', 'film', 'cross_attention'])
def test_fusion_cli_and_validation(fusion):
    from scripts.run_intention_ablation import parse_args
    arguments = ['--output', 'unused', '--variants', 'context_no_memory',
        '--task-context-checkpoint', 'observer.pt',
        '--task-context-split-manifest', 'manifest.json', '--task-context-fusion', fusion,
        '--task-context-fusion-dim', '16']
    args = parse_args(arguments)
    assert args.task_context_fusion == fusion and args.task_context_fusion_dim == 16
    with pytest.raises(SystemExit):
        parse_args(['--output', 'unused', '--task-context-fusion', fusion])
    with pytest.raises(SystemExit):
        parse_args(arguments[:-1] + ['0'])
    if fusion == 'cross_attention':
        with pytest.raises(SystemExit):
            parse_args(arguments[:-1] + ['15'])


def test_model_rejects_fusion_without_context():
    from models.align_intention import ALIGNIntentionModel
    with pytest.raises(ValueError, match='requires a task context'):
        ALIGNIntentionModel(task_context_fusion='film')
    with pytest.raises(ValueError, match='positive integer'):
        ALIGNIntentionModel(task_context_fusion_dim=0)
    # Helpers used by older callers expose the same defaults.
    assert context_policy_options(SimpleNamespace())['task_context_fusion'] == 'additive'
