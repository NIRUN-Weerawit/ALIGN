"""Independent summary/student tuning, optimizer coverage and saved architecture."""
import pytest
import torch
from tests.test_task_context_policy import bundle, FakeVision, inputs
from tests.test_context_aci_head import activate


@pytest.mark.parametrize('summary_trainable,student_trainable', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('kind', ['diffusion', 'flow_matching'])
@pytest.mark.parametrize('temporal', [False, True])
def test_independent_unfreeze_gradients_learning_rates_and_reload(monkeypatch, bundle, tmp_path,
                                                                summary_trainable, student_trainable, kind, temporal):
    from models.align_intention import ALIGNIntentionModel
    from data.task_context_policy_dataset import context_policy_options, policy_parameter_groups
    from eval.eval_intention import load_intention_model
    torch.set_num_threads(2)
    monkeypatch.setattr('models.align_intention.VisionEncoder', FakeVision)
    config = dict(state_dim=8, mamba_output_dim=16, num_cameras=1, compressed_dim=4,
        action_dim=7, chunk_size=4, history_size=1, head_d_model=8, head_type=kind,
        diffusion_loss_repeats=1, task_context_fusion='aci', task_context_fusion_dim=16,
        policy_temporal_mamba=temporal, policy_temporal_width=16,
        task_context_trainable=student_trainable, task_context_summary_trainable=summary_trainable)
    model = ALIGNIntentionModel(**config, task_context_checkpoint=str(bundle)).train()
    model._build_head_and_bank(1024)
    activate(model.intention_head)
    groups = policy_parameter_groups(model, .001, .0003, .0002)
    optimizer = torch.optim.AdamW(groups, lr=.001)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    for name, enabled, rate in [('summary', summary_trainable, .0002), ('encoder', student_trainable, .0003)]:
        params = list(getattr(model.task_context.observer, name).parameters())
        assert all(p.requires_grad == enabled for p in params)
        if enabled:
            group = next(g for g in optimizer.param_groups if any(p is params[0] for p in g['params']))
            assert group['lr'] == rate
    before = {name: {k:v.clone() for k,v in getattr(model.task_context.observer, name).state_dict().items()}
              for name in ('summary', 'encoder')}
    _, patches, cls, state, lengths, ticks = inputs()
    context = model.context_sequence(patches, cls, state, lengths, ticks)[:, 3]
    condition = model.action_condition(torch.randn(2, 1, 1024), torch.randn(2, 1, 8), context)
    loss = model.intention_head.loss(torch.randn(2, 4, 7), condition)
    loss.backward()
    for name, enabled in [('summary', summary_trainable), ('encoder', student_trainable)]:
        params = list(getattr(model.task_context.observer, name).parameters())
        if enabled:
            assert sum(float(p.grad.abs().sum()) for p in params if p.grad is not None) > 0
        else:
            assert all(p.grad is None for p in params)
    optimizer.step()
    for name, enabled in [('summary', summary_trainable), ('encoder', student_trainable)]:
        changed = any(not torch.equal(value, before[name][key])
                      for key, value in getattr(model.task_context.observer, name).state_dict().items())
        assert changed == enabled
    # Frozen student still propagates gradients to the summary above, while inference suppresses all graphs.
    model.eval()
    with torch.no_grad():
        inference_context = model.context_sequence(patches, cls, state, lengths, ticks)
    assert not inference_context.requires_grad
    config.update(context_policy_options(model), pool_out_dim=1024, frozen_vision_omitted=True)
    path = tmp_path / 'policy.pt'
    torch.save(dict(config=config, model_state_dict={k:v for k,v in model.state_dict().items()
               if not k.startswith('vision_encoder.')}), path)
    restored, _ = load_intention_model(str(path), torch.device('cpu'))
    assert restored.task_context.summary_trainable == summary_trainable
    assert restored.task_context.trainable == student_trainable
    with torch.no_grad():
        torch.testing.assert_close(restored.context_sequence(patches, cls, state, lengths, ticks), inference_context,
                                   rtol=0, atol=0)


def test_unfreeze_cli_and_lr_validation():
    from scripts.run_intention_ablation import parse_args
    base = ['--output', 'unused', '--variants', 'context_memory', '--task-context-checkpoint', 'observer.pt',
            '--task-context-split-manifest', 'manifest.json', '--task-context-fusion', 'aci',
            '--policy-temporal-mamba', '--unfreeze-summary-encoder', '--unfreeze-task-context-encoder',
            '--task-context-summary-lr', '0.00002', '--task-context-lr', '0.00001']
    args = parse_args(base)
    assert args.task_context_summary_trainable and args.task_context_trainable
    assert args.task_context_summary_lr == .00002 and args.task_context_lr == .00001
    with pytest.raises(SystemExit):
        parse_args(base + ['--task-context-summary-lr', 'nan'])


@pytest.mark.parametrize('kind', ['diffusion', 'flow_matching'])
def test_production_training_updates_unfrozen_modules(monkeypatch, bundle, kind):
    from types import SimpleNamespace
    from models.align_intention import ALIGNIntentionModel
    from training.train_intention import train_v4_epoch, validate
    from data.task_context_policy_dataset import policy_parameter_groups
    torch.set_num_threads(2)
    monkeypatch.setattr('models.align_intention.VisionEncoder', FakeVision)
    model = ALIGNIntentionModel(state_dim=8, mamba_output_dim=16, num_cameras=1,
        compressed_dim=4, action_dim=7, chunk_size=4, history_size=1, head_d_model=8,
        head_type=kind, diffusion_loss_repeats=1, task_context_checkpoint=str(bundle),
        task_context_fusion='aci', task_context_fusion_dim=16, policy_temporal_mamba=True,
        policy_temporal_width=16, task_context_trainable=True, task_context_summary_trainable=True,
        use_memory_bank=True, memory_bank_len=3, memory_patch_retrieval=True,
        memory_consolidation_key='temporal')
    model._build_head_and_bank(1024)
    activate(model.intention_head)
    optimizer = torch.optim.AdamW(policy_parameter_groups(model, .001, .0003, .0002))
    args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7, head_type=kind,
        skip_nan=False, grad_clip=1., no_sample_during_train=True, head_window_batch_size=2)
    batch = dict(frames_segment=torch.randn(2, 9, 257, 768), states_segment=torch.randn(2, 9, 7),
                 actions_segment=torch.rand(2, 9, 7), segment_len=torch.tensor([9, 7]),
                 observation_timesteps=torch.arange(9.).expand(2, -1))
    loss, _ = train_v4_epoch(model, [batch], optimizer, torch.device('cpu'), args)
    assert 0 < loss < float('inf')
    for module in (model.task_context.observer.summary, model.task_context.observer.encoder,
                   model.policy_temporal_encoder):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    val_loss, _, _ = validate(model, [batch], torch.device('cpu'), args)
    assert 0 < val_loss < float('inf')
