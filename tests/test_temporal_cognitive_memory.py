"""Temporal merge selection, age preservation, policy routing and training parity."""
import pytest
import torch
from models.memory_bank import EpisodicMemoryModule
from tests.test_task_context_policy import bundle, FakeVision, inputs


@pytest.mark.parametrize('mask_fields', [False, True])
@pytest.mark.parametrize('timestamp_policy', ['latest', 'mean'])
def test_cognitive_selection_merges_all_fields_and_retains_time(mask_fields, timestamp_policy):
    bank = EpisodicMemoryModule(4, 4, 4, bank_len=3, consolidation_key='cognitive',
                                mask_missing_fields=mask_fields, merge_timestamp=timestamp_policy)
    bank.reset(2, torch.device('cpu'))
    perceptual = torch.tensor([[1., 0, 0, 0], [2., 0, 0, 0], [0., 1, 0, 0], [1., 1, 0, 0]])
    cognitive = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0], [0., 2, 0, 0], [1., 1, 0, 0]])
    for step in range(4):
        bank.observe_only(perceptual[step].expand(2, -1), perceptual[step].expand(2, -1),
                          cognitive[step].expand(2, 1, -1),
                          observed_mask=torch.tensor([True, step < 2]),
                          timestamp=torch.full((2,), float(step * 10)))
    # Visual pair would be 0/1; cognitive pair is 1/2.
    torch.testing.assert_close(bank.perceptual_bank[0, 1], (perceptual[1] + perceptual[2]) / 2)
    torch.testing.assert_close(bank.cognitive_bank[0, 1], (cognitive[1] + cognitive[2]) / 2)
    assert bank.timestamps[0].tolist() == [0, 20 if timestamp_policy == 'latest' else 15, 30]
    assert bank.timestamps[1].tolist() == [0, 10, -1]
    if mask_fields:
        assert bank.cognitive_times[0, 1] == (20 if timestamp_policy == 'latest' else 15)
    assert bank._count.tolist() == [3, 2]


def make_temporal_model(monkeypatch, bundle, kind='diffusion'):
    from models.align_intention import ALIGNIntentionModel
    monkeypatch.setattr('models.align_intention.VisionEncoder', FakeVision)
    model = ALIGNIntentionModel(state_dim=8, mamba_output_dim=16, num_cameras=1,
        compressed_dim=4, action_dim=7, chunk_size=4, history_size=1, head_d_model=8,
        head_type=kind, diffusion_loss_repeats=1, task_context_checkpoint=str(bundle),
        task_context_fusion='aci', task_context_fusion_dim=16,
        policy_temporal_mamba=True, policy_temporal_width=16,
        use_memory_bank=True, memory_bank_len=3, memory_patch_retrieval=True,
        memory_consolidation_key='temporal')
    model._build_head_and_bank(1024)
    return model


def test_temporal_payload_storage_and_checkpoint_reload(monkeypatch, bundle, tmp_path):
    from data.task_context_policy_dataset import context_policy_options
    from eval.eval_intention import load_intention_model
    model = make_temporal_model(monkeypatch, bundle)
    _, patches, cls, state, lengths, ticks = inputs()
    context = model.context_sequence(patches, cls, state, lengths, ticks)
    bank = model.memory_module
    bank.reset(2, torch.device('cpu'))
    assert bank.cognitive_dim == 4
    for step in range(5):
        payload = context[:, step]
        visual = torch.randn(2, 1, 1024)
        encoded = torch.randn(2, 1, 8)
        _, _, fused = model.condition_actions(visual, encoded, payload)
        torch.testing.assert_close(bank.cognitive_bank[:, -1 if step >= 2 else step], payload[:, -1])
        torch.testing.assert_close(fused[:, :-1], payload[:, :-1], rtol=0, atol=0)
    cfg = dict(state_dim=8, mamba_output_dim=16, num_cameras=1, compressed_dim=4,
               action_dim=7, chunk_size=4, history_size=1, head_d_model=8, head_type='diffusion',
               use_memory_bank=True, memory_bank_len=3, memory_patch_retrieval=True,
               memory_mode='episodic', memory_detach_writes=True, memory_write_fused=False,
               pool_out_dim=1024, frozen_vision_omitted=True, **context_policy_options(model))
    path = tmp_path / 'temporal.pt'
    torch.save(dict(config=cfg, model_state_dict={k:v for k,v in model.state_dict().items()
               if not k.startswith('vision_encoder.')}), path)
    restored, _ = load_intention_model(str(path), torch.device('cpu'))
    assert restored.memory_module.consolidation_key == 'cognitive'
    assert restored.memory_module.merge_timestamp == 'latest'
    assert restored.memory_module.cognitive_dim == 4


@pytest.mark.parametrize('kind', ['diffusion', 'flow_matching'])
def test_train_validation_and_stream_raw_write(monkeypatch, bundle, kind):
    from types import SimpleNamespace
    from models.intention_stream import IntentionStream
    from training.train_intention import train_v4_epoch, validate
    model = make_temporal_model(monkeypatch, bundle, kind)
    from tests.test_context_aci_head import activate
    activate(model.intention_head)
    model.memory_module.reset(2, torch.device('cpu'))
    stream = IntentionStream(model, 1)
    output = stream.observe(torch.randn(2, 257, 768), torch.randn(2, 7), store_memory=True)
    torch.testing.assert_close(model.memory_module.cognitive_bank[:, 0], output['intent_emb'][:, -1])
    args = SimpleNamespace(history_size=1, chunk_size=4, action_dim=7, head_type=kind,
        skip_nan=False, grad_clip=1., no_sample_during_train=True, head_window_batch_size=2)
    batch = dict(frames_segment=torch.randn(2, 9, 257, 768), states_segment=torch.randn(2, 9, 7),
                 actions_segment=torch.rand(2, 9, 7), segment_len=torch.tensor([9, 7]),
                 observation_timesteps=torch.arange(9.).expand(2, -1),
                 loss_anchor_mask=torch.tensor([[False, True, False, True, False, True, False, False, False]] * 2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    loss, _ = train_v4_epoch(model, [batch], optimizer, torch.device('cpu'), args)
    assert 0 < loss < float('inf')
    assert model.policy_temporal_encoder.blocks[0].in_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.task_context.parameters())
    val_loss, _, _ = validate(model, [batch], torch.device('cpu'), args)
    assert 0 < val_loss < float('inf')


def test_cli_temporal_consolidation_validation():
    from scripts.run_intention_ablation import parse_args
    base = ['--output', 'unused', '--variants', 'context_memory', '--task-context-checkpoint', 'observer.pt',
            '--task-context-split-manifest', 'manifest.json', '--task-context-fusion', 'aci',
            '--policy-temporal-mamba', '--memory-consolidation-key', 'temporal']
    assert parse_args(base).memory_merge_timestamp == 'latest'
    with pytest.raises(SystemExit):
        parse_args(base + ['--no-memory-temporal-cognitive'])


@pytest.mark.parametrize('variants', [['context_memory', 'context_no_memory'], ['context_no_memory']])
def test_temporal_consolidation_accepts_shared_ablation_options(variants):
    from scripts.run_intention_ablation import parse_args
    args = parse_args(['--output', 'unused', '--variants', *variants,
        '--task-context-checkpoint', 'observer.pt', '--task-context-split-manifest', 'manifest.json',
        '--task-context-fusion', 'aci', '--policy-temporal-mamba', '--memory-consolidation-key', 'temporal'])
    assert args.variants == variants and args.memory_consolidation_key == 'temporal'


def test_no_memory_model_normalizes_unused_temporal_key(monkeypatch, bundle):
    from models.align_intention import ALIGNIntentionModel
    from data.task_context_policy_dataset import context_policy_options
    monkeypatch.setattr('models.align_intention.VisionEncoder', FakeVision)
    model = ALIGNIntentionModel(state_dim=8, mamba_output_dim=16, num_cameras=1,
        compressed_dim=4, action_dim=7, chunk_size=4, history_size=1, head_d_model=8,
        task_context_checkpoint=str(bundle), task_context_fusion='aci', task_context_fusion_dim=16,
        policy_temporal_mamba=True, policy_temporal_width=16, use_memory_bank=False,
        memory_consolidation_key='temporal')
    model._build_head_and_bank(1024)
    assert model.memory_module is None
    assert context_policy_options(model)['memory_consolidation_key'] == 'perceptual'
    _, patches, cls, state, lengths, ticks = inputs()
    payload = model.context_sequence(patches, cls, state, lengths, ticks)[:, 0]
    visual, encoded = torch.randn(2, 1, 1024), torch.randn(2, 1, 8)
    output = model.condition_actions(visual, encoded, payload)
    assert output[0] is visual and output[1] is encoded and output[2] is payload
