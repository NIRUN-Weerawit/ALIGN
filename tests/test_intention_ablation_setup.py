"""Ablation comparisons must isolate episodes and reproduce cached crops."""
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from scripts.run_intention_ablation import (CachedEpisodes, LengthBucketBatchSampler,
    collate_segments, episode_split)
from scripts.run_intention_ablation import parse_args, extend_resume_manifest


def test_smoke_validation_limit_does_not_consume_extra_batches():
    from scripts.run_intention_ablation import LimitedBatches
    visited = []
    class Loader:
        def __len__(self): return 5
        def __iter__(self):
            for index in range(5):
                visited.append(index)
                yield index
    limited = LimitedBatches(Loader(), 2)
    assert len(limited) == 2 and list(limited) == [0, 1]
    assert visited == [0, 1]
    assert len(LimitedBatches(Loader(), 0)) == 5
    assert list(LimitedBatches(Loader(), 9)) == list(range(5))
    assert parse_args(['--output', 'example']).max_val_batches == 0
    with pytest.raises(SystemExit):
        parse_args(['--output', 'example', '--max-val-batches', '-1'])


def test_deployed_snapshot_records_original_source_provenance(tmp_path):
    import json
    from scripts.run_intention_ablation import source_provenance
    provenance = {'commit': 'original', 'source_sha256': 'dirty-source-digest'}
    (tmp_path / 'source_snapshot.json').write_text(json.dumps(provenance))
    assert source_provenance(tmp_path) == provenance


@pytest.mark.parametrize("target", [80, 100])
def test_resume_manifest_allows_same_or_larger_total_epochs(target):
    original = dict(epochs=80, commit="original", lr=0.001,
                    num_workers=4, gpu_memory_fraction=0.9)
    current = dict(epochs=target, commit="resuming", lr=0.001,
                   num_workers=0, gpu_memory_fraction=1.0)
    merged = extend_resume_manifest(original, current)
    assert merged["epochs"] == target
    assert merged["commit"] == "original"
    assert merged["num_workers"] == 0
    assert merged["gpu_memory_fraction"] == 1.0
    assert original["epochs"] == 80


def test_resume_manifest_rejects_decreasing_total_epochs():
    with pytest.raises(ValueError, match="total target"):
        extend_resume_manifest(dict(epochs=80), dict(epochs=40))


def saved_run(tmp_path, options=()):
    import json
    args = parse_args(['--output', str(tmp_path), *options])
    # Manifest-only metadata must not become command-line arguments.
    manifest = dict(vars(args), commit='original', train_episodes=['ep_0'])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest, default=str))
    return args


@pytest.mark.parametrize('head', ['diffusion', 'flow_matching'])
def test_resume_restores_saved_architecture_and_runtime(tmp_path, head):
    original = saved_run(tmp_path, ['--variants', 'plain_history', '--plain-history-size', '8',
        '--head-type', head, '--vision-backbone', 'dinov2_vitb14', '--compressed-dim', '64',
        '--use-task-text', '--epochs', '200', '--num-workers', '2', '--batch-size', '8',
        '--micro-batch-size', '1', '--data', 'custom.h5', '--cache', 'custom.dinov2'])
    resumed = parse_args(['--output', str(tmp_path), '--resume'])
    for key, value in vars(original).items():
        if key != 'resume':
            assert getattr(resumed, key) == value, key
    assert resumed.resume and not hasattr(resumed, 'commit')


def test_resume_explicit_runtime_overrides_and_epoch_extension(tmp_path):
    saved_run(tmp_path, ['--epochs', '100', '--pin-memory', '--num-workers', '2',
                        '--batch-size', '8', '--micro-batch-size', '1'])
    args = parse_args(['--resume', '--output', str(tmp_path), '--epochs=200',
                      '--num-workers', '0', '--micro-batch-size', '2', '--no-pin-memory'])
    assert (args.epochs, args.num_workers, args.micro_batch_size, args.pin_memory) == (200, 0, 2, False)
    with pytest.raises(SystemExit):
        parse_args(['--resume', '--output', str(tmp_path), '--epochs', '99'])


def test_resume_restores_aci_temporal_before_validation(tmp_path):
    saved_run(tmp_path, ['--variants', 'context_no_memory', '--head-type', 'flow_matching',
        '--task-context-checkpoint', 'observer.pt', '--task-context-split-manifest', 'split.json',
        '--task-context-fusion', 'aci', '--policy-temporal-mamba'])
    args = parse_args(['--resume', '--output', str(tmp_path)])
    assert args.task_context_fusion == 'aci' and args.policy_temporal_mamba
    assert args.head_type == 'flow_matching' and args.variants == ['context_no_memory']


def test_resume_variant_directory_uses_parent_manifest(tmp_path):
    saved_run(tmp_path, ['--variants', 'plain_history', 'intent_memory', '--plain-history-size', '8'])
    variant = tmp_path / 'plain_history'
    variant.mkdir()
    # Derived per-variant history_size must not override the common history_size.
    (variant / 'config.json').write_text('{"history_size": 8}')
    args = parse_args(['--resume', '--output', str(variant)])
    assert args.output == str(tmp_path) and args.worker_variant == 'plain_history'
    assert args.history_size == 1 and args.plain_history_size == 8
    assert args.variants == ['plain_history', 'intent_memory']
    with pytest.raises(SystemExit):
        parse_args(['--resume', '--output', str(variant), '--worker-variant', 'intent_memory'])


def test_resume_does_not_restore_old_invocation_flags(tmp_path):
    import json
    saved_run(tmp_path)
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(worker_variant='intent_memory', evaluate_existing=True, output='/old/location')
    path.write_text(json.dumps(manifest))
    args = parse_args(['--resume', '--output', str(tmp_path)])
    assert args.output == str(tmp_path) and args.worker_variant is None
    assert not args.evaluate_existing


def test_resume_legacy_defaults_do_not_silently_change_model(tmp_path):
    import json
    saved_run(tmp_path)
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    for key in ('vision_backbone', 'raw_dim', 'temporal_sampling', 'memory_mode',
                'memory_detach_writes', 'memory_write_fused', 'memory_retrieval_depth',
                'diffusion_train_steps', 'diffusion_loss_repeats', 'gripper_state_source'):
        manifest.pop(key)
    path.write_text(json.dumps(manifest))
    args = parse_args(['--resume', '--output', str(tmp_path)])
    assert (args.vision_backbone, args.raw_dim) == ('dinov2_vitb14', 768)
    assert args.temporal_sampling == 'crop' and args.memory_mode == 'legacy'
    assert args.memory_write_fused and not args.memory_detach_writes
    assert args.diffusion_loss_repeats == 1 and args.diffusion_train_steps == 10
    assert args.gripper_state_source == 'legacy_zero'


@pytest.mark.parametrize('contents', [None, '{broken', '[]', '{}'])
def test_resume_missing_or_invalid_manifest_fails_before_training(tmp_path, contents):
    if contents is not None:
        (tmp_path / 'manifest.json').write_text(contents)
    with pytest.raises(SystemExit):
        parse_args(['--resume', '--output', str(tmp_path)])


def test_sparse_history_controls_are_runtime_resume_settings():
    merged = extend_resume_manifest(dict(epochs=200, plain_history_loader='dense',
        pin_memory=True, plain_head_batch_size=2), dict(epochs=200,
        plain_history_loader='sparse', pin_memory=False, plain_head_batch_size=32))
    assert merged['plain_history_loader'] == 'sparse'
    assert merged['pin_memory'] is False and merged['plain_head_batch_size'] == 32


def test_plain_history_prefetch_defaults_and_explicit_override():
    plain = parse_args(['--output', 'example', '--variants', 'plain_history'])
    assert plain.num_workers == 1 and plain.prefetch_factor == 1
    assert plain.plain_history_copy_mode == 'direct'
    explicit = parse_args(['--output', 'example', '--variants', 'plain_history', '--num-workers', '0'])
    assert explicit.num_workers == 0
    mixed = parse_args(['--output', 'example', '--variants', 'plain_history', 'intent_memory'])
    assert mixed.num_workers == 1


def test_sequential_cli_defaults_and_shared_overrides():
    defaults = parse_args(["--output", "example"])
    assert defaults.variants == ["no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory"]
    assert defaults.worker_variant is None
    assert defaults.history_size == 1 and defaults.epochs == 80 and defaults.max_steps == 0
    assert defaults.num_workers == 1 and defaults.prefetch_factor == 1
    args = parse_args(["--output", "example", "--head-type", "flow_matching", "--lr", "0.0003",
                       "--state-dim", "32", "--compressed-dim", "8", "--intent-dim", "64",
                       "--num-intent-tokens", "2", "--memory-bank-len", "16", "--chunk-size", "10",
                       "--segment-length", "32", "--gripper-loss-weight", "0.2"])
    assert args.head_type == "flow_matching" and args.lr == 0.0003
    assert args.state_dim == 32 and args.compressed_dim == 8
    assert args.num_intent_tokens == 2 and args.memory_bank_len == 16
    assert args.gripper_loss_weight == 0.2 and args.chunk_size == 10


@pytest.mark.parametrize("history_size", [1, 2, 4, 16])
@pytest.mark.parametrize("head_type", ["diffusion", "flow_matching"])
def test_plain_history_variant_has_independent_window_for_both_heads(history_size, head_type):
    args = parse_args(["--output", "example", "--variants", "plain_history", "intent_memory",
                       "--plain-history-size", str(history_size), "--head-type", head_type])
    assert args.history_size == 1 and args.plain_history_size == history_size
    assert args.variants == ["plain_history", "intent_memory"]
    assert args.head_type == head_type


@pytest.mark.parametrize("options", [["--history-size", "0"], ["--head-d-model", "10"],
    ["--compressed-dim", "3"], ["--chunk-size", "2"], ["--validation-fraction", "1"],
    ["--segment-length", "4"], ["--lr", "0"], ["--batch-size", "0"]])
def test_invalid_shared_arguments_fail_before_training(options):
    with pytest.raises(SystemExit):
        parse_args(["--output", "example", *options])


class FakeDataset:
    _episode_keys = [f"ep_{i:06d}" for i in range(12)]
    _h5 = {f"ep_{i:06d}/texts": np.array("task_a" if i < 6 else "task_b") for i in range(12)}

    def _get_episode_length(self, ep):
        return 100

    def _read_frames_dinov2(self, ep, start, length):
        return np.arange(start, start + length, dtype=np.float32)[:, None, None]

    def _read_poses(self, ep, start, length):
        return np.zeros((length, 6), np.float32)

    def _read_poses_gripper(self, ep, start, length):
        return np.arange(start - 1, start + length - 1, dtype=np.float32)

    def _read_actions(self, ep, start, length):
        return np.repeat(np.arange(start, start + length, dtype=np.float32)[:, None], 7, axis=1)


def test_episode_split_is_disjoint_stratified_and_reproducible():
    train, val = episode_split(FakeDataset(), 42, 0.2)
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(12))
    assert len(val) == 2 and any(ep < 6 for ep in val) and any(ep >= 6 for ep in val)
    assert (train, val) == episode_split(FakeDataset(), 42, 0.2)


def test_validation_crops_ignore_epoch_and_training_crops_are_reproducible():
    dataset = FakeDataset()
    val = CachedEpisodes(dataset, [0], 20, 42, False)
    first = val[0]
    val.epoch = 100
    np.testing.assert_array_equal(first["frames_segment"], val[0]["frames_segment"])
    np.testing.assert_array_equal(first["states_segment"][:, 6], first["actions_segment"][:, 6] - 1)
    left = CachedEpisodes(dataset, [0], 20, 42, True)
    right = CachedEpisodes(dataset, [0], 20, 42, True)
    for epoch in (1, 2, 3):
        left.epoch = right.epoch = epoch
        np.testing.assert_array_equal(left[0]["frames_segment"], right[0]["frames_segment"])
    batch = collate_segments([first, first])
    assert batch["frames_segment"].shape == (2, 20, 1, 1)
    assert torch.isfinite(batch["states_segment"]).all()


def test_collator_pads_once_and_preserves_variable_length_values():
    first = dict(segment_len=3, text="a",
        frames_segment=np.arange(18, dtype=np.float32).reshape(3, 2, 3),
        states_segment=np.ones((3, 7), np.float32), actions_segment=np.full((3, 7), 2, np.float32),
        loss_anchor_mask=np.array([True, False, True]), observation_timesteps=np.arange(3, dtype=np.float32))
    second = dict(segment_len=2, text="b",
        frames_segment=np.full((2, 2, 3), 9, np.float32),
        states_segment=np.full((2, 7), 3, np.float32), actions_segment=np.full((2, 7), 4, np.float32),
        loss_anchor_mask=np.array([False, True]), observation_timesteps=np.arange(2, dtype=np.float32))
    batch = collate_segments([first, second])
    np.testing.assert_array_equal(batch["frames_segment"][0, :3], first["frames_segment"])
    np.testing.assert_array_equal(batch["frames_segment"][1, :2], second["frames_segment"])
    assert torch.count_nonzero(batch["frames_segment"][1, 2]) == 0
    np.testing.assert_array_equal(batch["loss_anchor_mask"], [[True, False, True], [False, True, False]])
    assert batch["texts"] == ["a", "b"]


def test_length_bucket_sampler_is_reproducible_and_groups_lengths():
    lengths = [1, 2, 1, 2, 100, 101, 102, 103, 104]
    sampler = LengthBucketBatchSampler(lengths, batch_size=2, seed=13,
                                      bucket_multiplier=2, drop_last=True)
    sampler.set_epoch(4)
    batches = list(sampler)
    assert batches == list(sampler)
    assert len(batches) == 4
    selected = sorted(index for batch in batches for index in batch)
    assert len(selected) == 8 and len(set(selected)) == 8
    assert 8 in selected  # The longest episode is not systematically dropped.
    even_sampler = LengthBucketBatchSampler(lengths[:8], batch_size=2, seed=13,
                                            bucket_multiplier=2, drop_last=True)
    even_batches = list(even_sampler)
    assert all(max(lengths[i] for i in batch) - min(lengths[i] for i in batch) <= 3
               for batch in even_batches)


def test_microbatches_preserve_logical_episode_groups():
    lengths = list(range(30))
    full = LengthBucketBatchSampler(lengths, batch_size=10, seed=13)
    micro = LengthBucketBatchSampler(lengths, batch_size=10, seed=13,
                                     micro_batch_size=2)
    full.set_epoch(4)
    micro.set_epoch(4)
    groups = list(full)
    pieces = list(micro)
    assert len(pieces) == len(micro) == len(groups) * 5
    assert [sum(pieces[i:i + 5], []) for i in range(0, len(pieces), 5)] == groups


def test_persistent_workers_observe_training_epoch_updates():
    train = CachedEpisodes(FakeDataset(), [0], 20, 42, True)
    loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=1,
                        persistent_workers=True, multiprocessing_context="spawn",
                        collate_fn=collate_segments)
    train.epoch = 1
    first = next(iter(loader))["frames_segment"].clone()
    train.epoch = 2
    second = next(iter(loader))["frames_segment"].clone()
    assert not torch.equal(first, second)
    loader._iterator._shutdown_workers()


def test_binary_gripper_validation_uses_explicit_cutoff(monkeypatch):
    from types import SimpleNamespace
    import models.align_intention as module
    from models.align_intention import ALIGNIntentionModel
    from tests.test_intention_training_contracts import FakeVision
    from training.train_intention import validate

    monkeypatch.setattr(module, "VisionEncoder", FakeVision)
    model = ALIGNIntentionModel(state_dim=4, compressed_dim=4, mamba_output_dim=0,
        head_type="flow_matching", head_d_model=8, chunk_size=2)
    model._build_head_and_bank(8)
    model.intention_head.sample = lambda cond, num_steps=None: cond.new_full((cond.shape[0], 2, 7), 0.25)
    model.intention_head.loss = lambda target, cond, **kwargs: cond.new_tensor(0.)
    batch = dict(frames_segment=np.zeros((1, 3, 1, 4, 4, 3), np.uint8),
        states_segment=np.zeros((1, 3, 7), np.float32),
        actions_segment=np.ones((1, 3, 7), np.float32), segment_len=np.array([3]))
    args = SimpleNamespace(history_size=1, chunk_size=2, action_dim=7,
        head_type="flow_matching", skip_nan=False, gripper_threshold=0.5)
    _, _, metrics = validate(model, [batch], torch.device("cpu"), args)
    assert metrics["grip_acc"] == 0.0  # 0.25 predicts open for a binary 0/1 target.
    args.gripper_threshold = 0.0
    _, _, legacy_metrics = validate(model, [batch], torch.device("cpu"), args)
    assert legacy_metrics["grip_acc"] == 1.0


def test_probe_report_describes_actual_checkpoint_and_anchors():
    from scripts.probe_condition_dependence import render_comparison
    report={'protocol':{'held_out_episodes':44,'episode_anchors':True,
                       'checkpoint_selection':'final epoch 4, equal budgets'},
            'variants':{'memory':{'epoch':4,'results':{}}}}
    text=render_comparison(report)
    assert 'final epoch 4, equal budgets' in text
    assert 'beginning/middle/last valid common prefix' in text
    assert 'memory=4' in text
    assert 'predate' not in text and 't=6' not in text


def test_gripper_balanced_accuracy_exposes_majority_only_predictions():
    from scripts.probe_condition_dependence import gripper_class_metrics
    rates=gripper_class_metrics(dict(tp=90,tn=0,fp=10,fn=0))
    assert rates['target_label_1_fraction']==.9
    assert rates['label_1_recall']==1 and rates['label_0_recall']==0
    assert rates['balanced_accuracy']==.5
    single_class=gripper_class_metrics(dict(tp=10,tn=0,fp=0,fn=0))
    assert single_class['label_0_recall'] is None and single_class['balanced_accuracy'] is None
