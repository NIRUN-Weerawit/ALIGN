"""Refit partitions and fixed-budget training must not consult held-out metrics."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.task_context_dataset import SCHEMA, file_hash, json_hash
from tools.refit_task_context import prepare_targets, read_policy_split


@pytest.mark.parametrize('match_policy', [False, True])
def test_refit_preserves_heldout_and_refits_normalization_only_on_384(tmp_path, match_policy):
    source = tmp_path / 'source'
    source.mkdir()
    keys = [f'ep_{i:06d}' for i in range(428)]
    split = dict(train=keys[:340], development=keys[340:384], confirmation=keys[384:])
    for i, key in enumerate(keys):
        value = float(i) if i < 384 else 1e8
        np.save(source / f'{key}.npy', np.zeros((2, 1, 5, 4), np.float32))
        np.savez(source / f'{key}.npz', states=np.full((2, 10), value, np.float32),
            terminal_delta=np.full((1, 5, 4), value, np.float32),
            procedure_delta=np.full((3, 1, 5, 4), value, np.float32))
    np.savez(source / 'normalization.npz', state_mean=np.zeros(10, np.float32))
    manifest = dict(schema=SCHEMA, split=split, episodes={k: {} for k in keys},
        files={}, normalization_sha256=file_hash(source / 'normalization.npz'))
    manifest['manifest_sha256'] = json_hash(manifest)
    (source / 'manifest.json').write_text(json.dumps(manifest))
    original_hash = file_hash(source / 'manifest.json')
    destination = tmp_path / 'refit'
    policy = None
    expected_train, expected_heldout = keys[:384], keys[384:]
    if match_policy:
        expected_train, expected_heldout = keys[44:], keys[:44]
        policy = tmp_path / 'policy.json'
        policy.write_text(json.dumps(dict(train_episodes=expected_train, val_episodes=expected_heldout)))
    result = prepare_targets(source, destination, policy)
    assert len(result['split']['train']) == 384 and result['split']['development'] == []
    assert result['split']['train'] == expected_train
    assert result['split']['confirmation'] == expected_heldout
    assert not set(result['split']['train']) & set(expected_heldout)
    norm = dict(np.load(destination / 'normalization.npz'))
    expected_states = np.concatenate([np.load(source / f'{k}.npz')['states'] for k in expected_train])
    expected_terminals = np.stack([np.load(source / f'{k}.npz')['terminal_delta'] for k in expected_train])
    np.testing.assert_array_equal(norm['state_mean'], expected_states.mean(0))
    np.testing.assert_array_equal(norm['terminal_mean'], expected_terminals.mean(0))
    if match_policy:
        assert result['refit']['sha256'] == file_hash(policy)
    assert file_hash(source / 'manifest.json') == original_hash
    assert (destination / f'{keys[0]}.npy').is_symlink()
    assert result['manifest_sha256'] == json_hash({k:v for k,v in result.items() if k != 'manifest_sha256'})


@pytest.mark.parametrize('problem', ['overlap', 'duplicate', 'missing_key', 'count', 'camera'])
def test_policy_split_rejects_invalid_membership_and_cameras(tmp_path, problem):
    keys = [f'ep_{i:06d}' for i in range(428)]
    reference = dict(train_episodes=keys[:384], val_episodes=keys[384:], cameras=['image'])
    if problem == 'overlap':
        reference['val_episodes'][0] = keys[0]
    elif problem == 'duplicate':
        reference['train_episodes'][0] = keys[1]
    elif problem == 'missing_key':
        reference['val_episodes'][0] = 'ep_unknown'
    elif problem == 'count':
        reference['train_episodes'].pop()
    else:
        reference['cameras'] = ['wrist_image']
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        read_policy_split(path, keys, ['image'])


def reject_evaluation(*args, **kwargs):
    raise AssertionError('Final refit must not evaluate or select on a held-out partition')


def test_teacher_final_refit_never_loads_validation(monkeypatch, tmp_path):
    import training.train_task_context_teacher as trainer
    from tests.test_task_context_teacher import toy
    model, batch = toy()
    batch['keys'] = ['a', 'b', 'c']
    np.savez(tmp_path / 'a.npz', actions=np.ones((4, 7), np.float32))
    dataset = SimpleNamespace(manifest=dict(cameras=['image'], action_horizon=4,
        procedure_points=3, manifest_sha256='test', split=dict(development=[])), root=tmp_path, keys=['a'])
    splits = []
    def load_data(root, split, **kw):
        assert split == 'train'
        splits.append(split)
        return dataset
    monkeypatch.setattr(trainer, 'TaskContextEpisodes', load_data)
    monkeypatch.setattr(trainer, 'make_loader', lambda *a, **kw: [batch])
    monkeypatch.setattr(trainer, 'TaskContextTeacher', lambda **kw: model)
    monkeypatch.setattr(trainer, 'evaluate', reject_evaluation)
    output = tmp_path / 'teacher'
    monkeypatch.setattr(sys, 'argv', ['train', '--targets', str(tmp_path), '--output', str(output),
        '--device', 'cpu', '--epochs', '1', '--final-refit'])
    trainer.main()
    checkpoint = torch.load(output / 'teacher_final.pt', weights_only=True)
    assert checkpoint['checkpoint_selection'] == 'fixed_budget_final'
    assert checkpoint['steps'] == checkpoint['epoch'] == 1
    assert not (output / 'teacher_best.pt').exists()
    assert splits == ['train']


def test_student_final_refit_never_loads_validation(monkeypatch, tmp_path):
    import training.train_task_context_encoder as trainer
    from tests.test_task_context_encoder import toy
    from tests.test_task_context_teacher import toy as teacher_toy
    model, summary, lengths, anchors = toy()
    teacher, _ = teacher_toy()
    teacher_path = tmp_path / 'teacher.pt'
    teacher_path.write_bytes(b'frozen-teacher-hash')
    batch = dict(keys=['a', 'b'], summary=summary, lengths=lengths, anchors=anchors,
        cls=torch.randn(2, 9, 8), states=torch.randn(2, 9, 10), task_tokens=torch.randn(2, 4, 4))
    dataset = SimpleNamespace(manifest=dict(split=dict(development=[]), teacher_sha256=file_hash(teacher_path),
        width=16, shape=[4, 4], visual_width=8, manifest_sha256='test'),
        norm=dict(latent_mean=np.zeros((4, 4), np.float32), latent_std=np.ones((4, 4), np.float32)))
    splits = []
    def load_data(root, split, **kw):
        assert split == 'train'
        splits.append(split)
        return dataset
    monkeypatch.setattr(trainer, 'TaskContextPrefixes', load_data)
    monkeypatch.setattr(trainer, 'make_loader', lambda *a, **kw: [batch])
    monkeypatch.setattr(trainer, 'load_teacher', lambda *a, **kw: (teacher, {}))
    monkeypatch.setattr(trainer, 'TaskContextEncoder', lambda **kw: model)
    monkeypatch.setattr(trainer, 'evaluate', reject_evaluation)
    output = tmp_path / 'student'
    monkeypatch.setattr(sys, 'argv', ['train', '--cache', str(tmp_path), '--teacher', str(teacher_path),
        '--output', str(output), '--device', 'cpu', '--epochs', '1', '--outcome-weight', '0', '--final-refit'])
    trainer.main()
    checkpoint = torch.load(output / 'student_final.pt', weights_only=True)
    assert checkpoint['checkpoint_selection'] == 'fixed_budget_final'
    assert checkpoint['steps'] == checkpoint['epoch'] == 1
    assert not (output / 'student_best.pt').exists()
    assert splits == ['train']
