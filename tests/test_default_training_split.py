"""Ablation and context-token training share Piper's Goal episode split."""
import json
from types import SimpleNamespace

import pytest
from data.training_split import (canonical_split, piper_episode_indices, piper_partition,
    validate_goal_training_manifest, resolve_context_manifest)


class GoalWindows:
    def __init__(self):
        value = canonical_split()
        self._episode_keys = sorted(value['episode_lengths'])
        self.lengths = value['episode_lengths']
        self.h5_path = 'libero_goal.h5'

    def _get_episode_length(self, i):
        return self.lengths[self._episode_keys[i]]

def test_ablation_episodes_use_the_same_fixed_partition():
    from scripts.run_intention_ablation import episode_split
    value = canonical_split()
    dataset = GoalWindows()
    expected = [set(value['train_episodes']), set(value['val_episodes'])]
    for seed, fraction in ((42, .1), (7, .25)):
        episodes = episode_split(dataset, seed, fraction)
        for indices, keys in zip(episodes, expected):
            assert {dataset._episode_keys[i] for i in indices} == keys


def test_declared_goal_mismatches_fail_instead_of_silently_resplitting():
    dataset = GoalWindows()
    dataset.lengths = dict(dataset.lengths)
    dataset.lengths[dataset._episode_keys[0]] += 1
    with pytest.raises(ValueError, match='lengths differ'):
        piper_episode_indices(dataset)
    with pytest.raises(ValueError, match='episodes differ'):
        piper_partition(dataset._episode_keys[:-1], source_name='libero_goal.h5')


def test_legacy_goal_training_caches_require_an_explicit_override():
    value=canonical_split()
    manifest=dict(split=dict(train=value['train_episodes'], development=[], confirmation=value['val_episodes']))
    validate_goal_training_manifest(manifest)
    manifest['split']['train']=value['train_episodes'][:-44]
    manifest['split']['development']=value['train_episodes'][-44:]
    with pytest.raises(ValueError, match='legacy LIBERO Goal split'):
        validate_goal_training_manifest(manifest)


def test_default_context_manifest_resolves_without_overwriting_explicit_choice(tmp_path, monkeypatch):
    import data.training_split as split
    path=tmp_path/'manifest.json'
    path.write_text('{}')
    monkeypatch.setattr(split,'PIPER_TARGETS',path)
    args=SimpleNamespace(task_context_checkpoint='observer.pt',task_context_split_manifest=None)
    resolve_context_manifest(args)
    assert args.task_context_split_manifest==str(path)
    args.task_context_split_manifest='explicit.json'
    resolve_context_manifest(args)
    assert args.task_context_split_manifest=='explicit.json'


def test_piper_context_manifest_supplies_nonempty_heldout_loader_and_evaluation(tmp_path):
    from data.task_context_policy_dataset import policy_episode_split
    from data.task_context_dataset import SCHEMA, json_hash
    value=canonical_split()
    keys=sorted(value['episode_lengths'])
    manifest=dict(schema=SCHEMA,split=dict(train=value['train_episodes'],development=[],confirmation=value['val_episodes']),
        cameras=value['cameras'],fps=20.,episodes={k:dict(length=value['episode_lengths'][k]) for k in keys})
    manifest['manifest_sha256']=json_hash(manifest)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    dataset=SimpleNamespace(_episode_keys=keys,cameras=value['cameras'],
        _get_episode_length=lambda i:value['episode_lengths'][keys[i]])
    train,val,record=policy_episode_split(dataset,path,dict(target_manifest_sha256=manifest['manifest_sha256']))
    assert len(train)==384 and len(val)==44 and not set(train)&set(val)
    assert record['evaluation_partition']=='confirmation'
    assert {keys[i] for i in val}==set(value['val_episodes'])
