"""Episode-keyed privileged teacher conditioning for context-policy training."""
import json
from pathlib import Path

import numpy as np
import torch

from data.task_context_dataset import file_hash


def validate_teacher_context_arguments(args):
    source = getattr(args, 'task_context_source', 'student')
    path = getattr(args, 'task_context_teacher_targets', None)
    if source == 'teacher':
        if not path or not args.task_context_checkpoint or not args.task_context_split_manifest:
            raise ValueError('Teacher context requires --task-context-teacher-targets, --task-context-checkpoint and --task-context-split-manifest')
        variants = getattr(args, 'variants', ['context'])
        if not any(v.startswith('context') for v in variants):
            raise ValueError('Teacher context requires a context variant')
    elif path:
        raise ValueError('--task-context-teacher-targets requires --task-context-source teacher')


def load_policy_teacher_targets(args, dataset, train, val):
    """Validate provenance and exact partitions before workers or training start."""
    if getattr(args, 'task_context_source', 'student') != 'teacher':
        args.task_context_teacher_provenance = None
        return None
    from models.task_context_observer import load_observer_bundle
    from data.training_split import validation_partition
    bundle = load_observer_bundle(args.task_context_checkpoint)
    root = Path(args.task_context_teacher_targets)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    split = json.loads(Path(args.task_context_split_manifest).read_text())
    if manifest['schema'] != 'align-privileged-task-tokens-v1':
        raise ValueError('Expected exported privileged teacher task tokens')
    if manifest['teacher_sha256'] != bundle['metadata'].get('teacher_sha256'):
        raise ValueError('Teacher targets and observer teacher differ')
    if manifest['target_manifest_sha256'] != split['manifest_sha256'] or manifest['target_manifest_sha256'] != bundle['metadata']['target_manifest_sha256']:
        raise ValueError('Teacher targets, observer and policy split must share the target manifest')
    cfg = bundle['encoder_config']
    if manifest['shape'] != [cfg['tokens'], cfg['token_dim']]:
        raise ValueError('Teacher task-token shape differs from observer')
    banks = {}
    seen = set()
    for partition, record in manifest['splits'].items():
        path = root / f'{partition}.npz'
        if file_hash(path) != record['sha256']:
            raise ValueError(f'Teacher token file hash mismatch: {partition}')
        with np.load(path, allow_pickle=False) as values:
            keys = values['keys'].astype(str).tolist()
            tokens = values['task_tokens'].copy()
        if len(keys) != len(set(keys)) or seen.intersection(keys):
            raise ValueError('Duplicate teacher episode keys')
        seen.update(keys)
        if set(keys) != set(split['split'][partition]):
            raise ValueError(f'Teacher episode partition mismatch: {partition}')
        if tokens.shape != (len(keys), *manifest['shape']) or not np.isfinite(tokens).all():
            raise ValueError('Invalid teacher task-token values or shape')
        banks[partition] = dict(zip(keys, tokens.astype(np.float32)))
    if set(banks) != set(split['split']):
        raise ValueError('Missing teacher episode partitions')
    evaluation = validation_partition(split['split'])
    for indices, partition in ((train, 'train'), (val, evaluation)):
        if {dataset._episode_keys[i] for i in indices} != set(banks[partition]):
            raise ValueError('Policy episodes differ from teacher token partition')
    args.task_context_teacher_provenance = dict(manifest_sha256=file_hash(manifest_path),
        teacher_sha256=manifest['teacher_sha256'], target_manifest_sha256=manifest['target_manifest_sha256'],
        shape=manifest['shape'], validation_partition=evaluation, splits=manifest['splits'])
    return {key: value for bank in banks.values() for key, value in bank.items()}


def replace_task_tokens(payload, reference, count):
    """Replace task slots only, retaining live execution and optional policy slots."""
    reference = torch.as_tensor(reference, device=payload.device, dtype=payload.dtype).detach()
    if reference.shape != (payload.shape[0], count, payload.shape[-1]) or not torch.isfinite(reference).all():
        raise ValueError('Reference task tokens must be finite [batch, task_tokens, token_dim]')
    task = reference[:, None].expand(-1, payload.shape[1], -1, -1)
    return torch.cat((task, payload[:, :, count:]), dim=-2)


def policy_context_sequence(model, patches, cls, states, lengths, ticks, batch):
    payload = model.context_sequence(patches, cls, states, lengths, ticks)
    if getattr(model, 'task_context_source', 'student') == 'teacher':
        if 'teacher_task_tokens' not in batch:
            raise ValueError('Teacher-conditioned policy requires episode teacher_task_tokens')
        payload = replace_task_tokens(payload, batch['teacher_task_tokens'],
            model.task_context.config['encoder_config']['tokens'])
    return payload
