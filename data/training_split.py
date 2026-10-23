"""Piper episode defaults for the ablation and context-token training paths."""
import json
from pathlib import Path

from data.task_context_dataset import json_hash

ROOT = Path(__file__).resolve().parents[1]
PIPER_SPLIT = ROOT / 'configs/splits/libero_goal_piper_384_44.json'
PIPER_TARGETS = ROOT / 'reports/task_context_piper_384_20261023/targets/manifest.json'
PIPER_PREFIXES = ROOT / 'reports/task_context_piper_384_20261023/prefix_cache'
PIPER_TEACHER = ROOT / 'checkpoints/task_context_piper_384_20261023/teacher_frozen.pt'


def canonical_split():
    value = json.loads(PIPER_SPLIT.read_text())
    if json_hash({k:v for k,v in value.items() if k != 'manifest_sha256'}) != value['manifest_sha256']:
        raise ValueError('Canonical Piper split checksum mismatch')
    train, val = value['train_episodes'], value['val_episodes']
    if len(train) != 384 or len(val) != 44 or len(set(train + val)) != 428:
        raise ValueError('Invalid canonical Piper episode membership')
    return value


def piper_partition(keys, lengths=None, source_name=''):
    """Return exact train/val keys for Goal; leave unrelated datasets unchanged."""
    value = canonical_split()
    expected = set(value['train_episodes'] + value['val_episodes'])
    declared_goal = 'libero_goal' in Path(str(source_name)).name.lower()
    if set(keys) != expected or len(keys) != 428:
        if declared_goal:
            raise ValueError('LIBERO Goal episodes differ from canonical Piper 384/44 split')
        return None
    if lengths is not None and dict(lengths) != value['episode_lengths']:
        if declared_goal:
            raise ValueError('LIBERO Goal episode lengths differ from canonical Piper data')
        return None
    return list(value['train_episodes']), list(value['val_episodes'])


def piper_episode_indices(dataset):
    keys = list(dataset._episode_keys)
    lengths = ({k:dataset._get_episode_length(i) for i,k in enumerate(keys)}
        if hasattr(dataset, '_get_episode_length') else None)
    partition = piper_partition(keys, lengths, getattr(dataset, 'h5_path', ''))
    if partition is None:
        return None
    mapping = {k:i for i,k in enumerate(keys)}
    return tuple([mapping[k] for k in group] for group in partition)


def resolve_context_manifest(args):
    """Implicit context policies use the Piper-trained observer's bound manifest."""
    if getattr(args, 'task_context_checkpoint', None) and not getattr(args, 'task_context_split_manifest', None):
        if not PIPER_TARGETS.exists():
            raise ValueError('Piper context targets are missing; supply --task-context-split-manifest explicitly')
        args.task_context_split_manifest = str(PIPER_TARGETS)


def validation_partition(partitions):
    return 'development' if partitions.get('development') else 'confirmation'


def validate_goal_training_manifest(manifest):
    """Refuse legacy Goal caches by default; old experiments require an explicit opt-in."""
    if 'split' not in manifest:
        raise ValueError('Training cache has no episode split')
    keys = [k for rows in manifest['split'].values() for k in rows]
    partition = piper_partition(keys,
        {k:v['length'] for k,v in manifest['episodes'].items()} if 'episodes' in manifest else None)
    if partition is not None and set(manifest['split']['train']) != set(partition[0]):
        raise ValueError('Training cache uses a legacy LIBERO Goal split; rebuild on Piper 384/44 or pass --allow-legacy-split')
