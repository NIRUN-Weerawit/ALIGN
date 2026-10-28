"""Enforce pretraining episode partitions for every action-policy ablation arm."""
import json
from pathlib import Path

from data.task_context_dataset import SCHEMA, json_hash, file_hash


def policy_episode_split(dataset, path, observer_metadata=None, allow_unbound_manifest=False):
    path = Path(path)
    m = json.loads(path.read_text())
    if m.get('schema') != SCHEMA or json_hash({k:v for k,v in m.items() if k != 'manifest_sha256'}) != m.get('manifest_sha256'):
        raise ValueError('Invalid pretrained target split manifest')
    partitions = m['split']
    flat = [k for rows in partitions.values() for k in rows]
    if len(flat) != len(set(flat)) or set(flat) != set(dataset._episode_keys):
        raise ValueError('Pretrained split overlaps or does not match policy dataset episodes')
    if list(dataset.cameras) != m['cameras']:
        raise ValueError('Pretrained split camera order differs from policy observations')
    override = None
    if observer_metadata and observer_metadata['target_manifest_sha256'] != m['manifest_sha256']:
        if not allow_unbound_manifest:
            raise ValueError('Observer and policy split are not bound to the same target manifest')
        # Explicit, recorded repoint: the frozen observer's forward pass does not
        # depend on episode membership; this only relaxes the bookkeeping bind.
        override = dict(allowed_by='--task-context-split-override',
            observer_target_manifest_sha256=observer_metadata['target_manifest_sha256'],
            policy_manifest_sha256=m['manifest_sha256'])
    mapping = {k:i for i,k in enumerate(dataset._episode_keys)}
    from data.training_split import validation_partition
    evaluation_partition = validation_partition(partitions)
    for k in flat:
        if dataset._get_episode_length(mapping[k]) != m['episodes'][k]['length']:
            raise ValueError(f'Policy dataset episode length differs from pretraining: {k}')
    record: dict = dict(path=str(path.resolve()), sha256=file_hash(path), manifest_sha256=m['manifest_sha256'],
        train_episodes=partitions['train'], development_episodes=partitions['development'],
        confirmation_episodes=partitions['confirmation'], fps=m['fps'], cameras=m['cameras'])
    record['evaluation_partition'] = evaluation_partition
    if override is not None:
        record['split_override'] = override
    return [mapping[k] for k in partitions['train']], [mapping[k] for k in partitions[evaluation_partition]], record


def add_context_policy_arguments(parser):
    parser.add_argument('--task-context-checkpoint', default=None, help='Pretrained observer bundle; context variants replace legacy intention')
    parser.add_argument('--task-context-split-manifest', default=None, help='Exact teacher target manifest; required for context-policy runs and shared by all arms')
    parser.add_argument('--task-context-split-override', action='store_true',
        help='Allow a split manifest that is not hash-bound to the observer bundle (e.g. repointing to another '
             "family's train/val episodes). All other manifest checks still apply; the override is recorded in the checkpoint config.")
    parser.add_argument('--num-task-tokens','--task-context-tokens',type=int,default=None,
        help='Expected task-token count; inferred from observer when omitted. Requires matching pretraining.')
    parser.add_argument('--task-context-trainable', action='store_true', help='Fine-tune causal encoder with action loss; frozen summary is retained')
    parser.add_argument('--task-context-fusion', choices=('additive', 'gated_residual', 'film', 'cross_attention', 'aci'),
        default='additive', help='Context conditioning; aci selects alternating action-to-context/observation attention')
    parser.add_argument('--task-context-fusion-dim', type=int, default=128,
        help='Hidden width for context fusion/ACI attention (divisible by 4 for attention modes)')
    parser.add_argument('--policy-warm-start', default=None, help='Copy shape-compatible common policy weights from an existing checkpoint; context weights remain from the observer')


def context_policy_options(model):
    branch = getattr(model, 'task_context', None)
    return dict(task_context_tokens=branch.config['encoder_config']['tokens'] if branch is not None else None,
        task_context_config=branch.config if branch is not None else None,
        task_context_trainable=branch.trainable if branch is not None else False,
        task_context_initialization=branch.initialization if branch is not None else 'pretrained',
        task_context_fusion=getattr(model, 'task_context_fusion', 'additive'),
        task_context_fusion_dim=getattr(model, 'task_context_fusion_dim', 128))


def policy_parameter_groups(model, lr):
    """Frozen pilot trains adapters/policy; pretrained fine-tuning uses 0.1x LR."""
    branch = getattr(model, 'task_context', None)
    if branch is None or not branch.trainable or branch.initialization == 'scratch':
        return [p for p in model.parameters() if p.requires_grad]
    encoder = [p for p in branch.observer.encoder.parameters() if p.requires_grad]
    ids = {id(p) for p in encoder}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return [dict(params=other, lr=lr), dict(params=encoder, lr=lr*.1)]


def warm_start_policy(model, path):
    """Explicit common-module warm start; never replace the selected context space."""
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint.get('config', {})
    for key in ('action_dim','chunk_size','num_cameras','compressed_dim','head_type'):
        if key in cfg and cfg[key] != getattr(model,key):
            raise ValueError(f'Policy warm-start {key} differs from the target architecture')
    own = model.state_dict()
    source = checkpoint['model_state_dict']
    copied = {k:v for k,v in source.items() if k in own and own[k].shape == v.shape and
        not k.startswith(('vision_encoder.','task_context.','intention_encoder.','intention_head.context_'))
        and not k.endswith(('alpha_bar','sigma'))}
    if not any(k.startswith('intention_head.') for k in copied):
        raise ValueError('Warm-start checkpoint has no compatible action-policy parameters')
    model.load_state_dict(copied,strict=False)
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path), tensors_copied=len(copied))
