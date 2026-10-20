"""Evaluate new policy checkpoints on their saved pretraining-safe episode split."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


def evaluate_fixed_split(model, cfg, paths, batch_size, n_batches, device):
    from data.align_dataset import ALIGNDataset
    from scripts.run_intention_ablation import CachedEpisodes, LimitedBatches
    from training.cached_loading import collate_cached_segments
    from training.train_intention import validate
    if len(paths) != 1:
        raise ValueError('Context policy evaluation requires its single audited dataset')
    cache = cfg.get('cache') or cfg.get('dinov2_path')
    if not cache or not Path(cache).is_dir():
        raise ValueError('Cached fixed-split evaluation needs the saved DINO cache path; use trajectory rollout for live images')
    record = cfg['task_context_split']
    ds = ALIGNDataset(paths[0], mode='head', cameras=record['cameras'],
        traj_window=cfg['chunk_size'], dinov2_path=cache)
    if set(ds._episode_keys) != set(record['train_episodes'] + record['development_episodes'] + record['confirmation_episodes']):
        raise ValueError('Evaluation episodes differ from the saved pretraining split')
    indices = [ds._episode_keys.index(k) for k in record['development_episodes']]
    dataset = CachedEpisodes(ds, indices, cfg['chunk_size'], cfg['seed'] + 1000, False,
        temporal_sampling='episode', supervision_points=cfg.get('supervision_points',16), chunk_size=cfg['chunk_size'])
    loader = LimitedBatches(DataLoader(dataset, batch_size=batch_size, num_workers=0,
        collate_fn=collate_cached_segments), n_batches)
    args = SimpleNamespace(history_size=cfg['history_size'], chunk_size=cfg['chunk_size'], action_dim=7,
        head_type=cfg['head_type'], skip_nan=False, gripper_threshold=cfg.get('gripper_threshold',.5),
        gripper_loss_weight=cfg.get('gripper_loss_weight',1.))
    try:
        loss, _, metrics = validate(model, loader, device, args)
    finally:
        ds.close()
    return dict(validation_loss=loss, **metrics, split='saved development episodes',
        confirmation_opened=False, split_manifest_sha256=record['manifest_sha256'])
