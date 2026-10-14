#!/usr/bin/env python3
"""Compare dense/sparse training on identical saved plain-history episode groups."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models.align_intention as align
from data.align_dataset import ALIGNDataset
from scripts.run_intention_ablation import (CachedEpisodes, LengthBucketBatchSampler,
    collate_segments, seed_everything)
from training.plain_history_training import SparseHistoryEpisodes, collate_sparse_histories, plain_history_epoch
from training.train_intention import train_v4_epoch


class UnusedCachedVision(torch.nn.Module):
    """Raw DINOv2 is not involved in either cached-feature benchmark."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.empty(0), requires_grad=False)


class CountingLoader:
    def __init__(self, loader, mode, cameras):
        self.loader, self.mode, self.cameras = loader, mode, cameras
        self.payload = self.frames = 0
        self.first_batch_wait_s = 0.
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        start = time.perf_counter()
        first = True
        for batch in self.loader:
            if first:
                self.first_batch_wait_s = time.perf_counter()-start
                first = False
            frames = batch['frames' if self.mode == 'sparse' else 'frames_segment']
            self.payload += frames.numel() * frames.element_size()
            self.frames += frames.numel() // (self.cameras * 257 * 768)
            del frames
            yield batch
            del batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--mode', choices=['dense', 'sparse'], required=True)
    parser.add_argument('--microbatches', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-data-stages', action='store_true')
    parser.add_argument('--copy-mode', choices=['direct', 'materialized'], default='direct')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--warmup-microbatches', type=int, default=0,
                        help='Warm kernels and persistent workers before the timed sample.')
    args = parser.parse_args()
    cfg = json.loads((args.run_directory / 'config.json').read_text())
    torch.set_num_threads(cfg.get('cpu_threads', 2))
    seed_everything(cfg['seed'] + args.epoch)
    align.VisionEncoder = UnusedCachedVision
    model = align.ALIGNIntentionModel(state_dim=cfg['state_dim'], compressed_dim=cfg['compressed_dim'],
        mamba_output_dim=0, use_intent_tokens=False, use_memory_bank=False,
        num_cameras=cfg['num_cameras'], history_size=cfg['history_size'], head_history_mode='full',
        head_type='diffusion', head_d_model=cfg['head_d_model'], action_dim=7, chunk_size=cfg['chunk_size'],
        diffusion_train_steps=cfg['diffusion_train_steps'], diffusion_loss_repeats=cfg['diffusion_loss_repeats'],
        diffusion_clip_sample=cfg['diffusion_clip_sample'], use_text=cfg['use_text'],
        text_dim=cfg['text_dim'], text_encoder_type=cfg['text_encoder_type'], text_vocab=cfg['text_vocab'])
    model._build_head_and_bank(cfg['pool_out_dim'])
    saved = torch.load(args.run_directory / 'training_last.pt', map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(saved['model_state_dict'], strict=False)
    if unexpected or any(not key.startswith('vision_encoder.') for key in missing):
        raise ValueError((missing, unexpected))
    device = torch.device('cuda')
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['lr'], weight_decay=cfg['weight_decay'], fused=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    del saved
    dataset = ALIGNDataset(cfg['data'], mode='head', cameras=cfg['cameras'],
                           traj_window=cfg['segment_length'], dinov2_path=cfg['cache'])
    episodes = [dataset._episode_keys.index(key) for key in cfg['train_episodes']]
    parent = CachedEpisodes(dataset, episodes, cfg['segment_length'], cfg['seed'], True,
        'episode', cfg['supervision_points'], cfg['chunk_size'], cfg['observation_dropout_prob'],
        cfg['drop_state_with_all_views'])
    parent.epoch = args.epoch
    micro = cfg.get('micro_batch_size') or cfg['batch_size']
    sampler = LengthBucketBatchSampler([dataset._get_episode_length(ep) for ep in episodes],
        cfg['batch_size'], cfg['seed'], cfg['length_bucket_multiplier'], micro_batch_size=micro)
    sampler.set_epoch(args.epoch)
    sparse = args.mode == 'sparse'
    data = SparseHistoryEpisodes(parent, cfg['history_size'],
        profile=args.profile_data_stages, copy_mode=args.copy_mode) if sparse else parent
    options = dict(num_workers=args.num_workers, pin_memory=args.pin_memory)
    if args.num_workers:
        options.update(prefetch_factor=args.prefetch_factor, multiprocessing_context='spawn',
                       persistent_workers=True)
    loader = CountingLoader(DataLoader(data, batch_sampler=sampler,
        collate_fn=collate_sparse_histories if sparse else collate_segments, **options),
        args.mode, cfg['num_cameras'])
    loop = SimpleNamespace(history_size=cfg['history_size'], chunk_size=cfg['chunk_size'],
        action_dim=7, head_type='diffusion', skip_nan=False, grad_clip=cfg['grad_clip'],
        no_sample_during_train=True, head_window_batch_size=cfg['head_window_batch_size'],
        plain_head_batch_size=32, gradient_accum_steps=cfg['batch_size'] // micro,
        profile_data_stages=args.profile_data_stages,
        gripper_loss_weight=cfg['gripper_loss_weight'], gripper_threshold=cfg['gripper_threshold'])
    seed_everything(cfg['seed'] + args.epoch)
    if args.warmup_microbatches:
        if sparse:
            plain_history_epoch(model, loader, device, loop, optimizer, args.warmup_microbatches)
        else:
            train_v4_epoch(model, loader, optimizer, device, loop, args.warmup_microbatches)
        loader.payload = loader.frames = 0
        loader.first_batch_wait_s = 0.
        torch.cuda.synchronize()
        seed_everything(cfg['seed'] + args.epoch)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    def disk_read_bytes():
        path = Path('/proc/self/io')
        if not path.exists():
            return None
        return int(next(line.split(':')[1] for line in path.read_text().splitlines()
                        if line.startswith('read_bytes:')))
    initial_reads = disk_read_bytes()
    initial_faults = resource.getrusage(resource.RUSAGE_SELF)
    if sparse:
        loss, _ = plain_history_epoch(model, loader, device, loop, optimizer, args.microbatches)
    else:
        loss, _ = train_v4_epoch(model, loader, optimizer, device, loop, args.microbatches)
    torch.cuda.synchronize()
    final_reads = disk_read_bytes()
    final_faults = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(mode=args.mode, microbatches=args.microbatches, epoch=args.epoch,
        history_size=cfg['history_size'], wall_s=time.perf_counter() - start, loss=loss,
        copy_mode=args.copy_mode, num_workers=args.num_workers, pin_memory=args.pin_memory,
        warmup_microbatches=args.warmup_microbatches,
        prefetch_factor=args.prefetch_factor, first_batch_wait_s=loader.first_batch_wait_s,
        frame_count=loader.frames, feature_payload_mib=loader.payload / 1024**2,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        disk_read_mib=(final_reads-initial_reads)/1024**2
            if initial_reads is not None and not args.num_workers else None,
        major_page_faults=final_faults.ru_majflt-initial_faults.ru_majflt,
        minor_page_faults=final_faults.ru_minflt-initial_faults.ru_minflt,
        timing=loop._last_stage_times)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if loader.loader._iterator is not None:
        loader.loader._iterator._shutdown_workers()
    dataset.close()


if __name__ == '__main__':
    main()
