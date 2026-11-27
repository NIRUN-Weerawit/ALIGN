"""Train a fresh X-VLA action transformer on the fixed ALIGN LIBERO Goal split."""
import argparse
import bisect
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / 'configs/splits/libero_goal_piper_384_44.json'
VIEWS = ('image', 'wrist_image')


class GoalDataset(Dataset):
    def __init__(self, path, keys, chunk):
        self.path = path
        self.chunk = chunk
        self.file = None
        self.episodes = []
        self.ends = []
        with h5py.File(path) as h5:
            for key in keys:
                episode = h5[key]
                poses = episode['poses'][:]
                actions = episode['actions'][:]
                if poses.shape != (len(actions), 6) or actions.shape != (len(poses), 7):
                    raise ValueError(f'Invalid pose/action dimensions for {key}')
                if any(len(episode[f'frames/{view}']) != len(poses) for view in VIEWS):
                    raise ValueError(f'Unaligned camera frames for {key}')
                if not (np.isfinite(poses).all() and np.isfinite(actions).all()):
                    raise ValueError(f'Non-finite pose/action for {key}')
                if not np.isin(actions[:, 6], (0, 1)).all():
                    raise ValueError(f'Expected ALIGN binary gripper targets in {key}')
                count = len(poses) - chunk
                if count < 1:
                    raise ValueError(f'{key} is too short for chunk size {chunk}')
                states = np.zeros((len(poses), 20), dtype=np.float32)
                states[:, :3] = poses[:, :3]
                rotation = Rotation.from_rotvec(poses[:, 3:]).as_matrix()
                states[:, 3:6] = rotation[:, :, 0]
                states[:, 6:9] = rotation[:, :, 1]
                targets = states[1:].copy()
                targets[:, 9] = 1 - actions[:-1, 6]
                text = json.loads(episode['texts'][()])
                self.episodes.append((key, states, targets, text[0]))
                self.ends.append(count + (self.ends[-1] if self.ends else 0))
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        i = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[i - 1] if i else 0)
        key, states, targets, text = self.episodes[i]
        if self.file is None:
            self.file = h5py.File(self.path)
        batch = {'observation.state': torch.from_numpy(states[t]),
                 'action': torch.from_numpy(targets[t:t + self.chunk]), 'text': text}
        for camera, name in zip(VIEWS, ('image', 'image2')):
            image = torch.from_numpy(self.file[f'{key}/frames/{camera}'][t]).permute(2, 0, 1)
            batch[f'observation.images.{name}'] = (image.float() / 255 - self.mean) / self.std
        return batch


def load_policy(snapshot, chunk, device):
    from lerobot.policies.xvla import modeling_xvla as xvla
    from safetensors import safe_open

    config = xvla.XVLAConfig.from_pretrained(snapshot, local_files_only=True)
    if config.action_mode != 'ee6d' or config.max_state_dim != 20:
        raise ValueError('Expected an X-VLA LIBERO EE6D snapshot')
    config.device = str(device)
    config.dtype = 'float32'
    config.chunk_size = chunk
    config.n_action_steps = chunk
    config.freeze_vision_encoder = config.freeze_language_encoder = True
    config.train_policy_transformer = config.train_soft_prompts = True
    policy = xvla.XVLAPolicy(config)
    prefix = 'model.vlm.'
    with safe_open(snapshot / 'model.safetensors', framework='pt', device='cpu') as source:
        weights = {key: source.get_tensor(key) for key in source.keys() if key.startswith(prefix)}
    if xvla._is_vendored_florence_state_dict(weights):
        weights = xvla._remap_vendored_florence_state_dict(weights)
    shared = prefix + 'language_model.shared.weight'
    embedding = prefix + 'language_model.encoder.embed_tokens.weight'
    if shared in weights and embedding not in weights:
        weights[embedding] = weights[shared]
    if embedding in weights and shared not in weights:
        weights[shared] = weights[embedding]
    policy.model.vlm.load_state_dict({key.removeprefix(prefix): value for key, value in weights.items()})
    policy.model.vlm.requires_grad_(False)
    return policy.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/libero_goal.h5')
    parser.add_argument('--encoder-source', required=True, type=Path, help='Local X-VLA LIBERO snapshot')
    parser.add_argument('--tokenizer', required=True, type=Path, help='Local BART tokenizer snapshot')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-batches', type=int, help='Limit batches per phase for a smoke run')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.chunk_size) < 1 or args.num_workers < 0:
        parser.error('Invalid epoch, batch, chunk, or worker count')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    from transformers import AutoTokenizer
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split = json.loads(SPLIT.read_text())
    train, val = split['train_episodes'], split['val_episodes']
    with h5py.File(args.data) as h5:
        if len(train) != 384 or len(val) != 44 or len(set(train + val)) != 428 or set(train + val) != {
            key for key in h5 if key.startswith('ep_')
        }:
            raise ValueError('HDF5 episode IDs do not match ALIGN Goal split')
    datasets = [GoalDataset(args.data, keys, args.chunk_size) for keys in (train, val)]
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0), num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0) for i, ds in enumerate(datasets)]
    policy = load_policy(args.encoder_source, args.chunk_size, device)
    processor = args.encoder_source / 'policy_preprocessor.json'
    config = {step['registry_name']: step['config'] for step in json.loads(processor.read_text())['steps']}
    token = config['tokenizer_processor']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = token['padding_side']
    domain = config['xvla_add_domain_id']['domain_id']
    optimizer = torch.optim.AdamW(policy.model.transformer.parameters(), lr=args.lr, fused=device.type == 'cuda')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SPLIT, args.output / 'split.json')
    (args.output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')

    def run(loader, training):
        policy.train(training)
        policy.model.vlm.eval()
        loss_sum = count = 0
        for step, batch in enumerate(loader, 1):
            text = batch.pop('text')
            batch[OBS_LANGUAGE_TOKENS] = tokenizer(
                text, max_length=token['max_length'], padding=token['padding'],
                truncation=token['truncation'], return_tensors='pt')['input_ids']
            batch['domain_id'] = torch.full((len(text),), domain, dtype=torch.long)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.set_grad_enabled(training), torch.autocast(device.type, dtype=torch.bfloat16,
                                                                 enabled=device.type == 'cuda'):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite X-VLA loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.transformer.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            loss_sum += float(loss.detach()) * len(text)
            count += len(text)
            if step == args.max_batches:
                break
        return loss_sum / count

    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = run(loaders[0], True)
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(args.seed)
            val_loss = run(loaders[1], False)
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        with (args.output / 'metrics.jsonl').open('a') as metrics:
            metrics.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            path = args.output / 'best'
            policy.save_pretrained(path)
            shutil.copyfile(processor, path / processor.name)
            (path / 'selection.json').write_text(json.dumps(row, indent=2) + '\n')


if __name__ == '__main__':
    main()
