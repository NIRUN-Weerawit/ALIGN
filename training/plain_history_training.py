"""Sparse cached observations for stateless, ordered-history diffusion models."""
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from training.train_intention import sync_and_step


class SparseHistoryEpisodes(Dataset):
    """Reuse the dense sampler's anchors/dropout, touching only their histories."""
    def __init__(self, episodes, history_size):
        self.episodes = episodes
        self.history_size = history_size

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        item = self.episodes[index]
        anchors = np.flatnonzero(item['loss_anchor_mask'])
        times = np.maximum(0, anchors[:, None] + np.arange(1 - self.history_size, 1))
        unique, inverse = np.unique(times, return_inverse=True)
        result = dict(
            frames=item['frames_segment'][unique],
            states=item['states_segment'][unique],
            history_indices=inverse.reshape(times.shape),
            targets=item['actions_segment'][anchors[:, None] +
                                            np.arange(self.episodes.chunk_size)],
            anchors=anchors, text=item['text'])
        if 'observation_camera_mask' in item:
            result['camera_mask'] = item['observation_camera_mask'][unique]
        if 'observation_state_mask' in item:
            result['state_mask'] = item['observation_state_mask'][unique]
        # All selected arrays above own their data. Drop the cached mmap
        # reference so sparse reads do not retain hundreds of resident maps.
        handles = getattr(self.episodes.dataset, '_dinov2_memmaps', None)
        if handles is not None:
            handles.pop(self.episodes.episodes[index], None)
        return result


def collate_sparse_histories(items):
    """Pack unique observations without padding to the longest episode."""
    offsets = np.cumsum([0] + [len(item['frames']) for item in items[:-1]])
    histories = np.concatenate([x['history_indices'] + offset
                                for x, offset in zip(items, offsets)])
    episodes = np.concatenate([np.full(len(x['anchors']), i, dtype=np.int64)
                               for i, x in enumerate(items)])
    anchors = np.concatenate([x['anchors'] for x in items])
    # Match the dense trainer's chronological anchor/episode ordering.
    order = np.lexsort((episodes, anchors))
    result = dict(
        frames=torch.from_numpy(np.concatenate([x['frames'] for x in items])),
        states=torch.from_numpy(np.concatenate([x['states'] for x in items])),
        history_indices=torch.from_numpy(histories[order]),
        targets=torch.from_numpy(np.concatenate([x['targets'] for x in items])[order]),
        anchor_timesteps=torch.from_numpy(anchors[order]),
        episode_indices=torch.from_numpy(episodes[order]),
        texts=[x['text'] for x in items])
    for key in ('camera_mask', 'state_mask'):
        if key in items[0]:
            result[key] = torch.from_numpy(np.concatenate([x[key] for x in items]))
    return result


def history_conditions(model, batch, device):
    if model.use_memory_bank or model.intention_encoder is not None:
        raise ValueError('Sparse history requires a stateless plain model')
    frames = batch['frames'].to(device, non_blocking=True)
    states = batch['states'].to(device, dtype=torch.float32, non_blocking=True)
    count, tokens, _ = frames.shape
    cameras = tokens // 257
    features = frames.reshape(count, cameras, 257, 768)
    patches = features[:, :, :-1].reshape(count, cameras * 256, 768)
    visibility = batch.get('camera_mask')
    if visibility is not None:
        visibility = visibility.to(device)
        visible_patches = visibility[:, :, None, None].expand(-1, -1, 256, 768).reshape(patches.shape)
        patches = torch.where(visible_patches, patches, torch.zeros_like(patches))
    state_mask = batch.get('state_mask')
    if state_mask is not None:
        state_mask = state_mask.to(device)
        states = torch.where(state_mask[:, None], states, torch.zeros_like(states))
    encoded_states = model.state_encoder(states)
    if state_mask is not None:
        encoded_states = encoded_states * state_mask[:, None]
    encoded_visual = model.encode_patch_sequence(patches, encoded_states)
    if visibility is not None:
        encoded_visual = (encoded_visual.reshape(count, cameras, 256, -1) *
                          visibility[:, :, None, None]).reshape(count, cameras * 256, -1)
    indices = batch['history_indices'].to(device)
    visual = encoded_visual.flatten(1)[indices]
    states = encoded_states[indices]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        text = None
        if model.text_condition is not None:
            # The dense trainer encodes task text in FP32 before head AMP.
            with torch.amp.autocast('cuda', enabled=False):
                text = model.encode_task_text(batch['texts'])
            text = text[batch['episode_indices'].to(device)]
        return model.action_condition(visual, states, text_emb=text)


def plain_history_epoch(model, loader, device, args, optimizer=None, max_steps=0):
    """Vectorize anchor conditioning; bound diffusion calls and preserve weighting."""
    training = optimizer is not None
    model.train(training)
    start = time.perf_counter()
    data_wait = 0.
    events = {}

    def mark():
        if device.type != 'cuda':
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def finish(name, before):
        if before is not None:
            events.setdefault(name, []).append((before, mark()))

    limit = min(max_steps, len(loader)) if max_steps else len(loader)
    iterator = iter(loader)
    accumulation = max(1, getattr(args, 'gradient_accum_steps', 1))
    group_count = total_count = 0
    loss_sum = action_sum = 0.
    squared = absolute = None
    metric_count = grip_correct = grip_total = 0
    genuine_windows = 0
    dim_weights = torch.ones(args.action_dim, device=device)
    dim_weights[6] = getattr(args, 'gripper_loss_weight', 1.)
    for step in tqdm(range(limit), desc='  [plain history]', unit='microbatch', leave=False):
        before = time.perf_counter()
        batch = next(iterator)
        if not training:
            genuine_windows += int(torch.unique(batch['anchor_timesteps']).numel())
        data_wait += time.perf_counter() - before
        if training and step % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
            group_count = 0
        with torch.set_grad_enabled(training):
            encode_start = mark()
            condition = history_conditions(model, batch, device)
            target = batch['targets'].to(device, non_blocking=True)
            finish('feature_and_condition_gpu_s', encode_start)
            count = target.shape[0]
            weighted_losses = []
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                for offset in range(0, count, args.plain_head_batch_size):
                    stop = min(count, offset + args.plain_head_batch_size)
                    loss_start = mark()
                    loss = model.intention_head.loss(target[offset:stop], condition[offset:stop],
                                                    dim_weights=dim_weights)
                    weighted_losses.append(loss * (stop - offset))
                    finish('diffusion_loss_gpu_s', loss_start)
                    if not training:
                        pred = model.intention_head.sample(condition[offset:stop]).float()
                        diff = (pred - target[offset:stop]).double()
                        chunk_squared = diff.square().sum((0, 1)).cpu().numpy()
                        chunk_absolute = diff.abs().sum((0, 1)).cpu().numpy()
                        squared = chunk_squared if squared is None else squared + chunk_squared
                        absolute = chunk_absolute if absolute is None else absolute + chunk_absolute
                        metric_count += diff.shape[0] * diff.shape[1]
                        grip_correct += int(((pred[..., 6] > args.gripper_threshold) ==
                                             (target[offset:stop, ..., 6] > 0)).sum())
                        grip_total += pred.shape[0] * pred.shape[1]
                        action_sum += float(pred.abs().mean()) * (stop - offset)
            total_loss = sum(weighted_losses)
            if training:
                backward_start = mark()
                total_loss.backward()
                group_count += count
                if (step + 1) % accumulation == 0 or step + 1 == limit:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_count)
                    sync_and_step(model, optimizer, device, args.grad_clip)
                finish('backward_optimizer_gpu_s', backward_start)
        loss_sum += float(total_loss.detach())
        total_count += count
        # Release the previous batch before DataLoader allocates the next one.
        del batch, condition, target, weighted_losses, total_loss
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if training:
        args._last_stage_times = dict(
            train_wall_s=time.perf_counter() - start, data_wait_wall_s=data_wait,
            **{name: sum(a.elapsed_time(b) for a, b in pairs) / 1000
               for name, pairs in events.items()})
        return loss_sum / total_count, 0.
    mse, mae = squared / metric_count, absolute / metric_count
    metrics = dict(pos_mse=float(mse[:3].mean()), rot_mse=float(mse[3:6].mean()),
                   grip_mse=float(mse[6]), pos_mae=float(mae[:3].mean()),
                   rot_mae=float(mae[3:6].mean()), grip_acc=grip_correct / grip_total,
                   gripper_padded_batches=0, gripper_genuine_batches=genuine_windows)
    metrics.update({key: float(mse[i]) for i, key in enumerate(
        ('px_mse', 'py_mse', 'pz_mse', 'rx_mse', 'ry_mse', 'rz_mse'))})
    return loss_sum / total_count, action_sum / total_count, metrics
