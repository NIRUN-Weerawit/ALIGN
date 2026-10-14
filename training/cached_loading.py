"""Direct, bounded host staging for complete cached observation segments."""
import time
import numpy as np
import torch
from torch.utils.data import get_worker_info


def allocate_feature_batch(shape, dtype):
    if get_worker_info() is not None:
        torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        return torch.empty(shape, dtype=torch_dtype).share_memory_()
    return torch.from_numpy(np.empty(shape, dtype=dtype))


def collate_cached_segments(items):
    """Keep every frame in order; copy valid rows once and zero only padding."""
    profiling = 'data_stage_cpu_s' in items[0]
    start = time.perf_counter() if profiling else 0.
    length = max(x['segment_len'] for x in items)
    first = items[0]['frames_segment']
    frames = allocate_feature_batch((len(items), length, *first.shape[1:]), first.dtype)
    output = frames.numpy()
    read_s = pad_s = 0.
    for index, item in enumerate(items):
        source = item['frames_segment']
        before = time.perf_counter() if profiling else 0.
        np.copyto(output[index, :len(source)], source)
        after = time.perf_counter() if profiling else 0.
        output[index, len(source):] = 0
        if profiling:
            read_s += after-before
            pad_s += time.perf_counter()-after
    result = dict(frames_segment=frames, segment_len=np.array([x['segment_len'] for x in items]),
                  texts=[x.get('text', '') for x in items])
    for key in ('states_segment', 'actions_segment', 'loss_anchor_mask',
                'observation_timesteps', 'observation_camera_mask', 'observation_state_mask'):
        if key not in items[0]:
            continue
        source = items[0][key]
        batch = np.zeros((len(items), length, *source.shape[1:]), dtype=source.dtype)
        for index, item in enumerate(items):
            batch[index, :len(item[key])] = item[key]
        result[key] = torch.from_numpy(batch)
    if profiling:
        result['data_stage_cpu_s'] = {
            key: sum(x['data_stage_cpu_s'][key] for x in items)
            for key in items[0]['data_stage_cpu_s']}
        result['data_stage_cpu_s'].update(feature_materialization_wall_s=read_s,
            padding_wall_s=pad_s, batch_pack_wall_s=time.perf_counter()-start-read_s-pad_s)
    return result
