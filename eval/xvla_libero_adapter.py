"""LeRobot XVLA LIBERO policy for the shared ALIGN simulator loop.

The native XVLA LIBERO proprioception is 20-D EE6D despite the legacy 8-D
input_features entry in the published config. Match LeRobot's policy-specific
LIBERO processor, rather than its generic LIBERO processor or Piper adapter.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def resolve_snapshot(checkpoint, cache_dir=None):
    """Accept a snapshot, an HF models--... cache root, or a Hub repo ID."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        if not (path / 'config.json').is_file():
            ref = path / 'refs/main'
            if ref.is_file():
                path = path / 'snapshots' / ref.read_text().strip()
            else:
                candidates = sorted((path / 'snapshots').glob('*/config.json'))
                if len(candidates) != 1:
                    raise ValueError(f'Choose a specific HF snapshot directory: {path}')
                path = candidates[0].parent
        if not (path / 'model.safetensors').is_file():
            raise FileNotFoundError(f'Checkpoint download is incomplete: {path}')
        return path.resolve()
    if path.is_absolute() or str(checkpoint).startswith('.'):
        raise FileNotFoundError(checkpoint)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(checkpoint), cache_dir=cache_dir)).resolve()


def rotation6d_matrix(values):
    """XVLA encodes the first two matrix columns consecutively (not interleaved)."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('Expected six finite rotation values')
    first = values[:3]
    norm = np.linalg.norm(first)
    if norm < 1e-8:
        raise ValueError('Degenerate XVLA first rotation column')
    first = first / norm
    second = values[3:] - first * np.dot(first, values[3:])
    norm = np.linalg.norm(second)
    if norm < 1e-8:
        raise ValueError('Degenerate XVLA second rotation column')
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def native_observation(obs, orientation_matrix):
    """Native LIBERO views and zero-gripper, zero-inactive-arm EE6D state.

    Agentview is rotated 180 degrees; wrist is kept in raw simulator orientation.
    Only real views are supplied: XVLAPolicy pads the empty third view with a
    false visibility mask. Images use the native ImageNet normalization.
    """
    matrix = np.asarray(orientation_matrix, dtype=np.float32)
    position = np.asarray(obs['robot0_eef_pos'], dtype=np.float32)
    if matrix.shape != (3, 3) or position.shape != (3,):
        raise ValueError('Invalid simulator EEF pose')
    state = np.concatenate([position, matrix[:, 0], matrix[:, 1], [0.], np.zeros(10)]).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError('Non-finite simulator EEF pose')
    batch = {'observation.state': torch.from_numpy(state).unsqueeze(0)}
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    for source, destination, flip in [('agentview_image', 'image', True),
                                      ('robot0_eye_in_hand_image', 'image2', False)]:
        image = np.asarray(obs[source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f'{source} must be uint8 HWC RGB')
        if flip:
            image = image[::-1, ::-1]
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.
        batch[f'observation.images.{destination}'] = ((tensor - mean) / std).unsqueeze(0)
    return batch


def absolute_to_delta(action, position, orientation, controller):
    """Invert OSC delta scaling, recomputing against the current executed pose.

    Result uses ALIGN's 0=closed, 1=open convention; the shared loop performs
    the sole conversion to simulator +/-1. Pending chunks remain absolute until
    execution so action horizons do not use a stale reference pose.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (20,) or not np.isfinite(action).all():
        raise ValueError('XVLA must return a finite 20-D EE6D action')
    if not controller.use_delta or controller.control_dim != 6:
        raise ValueError('XVLA adapter requires a 6-D OSC delta controller')
    target_rotation = rotation6d_matrix(action[3:9])
    rotation_delta = Rotation.from_matrix(target_rotation @ np.asarray(orientation).T).as_rotvec()
    delta = np.concatenate([action[:3] - position, rotation_delta])
    low, high = np.asarray(controller.input_min), np.asarray(controller.input_max)
    out_low, out_high = np.asarray(controller.output_min), np.asarray(controller.output_max)
    scale = (out_high - out_low) / (high - low)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Unsupported OSC action scaling')
    command = (delta - (out_high + out_low) / 2) / scale + (high + low) / 2
    command = np.clip(command, low, high)
    return np.concatenate([command, [1. - np.clip(action[9], 0., 1.)]]).astype(np.float32)


def simulator_controller(env):
    controller = env.robots[0].controller
    controller.update()
    return controller


class XVLALiberoAdapter(torch.nn.Module):
    """Expose native action chunks while sharing reset, horizon and success logic."""
    history_size = 1
    use_memory_bank = False
    intention_encoder = None
    text_condition = None
    head_history_mode = 'mean'
    head_type = 'xvla'

    def __init__(self, checkpoint, device, *, cache_dir=None, tokenizer_path=None,
                 dtype=None, denoising_steps=None):
        super().__init__()
        try:
            from lerobot.policies.xvla.configuration_xvla import XVLAConfig
            from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError('XVLA needs its evaluation runtime; see scripts/README_xvla_libero.md') from exc
        snapshot = resolve_snapshot(checkpoint, cache_dir)
        config = XVLAConfig.from_pretrained(snapshot, local_files_only=True)
        if config.action_mode != 'ee6d' or config.max_state_dim != 20:
            raise ValueError('This adapter supports the XVLA LIBERO EE6D checkpoint only')
        config.device = str(device)
        config.dtype = dtype or ('bfloat16' if torch.device(device).type == 'cuda' else 'float32')
        if denoising_steps is not None:
            if denoising_steps < 1:
                raise ValueError('XVLA denoising steps must be positive')
            config.num_denoising_steps = denoising_steps
        preprocessing = json.loads((snapshot / 'policy_preprocessor.json').read_text())
        steps = {step['registry_name']: step['config'] for step in preprocessing['steps']}
        normalization = steps['normalizer_processor']['norm_map']
        if any(normalization.get(key) != 'IDENTITY' for key in ('STATE', 'ACTION', 'VISUAL')):
            raise ValueError('Unsupported checkpoint normalization; use its processor pipeline')
        token = steps['tokenizer_processor']
        if tokenizer_path is None and cache_dir is not None:
            token_root = Path(cache_dir) / ('models--' + token['tokenizer_name'].replace('/', '--'))
            ref = token_root / 'refs/main'
            candidates = ([token_root / 'snapshots' / ref.read_text().strip()] if ref.is_file()
                          else sorted((token_root / 'snapshots').glob('*')))
            if len(candidates) == 1 and (candidates[0] / 'tokenizer.json').is_file():
                tokenizer_path = str(candidates[0])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or token['tokenizer_name'],
            cache_dir=cache_dir, local_files_only=True)
        self.tokenizer.padding_side = token['padding_side']
        self.token_options = dict(max_length=token['max_length'], padding=token['padding'],
                                  truncation=token['truncation'], return_tensors='pt')
        self.domain_id = int(steps['xvla_add_domain_id']['domain_id'])
        self.policy = XVLAPolicy.from_pretrained(snapshot, config=config, local_files_only=True, strict=True)
        self.chunk_size = config.chunk_size
        self.device = torch.device(device)
        self._text_cache = {}
        self.metadata = dict(policy_type='xvla', checkpoint=str(snapshot),
            checkpoint_revision=snapshot.name, model_bytes=(snapshot / 'model.safetensors').stat().st_size,
            lerobot=importlib.metadata.version('lerobot'), transformers=importlib.metadata.version('transformers'),
            dtype=config.dtype, num_denoising_steps=config.num_denoising_steps, domain_id=self.domain_id,
            tokenizer=str(tokenizer_path or token['tokenizer_name']), tokenizer_max_length=token['max_length'],
            state_contract='XYZ + consecutive rotation columns + zero gripper + zero inactive arm (20D)',
            image_contract='agentview flipped H/W; wrist raw; ImageNet normalization; masked empty third view',
            action_contract='absolute EE6D; OSC delta conversion at execution; gripper inverted once')
        self.eval()

    def reset_episode(self):
        self.policy.reset()
        self._text_cache.clear()

    @torch.inference_mode()
    def predict_sim_action_chunk(self, obs, *, task_text, env, proprio=None):
        if not task_text:
            raise ValueError('XVLA LIBERO requires the task instruction')
        controller = simulator_controller(env)
        batch = {key: value.to(self.device) for key, value in
                 native_observation(obs, controller.ee_ori_mat).items()}
        if proprio is not None:
            proprio = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
            if proprio.shape != (20,) or not torch.isfinite(proprio).all():
                raise ValueError('XVLA proprio override must be a finite 20-D state')
            batch['observation.state'] = proprio.unsqueeze(0)
        if task_text not in self._text_cache:
            tokens = self.tokenizer(task_text, **self.token_options)
            self._text_cache[task_text] = tokens['input_ids'].to(self.device)
        batch['observation.language.tokens'] = self._text_cache[task_text]
        batch['domain_id'] = torch.tensor([self.domain_id], device=self.device)
        with torch.backends.cudnn.flags(enabled=True):
            actions = self.policy.predict_action_chunk(batch).float()
        if actions.shape != (1, self.chunk_size, 20) or not torch.isfinite(actions).all():
            raise ValueError(f'Invalid XVLA action chunk: {tuple(actions.shape)}')
        return actions

    def to_libero_action(self, action, obs, env):
        controller = simulator_controller(env)
        return absolute_to_delta(action, np.asarray(obs['robot0_eef_pos']), controller.ee_ori_mat, controller)
