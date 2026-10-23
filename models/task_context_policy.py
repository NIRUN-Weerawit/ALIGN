"""Opt-in causal context conditioning for ALIGN's existing diffusion policy."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from models.task_context_observer import TaskContextObserver
from models.task_context_encoder import TaskContextEncoder
from models.intention_head import DiffusionPolicyHead


def rotation_columns(rotvec):
    """Stable Rodrigues conversion, matching geometric_state's row-major 3x2."""
    theta2 = rotvec.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    a = torch.where(theta2 < 1e-6, 1 - theta2 / 6 + theta2.square() / 120, theta.sin() / theta)
    b = torch.where(theta2 < 1e-6, .5 - theta2 / 24 + theta2.square() / 720, (1 - theta.cos()) / theta2.clamp_min(1e-12))
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    matrix = eye + a[..., None] * skew + b[..., None] * (skew @ skew)
    return matrix[..., :2].flatten(-2)


def visual_slots(patches, cls, cameras):
    """Raw camera-major 16x16 patches -> normalized CLS and four spatial cells."""
    shape = patches.shape[:-2]
    if patches.shape[-2] != cameras * 256 or cls.shape[-2] != cameras:
        raise ValueError("Context requires exactly 256 patches and CLS per camera")
    grid = patches.reshape(*shape, cameras, 2, 8, 2, 8, patches.shape[-1])
    cells = grid.mean((-4, -2)).reshape(*shape, cameras, 4, patches.shape[-1])
    slots = torch.cat([cls.unsqueeze(-2), cells], -2).float()
    return F.layer_norm(slots, (slots.shape[-1],), eps=1e-5)


@dataclass
class PolicyContextState:
    recurrence: object
    task_snapshot: torch.Tensor | None = None
    last_task: torch.Tensor | None = None


class TaskContextPolicyBranch(nn.Module):
    """Frozen pretrained observation/context branch; optional action-only fine-tuning.

    Config plus the policy state_dict reconstruct this without the original
    bundle path. Neither whole-episode teacher nor training labels are loaded.
    """
    def __init__(self, checkpoint=None, config=None, trainable=False, initialization="pretrained"):
        super().__init__()
        if initialization not in ("pretrained", "scratch"):
            raise ValueError("Unknown task-context initialization")
        bundle = None
        if config is None:
            if not checkpoint:
                raise ValueError("Context needs a pretrained observer bundle or saved configuration")
            from data.task_context_dataset import file_hash
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if bundle.get("schema") != "align-causal-task-observer-v1":
                raise ValueError("Expected an exported causal observer bundle")
            config = dict(summary_config=bundle["summary_config"], encoder_config=bundle["encoder_config"],
                metadata=bundle["metadata"], bundle_sha256=file_hash(checkpoint))
        self.config = dict(config)
        if config['encoder_config'].get('execution_dim',128) != config['encoder_config']['token_dim']:
            raise ValueError('Policy payload requires equal task/execution token widths')
        self.observer = TaskContextObserver(config["summary_config"], config["encoder_config"])
        if bundle is not None:
            self.observer.summary.load_state_dict(bundle["summary"], strict=True)
            self.observer.encoder.load_state_dict(bundle["encoder"], strict=True)
            self.observer.state_mean.copy_(bundle["state_mean"])
            self.observer.state_std.copy_(bundle["state_std"])
        if initialization == "scratch" and bundle is not None:
            # Same frozen observation/normalization space; new causal backbone/readouts.
            original = self.observer.encoder
            self.observer.encoder = TaskContextEncoder(**config["encoder_config"])
            self.observer.encoder.latent_mean.copy_(original.latent_mean)
            self.observer.encoder.latent_std.copy_(original.latent_std)
        self.trainable = bool(trainable or initialization == "scratch")
        self.initialization = initialization
        self.observer.requires_grad_(False).eval()
        self.observer.encoder.requires_grad_(self.trainable)
        self.fps = float(config["metadata"]["fps"])
        if self.fps <= 0:
            raise ValueError("Context frame rate must be positive")

    def train(self, mode=True):
        super().train(mode)
        self.observer.summary.eval()
        if not self.trainable:
            self.observer.encoder.eval()
        return self

    def summaries(self, patches, cls, state, seconds, lengths=None):
        if state.shape[-1] != 7:
            raise ValueError("Context expects position + rotation vector + causal gripper")
        if lengths is not None:
            valid = torch.arange(state.shape[1], device=state.device)[None] < lengths[:, None]
            patches = torch.where(valid[..., None, None], patches, 0.)
            cls = torch.where(valid[..., None, None], cls, 0.)
            state = torch.where(valid[..., None], state, 0.)
            seconds = torch.where(valid, seconds, 0.)
        with torch.no_grad(), torch.autocast(device_type=state.device.type, enabled=False):
            slots = visual_slots(patches.float(), cls.float(), self.config["summary_config"]["cameras"])
            geometry = torch.cat([state[..., :3].float(), rotation_columns(state[..., 3:6].float()), state[..., 6:7].float()], -1)
            geometry = (geometry - self.observer.state_mean) / self.observer.state_std
            return self.observer.summary(slots, geometry, seconds.float())

    def sequence(self, patches, cls, state, lengths, timesteps):
        summaries = self.summaries(patches, cls, state, timesteps / self.fps, lengths)
        encoder = self.observer.encoder
        # Read at every real time. Padding readout indices are clamped and then
        # excluded by the production action-anchor mask; no future is consulted.
        anchors = torch.arange(state.shape[1], device=state.device)[None].expand(len(state), -1)
        anchors = torch.minimum(anchors, lengths[:, None] - 1)
        with torch.set_grad_enabled(self.training and self.trainable), torch.autocast(device_type=state.device.type, enabled=False):
            out = encoder(summaries, lengths, anchors)
            task = out['tokens']
            if self.training and self.trainable:
                # Forward remains the top-weight mode. Straight-through selector
                # gradients allow action-only scratch/fine-tune arms to learn weights.
                probability = out['logits'].softmax(-1)
                weight = F.one_hot(out['mode'], encoder.config['modes']).to(probability) + probability - probability.detach()
                native = out['means'] * encoder.latent_std + encoder.latent_mean
                task = (weight[..., None, None] * native).sum(-3)
        return torch.cat([task, out["execution_token"]], -2)

    def new_stream(self, batch_size):
        return PolicyContextState(self.observer.encoder.new_stream(batch_size))

    @torch.no_grad()
    def step(self, patches, cls, state, cache=None):
        cache = self.new_stream(len(state)) if cache is None else cache
        seconds = state.new_full((len(state), 1), cache.recurrence.steps / self.fps)
        summaries = self.summaries(patches[:, None], cls[:, None], state[:, None], seconds)[:, 0]
        with torch.autocast(device_type=state.device.type, enabled=False):
            out = self.observer.encoder.step(summaries, cache.recurrence)
        cache.last_task = out["tokens"].detach().clone()
        task = cache.task_snapshot if cache.task_snapshot is not None else out["tokens"]
        return torch.cat([task, out["execution_token"]], -2), cache

    @staticmethod
    def freeze_task(cache):
        if cache is None or cache.last_task is None:
            raise ValueError("Observe at least one frame before saving task context")
        cache.task_snapshot = cache.last_task.detach().clone()


class ContextDiffusionPolicyHead(DiffusionPolicyHead):
    """Live condition plus task/execution residuals and optional learned interaction.

    The default preserves the original parameter names and computation. Optional
    fusion modules start as the same additive model, so a common pretrained head
    provides an identical starting policy for controlled fine-tuning experiments.
    Inputs are visual/state windows with shape ``(B, H, C)`` and a context payload
    ``(B, task_tokens + 1, context_dim)``. The output has shape ``(B, H_cond, C)``;
    the mean-condition head uses ``H_cond = 1``.
    """
    FUSION_MODES = ('additive', 'gated_residual', 'film', 'cross_attention')

    def __init__(self, *args, context_tokens=4, context_dim=128,
                 context_fusion='additive', context_fusion_dim=128, **kwargs):
        if context_fusion not in self.FUSION_MODES:
            raise ValueError(f'Unknown task-context fusion: {context_fusion}')
        if not isinstance(context_fusion_dim, int) or isinstance(context_fusion_dim, bool) or context_fusion_dim < 1:
            raise ValueError('Task-context fusion width must be a positive integer')
        if context_fusion == 'cross_attention' and context_fusion_dim % 4:
            raise ValueError('Task-context attention width must be divisible by four')
        super().__init__(*args, **kwargs)
        self.context_tokens, self.context_dim = context_tokens, context_dim
        self.context_fusion = context_fusion
        self.context_fusion_dim = context_fusion_dim
        self.context_task = nn.Sequential(nn.LayerNorm(context_tokens * context_dim), nn.Linear(context_tokens * context_dim, self.cond_dim))
        self.context_execution = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, self.cond_dim))
        self.context_task_scale = nn.Parameter(torch.tensor(.01))
        self.context_execution_scale = nn.Parameter(torch.tensor(.01))

        if context_fusion == 'gated_residual':
            self.context_gate = nn.Sequential(nn.LayerNorm(self.cond_dim),
                nn.Linear(self.cond_dim, context_fusion_dim), nn.SiLU(),
                nn.Linear(context_fusion_dim, self.cond_dim))
            nn.init.zeros_(self.context_gate[-1].weight)
            nn.init.zeros_(self.context_gate[-1].bias)
        elif context_fusion == 'film':
            self.context_film = nn.Sequential(nn.LayerNorm(context_tokens * context_dim),
                nn.Linear(context_tokens * context_dim, context_fusion_dim), nn.SiLU(),
                nn.Linear(context_fusion_dim, 2 * self.cond_dim))
            nn.init.zeros_(self.context_film[-1].weight)
            nn.init.zeros_(self.context_film[-1].bias)
        elif context_fusion == 'cross_attention':
            self.context_query = nn.Sequential(nn.LayerNorm(self.cond_dim),
                nn.Linear(self.cond_dim, context_fusion_dim))
            self.context_key_value = nn.Sequential(nn.LayerNorm(context_dim),
                nn.Linear(context_dim, context_fusion_dim))
            self.context_attention = nn.MultiheadAttention(context_fusion_dim, 4,
                dropout=0., batch_first=True)
            self.context_attention_output = nn.Linear(context_fusion_dim, self.cond_dim)
            nn.init.zeros_(self.context_attention_output.weight)
            nn.init.zeros_(self.context_attention_output.bias)

    def conditioning_components(self, visual, states, intent_emb=None):
        """Return live, complete task, and execution terms with shape ``(B, H, C)``.

        All optional task interactions belong to the task contribution, making
        removal controls well-defined for every fusion mode.
        """
        if intent_emb is None or intent_emb.shape[-2:] != (self.context_tokens + 1, self.context_dim):
            raise ValueError("Context policy requires task tokens and one execution token")
        current = super()._build_per_step_cond(visual, states, None)
        task = self.context_task(intent_emb[:, :self.context_tokens].flatten(1))[:, None]
        execution = self.context_execution(intent_emb[:, -1])[:, None]
        task = self.context_task_scale * task
        execution = self.context_execution_scale * execution
        if self.context_fusion == 'gated_residual':
            # Unit gate at initialization; each observation can learn which task
            # features are relevant without freezing the live execution branch.
            task = task * (2 * self.context_gate(current).sigmoid())
        if self.context_fusion == 'film':
            scale, shift = self.context_film(intent_emb[:, :self.context_tokens].flatten(1)).chunk(2, -1)
            normalized = F.layer_norm(current, (self.cond_dim,))
            task = task + scale[:, None].tanh() * normalized + shift[:, None].tanh()
        elif self.context_fusion == 'cross_attention':
            query = self.context_query(current)
            task_keys = self.context_key_value(intent_emb[:, :self.context_tokens])
            read, _ = self.context_attention(query, task_keys, task_keys, need_weights=False)
            task = task + self.context_attention_output(read)
        return current, task.expand_as(current), execution.expand_as(current)

    def _build_per_step_cond(self, visual, states, intent_emb=None):
        current, task, execution = self.conditioning_components(visual, states, intent_emb)
        return current + task + execution
