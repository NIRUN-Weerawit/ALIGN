"""Alternating context/observation attention with state-and-time-only FiLM.

Conditions are transported as a tensor to preserve the production loss, masking,
chunking and sampling interfaces. Packed visual/context values are unpacked into
individual tokens before attention; they NEVER enter global FiLM conditioning.
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.attention_backend import attention_backend
from models.intention_head import (DiffusionPolicyHead, FlowMatchingPolicyHead,
    DiffusionPolicyUNet1D, VisualContextProjection, SinusoidalPositionalEncoding)


class ActionContextAttention(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, width)
        self.position = SinusoidalPositionalEncoding(width)
        self.output = nn.Linear(width, channels)
        self.width = width
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, key, value, mask=None):
        batch, _, steps = x.shape
        q = self.query(self.norm(x.transpose(1, 2)))
        q = q + self.position(torch.arange(steps, device=x.device).float()).to(q.dtype)[None]
        q = q.reshape(batch, steps, 4, self.width // 4).transpose(1, 2)
        with attention_backend(x.device):
            read = F.scaled_dot_product_attention(q, key, value, attn_mask=mask, dropout_p=0.)
        read = read.transpose(1, 2).reshape(batch, steps, self.width)
        update = self.output(read).transpose(1, 2)
        if mask is not None:
            # Suppress the output bias as well when ALL context tokens are removed.
            update = update * mask.any(-1).reshape(batch, 1, 1).to(update.dtype)
        return x + update


class ACIUNet1D(DiffusionPolicyUNet1D):
    """Existing temporal U-Net, with C/O/C/O attention after FiLM blocks."""
    def __init__(self, *, state_dim, visual_dim, visual_token_dim, num_cameras,
                 context_tokens, context_dim, attention_width, temporal_tokens=0, **kwargs):
        super().__init__(cond_dim=state_dim, history_size=1, **kwargs)
        self.state_dim, self.visual_dim = state_dim, visual_dim
        self.context_count, self.context_dim = context_tokens + 1 + temporal_tokens, context_dim
        self.context_width = self.context_count * context_dim
        self.packed_dim = state_dim + self.context_width + visual_dim + self.context_count
        self.visual_token_dim = visual_token_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_slot = nn.Embedding(self.context_count, attention_width)
        self.context_projection = nn.Linear(context_dim, attention_width)
        self.context_key = nn.Linear(attention_width, attention_width)
        self.context_value = nn.Linear(attention_width, attention_width)
        self.visual_projection = VisualContextProjection(visual_token_dim, num_cameras, attention_width)
        self.attention_width = attention_width
        self.aci_attention = nn.ModuleDict({
            'context_encoder': ActionContextAttention(self.hidden_dim * 2, attention_width),
            'observation_bottleneck': ActionContextAttention(self.hidden_dim * 4, attention_width),
            'context_decoder': ActionContextAttention(self.hidden_dim, attention_width),
            'observation_output': ActionContextAttention(self.hidden_dim, attention_width),
        })

    def unpack(self, cond):
        if cond.ndim != 3 or cond.shape[1:] != (1, self.packed_dim):
            raise ValueError(f'ACI requires packed condition B,1,{self.packed_dim}')
        state, context, visual, present = cond[:, 0].split(
            [self.state_dim, self.context_width, self.visual_dim, self.context_count], -1)
        return state, context.reshape(-1, self.context_count, self.context_dim), visual, present

    def prepare_visual_context(self, cond):
        """Cache BOTH context and visual K/V once for all sampler steps/noise draws."""
        _, context, visual, present = self.unpack(cond)
        tokens = self.context_projection(self.context_norm(context)) + self.context_slot.weight[None]
        batch, count, width = tokens.shape
        def split(value):
            return value.reshape(batch, count, 4, width // 4).transpose(1, 2)
        ck, cv = split(self.context_key(tokens)), split(self.context_value(tokens))
        vk, vv = self.visual_projection(visual.reshape(batch, -1, self.visual_token_dim))
        return ck, cv, vk, vv, present[:, None, None].bool()

    def forward(self, x_t, cond, t_emb, visual_context=None, visual_kv=None):
        state, _, _, _ = self.unpack(cond)
        ck, cv, vk, vv, mask = (self.prepare_visual_context(cond) if visual_kv is None else visual_kv)
        x = self.input_proj(x_t.transpose(1, 2))
        skip0 = x
        x = self.enc1(self.down1(x), state, t_emb)
        skip1 = x
        x = self.enc2(x, state, t_emb)
        x = self.aci_attention['context_encoder'](x, ck, cv, mask)
        x = self.enc3(self.down2(x), state, t_emb)
        skip2 = x
        x = self.enc4(x, state, t_emb)
        x = self.bottleneck(x, state, t_emb)
        x = self.aci_attention['observation_bottleneck'](x, vk, vv)
        x = self.up2(x, target_len=skip2.shape[-1])
        x = self.dec2(torch.cat([x, skip2], 1), state, t_emb)
        x = self.up1(x, target_len=skip1.shape[-1])
        x = self.dec1(torch.cat([x, skip1], 1), state, t_emb)
        x = self.aci_attention['context_decoder'](x, ck, cv, mask)
        if x.shape[-1] != skip0.shape[-1]:
            x = F.interpolate(x, size=skip0.shape[-1], mode='linear')
        x = self.dec0(torch.cat([x, skip0], 1), state, t_emb)
        x = self.aci_attention['observation_output'](x, vk, vv)
        return self.output_proj(x).transpose(1, 2)


class ACIConditioning:
    def __init__(self, *, cond_dim, state_dim, visual_token_dim, num_cameras,
                 context_tokens=4, context_dim=128, context_fusion='aci',
                 context_fusion_dim=128, history_size=1, temporal_tokens=0, **kwargs):
        if context_fusion != 'aci' or history_size != 1:
            raise ValueError('ACI requires aci fusion and a single current observation')
        if context_fusion_dim < 4 or context_fusion_dim % 4:
            raise ValueError('ACI attention width must be positive and divisible by four')
        visual_dim = cond_dim - state_dim
        if visual_dim != num_cameras * 256 * visual_token_dim:
            raise ValueError('ACI requires camera-major 16x16 visual patch grids')
        context_count = context_tokens + 1 + temporal_tokens
        packed_dim = state_dim + visual_dim + context_count * (context_dim + 1)
        # The inherited scheduler/loss/sampler are unchanged. Only the denoiser
        # and condition construction differ from the existing generative heads.
        super().__init__(cond_dim=state_dim, history_size=1, **kwargs)
        self.cond_dim = packed_dim
        self.state_dim, self.visual_dim = state_dim, visual_dim
        self.context_tokens, self.context_dim = context_tokens, context_dim
        self.context_count, self.temporal_tokens = context_count, temporal_tokens
        self.context_fusion, self.context_fusion_dim = context_fusion, context_fusion_dim
        self.unet = ACIUNet1D(state_dim=state_dim, visual_dim=visual_dim,
            visual_token_dim=visual_token_dim, num_cameras=num_cameras,
            context_tokens=context_tokens, context_dim=context_dim, temporal_tokens=temporal_tokens,
            attention_width=context_fusion_dim, action_dim=self.action_dim,
            time_dim=self.time_dim, hidden_dim=kwargs.get('hidden_dim', 128))

    def _build_per_step_cond(self, visual, states, intent_emb=None):
        if visual.ndim != 3 or visual.shape[1:] != (1, self.visual_dim):
            raise ValueError('ACI expects one current visual observation')
        if states.shape != (len(visual), 1, self.state_dim):
            raise ValueError('ACI expects one current encoded robot state')
        if intent_emb is None or intent_emb.shape != (len(visual), self.context_count, self.context_dim):
            raise ValueError('ACI requires task tokens and one execution token')
        present = states.new_ones(len(states), 1, self.context_count)
        return torch.cat([states, intent_emb.flatten(1)[:, None], visual, present], -1)

    def remove_context(self, cond, *, task=False, execution=False, temporal=False):
        """True attention masking, including slot embeddings and output biases."""
        cond = cond.clone()
        if task:
            cond[..., -self.context_count:-self.context_count + self.context_tokens] = 0
        if execution:
            cond[..., -self.context_count + self.context_tokens] = 0
        if temporal and self.temporal_tokens:
            cond[..., -self.temporal_tokens:] = 0
        return cond

    def loss(self, actions_target, cond, dim_weights=None, sample_mask=None, visual_context=None):
        if visual_context is not None:
            raise ValueError('ACI packs observation/context tokens automatically; do not supply visual_context')
        return super().loss(actions_target, cond, dim_weights=dim_weights,
            sample_mask=sample_mask, visual_context=cond)

    @torch.no_grad()
    def sample(self, cond, num_steps=None, visual_context=None):
        if visual_context is not None:
            raise ValueError('ACI packs observation/context tokens automatically; do not supply visual_context')
        return super().sample(cond, num_steps=num_steps, visual_context=cond)


class ACIContextDiffusionPolicyHead(ACIConditioning, DiffusionPolicyHead):
    """DDPM/DDIM head with alternating context and observation attention."""


class ACIContextFlowMatchingPolicyHead(ACIConditioning, FlowMatchingPolicyHead):
    """Flow head with identical ACI conditioning and existing Euler/Heun solvers."""
