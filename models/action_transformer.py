"""Transformer denoiser sharing the existing diffusion/flow policy interfaces."""
import torch
from torch import nn
from torch.nn import functional as F
from models.attention_backend import attention_backend
from models.intention_head import SinusoidalPositionalEncoding, VisualContextProjection


class ActionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in range(3)])
        self.qkv = nn.Linear(width, width * 3)
        self.self_out = nn.Linear(width, width)
        self.query = nn.Linear(width, width)
        self.cross_out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

    def attention(self, q, k, v, mask=None):
        with attention_backend(q.device):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return y.transpose(1, 2).flatten(2)

    def forward(self, x, global_condition, kv):
        mods = self.modulation(global_condition).chunk(6, -1)
        def norm(i):
            return self.norms[i](x) * (1 + mods[2*i][:, None]) + mods[2*i+1][:, None]
        q, k, v = self.qkv(norm(0)).chunk(3, -1)
        x = x + self.self_out(self.attention(*[self.split(t) for t in (q,k,v)]))
        k, v, mask = kv
        update = self.cross_out(self.attention(self.split(self.query(norm(1))), k, v, mask))
        if mask is not None:
            update = update * mask.any(-1).reshape(len(x), 1, 1).to(update.dtype)
        x = x + update
        return x + self.ff(norm(2))


class ActionTransformer(nn.Module):
    """ACI uses separate context/visual banks; legacy conditions retain their semantics.

    Action horizon is independent of observation history. The attribute name
    ``unet`` on the owning policy is retained for sampler/checkpoint integration.
    """
    def __init__(self, *, action_dim, cond_dim, time_dim, width, layers=4, heads=8,
                 aci=None):
        super().__init__()
        if heads < 1 or width < 4 or width % 4 or width % heads or layers < 2:
            raise ValueError('Transformer width must divide heads and four; at least two layers and positive heads are required')
        self.width, self.heads = width, heads
        self.input_proj = nn.Linear(action_dim, width)
        self.position = SinusoidalPositionalEncoding(width)
        self.time = nn.Linear(time_dim, width)
        self.blocks = nn.ModuleList([ActionBlock(width, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        self.aci = aci is not None
        self.visual_projection = None
        if self.aci:
            self.state_dim = aci.state_dim
            self.context_count, self.context_dim = aci.context_count, aci.context_dim
            self.visual_dim, self.visual_token_dim = aci.visual_dim, aci.visual_token_dim
            self.state = nn.Linear(self.state_dim, width)
            self.context = nn.Sequential(nn.LayerNorm(self.context_dim), nn.Linear(self.context_dim, width))
            self.slot = nn.Embedding(self.context_count, width)
            self.visual_projection = VisualContextProjection(self.visual_token_dim, aci.visual_projection.num_cameras, width, heads)
        else:
            self.condition = nn.Linear(cond_dim, width)
        self.key, self.value = nn.Linear(width, width), nn.Linear(width, width)

    def kv(self, tokens, mask=None):
        def split(t):return t.reshape(len(t), t.shape[1], self.heads, -1).transpose(1,2)
        return split(self.key(tokens)), split(self.value(tokens)), mask

    def configure_visual_context(self, token_dim, num_cameras):
        self.visual_projection = VisualContextProjection(token_dim, num_cameras, self.width, self.heads)

    def configure_visual_attention(self, visual_dim, token_dim):
        self.inline_visual = (visual_dim, token_dim)
        self.configure_visual_context(token_dim, visual_dim // (256 * token_dim))

    def unpack(self, cond):
        sizes = [self.state_dim, self.context_count*self.context_dim, self.visual_dim, self.context_count]
        if cond.ndim != 3 or cond.shape[1] != 1 or cond.shape[2] != sum(sizes):
            raise ValueError('Invalid packed ACI condition')
        return cond[:,0].split(sizes, -1)

    def prepare_visual_context(self, context):
        if self.aci:
            _, tokens, visual, present = self.unpack(context)
            tokens = self.context(tokens.reshape(len(tokens), self.context_count, self.context_dim)) + self.slot.weight[None]
            ck = self.kv(tokens, present[:,None,None].bool())
            vk = (*self.visual_projection(visual.reshape(len(visual), -1, self.visual_token_dim)), None)
            return ck[0], ck[1], vk[0], vk[1], ck[2]
        return self.visual_projection(context)

    def forward(self, x_t, cond, t_emb, visual_context=None, visual_kv=None):
        x = self.input_proj(x_t)
        x = x + self.position(torch.arange(x.shape[1], device=x.device).float()).to(x.dtype)[None]
        if self.aci:
            state, _, _, _ = self.unpack(cond)
            global_condition = self.time(t_emb) + self.state(state)
            ck, cv, vk, vv, mask = visual_kv if visual_kv is not None else self.prepare_visual_context(cond)
            banks = [(ck, cv, mask), (vk, vv, None)]
        else:
            tokens = self.condition(cond)
            tokens = tokens + self.position(torch.arange(tokens.shape[1], device=x.device).float()).to(tokens.dtype)[None]
            global_condition = self.time(t_emb)
            banks = [self.kv(tokens)]
            if visual_kv is None and visual_context is not None:
                visual_kv = self.prepare_visual_context(visual_context)
            if visual_kv is None and hasattr(self, 'inline_visual'):
                dim, token_dim = self.inline_visual
                visual_kv = self.prepare_visual_context(cond[:, -1, :dim].reshape(len(cond), -1, token_dim))
            if visual_kv is not None:banks.append((*visual_kv, None))
        for i, block in enumerate(self.blocks):
            x = block(x, global_condition, banks[i % len(banks)])
        return self.output(x)
