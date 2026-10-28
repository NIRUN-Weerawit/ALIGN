"""Policy-loss-trained Mamba memory, independent of the frozen task observer."""
from dataclasses import dataclass

import torch
from torch import nn

from models.task_context_encoder import reference_step


@dataclass
class PolicyTemporalState:
    layers: list
    steps: int = 0


class PolicyTemporalEncoder(nn.Module):
    def __init__(self, input_dim, token_dim, width=128, layers=1):
        super().__init__()
        from mamba_ssm import Mamba
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
               for x in (input_dim, token_dim, width, layers)):
            raise ValueError('Policy temporal dimensions/layers must be positive integers')
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.blocks = nn.ModuleList([Mamba(d_model=width, d_state=16, d_conv=4,
            expand=2, layer_idx=i) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, token_dim))

    def forward(self, summaries, lengths):
        if summaries.ndim != 3 or lengths.shape != summaries.shape[:1]:
            raise ValueError('Policy temporal encoder requires B,T,D summaries and B lengths')
        if (lengths < 1).any() or (lengths > summaries.shape[1]).any():
            raise ValueError('Invalid policy temporal sequence lengths')
        valid = torch.arange(summaries.shape[1], device=summaries.device)[None] < lengths[:, None]
        # Summary/observer are frozen; padded NaNs must not reach ANY projection.
        summaries = torch.where(valid[..., None], summaries.detach(), 0.)
        with torch.autocast(device_type=summaries.device.type, enabled=False):
            x = self.input(summaries.float())
            for norm, block in zip(self.norms, self.blocks):
                value = norm(x)
                if x.is_cuda:
                    y = block(value)
                else:
                    conv, ssm = block.allocate_inference_cache(len(x), x.shape[1])
                    values = []
                    for t in range(x.shape[1]):
                        y, conv, ssm = reference_step(block, value[:, t], conv, ssm)
                        values.append(y)
                    y = torch.stack(values, 1)
                x = x + y
            tokens = self.output(x)
        return torch.where(valid[..., None], tokens, 0.).unsqueeze(-2)

    def new_stream(self, batch_size):
        return PolicyTemporalState([block.allocate_inference_cache(batch_size, 1) for block in self.blocks])

    @torch.no_grad()
    def step(self, summary, state):
        with torch.autocast(device_type=summary.device.type, enabled=False):
            x = self.input(summary.float())
            for i, (norm, block) in enumerate(zip(self.norms, self.blocks)):
                value = norm(x)
                conv, ssm = state.layers[i]
                if x.is_cuda:
                    y, conv, ssm = block.step(value[:, None], conv, ssm)
                    y = y[:, 0]
                else:
                    y, conv, ssm = reference_step(block, value, conv, ssm)
                state.layers[i] = conv, ssm
                x = x + y
            token = self.output(x).unsqueeze(-2)
        state.steps += 1
        return token
