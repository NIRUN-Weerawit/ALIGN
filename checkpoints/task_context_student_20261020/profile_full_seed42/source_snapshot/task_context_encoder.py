"""Causal observation-prefix belief and execution state in a frozen teacher space.

Only per-observation summaries enter this module. Completed-episode tokens,
groups, future targets, and episode duration are supervision, never features.
The CPU reference implements the same Mamba1 equations for contract tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def reference_step(block, value, conv, ssm):
    """Differentiable Mamba1 reference; shared weights with the CUDA implementation."""
    x, z = block.in_proj(value).chunk(2, -1)
    conv = torch.cat([conv[..., 1:], x[..., None]], -1)
    x = F.silu((conv * block.conv1d.weight[:, 0]).sum(-1) + block.conv1d.bias)
    dt, b, c = torch.split(block.x_proj(x), [block.dt_rank, block.d_state, block.d_state], -1)
    dt = F.softplus(F.linear(dt, block.dt_proj.weight, block.dt_proj.bias)).float()
    a = -block.A_log.float().exp()
    ssm = ssm * torch.exp(dt[..., None] * a) + dt[..., None] * b[:, None].float() * x[..., None].float()
    y = (ssm.to(x.dtype) * c[:, None]).sum(-1) + block.D.to(x.dtype) * x
    return block.out_proj(y * F.silu(z)), conv, ssm


@dataclass
class ContextStreamState:
    layers: list
    # Used only by bounded-history controls. Full-prefix deployment stores no frames.
    recent: list
    steps: int = 0


class TaskContextEncoder(nn.Module):
    def __init__(self, width=256, tokens=4, token_dim=128, modes=4, layers=2,
                 d_state=16, d_conv=4, expand=2, execution_dim=128,
                 visual_width=1536, horizons=(16, 64), history="full", window=16):
        super().__init__()
        from mamba_ssm import Mamba
        if history not in ("full", "current", "short"):
            raise ValueError("Unknown history control")
        if min(width, tokens, token_dim, modes, layers, d_state, d_conv, expand, window) < 1:
            raise ValueError("Context dimensions must be positive")
        if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
            raise ValueError("Future horizons must be distinct positive offsets")
        self.config = dict(width=width, tokens=tokens, token_dim=token_dim, modes=modes,
            layers=layers, d_state=d_state, d_conv=d_conv, expand=expand,
            execution_dim=execution_dim, visual_width=visual_width,
            horizons=list(horizons), history=history, window=window)
        self.blocks = nn.ModuleList([Mamba(d_model=width, d_state=d_state, d_conv=d_conv,
            expand=expand, layer_idx=i) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        flat = tokens * token_dim
        self.belief = nn.Sequential(nn.Linear(width, 512), nn.SiLU())
        self.logits = nn.Linear(512, modes)
        self.means = nn.Linear(512, modes * flat)
        self.log_stds = nn.Linear(512, modes * flat)
        nn.init.zeros_(self.log_stds.weight)
        nn.init.zeros_(self.log_stds.bias)
        self.execution = nn.Sequential(nn.Linear(width, execution_dim), nn.LayerNorm(execution_dim))
        self.future = nn.Sequential(nn.Linear(execution_dim, 256), nn.SiLU(),
            nn.Linear(256, len(horizons) * (visual_width + 10)))
        self.register_buffer("latent_mean", torch.zeros(tokens, token_dim))
        self.register_buffer("latent_std", torch.ones(tokens, token_dim))

    def _sequence(self, x):
        for norm, block in zip(self.norms, self.blocks):
            normalized = norm(x)
            if x.is_cuda:
                y = block(normalized)
            else:
                conv, ssm = block.allocate_inference_cache(len(x), x.shape[1])
                ys = []
                for t in range(x.shape[1]):
                    y, conv, ssm = reference_step(block, normalized[:, t], conv, ssm)
                    ys.append(y)
                y = torch.stack(ys, 1)
            x = x + y
        return self.final_norm(x)

    def readout(self, hidden):
        cfg = self.config
        belief = self.belief(hidden)
        shape = (*hidden.shape[:-1], cfg["modes"], cfg["tokens"], cfg["token_dim"])
        means = self.means(belief).reshape(shape)
        log_std = self.log_stds(belief).reshape(shape).float().clamp(-3, 1)
        logits = self.logits(belief).float()
        mode = logits.argmax(-1)
        native = means * self.latent_std + self.latent_mean
        selected = native.gather(-3, mode[..., None, None, None].expand(*mode.shape, 1,
            cfg["tokens"], cfg["token_dim"])).squeeze(-3)
        execution = self.execution(hidden)
        future = self.future(execution).reshape(*hidden.shape[:-1], len(cfg["horizons"]), cfg["visual_width"] + 10)
        return dict(logits=logits, means=means, log_stds=log_std, tokens=selected,
            mode=mode, execution_token=execution.unsqueeze(-2), future=future)

    def forward(self, summaries, lengths, anchors):
        """Read chosen prefixes; lengths are padding masks, not learned inputs."""
        if summaries.ndim != 3 or anchors.ndim != 2 or lengths.shape != summaries.shape[:1]:
            raise ValueError("Expected summaries B,T,D; lengths B; anchors B,A")
        if (lengths < 1).any() or (lengths > summaries.shape[1]).any():
            raise ValueError("Invalid real sequence length")
        if (anchors < 0).any() or (anchors >= lengths[:, None]).any():
            raise ValueError("Anchor outside real observation prefix")
        cfg = self.config
        bi = torch.arange(len(summaries), device=summaries.device)[:, None]
        if cfg["history"] == "full":
            valid = torch.arange(summaries.shape[1], device=summaries.device)[None] < lengths[:, None]
            x = torch.where(valid[..., None], summaries, torch.zeros_like(summaries))
            hidden = self._sequence(x)[bi, anchors]
        else:
            # Gather BEFORE any projection: this control cannot access older frames.
            window = 1 if cfg["history"] == "current" else cfg["window"]
            offsets = torch.arange(window, device=summaries.device)
            start = (anchors - window + 1).clamp_min(0)
            index = start[..., None] + offsets
            valid = index <= anchors[..., None]
            index = torch.minimum(index, anchors[..., None])
            x = summaries[bi[..., None], index]
            x = torch.where(valid[..., None], x, torch.zeros_like(x))
            hidden = self._sequence(x.flatten(0, 1)).reshape(len(x), x.shape[1], window, -1)
            local = (anchors - start)[..., None, None].expand(-1, -1, 1, cfg["width"])
            hidden = hidden.gather(2, local).squeeze(2)
        return self.readout(hidden)

    def new_stream(self, batch_size=1):
        return ContextStreamState([block.allocate_inference_cache(batch_size, 1) for block in self.blocks], [])

    @torch.no_grad()
    def step(self, summary, state):
        if summary.ndim != 2:
            raise ValueError("One real B,D observation is required")
        if self.config["history"] != "full":
            window = 1 if self.config["history"] == "current" else self.config["window"]
            state.recent = (state.recent + [summary.clone()])[-window:]
            x = torch.stack(state.recent, 1)
            output = self.readout(self._sequence(x)[:, -1])
        else:
            x = summary
            for i, (norm, block) in enumerate(zip(self.norms, self.blocks)):
                value = norm(x)
                conv, ssm = state.layers[i]
                if x.is_cuda:
                    y, conv, ssm = block.step(value[:, None], conv, ssm)
                    y = y[:, 0]
                else:
                    y, conv, ssm = reference_step(block, value, conv, ssm)
                state.layers[i] = (conv, ssm)
                x = x + y
            output = self.readout(self.final_norm(x))
        state.steps += 1
        return output


def mixture_log_prob(output, target):
    """FP32 component joint log density in train-normalized token coordinates."""
    mu, log_std = output["means"].float(), output["log_stds"].float()
    target = target.float().unsqueeze(-3)
    component = (-.5 * ((target - mu) * (-log_std).exp()).square() - log_std - .5 * math.log(2 * math.pi)).sum((-1, -2))
    return component + output["logits"].float().log_softmax(-1)


def belief_loss(model, output, native_target):
    target = (native_target - model.latent_mean) / model.latent_std
    if target.ndim == 3:
        target = target[:, None].expand(-1, output["means"].shape[1], -1, -1)
    joint = mixture_log_prob(output, target)
    return -torch.logsumexp(joint, -1).mean() / (model.config["tokens"] * model.config["token_dim"])


def execution_loss(model, output, batch):
    """Masked +16/+64 observed CLS/geometric changes; no repeated end padding."""
    anchors = batch["anchors"]
    bi = torch.arange(len(anchors), device=anchors.device)[:, None]
    total = output["future"].float().sum() * 0
    metrics = {}
    for hi, horizon in enumerate(model.config["horizons"]):
        valid = anchors + horizon < batch["lengths"][:, None]
        n = int(valid.sum())
        metrics[f"pairs_h{horizon}"] = n
        if not n:
            continue
        # Clamp only for safe gather. Unavailable examples are excluded afterward.
        index = torch.minimum(anchors + horizon, batch["lengths"][:, None] - 1)
        visual = batch["cls"][bi, index] - batch["cls"][bi, anchors]
        state = batch["states"][bi, index] - batch["states"][bi, anchors]
        pred_visual, pred_state = output["future"][:, :, hi].float().split([model.config["visual_width"], 10], -1)
        target_visual, target_state = visual[valid].detach().float(), state[valid].detach().float()
        visual_loss = F.smooth_l1_loss(pred_visual[valid], target_visual, reduction="none").mean(-1)
        state_loss = F.smooth_l1_loss(pred_state[valid], target_state, reduction="none").mean(-1)
        total = total + (visual_loss + state_loss).sum() / 2
        metrics[f"loss_sum_h{horizon}"] = (visual_loss + state_loss).detach().sum() / 2
        metrics[f"persistence_sum_h{horizon}"] = (F.smooth_l1_loss(torch.zeros_like(target_visual), target_visual, reduction="none").mean(-1)
            + F.smooth_l1_loss(torch.zeros_like(target_state), target_state, reduction="none").mean(-1)).sum() / 2
    return total / (anchors.numel() * len(model.config["horizons"])), metrics
