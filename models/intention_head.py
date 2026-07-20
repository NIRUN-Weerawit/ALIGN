#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intention heads: K past states + intent tokens → K future actions.

V4: All heads consume intent tokens (B, N, intent_dim) instead of
h_current (B, mamba_output_dim). Text conditioning removed.

Two head architectures:
  - IntentionTransformerHead: standard transformer (K+N tokens)
  - MambaActionHead: Mamba recurrent head (O(1) inference)
  - DiffusionPolicyHead: 1D U-Net (Chi et al. 2023)

All consume:
  - z_v_pooled_window: (B, K, pool_out_dim) — K past pooled visions
  - z_t_window:        (B, K, state_dim)    — K past states
  - intent_emb:        (B, N, intent_dim)   — N intent tokens (V4)

Output:
  - actions: (B, K, action_dim) — K future actions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ================================================================
# Intention Transformer Head
# ================================================================

class IntentionTransformerHead(nn.Module):
    """Transformer head for action prediction from intention state.

    Prepends N intent tokens as context tokens, then K per-timestep
    tokens from concat[z_v_pooled, z_t]. Transformer encoder over
    (K+N) tokens. Output K actions.

    Args:
        pool_out_dim:     dim of z_v_pooled (V * vision_dim)
        state_dim:        robot state dim (e.g., 256)
        intent_dim:       intent token dim (e.g., 512). Set to 0 to disable.
        action_dim:       action output dim (e.g., 7)
        chunk_size:       K — number of past steps / future actions
        num_intent_tokens: N — number of intent tokens (default 2)
        d_model:          internal transformer dim
        nhead:            attention heads
        num_layers:       transformer layers
        dim_feedforward:  FFN dim
        dropout:          dropout
    """
    def __init__(self, pool_out_dim: int = 256, state_dim: int = 256,
                 intent_dim: int = 512, action_dim: int = 6,
                 chunk_size: int = 10, num_intent_tokens: int = 2,
                 d_model: int = 384, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.pool_out_dim = pool_out_dim
        self.state_dim = state_dim
        self.d_model = d_model
        self.use_intent = intent_dim > 0
        self.num_intent_tokens = num_intent_tokens
        self.intent_dim = intent_dim

        # Per-timestep projection: concat[z_v_pooled, z_t] → d_model
        per_step_in_dim = pool_out_dim + state_dim
        self.input_proj = nn.Linear(per_step_in_dim, d_model)

        # Intent token projection (prepended as context tokens)
        if self.use_intent:
            self.intent_proj = nn.Linear(intent_dim, d_model)
        else:
            self.intent_proj = None

        # Positional encoding (learned) — sized for K+N tokens
        self.pos_emb = nn.Parameter(
            torch.randn(chunk_size + num_intent_tokens, d_model) * 0.02
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
            dropout=dropout, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output: K actions (one per timestep)
        self.output_proj = nn.Linear(d_model, action_dim)

        # Initialize output near zero (identity-ish start)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_v_pooled_window: torch.Tensor,
                z_t_window: torch.Tensor,
                intent_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_v_pooled_window: (B, K, pool_out_dim) — K past pooled visions
            z_t_window:        (B, K, state_dim)    — K past states
            intent_emb:        (B, N, intent_dim)   — N intent tokens (or None)
        Returns:
            actions: (B, K, action_dim) — K future actions
        """
        B, K = z_v_pooled_window.shape[:2]
        assert K == self.chunk_size, (
            f"Window size {K} doesn't match chunk_size {self.chunk_size}"
        )

        # Per-timestep input: concat[z_v_pooled, z_t]
        per_step_in = torch.cat([z_v_pooled_window, z_t_window], dim=-1)
        x = self.input_proj(per_step_in)  # (B, K, d_model)

        # Prepend intent tokens as context
        if self.use_intent and intent_emb is not None:
            intent_tokens = self.intent_proj(intent_emb)  # (B, N, d_model)
            x = torch.cat([intent_tokens, x], dim=1)  # (B, K+N, d_model)

        # Add positional encoding
        x = x + self.pos_emb[:x.size(1)].unsqueeze(0)

        # Transformer (use math backend to avoid cuDNN issues with Mamba)
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                x = self.transformer(x)
        except ImportError:
            x = self.transformer(x)

        # Drop intent tokens; keep K timestep tokens
        if self.use_intent and intent_emb is not None:
            N = intent_emb.shape[1]
            x = x[:, N:]  # (B, K, d_model)

        # Output: K actions
        actions = self.output_proj(x)  # (B, K, action_dim)
        return actions


# ================================================================
# Mamba Action Head
# ================================================================

try:
    from mamba_ssm import Mamba as _MambaSSM
    _HAS_MAMBA_HEAD = True
except ImportError:
    _HAS_MAMBA_HEAD = False


class MambaActionHead(nn.Module):
    """Mamba-based action head: K past (z_v_pooled, z_t) + intent → K future actions.

    Architecture:
      input[t] = concat[z_v_pooled[t], z_t[t], intent_pooled]
      mamba_seq = Mamba(input_seq)               # (B, K, hidden_dim)
      actions = output_proj(mamba_seq)            # (B, K, action_dim)

    Intent tokens are pooled to a single vector and repeated per-step.
    """
    def __init__(self, pool_out_dim: int = 256, state_dim: int = 256,
                 intent_dim: int = 512, action_dim: int = 6,
                 chunk_size: int = 10, mamba_d_state: int = 16,
                 mamba_d_conv: int = 4, mamba_expand: int = 2,
                 use_intent: bool = True):
        super().__init__()
        if not _HAS_MAMBA_HEAD:
            raise ImportError(
                "mamba_ssm not installed. Run: pip install mamba-ssm causal-conv1d"
            )

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.intent_dim = intent_dim
        self.use_intent = use_intent and intent_dim > 0

        # Input dim: per-step = pool + state
        # Intent tokens are projected to this same dim and prepended to the
        # K-step sequence (Mamba processes the full (B, N+K, D) sequence).
        per_step_in = pool_out_dim + state_dim
        self.input_dim = per_step_in
        if self.use_intent:
            # Project N intent tokens (B, N, intent_dim) to per_step_in_dim
            self.intent_proj = nn.Linear(intent_dim, per_step_in)

        # Mamba block
        self.mamba = _MambaSSM(
            d_model=self.input_dim,
            d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
        )
        # Output projection
        self.output_proj = nn.Linear(self.input_dim, action_dim)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_v_pooled_window: torch.Tensor,
                z_t_window: torch.Tensor,
                intent_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_v_pooled_window: (B, K, pool_out_dim) — K past pooled visions
            z_t_window:        (B, K, state_dim)    — K past states
            intent_emb:        (B, N, intent_dim)   — N intent tokens (or None)
        Returns:
            actions: (B, K, action_dim) — K future actions
        """
        B, K = z_v_pooled_window.shape[:2]
        assert K == self.chunk_size, (
            f"Window size {K} doesn't match chunk_size {self.chunk_size}"
        )

        # Per-timestep input: concat[z_v_pooled, z_t]
        per_step_in = torch.cat([z_v_pooled_window, z_t_window], dim=-1)
        # (B, K, per_step_in_dim) where per_step_in_dim = pool_out_dim + state_dim

        # For V4: prepend N intent tokens as context (similar to transformer head).
        # Mamba processes the full (B, K+N, per_step_in_dim) sequence, with
        # intent tokens as context. The output is sliced to (B, K, ...) for
        # the per-step actions.
        if self.use_intent and intent_emb is not None:
            # Project intent tokens to per_step_in_dim
            intent_tokens = self.intent_proj(intent_emb)  # (B, N, per_step_in_dim)
            # Pad per_step_in: needs to match intent_tokens dim. Easiest: project
            # intent to per_step_in dim, then prepend.
            # Actually use the same per_step_in dim for both.
            per_step_in = torch.cat([intent_tokens, per_step_in], dim=1)  # (B, N+K, D)

        # Mamba: (B, N+K, per_step_in_dim) -> (B, N+K, per_step_in_dim)
        out = self.mamba(per_step_in)
        # Slice off the first N tokens (intent context) — we only want per-step actions
        if self.use_intent and intent_emb is not None:
            N = intent_emb.shape[1]
            out = out[:, N:, :]  # (B, K, per_step_in_dim)

        # Output projection: per-timestep action
        actions = self.output_proj(out)  # (B, K, action_dim)
        return actions


# ================================================================
# Diffusion Policy Head (Chi et al. 2023) — 1D Conditional U-Net
# ================================================================

class DiffusionPolicyUNet1D(nn.Module):
    """1D Conditional U-Net for Diffusion Policy (Chi et al. 2023).

    Architecture:
        Input:  x_t (B, K, action_dim) + cond (B, K, cond_dim) → concat
                → (B, K, action_dim + cond_dim) → permute to (B, C, K)
        Encoder: Conv1dBlock(in, h) → Down → Conv1dBlock(h, 2h) → Down → Conv1dBlock(2h, 4h)
        Bottleneck: Conv1dBlock(4h, 4h)
        Decoder: Upsample + Conv1dBlock(4h + 4h, 2h)
                  + Upsample + Conv1dBlock(2h + 2h, h)
                  + Conv1dBlock(h + h, h)
        Output: Conv1d(h, action_dim)  →  permute back to (B, K, action_dim)

    All conv blocks are FiLM-conditioned on (cond_global, t_emb).
    """
    def __init__(self, action_dim: int, cond_dim: int, time_dim: int = 64,
                 hidden_dim: int = 128, n_groups: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv1d(action_dim + cond_dim, hidden_dim,
                                    kernel_size=3, padding=1)

        # Encoder
        self.down1 = _Downsample1d(hidden_dim)
        self.enc1 = _Conv1dBlock(hidden_dim, hidden_dim, cond_dim, time_dim, n_groups)
        self.enc2 = _Conv1dBlock(hidden_dim, hidden_dim * 2, cond_dim, time_dim, n_groups)

        self.down2 = _Downsample1d(hidden_dim * 2)
        self.enc3 = _Conv1dBlock(hidden_dim * 2, hidden_dim * 2, cond_dim, time_dim, n_groups)
        self.enc4 = _Conv1dBlock(hidden_dim * 2, hidden_dim * 4, cond_dim, time_dim, n_groups)

        # Bottleneck
        self.bottleneck = _Conv1dBlock(hidden_dim * 4, hidden_dim * 4,
                                        cond_dim, time_dim, n_groups)

        # Decoder with skip connections
        self.up2 = _Upsample1d(hidden_dim * 4)
        self.dec2 = _Conv1dBlock(hidden_dim * 4 + hidden_dim * 2,
                                hidden_dim * 2, cond_dim, time_dim, n_groups)

        self.up1 = _Upsample1d(hidden_dim * 2)
        self.dec1 = _Conv1dBlock(hidden_dim * 2 + hidden_dim,
                                hidden_dim, cond_dim, time_dim, n_groups)

        self.dec0 = _Conv1dBlock(hidden_dim + hidden_dim,
                                hidden_dim, cond_dim, time_dim, n_groups)

        self.output_proj = nn.Conv1d(hidden_dim, action_dim, kernel_size=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        B, K, _ = x_t.shape
        cond_global = cond.mean(dim=1)  # (B, cond_dim)

        x = torch.cat([x_t, cond], dim=-1)              # (B, K, action+cond)
        x = x.permute(0, 2, 1)                          # (B, C, K)
        x = self.input_proj(x)                          # (B, hidden, K)

        skip0 = x

        x = self.down1(x)
        x = self.enc1(x, cond_global, t_emb)
        skip1 = x

        x = self.enc2(x, cond_global, t_emb)

        x = self.down2(x)
        x = self.enc3(x, cond_global, t_emb)
        skip2 = x

        x = self.enc4(x, cond_global, t_emb)

        x = self.bottleneck(x, cond_global, t_emb)

        x = self.up2(x, target_len=skip2.shape[-1])
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x, cond_global, t_emb)

        x = self.up1(x, target_len=skip1.shape[-1])
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, cond_global, t_emb)

        if x.shape[-1] != skip0.shape[-1]:
            x = nn.functional.interpolate(x, size=skip0.shape[-1], mode="linear")
        x = torch.cat([x, skip0], dim=1)
        x = self.dec0(x, cond_global, t_emb)

        x = self.output_proj(x)
        x = x.permute(0, 2, 1)
        return x


class _Conv1dBlock(nn.Module):
    """Conv1d + GroupNorm + SiLU with FiLM conditioning."""
    def __init__(self, in_channels, out_channels, cond_dim, time_dim, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
        )
        # FiLM: scale and shift from (cond + time)
        film_dim = cond_dim + time_dim
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(film_dim, out_channels * 2),
        )

    def forward(self, x, cond_global, t_emb):
        h = self.block(x)
        # FiLM conditioning
        film_input = torch.cat([cond_global, t_emb], dim=-1)  # (B, cond+time)
        film_params = self.film(film_input).unsqueeze(-1)      # (B, 2*C, 1)
        scale, shift = film_params.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        return h


class _Downsample1d(nn.Module):
    """Average pooling downsample by 2."""
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class _Upsample1d(nn.Module):
    """Nearest neighbor upsample to target length."""
    def __init__(self, dim):
        super().__init__()

    def forward(self, x, target_len):
        return nn.functional.interpolate(x, size=target_len, mode="nearest")


class DiffusionPolicyHead(nn.Module):
    """Diffusion Policy head (Chi et al. 2023) with 1D Conditional U-Net.

    Conditioned on intent tokens (pooled) + z_v_pooled + z_t.
    No text conditioning.

    Args:
        cond_dim:          per-step condition dim (z_v + z_t + intent)
        action_dim:        output dim (default 6)
        hidden_dim:        U-Net base channels (default 128)
        num_inference_steps: DDIM denoising steps (default 10)
        time_dim:          sinusoidal time embedding dim (default 64)
        chunk_size:        K — number of past steps / future actions
    """
    def __init__(self, cond_dim: int = 768, action_dim: int = 6,
                 hidden_dim: int = 128, num_inference_steps: int = 10,
                 time_dim: int = 64, chunk_size: int = 10):
        super().__init__()
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.time_dim = time_dim
        self.chunk_size = chunk_size
        self.cond_dim = cond_dim

        # Sinusoidal time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # DDPM noise schedule (cosine)
        T_steps = num_inference_steps
        s = 0.008
        t_vals = torch.arange(T_steps + 1, dtype=torch.float64)
        theta = torch.tensor(math.pi / 2, dtype=torch.float64) * (
            t_vals / T_steps + s) / (1 + s)
        alpha_bar = torch.cos(theta).pow(2)
        sigma = torch.sqrt(1 - alpha_bar)
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("sigma", sigma.float())

        # The U-Net noise predictor
        self.unet = DiffusionPolicyUNet1D(
            action_dim=action_dim,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
        )

    def _build_per_step_cond(self, z_v_pooled_window: torch.Tensor,
                              z_t_window: torch.Tensor,
                              intent_emb: torch.Tensor = None) -> torch.Tensor:
        """Build per-step condition (B, K, cond_dim).

        Condition = concat[z_v_pooled, z_t, intent_pooled] per step.
        Intent tokens are pooled to a single vector and repeated per-step.
        """
        B, K = z_v_pooled_window.shape[:2]
        parts = [z_v_pooled_window, z_t_window]
        if intent_emb is not None:
            if intent_emb.ndim == 3:
                intent_pooled = intent_emb.mean(dim=1)  # (B, intent_dim)
            else:
                intent_pooled = intent_emb
            intent_repeated = intent_pooled.unsqueeze(1).expand(-1, K, -1)
            parts.append(intent_repeated)
        return torch.cat(parts, dim=-1)

    def forward(self, z_v_pooled_window: torch.Tensor,
                z_t_window: torch.Tensor,
                intent_emb: torch.Tensor = None) -> torch.Tensor:
        """Build per-step condition. Returns cond for loss/sample."""
        return self._build_per_step_cond(
            z_v_pooled_window, z_t_window, intent_emb,
        )

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                      cond: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy actions and timestep."""
        t_emb = self.time_emb(t.float() / self.num_inference_steps)
        return self.unet(x_t, cond, t_emb)

    def loss(self, actions_target: torch.Tensor,
             cond: torch.Tensor) -> torch.Tensor:
        """DDPM noise-prediction training loss."""
        B, K = actions_target.shape[:2]
        device = actions_target.device

        t_indices = torch.randint(0, self.num_inference_steps + 1,
                                  (B,), device=device).long()
        alpha_bar_t = self.alpha_bar[t_indices][:, None, None]
        sigma_t = self.sigma[t_indices][:, None, None]

        noise = torch.randn_like(actions_target, dtype=torch.float32)
        x_t = alpha_bar_t.sqrt() * actions_target.float() + sigma_t * noise

        predicted_noise = self.predict_noise(x_t, t_indices, cond)
        return F.mse_loss(predicted_noise.float(), noise.float())

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """DDIM deterministic sampling (eta=0) from noise to actions."""
        if num_steps is None:
            num_steps = self.num_inference_steps
        B, K, D = cond.shape[0], cond.shape[1], self.action_dim
        device = cond.device

        x = torch.randn(B, K, D, device=device)

        for i in range(num_steps - 1, -1, -1):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self.predict_noise(x, t, cond)

            a_bar_t = self.alpha_bar[i]
            a_bar_prev = (self.alpha_bar[max(i - 1, 0)]
                          if i > 0
                          else torch.tensor(1.0, dtype=torch.float32, device=device))
            sigma_t = self.sigma[i]

            x_0_pred = (x - sigma_t * eps) / a_bar_t.sqrt()
            x = a_bar_prev.sqrt() * x_0_pred + (1.0 - a_bar_prev).sqrt() * eps
        return x


# ================================================================
# Sinusoidal time embedding
# ================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps.

    Input: t of any shape, e.g. (B, K) or (B, 1, 1)
    Output: same shape with last dim = time_dim
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        if half_dim < 1:
            half_dim = 1
        exponent = torch.arange(half_dim, device=device, dtype=t.dtype)
        exponent = exponent * (-torch.log(torch.tensor(10000.0)) / max(half_dim - 1, 1))
        freqs = torch.exp(exponent)
        original_shape = t.shape
        t_flat = t.reshape(-1)
        angles = t_flat.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        out_shape = original_shape + (self.dim,)
        return emb.reshape(out_shape)
