#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intention heads: K past states + 1 h → K future actions.

Two head architectures:
  - IntentionTransformerHead: standard transformer (K+1 tokens)
  - MambaActionHead: Mamba recurrent head (O(1) inference, variable horizon)

Both consume:
  - z_v_pooled_window: (B, K, vision_dim * num_cameras) — K past pooled visions
  - z_t_window:        (B, K, state_dim)               — K past states
  - h_current:         (B, mamba_output_dim)           — current Mamba state (or None)

Output:
  - actions: (B, K, action_dim) — K future actions

Architecture:
  - h as a context token (prepended)
  - Per-timestep tokens from concat[z_v_pooled, z_t]
  - Transformer encoder over the (K+1) tokens
  - Output projection to K actions
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IntentionTransformerHead(nn.Module):
    """Transformer head for action prediction from intention state.

    Args:
        vision_dim:       per-patch dim (e.g., 256)
        state_dim:        robot state dim (e.g., 256)
        mamba_output_dim: mamba hidden state dim (e.g., 512). Set to 0 to disable.
        text_dim:         text encoder dim (e.g., 256). Set to 0 to disable.
        action_dim:       action output dim (e.g., 7)
        chunk_size:       K — number of past steps / future actions
        d_model:          internal transformer dim
        nhead:            attention heads
        num_layers:       transformer layers
        dim_feedforward:  FFN dim
        dropout:          dropout
        pool_out_dim:     actual input dim of z_v_pooled (e.g., 2*vision_dim for 2 cams)
    """
    def __init__(self, vision_dim: int = 256, state_dim: int = 256,
                 mamba_output_dim: int = 512, text_dim: int = 0,
                 action_dim: int = 6,
                 chunk_size: int = 10, d_model: int = 384, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 1024,
                 dropout: float = 0.0, pool_out_dim: Optional[int] = None):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.pool_out_dim = pool_out_dim or vision_dim
        self.d_model = d_model
        self.use_history = mamba_output_dim > 0
        self.use_text = text_dim > 0
        self.text_dim = text_dim

        # Per-timestep projection: concat[z_v_pooled, z_t, z_text?] → d_model
        per_step_in_dim = self.pool_out_dim + state_dim + text_dim
        self.input_proj = nn.Linear(per_step_in_dim, d_model)
        # h projection (h goes in as a context token) — only if use_history
        if self.use_history:
            self.h_proj = nn.Linear(mamba_output_dim, d_model)
        else:
            self.h_proj = None
        # text projection (text goes in as a context token) — only if use_text
        if self.use_text:
            self.text_proj = nn.Linear(text_dim, d_model)
        else:
            self.text_proj = None

        # Positional encoding (learned)
        self.pos_emb = nn.Parameter(
            torch.randn(chunk_size + 1, d_model) * 0.02
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
                h_current: torch.Tensor,
                z_text: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_v_pooled_window: (B, K, pool_out_dim) — K past pooled visions
            z_t_window:        (B, K, state_dim)    — K past states
            h_current:         (B, mamba_output_dim) — current Mamba state (or None)
            z_text:            (B, text_dim) — task text embedding (or None)
        Returns:
            actions: (B, K, action_dim) — K future actions
        """
        B, K = z_v_pooled_window.shape[:2]
        assert K == self.chunk_size, (
            f"Window size {K} doesn't match chunk_size {self.chunk_size}"
        )

        # Per-timestep input: concat[z_v_pooled, z_t, z_text?]
        per_step_parts = [z_v_pooled_window, z_t_window]
        if self.use_text and z_text is not None:
            # Expand z_text to per-step: (B, 1, text_dim) → (B, K, text_dim)
            z_text_expanded = z_text.unsqueeze(1).expand(-1, K, -1)
            per_step_parts.append(z_text_expanded)
        per_step_in = torch.cat(per_step_parts, dim=-1)
        # (B, K, pool_out_dim + state_dim + text_dim?)
        x = self.input_proj(per_step_in)  # (B, K, d_model)

        # h as a context token (prepended) — only if use_history
        if self.use_history and h_current is not None:
            h_token = self.h_proj(h_current).unsqueeze(1)  # (B, 1, d_model)
            x = torch.cat([h_token, x], dim=1)  # (B, K+1, d_model)
        # else: no h token, x stays (B, K, d_model)

        # Add positional encoding (size matches x)
        # Note: pos_emb was sized chunk_size+1 (for K + 1 h token). With text
        # added per-step (not as a separate token), no size change needed.
        x = x + self.pos_emb[:x.size(1)].unsqueeze(0)

        # Transformer (use math backend to avoid cuDNN issues with Mamba)
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                x = self.transformer(x)
        except ImportError:
            x = self.transformer(x)  # fallback if SDPA API not available

        # If h was prepended, drop it; otherwise keep all K
        if self.use_history and h_current is not None:
            x = x[:, 1:]  # (B, K, d_model) — drop h context token

        # Drop the h context token, keep K timestep tokens
        # (already done above if use_history; if not, x is already (B, K, d_model))

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
    """Mamba-based action head: K past (z_v_pooled, z_t) + 1 h → K future actions.

    Architecture:
      input[t] = concat[z_v_pooled[t], z_t[t]]  (optionally + h_current repeated + z_text)
      mamba_seq = Mamba(input_seq)               # (B, K, hidden_dim)
      actions = output_proj(mamba_seq)            # (B, K, action_dim)

    Compared to IntentionTransformerHead:
      + O(K) training compute vs O(K²) for self-attention
      + O(1) inference per step (use Mamba.step with persistent state)
      + Variable horizon (predict any K)
      - Slightly less expressive on rich data
    """
    def __init__(self, pool_out_dim: int = 256, state_dim: int = 256,
                 mamba_output_dim: int = 512, text_dim: int = 0,
                 action_dim: int = 6,
                 chunk_size: int = 10, mamba_d_state: int = 16,
                 mamba_d_conv: int = 4, mamba_expand: int = 2,
                 use_history: bool = True):
        super().__init__()
        if not _HAS_MAMBA_HEAD:
            raise ImportError(
                "mamba_ssm not installed. Run: pip install mamba-ssm causal-conv1d"
            )

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_history = use_history
        self.use_text = text_dim > 0
        self.text_dim = text_dim

        # Input dim: per-step = pool + state + text
        per_step_in = pool_out_dim + state_dim + text_dim
        if use_history and mamba_output_dim > 0:
            self.input_dim = per_step_in + mamba_output_dim
        else:
            self.input_dim = per_step_in
            self.use_history = False  # force off if no history dim

        # Mamba block
        self.mamba = _MambaSSM(
            d_model=self.input_dim,
            d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
        )
        # Output projection
        self.output_proj = nn.Linear(self.input_dim, action_dim)
        # Init output near zero (identity-ish start)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_v_pooled_window: torch.Tensor,
                z_t_window: torch.Tensor,
                h_current: torch.Tensor = None,
                z_text: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_v_pooled_window: (B, K, pool_out_dim) — K past pooled visions
            z_t_window:        (B, K, state_dim)    — K past states
            h_current:         (B, mamba_output_dim) — current Mamba state (or None)
            z_text:            (B, text_dim) — task text embedding (or None)
        Returns:
            actions: (B, K, action_dim) — K future actions
        """
        B, K = z_v_pooled_window.shape[:2]
        assert K == self.chunk_size, (
            f"Window size {K} doesn't match chunk_size {self.chunk_size}"
        )

        # Per-timestep input: concat[z_v_pooled, z_t, z_text?]
        per_step_parts = [z_v_pooled_window, z_t_window]
        if self.use_text and z_text is not None:
            # Expand z_text to per-step: (B, 1, text_dim) → (B, K, text_dim)
            z_text_expanded = z_text.unsqueeze(1).expand(-1, K, -1)
            per_step_parts.append(z_text_expanded)
        per_step_in = torch.cat(per_step_parts, dim=-1)
        # (B, K, pool_out_dim + state_dim + text_dim?)
        if self.use_history and h_current is not None:
            h_repeated = h_current.unsqueeze(1).expand(-1, K, -1)  # (B, K, mamba_output_dim)
            per_step_in = torch.cat([per_step_in, h_repeated], dim=-1)
        # (B, K, input_dim)

        # Mamba: (B, K, input_dim) -> (B, K, input_dim)
        out = self.mamba(per_step_in)

        # Output projection: per-timestep action
        actions = self.output_proj(out)  # (B, K, action_dim)
        return actions


# ================================================================
# Diffusion Action Head (DDPM baseline)
# ================================================================


class DiffusionActionHead(nn.Module):
    """DDPM-style diffusion head for action prediction.

    Baseline for comparing against flow-matching: trains a noise predictor
    ε_θ(x_T, t, c) and denoises iteratively back to x_0 ≈ actions.

    Architecture (same conditioning backbone as flow-matching head):
      - Per-step condition: concat[z_v_pooled[t], z_t[t], z_text?, h_current]
      - Noise predictor: MLP with sinusoidal time emb + FiLM gating
      - Training: noise prediction objective  (DDPM)
      - Inference: deterministic denoising (no variance scheduling)

    Compared to flow-matching:
      + Proven baseline — Diffusion Policy paper, widely adopted
      - ~20 inference steps needed (vs ~10 for FM at similar quality)
      - More training samples per forward than direct regression

    Args:
        cond_dim:          condition dim = pool_out + state + text + h
        action_dim:        output dim (default 6)
        hidden_dim:        MLP hidden (default 256, maps via --head-d-model)
        num_inference_steps: denoising steps (default 20)
        time_dim:          sinusoidal time emb dim (default 64)
        chunk_size:        K — number of past steps / future actions
    """

    def __init__(self, cond_dim: int = 768, action_dim: int = 6,
                 hidden_dim: int = 256, num_inference_steps: int = 20,
                 time_dim: int = 64, chunk_size: int = 10):
        super().__init__()
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.time_dim = time_dim
        self.chunk_size = chunk_size
        self.cond_dim = cond_dim

        # Cosine noise schedule (Lin et al. 2023): ᾱ_t = cos^2(π/2 · (t/T + s)/(1+s))
        # σ_t = sqrt(1 - ᾱ_t) controls signal-to-noise ratio at step t
        T_steps = num_inference_steps
        s = 0.008
        t_vals = torch.arange(T_steps + 1, dtype=torch.float64)         # 0…T
        theta = torch.tensor(math.pi / 2, dtype=torch.float64) * \
                (t_vals / T_steps + s) / (1 + s)
        alpha_bar = torch.cos(theta).pow(2)                             # ᾱ_t
        sigma = torch.sqrt(1 - alpha_bar)                               # √(1-ᾱ)
        self.register_buffer("alpha_bar", alpha_bar.float())            # (T+1,) 0→~1
        self.register_buffer("sigma", sigma.float())                    # (T+1,)  0→~1

        # Per-sample cosine-squared values θ(t) for time embedding:
        # cos_s_value[i] = cos²((i + 0.5)*π / (2*T)) for i in [0, T-1]
        self.register_buffer("cos_s_values", torch.tensor(
            [math.cos(math.pi * (i + 0.5) / (2 * T_steps)) ** 2
             for i in range(T_steps)], dtype=torch.float32))

        # Time embedding: cosine² → sinusoidal scaling → MLP
        self.time_emb_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Input: cat(action_t, cond)
        self.input_proj = nn.Linear(action_dim + cond_dim, hidden_dim)
        self.cond_scale = nn.Parameter(torch.ones(hidden_dim))
        self.cond_shift = nn.Parameter(torch.zeros(hidden_dim))

        # Noise predictor body (shared across timesteps, conditioned on c and t_emb)
        self.body = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output: hidden → noise prediction (ε̂)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        # Small init for stable early-step training (avoids blowing up ᾱ=0 regime)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    # ---------------------------------------------------------------
    # Noise predictor: ε_θ(x_t , t, c)  →  (B, K, action_dim)
    # ---------------------------------------------------------------
    def _cosine_s_emb(self, t_indices: torch.Tensor) -> torch.Tensor:
        """Cosine-squared time embedding (Lin et al. 2023).

        t_indices: integer step indices, any shape (e.g. (B,), (B,K))
        Returns:   scaled sinusoidal embedding of shape (*t.shape, time_dim)

        Uses cos²(θ_t) to scale the sinusoidal frequencies so nearby timesteps
        get similar embeddings — smoother than raw sin/cos of t.
        """
        clamped = torch.clamp(t_indices.long(), 0, len(self.cos_s_values) - 1)
        cos2_t = self.cos_s_values[clamped]                              # (*t.shape,)
        half_dim = self.time_dim // 2
        exponents = torch.arange(half_dim, device=cos2_t.device, dtype=torch.float32)
        freqs = torch.exp(exponents * (-math.log(10000.0) / max(half_dim - 1, 1)))
        scaled = cos2_t.unsqueeze(-1) * freqs                            # (*t.shape, half_dim)
        raw_emb = torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)  # (*, time_dim)
        return self.time_emb_mlp(raw_emb)

    def _build_per_step_cond(self, z_v_pooled_window: torch.Tensor,
                              z_t_window: torch.Tensor,
                              h_current: torch.Tensor = None,
                              z_text: torch.Tensor = None) -> torch.Tensor:
        """Build per-step condition (B, K, cond_dim)."""
        B, K = z_v_pooled_window.shape[:2]
        parts = [z_v_pooled_window, z_t_window]
        if z_text is not None:
            parts.append(z_text.unsqueeze(1).expand(-1, K, -1))
        if h_current is not None:
            parts.append(h_current.unsqueeze(1).expand(-1, K, -1))
        return torch.cat(parts, dim=-1)

    def forward(self, z_v_pooled_window: torch.Tensor,
                z_t_window: torch.Tensor,
                h_current: torch.Tensor = None,
                z_text: torch.Tensor = None) -> torch.Tensor:
        """Build per-step condition for training loss.

        Returns cond tensor; caller passes it to .loss() (see FlowMatchingActionHead).
        """
        return self._build_per_step_cond(
            z_v_pooled_window, z_t_window, h_current, z_text,
        )

    def predict_noise(self, noisy_actions: torch.Tensor,
                      timesteps: int, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise given action observation at discrete step *timesteps* (0…T).

        Unlike the flow head where *t* is continuous [0,1], diffusion works in
        integer steps here for simplicity and to match DDPM/DDIM convention.
        For batched training with per-sample timesteps use .loss() directly.

        Args:
            noisy_actions: (B, K, action_dim) — x_current
            timesteps:     step index 0…T  (scalar, same for entire batch+chunk)
            cond:          (B, K, cond_dim)  — per-step condition
        Returns:
            predicted_noise: (B, K, action_dim) — ε̂
        """
        B, K = noisy_actions.shape[:2]

        # Time embedding: scalar t → (time_dim,) → (K, time_dim) broadcast
        t_tensor = torch.full((K,), float(timesteps), device=cond.device)   # (K,)
        emb = self._cosine_s_emb(t_tensor).unsqueeze(0)                      # (1, K, time_dim)

        h_base = self.input_proj(torch.cat([noisy_actions, cond], -1))     # (B, K, hidden)
        h_gate = h_base * self.cond_scale + self.cond_shift                # FiLM gating on cond path

        body_in = torch.cat([h_gate.expand_as(h_base), emb.expand(B, -1, -1).expand_as(h_base)], -1)  # (B, K, hidden+time)
        h_body = self.body(body_in)                                        # (B, K, hidden)
        return self.output_proj(h_body).float()                           # (B, K, action_dim)

    def loss(self, actions_target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """DDPM noise-prediction training loss.

        Sample t uniformly from 0…T, add Gaussian noise scaled by sigma_t,
        ask the model to predict that noise back given x_t.

        Args:
            actions_target: (B, K, action_dim) — ground truth actions
            cond:           (B, K, cond_dim)   — per-step condition
        Returns:
            scalar MSE between predicted noise and actual injected Gaussian noise
        """
        B, K = actions_target.shape[:2]
        device = actions_target.device

        # Sample t ~ Uniform(0, T) for each sample in the batch (share across chunk dim)
        t_indices = torch.randint(0, self.num_inference_steps + 1,
                                  (B,), device=device).long()             # (B,)

        alpha_bar_t = self.alpha_bar[t_indices][:, None, None]            # (B, 1, 1)
        sigma_t     = self.sigma[t_indices][:, None, None]

        noise = torch.randn_like(actions_target, dtype=torch.float32)      # ε ~ N(0,I)
        x_t = alpha_bar_t.sqrt() * actions_target.float() + sigma_t * noise

        predicted_noise = self._predict_noise_with_index(x_t, t_indices, cond)
        return F.mse_loss(predicted_noise.float(), noise.float())

    def _predict_noise_with_index(self, x_t: torch.Tensor,
                                   t_indices: torch.Tensor,
                                   cond: torch.Tensor) -> torch.Tensor:
        """Core forward inside loss() — handles per-sample (batch-level) timestep indices."""
        B = x_t.shape[0]
        K = x_t.shape[1]

        # Time embedding from integer t → float in [0, T_steps] scaled to [0, 1]
        t_float = (t_indices.to(cond.dtype).unsqueeze(1)) \
                  .expand(-1, K)                                     # (B, K)
        emb = self._cosine_s_emb(t_float)                             # (B, K, time_dim)

        h_base = self.input_proj(torch.cat([x_t, cond], -1))          # (B, K, hidden)
        h_gate = h_base * self.cond_scale + self.cond_shift           # FiLM gating on cond path

        body_in = torch.cat([h_gate, emb], -1)                         # (B, K, hidden+time)
        h_body = self.body(body_in)                                   # (B, K, hidden)
        return self.output_proj(h_body).float()                        # (B, K, action_dim)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Deterministic DDIM-style denoising from noise to actions.

        Runs in reverse: x_T → x_{T-1} → … → x_0 ≈ actions.

        Args:
            cond:     (B, K, cond_dim) — per-step condition
            num_steps: override default inference step count
        Returns:
            actions:  (B, K, action_dim)
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        B, K = cond.shape[:2]
        device = cond.device

        # Start from pure noise at step T (the end of the schedule)
        x = torch.randn(B, K, self.action_dim, dtype=torch.float32, device=device)

        for i in range(num_steps - 1, -1, -1):   # t = T-1 → 0
            t = torch.tensor([i], dtype=torch.long, device=device)
            x_next = self._step(x, i, cond)
            x = x_next

        return x.float()

    def _step(self, x: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        """One denoising step: x_t → x_{t-1} (deterministic)."""
        predicted_noise = self._predict_noise_with_index(
            x,
            # Single timestep repeated for whole batch — shape (B,)
            torch.full((x.shape[0],), t, dtype=torch.long, device=x.device),
            cond,
        )

        sigma_t     = self.sigma[t].float()                          # (1,)
        sigma_prev  = self.sigma[max(t - 1, 0)].float()

        a_bar_t      = self.alpha_bar[t].float()
        if t > 0:
            a_bar_prev = self.alpha_bar[t - 1].float()
        else:
            a_bar_prev = torch.tensor(1.0, dtype=torch.float32, device=x.device)

        # Mean prediction (DDPM/DDIM common term):
        x_hat = ((x - sigma_t * predicted_noise / a_bar_t.sqrt()) /
                  sigma_t * (a_bar_prev.sqrt() - a_bar_t.sqrt()))

        # Add variance term: σ_{t-1}·(σ_t^2 - (ᾱ_t - ᾱ_{t-1})²)^(1/2) for diversity
        if t > 1:
            var_coeff = sigma_prev * torch.sqrt(
                sigma_t.pow(2) - (a_bar_t - a_bar_prev).pow(2) + 1e-8,
            )
        else:
            var_coeff = sigma_prev

        return x_hat + predicted_noise * var_coeff[:, None, None]


# ================================================================
# Flow-Matching Action Head
# ================================================================

class FlowMatchingActionHead(nn.Module):
    """Flow-matching action head (Lipman et al. 2023).

    Architecture:
      - Per-step condition: concat[z_v_pooled[t], z_t[t], z_text?, h_current]
      - Velocity field: small MLP that takes (noisy_action, t, cond) → velocity
      - Training: predict velocity field
      - Inference: integrate ODE from t=0 to t=1 to get actions

    Args:
        cond_dim:          dimension of per-step condition (z_v + z_t + text + h)
        action_dim:        action output dim (default 6)
        hidden_dim:        velocity MLP hidden dim (default 256)
        num_inference_steps: ODE integration steps (default 10)
        time_dim:          time embedding dim (default 64)
        chunk_size:        K — number of past steps / future actions

    Compared to direct regression (Mamba/Transformer head):
      + Can model multi-modal action distributions
      + Better sample quality on complex tasks
      - Slower inference (N forward passes vs 1)
      - More training compute

    Flow-matching recap (for context):
      Training:
        Sample x0 ~ N(0, I), t ~ U[0, 1]
        x_t = (1-t) * x0 + t * a    (linear interpolation)
        v_target = a - x0           (velocity)
        v_pred = v_net(x_t, t, cond)
        loss = ||v_pred - v_target||^2
      Inference (Euler ODE):
        x_0 ~ N(0, I)
        for i in 0..N-1:
          t = i / N
          x_{i+1} = x_i + v_net(x_i, t, cond) * dt
        return x_N
    """
    def __init__(self, cond_dim: int = 768, action_dim: int = 6,
                 hidden_dim: int = 256, num_inference_steps: int = 10,
                 time_dim: int = 64, chunk_size: int = 10,
                 use_history: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.time_dim = time_dim
        self.chunk_size = chunk_size
        self.use_history = use_history
        self.cond_dim = cond_dim

        # Time embedding (sinusoidal → MLP)
        self.time_emb = nn.Sequential(
            SinusoidalPositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Velocity field network: (noisy_action, t_emb, cond) → velocity
        self.v_net = nn.Sequential(
            nn.Linear(action_dim + time_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Initialize last layer small for stable training
        nn.init.normal_(self.v_net[-1].weight, std=1e-3)
        nn.init.zeros_(self.v_net[-1].bias)

    def _build_per_step_cond(self, z_v_pooled_window: torch.Tensor,
                              z_t_window: torch.Tensor,
                              h_current: torch.Tensor = None,
                              z_text: torch.Tensor = None) -> torch.Tensor:
        """Build per-step condition (B, K, cond_dim)."""
        B, K = z_v_pooled_window.shape[:2]
        per_step_parts = [z_v_pooled_window, z_t_window]
        if z_text is not None:
            z_text_expanded = z_text.unsqueeze(1).expand(-1, K, -1)
            per_step_parts.append(z_text_expanded)
        per_step_in = torch.cat(per_step_parts, dim=-1)
        if self.use_history and h_current is not None:
            h_repeated = h_current.unsqueeze(1).expand(-1, K, -1)
            per_step_in = torch.cat([per_step_in, h_repeated], dim=-1)
        return per_step_in

    def compute_velocity(self, noisy_actions: torch.Tensor, t: torch.Tensor,
                         cond: torch.Tensor) -> torch.Tensor:
        """Predict velocity field.

        Args:
            noisy_actions: (B, K, action_dim)
            t:             (B, K) or (B, K, 1) timestep in [0, 1]
            cond:          (B, K, cond_dim)
        Returns:
            velocity: (B, K, action_dim)
        """
        # Squeeze trailing dim if present (B, K, 1) → (B, K)
        if t.ndim == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        # Time embedding: t (B, K) → t_emb (B, K, time_dim)
        t_emb = self.time_emb(t)  # (B, K, time_dim)
        # If t was (B, 1) (for sample() with single t), broadcast over K
        if t_emb.shape[1] == 1 and cond.shape[1] != 1:
            t_emb = t_emb.expand(-1, cond.shape[1], -1)
        # Concatenate inputs: noisy_actions (B, K, action_dim) +
        #                    t_emb (B, K, time_dim) +
        #                    cond (B, K, cond_dim)
        inp = torch.cat([noisy_actions, t_emb, cond], dim=-1)
        return self.v_net(inp)

    def loss(self, actions_target: torch.Tensor,
             cond: torch.Tensor) -> torch.Tensor:
        """Flow-matching training loss.

        Args:
            actions_target: (B, K, action_dim) — ground truth actions
            cond:           (B, K, cond_dim) — per-step condition
        Returns:
            loss: scalar — MSE between predicted and target velocity
        """
        B, K, D = actions_target.shape
        # Sample noise
        x0 = torch.randn_like(actions_target)
        # Sample timestep uniformly in [0, 1]
        t = torch.rand(B, K, 1, device=actions_target.device)
        # Linear interpolation
        x_t = (1 - t) * x0 + t * actions_target
        # Velocity target: derivative of x_t w.r.t. t
        v_target = actions_target - x0
        # Predicted velocity
        v_pred = self.compute_velocity(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)

    def forward(self, z_v_pooled_window: torch.Tensor,
                z_t_window: torch.Tensor,
                h_current: torch.Tensor = None,
                z_text: torch.Tensor = None) -> torch.Tensor:
        """Training forward: returns per-step condition for loss computation.

        Args:
            z_v_pooled_window: (B, K, pool_out_dim)
            z_t_window:        (B, K, state_dim)
            h_current:         (B, mamba_output_dim) or None
            z_text:            (B, text_dim) or None
        Returns:
            cond: (B, K, cond_dim) — per-step condition (passed to loss())
        """
        return self._build_per_step_cond(
            z_v_pooled_window, z_t_window, h_current, z_text,
        )

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """ODE integration: x_0 ~ N(0, I) → x_1 = action via Euler.

        Args:
            cond: (B, K, cond_dim) — per-step condition
            num_steps: ODE integration steps (default: self.num_inference_steps)
        Returns:
            actions: (B, K, action_dim)
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        B, K, D = cond.shape[0], cond.shape[1], self.action_dim
        device = cond.device
        # Initialize from noise
        x = torch.randn(B, K, D, device=device)
        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B, K, 1), i * dt, device=device)
            v = self.compute_velocity(x, t, cond)
            x = x + v * dt
        return x


# ================================================================
# Sinusoidal time embedding (used by FlowMatchingActionHead)
# ================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps (used by diffusion/FM).

    Input: t of any shape, e.g. (B, K) or (B, 1, 1)
    Output: same shape with last dim = time_dim
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: any shape, e.g. (B, K, 1), (B, 1, 1), (B, K), (B, 1)
        # We want output of shape t.shape + (dim,)
        device = t.device
        half_dim = self.dim // 2
        if half_dim < 1:
            half_dim = 1
        # Frequency factors: (half_dim,)
        # Standard sinusoidal: 1 / 10000^(2k/dim) for k in [0, half_dim)
        exponent = torch.arange(half_dim, device=device, dtype=t.dtype)
        exponent = exponent * (-torch.log(torch.tensor(10000.0)) / max(half_dim - 1, 1))
        freqs = torch.exp(exponent)
        # Flatten t to (N,) where N = prod(t.shape)
        original_shape = t.shape
        t_flat = t.reshape(-1)  # (N,)
        # Compute angles: (N, half_dim)
        angles = t_flat.unsqueeze(-1) * freqs.unsqueeze(0)  # (N, half_dim)
        # Concat sin and cos: (N, dim)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        # Reshape back: t.shape + (dim,)
        out_shape = original_shape + (self.dim,)
        return emb.reshape(out_shape)
