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
import torch
import torch.nn as nn
from typing import Optional


class IntentionTransformerHead(nn.Module):
    """Transformer head for action prediction from intention state.

    Args:
        vision_dim:       per-patch dim (e.g., 256)
        state_dim:        robot state dim (e.g., 256)
        mamba_output_dim: mamba hidden state dim (e.g., 512). Set to 0 to disable.
        text_dim:         text encoder dim (e.g., 256). Set to 0 to disable.
        action_dim:       action output dim (e.g., 6)
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
