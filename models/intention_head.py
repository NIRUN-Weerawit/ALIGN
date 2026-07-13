#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intention transformer head: K past states + 1 h → K future actions.

The head consumes:
  - z_v_pooled_window: (B, K, vision_dim * num_cameras) — K past pooled visions
  - z_t_window:        (B, K, state_dim)               — K past states
  - h_current:         (B, mamba_output_dim)           — current Mamba state

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
        mamba_output_dim: mamba hidden state dim (e.g., 512)
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
                 mamba_output_dim: int = 512, action_dim: int = 6,
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

        # Per-timestep projection: concat[z_v_pooled, z_t] → d_model
        self.input_proj = nn.Linear(self.pool_out_dim + state_dim, d_model)
        # h projection (h goes in as a context token)
        self.h_proj = nn.Linear(mamba_output_dim, d_model)

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
                h_current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_v_pooled_window: (B, K, pool_out_dim) — K past pooled visions
            z_t_window:        (B, K, state_dim)    — K past states
            h_current:         (B, mamba_output_dim) — current Mamba state
        Returns:
            actions: (B, K, action_dim) — K future actions
        """
        B, K = z_v_pooled_window.shape[:2]
        assert K == self.chunk_size, (
            f"Window size {K} doesn't match chunk_size {self.chunk_size}"
        )

        # Per-timestep input
        per_step_in = torch.cat([z_v_pooled_window, z_t_window], dim=-1)
        # (B, K, pool_out_dim + state_dim)
        x = self.input_proj(per_step_in)  # (B, K, d_model)

        # h as a context token (prepended)
        h_token = self.h_proj(h_current).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([h_token, x], dim=1)  # (B, K+1, d_model)

        # Add positional encoding
        x = x + self.pos_emb.unsqueeze(0)  # (B, K+1, d_model)

        # Transformer
        x = self.transformer(x)  # (B, K+1, d_model)

        # Drop the h context token, keep K timestep tokens
        x = x[:, 1:]  # (B, K, d_model)

        # Output: K actions
        actions = self.output_proj(x)  # (B, K, action_dim)
        return actions
