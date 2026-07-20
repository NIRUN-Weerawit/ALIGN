#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intention encoder: Mamba-based history summarization.

Architecture (per timestep t):
  z_v_patches: (B, P, vision_dim)            ← 16 patch tokens from DINOv2
  z_t:         (B, state_dim)               ← robot state embedding
  z_v_pooled   = StateConditionedPool(z_v_patches, z_t)   # (B, vision_dim)
  mamba_in     = concat[z_v_pooled, z_t]                 # (B, vision_dim + state_dim)
  h(t)         = Mamba(h(t-1), mamba_in)                 # (B, mamba_output_dim)

The state-conditioned attention pool uses z_t as the query and z_v patches as
keys/values. This makes the visual summary "state-aware" — it focuses on
patches relevant to the current robot state.

Training: use Mamba(x_seq) for batched T-step forward.
Inference: use Mamba.step(x_t, conv_state, ssm_state) for one-step-at-a-time.

Requires official mamba_ssm: pip install mamba-ssm causal-conv1d
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ================================================================
# State-Conditioned Attention Pool
# ================================================================

class StateConditionedAttentionPool(nn.Module):
    """N-query cross-attention where robot state z_t generates N independent queries.

    Instead of a single pooled token, each query learns to attend to a different
    spatial region of the patches — effectively a state-conditioned spatial summary
    with N diverse foci per timestep. The queries are produced by an independent
    linear projection per query (no parameter sharing between queries), so they
    naturally diverge to cover different visual regions.

    Args:
        vision_dim: patch feature dim (e.g., 256)
        state_dim:  robot state dim (e.g., 256)
        num_queries: number of independent queries per camera (default 8)
        num_heads:   attention heads within each query (default 4)
        dropout:     attention dropout

    Input:
        z_v_patches: (B, P, vision_dim)  — P patch tokens
        z_t:         (B, state_dim)      — current robot state

    Output:
        (B, num_queries * vision_dim) — concatenated N pooled tokens
    """
    def __init__(self, vision_dim: int = 256, state_dim: int = 256,
                 num_queries: int = 8, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.num_queries = num_queries

        # N independent queries from z_t — each learns a different spatial focus
        # Output: (B, N*vision_dim) → reshape to (B, N, vision_dim)
        self.query_proj = nn.Linear(state_dim, num_queries * vision_dim)
        # Cross-attention: N queries attend to P patches each
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vision_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(vision_dim)

    def forward(self, z_v_patches: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_v_patches: (B, P, vision_dim)
            z_t:         (B, state_dim)
        Returns:
            (B, num_queries * vision_dim) — N pooled tokens, concatenated
        """
        B = z_t.shape[0]
        # N independent queries from state — each is a separate linear projection
        q = self.query_proj(z_t).reshape(B, self.num_queries, self.vision_dim)  # (B, N, D_viz)
        k = v = z_v_patches  # (B, P, D_viz)
        # Cross-attention: each of N queries attends to all P patches
        attn_out, _ = self.cross_attn(q, k, v)  # (B, N, D_viz)
        # Residual + LayerNorm
        out = self.norm(attn_out + q)            # (B, N, D_viz)
        return out.reshape(B, -1)                # (B, N*D_viz)


# ================================================================
# Per-Camera Pool (multi-camera support)
# ================================================================

class PerCameraStateConditionedPool(nn.Module):
    """Apply N-query state-conditioned pool to each camera, then concatenate.

    For num_cameras=1: output (B, num_queries * vision_dim).
    For num_cameras>1: output (B, V * num_queries * vision_dim) — per-cam results concatenated.

    Args:
        vision_dim: per-patch dim
        state_dim:  state embedding dim
        num_cameras: number of cameras
        num_queries: independent queries per camera (N-query pool)
        num_heads:   attention heads
    """
    def __init__(self, vision_dim: int = 256, state_dim: int = 256,
                 num_cameras: int = 1, num_queries: int = 8,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_cameras = num_cameras
        self.num_queries = num_queries
        self.vision_dim = vision_dim
        self.pools = nn.ModuleList([
            StateConditionedAttentionPool(
                vision_dim=vision_dim, state_dim=state_dim,
                num_queries=num_queries, num_heads=num_heads, dropout=dropout,
            )
            for _ in range(num_cameras)
        ])

    def forward(self, z_v_patches_multi: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_v_patches_multi: (B, V, P, vision_dim) — V cameras × P patches
            z_t: (B, state_dim)
        Returns:
            (B, V * num_queries * vision_dim) — concatenated per-camera N-query pools
        """
        if self.num_cameras == 1:
            return self.pools[0](z_v_patches_multi.squeeze(1), z_t)
        outs = []
        for v in range(self.num_cameras):
            z_v_v = z_v_patches_multi[:, v]  # (B, P, vision_dim)
            outs.append(self.pools[v](z_v_v, z_t))
        return torch.cat(outs, dim=-1)  # (B, V * num_queries * vision_dim)


# ================================================================
# Intention Encoder (Mamba-based)
# ================================================================

class IntentionEncoder(nn.Module):
    """Mamba-based intention encoder with N-query visual pooling.

    Combines:
      - N-query State-Conditioned Attention Pool (per-camera, then concat)
      - Mamba recurrence (single hidden state h(t))

    Per timestep:
      z_v_patches(t) = (B, [V,] P, vision_dim)              from VisionEncoder
      z_t(t)         = (B, state_dim)                        from StateEncoder
      z_v_pooled(t)  = N-Query-Pool(z_v_patches, z_t)        (B, V*NQ*vision_dim)
      mamba_in(t)    = concat[z_v_pooled, z_t]               (B, V*NQ*vision_dim + state_dim)
      h(t)           = Mamba(h(t-1), mamba_in)               (B, mamba_output_dim)

    Args:
        vision_dim:       per-patch dim (e.g., 256)
        state_dim:        robot state dim (e.g., 256)
        mamba_output_dim: output dim of Mamba hidden state (e.g., 512)
        num_cameras:      number of cameras (default 1)
        num_queries:      N-query pool size per camera (default 8)
        mamba_d_state:    SSM state dim (default 16)
        mamba_d_conv:     local conv width (default 4)
        mamba_expand:     Mamba block expand (default 2)
    """
    def __init__(self, vision_dim: int = 256, state_dim: int = 256,
                 mamba_output_dim: int = 512, num_cameras: int = 1,
                 num_queries: int = 8,
                 mamba_d_state: int = 16, mamba_d_conv: int = 4,
                 mamba_expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm not installed. "
                "Run: pip install mamba-ssm causal-conv1d"
            )

        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.mamba_output_dim = mamba_output_dim
        self.num_cameras = num_cameras
        self.num_queries = num_queries

        # Per-camera N-query state-conditioned attention pool
        self.pool = PerCameraStateConditionedPool(
            vision_dim=vision_dim, state_dim=state_dim,
            num_cameras=num_cameras, num_queries=num_queries,
        )
        # Pool output dim = V * NQ * D_viz
        self.pool_out_dim = num_cameras * num_queries * vision_dim

        # Mamba input dim = pool_out_dim + state_dim
        self.mamba_in_dim = self.pool_out_dim + state_dim

        # Mamba block
        self.mamba = Mamba(
            d_model=self.mamba_in_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        # Project Mamba output (mamba_in_dim) → mamba_output_dim
        self.mamba_to_hidden = nn.Linear(self.mamba_in_dim, mamba_output_dim)

    def pool_patches(self, z_v_patches: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Apply N-query state-conditioned attention pool.

        Args:
            z_v_patches: (B, P, vision_dim) or (B, V, P, vision_dim)
            z_t: (B, state_dim)
        Returns:
            (B, V * num_queries * vision_dim) — N pooled tokens per camera
        """
        if z_v_patches.ndim == 3:
            # (B, P, vision_dim) — single camera
            z_v_patches = z_v_patches.unsqueeze(1)  # (B, 1, P, vision_dim)
        return self.pool(z_v_patches, z_t)

    def forward(self, z_v_patches_seq: torch.Tensor, z_t_seq: torch.Tensor
                ) -> torch.Tensor:
        """Batched T-step forward (training).

        Args:
            z_v_patches_seq: (B, T, P, vision_dim) or (B, T, V, P, vision_dim)
            z_t_seq:         (B, T, state_dim)
        Returns:
            h_seq: (B, T, mamba_output_dim)
        """
        B, T = z_v_patches_seq.shape[:2]

        # Pool each timestep's patches
        z_v_pooled_seq = []
        for t in range(T):
            z_v_t = z_v_patches_seq[:, t]  # (B, P or V, P, vision_dim)
            z_t_t = z_t_seq[:, t]          # (B, state_dim)
            z_v_pooled_t = self.pool_patches(z_v_t, z_t_t)  # (B, V*vision_dim)
            z_v_pooled_seq.append(z_v_pooled_t)
        z_v_pooled_seq = torch.stack(z_v_pooled_seq, dim=1)  # (B, T, V*vision_dim)

        # Mamba input
        mamba_in = torch.cat([z_v_pooled_seq, z_t_seq], dim=-1)  # (B, T, mamba_in_dim)
        # Mamba: returns sequence of outputs
        h_seq = self.mamba(mamba_in)                            # (B, T, mamba_in_dim)
        # Project to mamba_output_dim
        h_seq = self.mamba_to_hidden(h_seq)                     # (B, T, mamba_output_dim)
        return h_seq

    def forward_step(self, z_v_patches: torch.Tensor, z_t: torch.Tensor,
                     h_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                     ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single recurrent step (inference).

        Args:
            z_v_patches: (B, P, vision_dim) or (B, V, P, vision_dim)
            z_t:         (B, state_dim)
            h_states:    (conv_state, ssm_state) from previous step, or None for first step

        Returns:
            h_new:        (B, mamba_output_dim) — output for this step
            h_states_new: (conv_state, ssm_state) for next step
        """
        B = z_v_patches.shape[0]

        # Allocate inference cache if first step
        if h_states is None:
            h_states = self.mamba.allocate_inference_cache(
                batch_size=B, max_seqlen=1,
            )
        conv_state, ssm_state = h_states

        # Pool
        z_v_pooled = self.pool_patches(z_v_patches, z_t)  # (B, V*vision_dim)

        # Mamba input: concat[z_v_pooled, z_t]
        mamba_in = torch.cat([z_v_pooled, z_t], dim=-1)  # (B, mamba_in_dim)

        # Mamba step: process 1 token
        mamba_out, conv_state, ssm_state = self.mamba.step(
            mamba_in.unsqueeze(1),  # (B, 1, mamba_in_dim) — 1 token
            conv_state, ssm_state,
        )  # mamba_out: (B, 1, mamba_in_dim)

        # Project to mamba_output_dim
        h_new = self.mamba_to_hidden(mamba_out.squeeze(1))  # (B, mamba_output_dim)
        return h_new, (conv_state, ssm_state)

    def allocate_state(self, batch_size: int, device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate inference cache (call before first step)."""
        return self.mamba.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=1,
        ).to(device)
