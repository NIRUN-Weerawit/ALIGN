#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-Attention Mixer for ALIGN.

Replaces simple concatenation of (z_v, z_t, z_text) with learned
bidirectional cross-attention. Each modality is enriched by attending
to the other two before the heads consume them.

Architecture (per block, 2 blocks total):
  1. Trajectory attends to (Vision, Text)  — K=10-20 tokens query, 2 keys
  2. Vision attends to (Trajectory', Text) — 1 token query, K+1 keys
  3. Text attends to (Vision', Trajectory') — 1 token query, K+1 keys

Trajectory goes first because it has the most information (K tokens vs 1).
Vision and text are sparse (1 token each) and benefit from the richer
trajectory context.

Initialization:
  - All GatedCrossAttention gates start near 0.7 (sigmoid(1.0)) → pass-through
  - This preserves pretrained encoder output during early training
  - Position embeddings use sinusoidal (frozen) + learned offset (init=0)

Reference: Flamingo (Alayrac 2022) gated cross-attention,
           BLIP-2 (Li 2023) Q-Former style mixer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path so 'models.X' imports work whether this file is
# imported as part of the package OR run as a script (e.g. `python models/cross_attention_mixer.py`)
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from models.sinusoidal_pos_emb import SinusoidalPositionalEncoding
except (ImportError, ValueError):
    # Fallback for direct script execution
    from sinusoidal_pos_emb import SinusoidalPositionalEncoding


# ================================================================
# Modality-specific LayerNorm
# ================================================================

class ModLN(nn.Module):
    """LayerNorm with per-modality scale and shift.

    Vision (DINOv2 features) and text (CLIP features) have very different
    statistics — averaging them with shared LayerNorm loses information.
    Each modality gets its own learned (scale, shift).
    """

    MODALITY_IDS = {"vision": 0, "trajectory": 1, "text": 2}

    def __init__(self, dim: int, n_modalities: int = 3):
        super().__init__()
        self.dim = dim
        # Initialize with small per-modality perturbations so each modality
        # has slightly different statistics at init. This breaks the symmetry
        # of "vision == trajectory == text" at the start of training.
        self.scale = nn.Parameter(torch.ones(n_modalities, dim) + 0.02 * torch.randn(n_modalities, dim))
        self.shift = nn.Parameter(0.02 * torch.randn(n_modalities, dim))

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Apply per-modality LayerNorm.

        Args:
            x: (B, [K], D) tensor
            modality: 'vision' | 'trajectory' | 'text'

        Returns:
            Same shape, normalized per modality
        """
        mid = self.MODALITY_IDS[modality]
        ln = F.layer_norm(x, normalized_shape=(self.dim,))
        return ln * self.scale[mid] + self.shift[mid]


# ================================================================
# Gated Cross-Attention
# ================================================================

class GatedCrossAttention(nn.Module):
    """Multihead cross-attention with sigmoid gate and residual.

    Gate initialized to ~0.7 (sigmoid(1.0)) so the block starts as
    a near-pass-through. This means the mixer is initially similar
    to the identity function — pretrained encoder features are
    preserved while the mixer learns to mix.

    Args:
        d_model: hidden dim (512)
        nhead: number of attention heads (8)
        dropout: attention and residual dropout
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.mha = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        # Gate: produces a per-feature scalar in [0, 1]
        self.gate_proj = nn.Linear(d_model, d_model)
        # Output norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize gate to ~0.7: bias=+1.0, weight=small
        # sigmoid(0.1 * 0 + 1.0) ≈ 0.73 at init
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, 1.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply gated cross-attention.

        Args:
            q: (B, Q, D) — query tokens
            k, v: (B, K, D) — key/value tokens (same length)

        Returns:
            (B, Q, D) — gated, attended, residual-merged
        """
        attn_out, _ = self.mha(q, k, v)  # (B, Q, D)
        gate = torch.sigmoid(self.gate_proj(q))  # (B, Q, D), in [0, 1]
        out = self.dropout(gate * attn_out)
        return self.norm(q + out)


# ================================================================
# Cross-Attention Mixer (the main module)
# ================================================================

class CrossAttentionMixer(nn.Module):
    """Bidirectional cross-attention mixer for ALIGN.

    Input:  z_v [B, 256], z_t [B, K, 256], z_text [B, 256]
    Output: z_v' [B, 256], z_t' [B, K, 256], z_text' [B, 256]

    The mixer is two stacked blocks. Each block has 3 cross-attention
    operations in this order: T → V → Text (trajectory conditions
    the sparser modalities first).

    Args:
        enc_dim: encoder output dim (256) — input/output
        mixer_dim: hidden dim of the mixer (512)
        num_blocks: number of mixer blocks (2)
        nhead: number of attention heads per block (8)
        max_traj_len: max K for position embedding (default 64)
        dropout: dropout inside attention
    """

    def __init__(
        self,
        enc_dim: int = 256,
        mixer_dim: int = 512,
        num_blocks: int = 2,
        nhead: int = 8,
        max_traj_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.mixer_dim = mixer_dim
        self.num_blocks = num_blocks

        # Input projections: enc_dim → mixer_dim
        self.input_proj = nn.ModuleDict({
            "vision": nn.Sequential(
                nn.Linear(enc_dim, mixer_dim), nn.LayerNorm(mixer_dim)
            ),
            "trajectory": nn.Sequential(
                nn.Linear(enc_dim, mixer_dim), nn.LayerNorm(mixer_dim)
            ),
            "text": nn.Sequential(
                nn.Linear(enc_dim, mixer_dim), nn.LayerNorm(mixer_dim)
            ),
        })

        # Output projections: mixer_dim → enc_dim
        self.output_proj = nn.ModuleDict({
            "vision": nn.Linear(mixer_dim, enc_dim),
            "trajectory": nn.Linear(mixer_dim, enc_dim),
            "text": nn.Linear(mixer_dim, enc_dim),
        })

        # Position embedding for trajectory (sinusoidal + learned offset)
        self.pos_emb_sin = SinusoidalPositionalEncoding(
            max_len=max_traj_len, d_model=mixer_dim
        )
        self.pos_emb_learned = nn.Parameter(
            torch.zeros(max_traj_len, mixer_dim),
            requires_grad=True,
        )
        nn.init.normal_(self.pos_emb_learned, std=1e-3)  # small init

        # Per-block GatedCrossAttention + ModLN
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleDict({
                # Trajectory first (most informative)
                "gca_traj": GatedCrossAttention(mixer_dim, nhead, dropout),
                # Vision
                "gca_vis": GatedCrossAttention(mixer_dim, nhead, dropout),
                # Text
                "gca_text": GatedCrossAttention(mixer_dim, nhead, dropout),
                # ModLNs
                "modln_traj": ModLN(mixer_dim, n_modalities=3),
                "modln_vis": ModLN(mixer_dim, n_modalities=3),
                "modln_text": ModLN(mixer_dim, n_modalities=3),
            })
            self.blocks.append(block)

        self._init_weights()

    def _init_weights(self):
        """Initialize output projections to near-zero for stable start.

        Combined with gate init, this means:
          z_out ≈ input_proj(x) + small_perturbation
                ≈ x (after output_proj cancels input_proj effect)
        """
        for proj in self.output_proj.values():
            nn.init.normal_(proj.weight, std=1e-3)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        z_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply cross-attention mixer.

        Args:
            z_v: (B, D) or (B, K, D) — vision (single or batched frames)
            z_t: (B, K, D) trajectory
            z_text: (B, D) text

        Returns:
            (z_v', z_t', z_text') — same shapes as inputs
        """
        # Handle both single vision token (B, D) and batched (B, K, D)
        v_is_batched = z_v.dim() == 3
        if v_is_batched:
            B, K_v, D = z_v.shape
            v_h = self.input_proj["vision"](z_v)  # (B, K_v, mixer_dim)
        else:
            B, D = z_v.shape
            v_h = self.input_proj["vision"](z_v).unsqueeze(1)  # (B, 1, mixer_dim)

        t_h = self.input_proj["trajectory"](z_t)             # (B, K, mixer_dim)
        x_h = self.input_proj["text"](z_text).unsqueeze(1)   # (B, 1, mixer_dim)

        # Add position embedding to trajectory (sinusoidal + learned)
        K = t_h.size(1)
        pos = self.pos_emb_sin(t_h) + self.pos_emb_learned[:K].unsqueeze(0)
        t_h = t_h + pos

        for block in self.blocks:
            # 1. Trajectory attends to (Vision, Text) — K tokens query, K_v+1 keys
            kv = torch.cat([v_h, x_h], dim=1)  # (B, K_v+1, mixer_dim)
            t_out = block["gca_traj"](t_h, kv, kv)
            t_h = block["modln_traj"](t_out, "trajectory")

            # 2. Vision attends to (Trajectory', Text) — K_v tokens query, K+1 keys
            kv = torch.cat([t_h, x_h], dim=1)  # (B, K+1, mixer_dim)
            v_out = block["gca_vis"](v_h, kv, kv)
            v_h = block["modln_vis"](v_out, "vision")

            # 3. Text attends to (Vision', Trajectory') — 1 token query, K_v+K keys
            kv = torch.cat([v_h, t_h], dim=1)  # (B, K_v+K, mixer_dim)
            x_out = block["gca_text"](x_h, kv, kv)
            x_h = block["modln_text"](x_out, "text")

        # Project back to enc_dim
        if v_is_batched:
            z_v_out = z_v + self.output_proj["vision"](v_h)  # (B, K_v, D)
        else:
            z_v_out = z_v + self.output_proj["vision"](v_h).squeeze(1)  # (B, D)
        z_t_out = z_t + self.output_proj["trajectory"](t_h)
        z_text_out = z_text + self.output_proj["text"](x_h).squeeze(1)
        return z_v_out, z_t_out, z_text_out


# ================================================================
# Smoke test
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CrossAttentionMixer smoke test")
    print("=" * 60)

    enc_dim = 256
    mixer_dim = 512
    B, K = 4, 20

    mixer = CrossAttentionMixer(
        enc_dim=enc_dim, mixer_dim=mixer_dim, num_blocks=2
    )

    z_v = torch.randn(B, enc_dim)
    z_t = torch.randn(B, K, enc_dim)
    z_text = torch.randn(B, enc_dim)

    # Test forward
    z_v_out, z_t_out, z_text_out = mixer(z_v, z_t, z_text)
    print(f"  Input  z_v:    {z_v.shape}")
    print(f"  Input  z_t:    {z_t.shape}")
    print(f"  Input  z_text: {z_text.shape}")
    print(f"  Output z_v':   {z_v_out.shape}")
    print(f"  Output z_t':   {z_t_out.shape}")
    print(f"  Output z_text':{z_text_out.shape}")

    # Test identity-ish init: output should be close to input
    diff_v = (z_v_out - z_v).abs().mean().item()
    diff_t = (z_t_out - z_t).abs().mean().item()
    diff_x = (z_text_out - z_text).abs().mean().item()
    print(f"\n  Mean |Δ| vision:     {diff_v:.4f}")
    print(f"  Mean |Δ| trajectory: {diff_t:.4f}")
    print(f"  Mean |Δ| text:       {diff_x:.4f}")
    assert diff_v < 0.1, f"Vision output drifted too far from input: {diff_v}"
    assert diff_t < 0.1, f"Trajectory output drifted too far: {diff_t}"
    assert diff_x < 0.1, f"Text output drifted too far: {diff_x}"

    # Test gate values
    for name, block in mixer.blocks.named_children():
        gca = block["gca_traj"]
        gate_w = gca.gate_proj.weight.abs().mean().item()
        gate_b = gca.gate_proj.bias.abs().mean().item()
        print(f"  {name}/gca_traj  gate weight={gate_w:.4f}  bias={gate_b:.4f}")
        # gate ≈ sigmoid(bias) at init since weight is small
        import math
        expected = 1.0 / (1.0 + math.exp(-gate_b))
        print(f"    initial gate ≈ {expected:.3f}  (target: ~0.7)")

    # Test parameter count
    n_params = sum(p.numel() for p in mixer.parameters() if p.requires_grad)
    print(f"\n  Trainable params: {n_params:,} ({n_params/1e6:.2f}M)")
    assert 5e6 < n_params < 12e6, f"Unexpected param count: {n_params}"

    # Test gradient flow
    z_v_out, z_t_out, z_text_out = mixer(z_v, z_t, z_text)
    loss = (z_v_out ** 2).sum() + (z_t_out ** 2).sum() + (z_text_out ** 2).sum()
    loss.backward()
    n_grad = sum(1 for p in mixer.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in mixer.parameters())
    print(f"  Params with gradient: {n_grad}/{n_total}")
    assert n_grad == n_total, "Some mixer params received no gradient"

    print("\n✅ All smoke tests passed")
