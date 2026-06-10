#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinusoidal position embeddings for trajectory tokens.

Adds temporal structure to the K-frame trajectory window so the
cross-attention mixer knows "this is the *current* frame" vs
"this is 1 second ago" — without it, frame 0 and frame 19 are
indistinguishable to attention.

Usage:
    pos_emb = SinusoidalPositionalEncoding(max_len=128, d_model=512)
    z_t = z_t + pos_emb(z_t)  # broadcast over batch
"""

import math
import torch
import torch.nn as nn


def sinusoidal_embedding(max_len: int, d_model: int, device=None) -> torch.Tensor:
    """Standard sinusoidal position embeddings (Vaswani 2017).

    Returns:
        (max_len, d_model) tensor. Position 0 is the "current" frame
        (zero vector), increasing positions go further into the past.
    """
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal position embeddings as a module (buffer, not param).

    Returns additive positional encoding for sequences of length up to max_len.
    """

    def __init__(self, max_len: int = 128, d_model: int = 512):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        pe = sinusoidal_embedding(max_len, d_model)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position encoding to x.

        Args:
            x: (B, K, D) tensor

        Returns:
            (B, K, D) tensor with position info added
        """
        K = x.size(1)
        assert K <= self.max_len, f"Sequence length {K} exceeds max_len {self.max_len}"
        return self.pe[:K].unsqueeze(0)  # (1, K, D) for broadcast over batch


if __name__ == "__main__":
    # Quick sanity check
    pos = SinusoidalPositionalEncoding(max_len=20, d_model=512)
    x = torch.zeros(2, 20, 512)
    out = pos(x)
    print(f"x:       {x.shape}")
    print(f"x + pos: {out.shape}")
    print(f"pos[0]:  {out[0, 0, :5].tolist()}  (should be ~0 at first position)")
    print(f"pos[5]:  {out[0, 5, :5].tolist()}  (should be non-trivial)")
    print(f"pos[19]: {out[0, 19, :5].tolist()}")
    # Different positions should produce different encodings
    assert not torch.allclose(out[0, 0], out[0, 5])
    print("Position embeddings differentiate frames ✅")
