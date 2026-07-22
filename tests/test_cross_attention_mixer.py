#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for CrossAttentionMixer.

Run: python tests/test_cross_attention_mixer.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from models.cross_attention_mixer import (
    CrossAttentionMixer,
    GatedCrossAttention,
    ModLN,
)
from models.sinusoidal_pos_emb import SinusoidalPositionalEncoding


def test_modln_shape():
    """ModLN should preserve input shape."""
    x = torch.randn(2, 10, 512)
    modln = ModLN(512)
    for modality in ["vision", "trajectory", "text"]:
        y = modln(x, modality)
        assert y.shape == x.shape, f"{modality}: {y.shape} != {x.shape}"
    print("  ✅ test_modln_shape")


def test_modln_per_modality():
    """ModLN should produce different outputs for different modalities."""
    x = torch.randn(2, 512)
    modln = ModLN(512)
    y_v = modln(x, "vision")
    y_t = modln(x, "trajectory")
    y_x = modln(x, "text")
    assert not torch.allclose(y_v, y_t), "vision and trajectory should differ"
    assert not torch.allclose(y_v, y_x), "vision and text should differ"
    assert not torch.allclose(y_t, y_x), "trajectory and text should differ"
    print("  ✅ test_modln_per_modality")


def test_gated_cross_attention_shape():
    """GCA should output same shape as query."""
    B, Kq, Kkv, D = 2, 10, 5, 512
    gca = GatedCrossAttention(D, nhead=8)
    q = torch.randn(B, Kq, D)
    k = torch.randn(B, Kkv, D)
    v = torch.randn(B, Kkv, D)
    y = gca(q, k, v)
    assert y.shape == q.shape
    print("  ✅ test_gated_cross_attention_shape")


def test_gated_cross_attention_identity_init():
    """At init, gate ≈ sigmoid(1) = 0.73, so output is dominated by q (residual)."""
    D = 512
    gca = GatedCrossAttention(D, nhead=8)

    # Check gate init: bias=1.0, weight≈0
    bias_mean = gca.gate_proj.bias.abs().mean().item()
    weight_std = gca.gate_proj.weight.std().item()
    assert abs(bias_mean - 1.0) < 1e-5, f"gate bias not 1.0: {bias_mean}"
    assert weight_std < 0.1, f"gate weight std too large: {weight_std}"

    # Initial gate value when q is zero
    q = torch.zeros(1, 1, D)
    k = torch.randn(1, 1, D) * 10
    v = torch.randn(1, 1, D) * 10
    y = gca(q, k, v)
    # y should be near-zero (residual of zero q + small attention)
    # Not exact: attention output scaled by gate, residual added to q
    assert y.abs().mean().item() < 5.0, "Output too large for identity-like init"
    print("  ✅ test_gated_cross_attention_identity_init")


def test_mixer_output_shape():
    """Mixer should output 256d embeddings for v, text, and (B, K, 256) for t."""
    enc_dim = 256
    mixer = CrossAttentionMixer(enc_dim=enc_dim, mixer_dim=512, num_blocks=2)
    B, K = 4, 20
    z_v = torch.randn(B, enc_dim)
    z_s = torch.randn(B, K, enc_dim)
    z_sext = torch.randn(B, enc_dim)

    z_v2, z_s2, z_sext2 = mixer(z_v, z_s, z_sext)
    assert z_v2.shape == z_v.shape, f"vision: {z_v2.shape} != {z_v.shape}"
    assert z_s2.shape == z_s.shape, f"trajectory: {z_s2.shape} != {z_s.shape}"
    assert z_sext2.shape == z_sext.shape, f"text: {z_sext2.shape} != {z_sext.shape}"
    print("  ✅ test_mixer_output_shape")


def test_mixer_preserves_dim_for_different_K():
    """Mixer should work with various trajectory lengths K."""
    enc_dim = 256
    mixer = CrossAttentionMixer(enc_dim=enc_dim, mixer_dim=512, num_blocks=2)
    z_v = torch.randn(2, enc_dim)
    z_sext = torch.randn(2, enc_dim)
    for K in [1, 5, 10, 20, 50]:
        z_s = torch.randn(2, K, enc_dim)
        z_v2, z_s2, z_sext2 = mixer(z_v, z_s, z_sext)
        assert z_s2.shape == (2, K, enc_dim), f"K={K}: {z_s2.shape}"
    print("  ✅ test_mixer_preserves_dim_for_different_K")


def test_mixer_near_identity_at_init():
    """At init, output should be close to input (small delta)."""
    enc_dim = 256
    mixer = CrossAttentionMixer(enc_dim=enc_dim, mixer_dim=512, num_blocks=2)
    B, K = 2, 10
    z_v = torch.randn(B, enc_dim)
    z_s = torch.randn(B, K, enc_dim)
    z_sext = torch.randn(B, enc_dim)
    z_v2, z_s2, z_sext2 = mixer(z_v, z_s, z_sext)
    for name, z, z2 in [("v", z_v, z_v2), ("t", z_s, z_s2), ("text", z_sext, z_sext2)]:
        delta = (z2 - z).abs().mean().item()
        assert delta < 0.1, f"{name} drift too large: {delta}"
    print("  ✅ test_mixer_near_identity_at_init")


def test_mixer_gradients_flow():
    """Backward pass should produce gradients for all mixer params."""
    mixer = CrossAttentionMixer(enc_dim=256, mixer_dim=512, num_blocks=2)
    B, K = 2, 10
    z_v = torch.randn(B, 256, requires_grad=False)
    z_s = torch.randn(B, K, 256, requires_grad=False)
    z_sext = torch.randn(B, 256, requires_grad=False)
    z_v2, z_s2, z_sext2 = mixer(z_v, z_s, z_sext)
    loss = (z_v2 ** 2).sum() + (z_s2 ** 2).sum() + (z_sext2 ** 2).sum()
    loss.backward()
    n_with_grad = sum(1 for p in mixer.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in mixer.parameters())
    assert n_with_grad == n_total, f"Only {n_with_grad}/{n_total} params got gradient"
    print("  ✅ test_mixer_gradients_flow")


def test_mixer_block_count():
    """Num blocks should be configurable (1, 2, 3)."""
    for num_blocks in [1, 2, 3]:
        mixer = CrossAttentionMixer(enc_dim=256, mixer_dim=512, num_blocks=num_blocks)
        n = sum(p.numel() for p in mixer.parameters() if p.requires_grad)
        assert n > 0
        # Each block adds ~3.5M params
    print("  ✅ test_mixer_block_count")


def test_mixer_param_range():
    """Mixer with d=512, blocks=2 should be ~7-10M params."""
    mixer = CrossAttentionMixer(enc_dim=256, mixer_dim=512, num_blocks=2)
    n = sum(p.numel() for p in mixer.parameters() if p.requires_grad)
    assert 5e6 < n < 12e6, f"Unexpected param count: {n}"
    print(f"  ✅ test_mixer_param_range (n={n/1e6:.2f}M)")


def test_positional_embedding_differentiates_frames():
    """Position embeddings should be different for different positions."""
    pos = SinusoidalPositionalEncoding(max_len=20, d_model=512)
    x = torch.zeros(2, 20, 512)
    y = pos(x)  # (1, 20, 512)
    for i in range(20):
        for j in range(i + 1, 20):
            assert not torch.allclose(y[0, i], y[0, j]), \
                f"positions {i} and {j} have same embedding"
    print("  ✅ test_positional_embedding_differentiates_frames")


def test_mixer_does_not_modify_input():
    """Mixer should be a function (no in-place modification of inputs)."""
    mixer = CrossAttentionMixer(enc_dim=256, mixer_dim=512, num_blocks=2)
    B, K = 2, 10
    z_v = torch.randn(B, 256)
    z_s = torch.randn(B, K, 256)
    z_sext = torch.randn(B, 256)
    z_v_orig = z_v.clone()
    z_s_orig = z_s.clone()
    z_sext_orig = z_sext.clone()
    mixer(z_v, z_s, z_sext)
    # Inputs should be unchanged (mixer is a function, residual is on the output)
    assert torch.equal(z_v, z_v_orig)
    assert torch.equal(z_s, z_s_orig)
    assert torch.equal(z_sext, z_sext_orig)
    print("  ✅ test_mixer_does_not_modify_input")


if __name__ == "__main__":
    print("=" * 60)
    print("CrossAttentionMixer unit tests")
    print("=" * 60)
    test_modln_shape()
    test_modln_per_modality()
    test_gated_cross_attention_shape()
    test_gated_cross_attention_identity_init()
    test_mixer_output_shape()
    test_mixer_preserves_dim_for_different_K()
    test_mixer_near_identity_at_init()
    test_mixer_gradients_flow()
    test_mixer_block_count()
    test_mixer_param_range()
    test_positional_embedding_differentiates_frames()
    test_mixer_does_not_modify_input()
    print("=" * 60)
    print("All tests passed ✅")
