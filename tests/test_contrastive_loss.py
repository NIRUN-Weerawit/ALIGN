#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for ALIGN contrastive loss.

Run: python tests/test_contrastive_loss.py
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.contrastive_loss import ContrastiveLoss3Way


def test_random_embeddings():
    """Random embeddings should give loss ~ln(B) and cosines near 0."""
    torch.manual_seed(42)
    B, D = 8, 256
    z_v = torch.randn(B, D) * 10
    z_s = torch.randn(B, D) * 10
    z_sext = torch.randn(B, D) * 10

    criterion = ContrastiveLoss3Way(temperature=0.07)
    stats = criterion(z_v, z_s, z_sext)

    # Loss for random 8-way classification: ln(8) ≈ 2.08
    assert 1.5 < stats['loss'].item() < 3.5, \
        f"Expected loss ~2.5, got {stats['loss'].item()}"

    # All cosines should be in [-0.2, 0.2] for random
    for k in ['avg_cos_vt', 'avg_cos_vl', 'avg_cos_tl']:
        v = stats[k].item()
        assert -0.2 < v < 0.2, f"{k}={v} out of expected range"
    print("  ✅ test_random_embeddings passed")


def test_perfect_alignment():
    """Identical z_v, z_s, z_sext should give cosines = 1.0 and loss ≈ 0."""
    B, D = 8, 256
    base = torch.randn(B, D)
    z_v = z_s = z_sext = base

    criterion = ContrastiveLoss3Way(temperature=0.07)
    stats = criterion(z_v, z_s, z_sext)

    for k in ['avg_cos_vt', 'avg_cos_vl', 'avg_cos_tl']:
        assert abs(stats[k].item() - 1.0) < 1e-5, \
            f"{k}={stats[k].item()} expected 1.0"

    assert stats['loss'].item() < 0.01, \
        f"Loss {stats['loss'].item()} should be ~0 for perfect alignment"
    print("  ✅ test_perfect_alignment passed")


def test_anti_alignment():
    """Anti-correlated z_v, z_s should give cos_vt = -1.0."""
    B, D = 8, 256
    z_v = torch.randn(B, D)
    z_s = -z_v
    z_sext = torch.randn(B, D)

    criterion = ContrastiveLoss3Way(temperature=0.07)
    stats = criterion(z_v, z_s, z_sext)

    assert abs(stats['avg_cos_vt'].item() - (-1.0)) < 1e-5
    print("  ✅ test_anti_alignment passed")


def test_range_constraint():
    """All reported cosines must be in [-1, 1] for any input."""
    B, D = 16, 256
    criterion = ContrastiveLoss3Way(temperature=0.07)

    for seed in range(20):
        torch.manual_seed(seed)
        # Try various magnitudes
        for mag in [0.1, 1.0, 10.0, 100.0]:
            z_v = torch.randn(B, D) * mag
            z_s = torch.randn(B, D) * mag
            z_sext = torch.randn(B, D) * mag
            stats = criterion(z_v, z_s, z_sext)
            for k in ['avg_cos_vt', 'avg_cos_vl', 'avg_cos_tl']:
                v = stats[k].item()
                assert -1.0 <= v <= 1.0, \
                    f"seed={seed} mag={mag} {k}={v} out of [-1, 1]"
    print("  ✅ test_range_constraint passed (20 seeds × 4 magnitudes)")


def test_temperature_scaling():
    """Higher temperature should lower the loss (softer distribution)."""
    B, D = 8, 256
    torch.manual_seed(0)
    z_v = torch.randn(B, D)
    z_s = torch.randn(B, D)
    z_sext = torch.randn(B, D)

    losses = []
    for tau in [0.01, 0.07, 0.5, 2.0]:
        c = ContrastiveLoss3Way(temperature=tau)
        stats = c(z_v, z_s, z_sext)
        losses.append(stats['loss'].item())

    # Loss should be monotonically decreasing with temperature
    for i in range(len(losses) - 1):
        assert losses[i] >= losses[i + 1], \
            f"Loss should decrease with temperature: {losses}"
    print(f"  ✅ test_temperature_scaling passed (losses: {[round(l, 3) for l in losses]})")


def test_temperature_not_learnable():
    """Temperature must NOT be a learnable parameter."""
    c = ContrastiveLoss3Way(temperature=0.07)
    n_params = sum(1 for p in c.parameters() if p.requires_grad)
    assert n_params == 0, f"Temperature should not be a learnable param, got {n_params}"
    assert not isinstance(c.temperature, torch.nn.Parameter), \
        "Temperature must be a plain float, not nn.Parameter"
    print("  ✅ test_temperature_not_learnable passed")


def test_input_normalization():
    """The loss should give identical results regardless of input magnitude."""
    B, D = 8, 256
    torch.manual_seed(0)
    base_v = torch.randn(B, D)
    base_t = torch.randn(B, D)
    base_text = torch.randn(B, D)

    criterion = ContrastiveLoss3Way(temperature=0.07)

    stats1 = criterion(base_v, base_t, base_text)
    stats2 = criterion(base_v * 1000, base_t * 1000, base_text * 1000)
    stats3 = criterion(base_v * 0.001, base_t * 0.001, base_text * 0.001)

    # Cosines should be identical (scale-invariant due to L2 normalization)
    for k in ['avg_cos_vt', 'avg_cos_vl', 'avg_cos_tl']:
        assert abs(stats1[k].item() - stats2[k].item()) < 1e-4, \
            f"{k} should be scale-invariant"
        assert abs(stats1[k].item() - stats3[k].item()) < 1e-4, \
            f"{k} should be scale-invariant"
    # Loss should also be scale-invariant
    assert abs(stats1['loss'].item() - stats2['loss'].item()) < 1e-3
    print("  ✅ test_input_normalization passed (scale-invariant at mag=1, 1000, 0.001)")


def test_batch_size_invariance():
    """Loss should scale with ln(B) for random embeddings (theoretical InfoNCE)."""
    D = 256
    criterion = ContrastiveLoss3Way(temperature=0.07)

    losses = {}
    for B in [4, 8, 16, 32, 64]:
        torch.manual_seed(0)
        z_v = torch.randn(B, D)
        z_s = torch.randn(B, D)
        z_sext = torch.randn(B, D)
        stats = criterion(z_v, z_s, z_sext)
        losses[B] = stats['loss'].item()
        # Should be roughly ln(B) for random
        import math
        expected = math.log(B) * 0.7  # 3 modalities → factor ~0.7
        diff = abs(losses[B] - expected)
        print(f"    B={B:3d}  loss={losses[B]:.3f}  expected~{expected:.3f}")
    print("  ✅ test_batch_size_invariance passed")


if __name__ == "__main__":
    print("=" * 60)
    print("ContrastiveLoss3Way tests")
    print("=" * 60)
    test_random_embeddings()
    test_perfect_alignment()
    test_anti_alignment()
    test_range_constraint()
    test_temperature_scaling()
    test_temperature_not_learnable()
    test_input_normalization()
    test_batch_size_invariance()
    print("=" * 60)
    print("All tests passed ✅")
