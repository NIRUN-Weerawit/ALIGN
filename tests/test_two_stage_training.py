#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for two-stage training of cross-attention mixer.

Verifies the contract:
  - set_mixer_trainable(False) freezes the mixer (no gradients)
  - set_mixer_trainable(True) unfreezes it (gradients flow)
  - The mixer can be turned on/off without breaking forward pass
  - CLI flags are wired through correctly

Run: python tests/test_two_stage_training.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.align_model import ALIGNModel


def test_mixer_can_be_frozen_after_init():
    """Mixer should be freezable at any point after init."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        use_cross_attention=True, mixer_dim=512, num_mixer_blocks=2,
    )

    # Initially trainable
    n_train = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_train > 0, "Mixer should be trainable at init"

    # Freeze
    model.set_mixer_trainable(False)
    n_frozen = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_frozen == 0, f"Mixer should be fully frozen, got {n_frozen} trainable"
    assert not model.cross_attention_mixer.training, "Mixer should be in eval mode"

    # Unfreeze
    model.set_mixer_trainable(True)
    n_unfrozen = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_unfrozen == n_train, "Mixer should be fully unfrozen"
    assert model.cross_attention_mixer.training, "Mixer should be in train mode"

    print("  ✅ test_mixer_can_be_frozen_after_init")


def test_frozen_mixer_receives_no_gradients():
    """When frozen, mixer params should receive no gradients during backward."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        use_cross_attention=True, mixer_dim=512, num_mixer_blocks=2,
    )
    model.set_mixer_trainable(False)

    B, K = 2, 10
    z_v = torch.randn(B, 256, requires_grad=True)
    z_t = torch.randn(B, K, 256, requires_grad=True)
    z_text = torch.randn(B, 256, requires_grad=True)

    z_v2, z_t2, z_text2 = model.cross_attention_mixer(z_v, z_t, z_text)
    loss = (z_v2 ** 2).sum() + (z_t2 ** 2).sum() + (z_text2 ** 2).sum()
    loss.backward()

    # Mixer params should have NO gradient (frozen)
    for name, p in model.cross_attention_mixer.named_parameters():
        if p.requires_grad:
            assert p.grad is None or p.grad.abs().sum() == 0, \
                f"Frozen mixer param {name} should not have gradient"

    print("  ✅ test_frozen_mixer_receives_no_gradients")


def test_trainable_mixer_receives_gradients():
    """When trainable, mixer params should receive gradients."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        use_cross_attention=True, mixer_dim=512, num_mixer_blocks=2,
    )
    # Don't freeze

    B, K = 2, 10
    z_v = torch.randn(B, 256)
    z_t = torch.randn(B, K, 256)
    z_text = torch.randn(B, 256)

    z_v2, z_t2, z_text2 = model.cross_attention_mixer(z_v, z_t, z_text)
    loss = (z_v2 ** 2).sum() + (z_t2 ** 2).sum() + (z_text2 ** 2).sum()
    loss.backward()

    n_with_grad = sum(
        1 for p in model.cross_attention_mixer.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_with_grad == n_total, \
        f"Trainable mixer: only {n_with_grad}/{n_total} params got gradient"

    print("  ✅ test_trainable_mixer_receives_gradients")


def test_freeze_toggle_preserves_optimizer_state():
    """Toggling trainable should not corrupt the model — re-freeze should still work."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        use_cross_attention=True, mixer_dim=512, num_mixer_blocks=2,
    )

    # Toggle multiple times
    for _ in range(3):
        model.set_mixer_trainable(False)
        n_frozen = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
        assert n_frozen == 0
        model.set_mixer_trainable(True)
        n_train = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
        assert n_train > 0

    print("  ✅ test_freeze_toggle_preserves_optimizer_state")


def test_no_mixer_set_trainable_is_noop():
    """If model has no mixer, set_mixer_trainable should not crash."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')
    # No mixer — this should not raise
    model.set_mixer_trainable(False)
    model.set_mixer_trainable(True)
    assert model.cross_attention_mixer is None
    print("  ✅ test_no_mixer_set_trainable_is_noop")


def test_two_stage_pretrain_includes_pretrain_a_then_pretrain():
    """Verify the run_streaming_pipeline signature has stage_a_epochs."""
    from training.pretrain_streaming import run_streaming_pipeline
    import inspect
    sig = inspect.signature(run_streaming_pipeline)
    assert 'stage_a_epochs' in sig.parameters, "stage_a_epochs missing"
    assert 'use_cross_attention' in sig.parameters, "use_cross_attention missing"
    assert 'head_loss_weight' in sig.parameters, "head_loss_weight missing"
    assert 'modality_dropout' in sig.parameters, "modality_dropout missing"
    print("  ✅ test_two_stage_pretrain_includes_pretrain_a_then_pretrain")


def test_pretrain_from_stream_has_freeze_mixer():
    """Verify the pretrain_from_stream signature has freeze_mixer."""
    from training.pretrain_streaming import pretrain_from_stream
    import inspect
    sig = inspect.signature(pretrain_from_stream)
    assert 'freeze_mixer' in sig.parameters, "freeze_mixer missing from pretrain_from_stream"
    assert sig.parameters['freeze_mixer'].default == False, \
        "freeze_mixer should default to False (mixer trainable by default)"
    print("  ✅ test_pretrain_from_stream_has_freeze_mixer")


def test_train_heads_has_combined_loss_params():
    """Verify the train_heads_from_stream has the new params."""
    from training.pretrain_streaming import train_heads_from_stream
    import inspect
    sig = inspect.signature(train_heads_from_stream)
    for name in ['head_loss_weight', 'modality_dropout', 'use_cross_attention',
                 'mixer_dim', 'num_mixer_blocks', 'temperature']:
        assert name in sig.parameters, f"{name} missing from train_heads_from_stream"
    assert sig.parameters['head_loss_weight'].default == 0.1
    assert sig.parameters['modality_dropout'].default == 0.0
    print("  ✅ test_train_heads_has_combined_loss_params")


def test_cli_flags_exist():
    """Verify the CLI has the new flags for two-stage training."""
    # Parse the file's argparse by importing it
    import argparse
    import ast
    tree = ast.parse(open("training/pretrain_streaming.py").read())
    # Look for add_argument calls in main()
    flag_strs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    flag_strs.append(node.args[0].value)
    for flag in ['--use-cross-attention', '--mixer-dim', '--num-mixer-blocks',
                 '--freeze-mixer', '--stage-a-epochs', '--stage-b-head-loss-weight',
                 '--modality-dropout']:
        assert flag in flag_strs, f"CLI flag {flag} not declared in main()"
    print("  ✅ test_cli_flags_exist")


if __name__ == "__main__":
    print("=" * 60)
    print("Two-stage training tests")
    print("=" * 60)
    test_mixer_can_be_frozen_after_init()
    test_frozen_mixer_receives_no_gradients()
    test_trainable_mixer_receives_gradients()
    test_freeze_toggle_preserves_optimizer_state()
    test_no_mixer_set_trainable_is_noop()
    test_two_stage_pretrain_includes_pretrain_a_then_pretrain()
    test_pretrain_from_stream_has_freeze_mixer()
    test_train_heads_has_combined_loss_params()
    test_cli_flags_exist()
    print("=" * 60)
    print("All tests passed ✅")
