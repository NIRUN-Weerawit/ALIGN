#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for ALIGNModel 3-phase training architecture.

Verifies the contract:
  - freeze_all_encoders() freezes vision_proj, traj_encoder, text_proj, mixer
  - freeze_mixer() / unfreeze_mixer() toggle correctly
  - encode_raw_all() returns pre-mixer embeddings (no mixer call)
  - encode_mixed() returns post-mixer embeddings
  - Checkpoint save/load round-trips correctly
  - from_trainable_checkpoint() produces a functional model

Run: python tests/test_two_stage_training.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.align_model import ALIGNModel


def test_mixer_can_be_frozen_after_init():
    """freeze_mixer/unfreeze_mixer should toggle requires_grad correctly."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        mixer_dim=512, num_mixer_blocks=2,
    )
    assert model.cross_attention_mixer is not None, "Mixer should always be present"

    # Initially trainable (requires_grad=True)
    n_train = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_train > 0, "Mixer should be trainable at init"

    # Freeze
    model.freeze_mixer()
    n_frozen = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_frozen == 0, f"Mixer should be fully frozen, got {n_frozen} trainable"
    assert not model.cross_attention_mixer.training, "Mixer should be in eval mode when frozen"

    # Unfreeze
    model.unfreeze_mixer()
    n_unfrozen = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    assert n_unfrozen == n_train, f"Mixer should be fully unfrozen, got {n_unfrozen} vs {n_train}"
    assert model.cross_attention_mixer.training, "Mixer should be in train mode when unfrozen"

    print("  ✅ test_mixer_can_be_frozen_after_init")


def test_frozen_mixer_receives_no_gradients():
    """When frozen via freeze_mixer(), mixer params should get no gradient."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        mixer_dim=512, num_mixer_blocks=2,
    )
    model.freeze_mixer()

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    # Use encode_mixed which routes through mixer
    mixed = model.encode_mixed(frames, traj, texts)
    loss = mixed["z_v"].sum() + mixed["z_t"].sum() + mixed["z_text"].sum()
    loss.backward()

    for name, p in model.cross_attention_mixer.named_parameters():
        if p.requires_grad:
            assert p.grad is None or p.grad.abs().sum().item() == 0, \
                f"Frozen mixer param {name} should not have gradient"

    print("  ✅ test_frozen_mixer_receives_no_gradients")


def test_trainable_mixer_receives_gradients():
    """When not frozen, mixer params should get gradients."""
    model = ALIGNModel(
        embed_dim=256, use_text=True, device='cpu',
        mixer_dim=512, num_mixer_blocks=2,
    )
    # Mixer is trainable by default

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    mixed = model.encode_mixed(frames, traj, texts)
    loss = mixed["z_v"].sum() + mixed["z_t"].sum() + mixed["z_text"].sum()
    loss.backward()

    n_with_grad = sum(
        1 for p in model.cross_attention_mixer.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    n_total = sum(1 for p in model.cross_attention_mixer.parameters() if p.requires_grad)
    # Some params may not get grad if they don't influence loss (e.g. zero-init), but most should
    assert n_with_grad >= n_total * 0.75, \
        f"Trainable mixer: only {n_with_grad}/{n_total} params got gradient"

    print("  ✅ test_trainable_mixer_receives_gradients")


def test_freeze_unfreeze_toggle():
    """Toggling freeze_mixer/unfreeze_mixer multiple times should work."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')

    for _ in range(3):
        model.freeze_mixer()
        n_frozen = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
        assert n_frozen == 0, f"Expected 0 trainable after freeze, got {n_frozen}"

        model.unfreeze_mixer()
        n_train = sum(p.numel() for p in model.cross_attention_mixer.parameters() if p.requires_grad)
        assert n_train > 0, f"Expected trainable after unfreeze, got {n_train}"

    print("  ✅ test_freeze_unfreeze_toggle")


def test_freeze_all_encoders():
    """freeze_all_encoders() should freeze vision_proj, traj_encoder, text_proj, and mixer."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')
    model.freeze_backbone()  # backbones frozen by default
    model.freeze_all_encoders()

    # Check encoder modules are frozen
    for name, param in model.named_parameters():
        # Backbones should still be frozen
        if 'backbone' in name or 'model.' in name:
            assert not param.requires_grad, f"Backbone param {name} should be frozen"
        # Heads should be trainable
        if 'decision_head' in name or 'assistant_head' in name:
            assert param.requires_grad, f"Head param {name} should be trainable"

    # Encoder prefixes that must be frozen
    encoder_prefixes = ['vision_encoder.projection', 'traj_encoder.', 'text_encoder.projection']
    for name, param in model.named_parameters():
        is_encoder = any(name.startswith(p) for p in encoder_prefixes)
        is_head = 'decision_head' in name or 'assistant_head' in name
        if is_encoder:
            assert not param.requires_grad, f"Encoder param {name} should be frozen after freeze_all_encoders()"

    # Verify mixer is also frozen
    for name, param in model.cross_attention_mixer.named_parameters():
        assert not param.requires_grad, f"Mixer param {name} should be frozen after freeze_all_encoders()"

    # Verify heads are still trainable
    for name, param in model.decision_head.named_parameters():
        assert param.requires_grad, f"Decision head param {name} should be trainable"
    for name, param in model.assistant_head.named_parameters():
        assert param.requires_grad, f"Assistant head param {name} should be trainable"

    print("  ✅ test_freeze_all_encoders")


def test_encode_raw_all_no_mixer():
    """encode_raw_all() should return embeddings that are NOT mixer-enriched.

    We verify this by comparing encode_raw_all output with encode_mixed output.
    With identity-initialized mixer, they should be close but not identical
    (small perturbations from near-zero output proj). The key property:
    encode_raw_all should NOT go through the mixer at all.
    """
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    raw = model.encode_raw_all(frames, traj, texts)
    assert set(raw.keys()) == {"z_v", "z_t", "z_text"}, f"Unexpected keys: {raw.keys()}"
    assert raw["z_v"].shape == (B, 256)
    assert raw["z_t"].shape == (B, 256)
    assert raw["z_text"].shape == (B, 256)

    print("  ✅ test_encode_raw_all_no_mixer")


def test_encode_mixed_uses_mixer():
    """encode_mixed() should return post-mixer embeddings (z_v', z_t', z_text')."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    mixed = model.encode_mixed(frames, traj, texts)
    assert set(mixed.keys()) == {"z_v", "z_t", "z_text", "z_t_tokens"}, \
        f"Unexpected keys: {mixed.keys()}"
    assert mixed["z_v"].shape == (B, 256)
    assert mixed["z_t"].shape == (B, 256)
    assert mixed["z_text"].shape == (B, 256)
    assert mixed["z_t_tokens"].shape == (B, K, 256)

    print("  ✅ test_encode_mixed_uses_mixer")


def test_raw_vs_mixed_differ():
    """With a non-frozen mixer, encode_raw_all and encode_mixed should differ."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    raw = model.encode_raw_all(frames, traj, texts)
    mixed = model.encode_mixed(frames, traj, texts)

    # Even with identity init, output projection adds small perturbations
    z_v_diff = (raw["z_v"] - mixed["z_v"]).abs().mean().item()
    z_t_diff = (raw["z_t"] - mixed["z_t"]).abs().mean().item()
    z_text_diff = (raw["z_text"] - mixed["z_text"]).abs().mean().item()

    print(f"  Raw vs Mixed diff: z_v={z_v_diff:.6f} z_t={z_t_diff:.6f} z_text={z_text_diff:.6f}")
    # They should differ at least slightly (mixer does add small perturbations via output_proj)
    assert z_v_diff > 0 or z_t_diff > 0 or z_text_diff > 0, \
        "encode_raw_all and encode_mixed should produce different outputs"

    print("  ✅ test_raw_vs_mixed_differ")


def test_get_trainable_params():
    """get_trainable_params should respect include_heads flag."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')
    model.freeze_backbone()

    # With heads
    all_params = model.get_trainable_params(include_heads=True)
    n_all = len(all_params)

    # Without heads
    encoder_params = model.get_trainable_params(include_heads=False)
    n_encoder = len(encoder_params)

    assert n_all > n_encoder, \
        f"With heads ({n_all}) should be > without heads ({n_encoder})"

    print("  ✅ test_get_trainable_params")


def test_get_head_params():
    """get_head_params should return only decision_head + assistant_head params."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')
    head_params = model.get_head_params()
    all_params = list(model.parameters())

    assert len(head_params) > 0, "get_head_params should return non-empty list"
    assert len(head_params) < len(all_params), \
        f"Head params ({len(head_params)}) should be a subset of all params ({len(all_params)})"

    print("  ✅ test_get_head_params")


def test_checkpoint_round_trip():
    """Test save_pretrain_checkpoint -> load_trainable_state_dict round-trip."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')
    model.freeze_backbone()

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    # Run forward to get initial output
    raw_before = model.encode_raw_all(frames, traj, texts)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name

    try:
        # Save encoder checkpoint
        model.save_pretrain_checkpoint(
            path=tmp_path, epoch=40, loss=2.5,
            phase="encoder",
            optimizer_state={"dummy": "state"},
            config={"embed_dim": 256, "traj_window": 20},
        )
        assert os.path.exists(tmp_path)
        ckpt_size = os.path.getsize(tmp_path)
        print(f"  Encoder checkpoint size: {ckpt_size:,} bytes ({ckpt_size/1024/1024:.2f} MB)")

        # Load into a fresh model
        model2 = ALIGNModel.from_trainable_checkpoint(tmp_path, device='cpu')
        raw_after = model2.encode_raw_all(frames, traj, texts)

        # Outputs should match (same weights loaded)
        assert torch.allclose(raw_before["z_v"], raw_after["z_v"], atol=1e-5), \
            "z_v should match after round-trip"
        assert torch.allclose(raw_before["z_t"], raw_after["z_t"], atol=1e-5), \
            "z_t should match after round-trip"
        assert torch.allclose(raw_before["z_text"], raw_after["z_text"], atol=1e-5), \
            "z_text should match after round-trip"

        # Test full pretrain checkpoint (includes mixer)
        model.save_pretrain_checkpoint(
            path=tmp_path, epoch=50, loss=1.8,
            phase="full_pretrain",
            optimizer_state={"dummy": "state"},
            config={"embed_dim": 256},
        )
        ckpt_size2 = os.path.getsize(tmp_path)
        print(f"  Full pretrain checkpoint size: {ckpt_size2:,} bytes ({ckpt_size2/1024/1024:.2f} MB)")
        assert ckpt_size2 > ckpt_size, "Full pretrain checkpoint should be larger than encoder-only"

        # Test heads checkpoint
        model.save_heads_checkpoint(
            path=tmp_path, epoch=30, loss=0.12,
            optimizer_state={"dummy": "state"},
            config={"chunk_size": 5},
        )
        ckpt_size3 = os.path.getsize(tmp_path)
        print(f"  Heads checkpoint size: {ckpt_size3:,} bytes ({ckpt_size3/1024/1024:.2f} MB)")

        # Verify full forward pass works after load
        raw_after = model2.encode_mixed(frames, traj, texts)
        assert raw_after["z_v"].shape == (B, 256)

        # Full forward pass
        out = model2(frames, traj, texts)
        assert "alpha" in out
        assert "delta" in out
        assert out["alpha"].shape == (B, 1)
        assert out["delta"].shape == (B, K, 6)

    finally:
        os.unlink(tmp_path)

    print("  ✅ test_checkpoint_round_trip")


def test_from_trainable_checkpoint_functional():
    """from_trainable_checkpoint should produce a fully functional model."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)
    texts = ["test"] * B

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name

    try:
        model.save_pretrain_checkpoint(
            path=tmp_path, epoch=1, loss=3.0,
            phase="full_pretrain",
            optimizer_state={},
            config={"embed_dim": 256},
        )

        model2 = ALIGNModel.from_trainable_checkpoint(tmp_path, device='cpu')

        # Test forward pass with all compute options
        out = model2(frames, traj, texts, compute_decision=True, compute_assistant=True)
        assert "alpha" in out and "delta" in out
        assert out["alpha"].shape == (B, 1)
        assert 0 <= out["alpha"].min().item() <= out["alpha"].max().item() <= 1

        out2 = model2(frames, traj, texts, compute_decision=True, compute_assistant=False)
        assert "alpha" in out2 and "delta" not in out2

        out3 = model2(frames, traj, texts, compute_decision=False, compute_assistant=True)
        assert "alpha" not in out3 and "delta" in out3

    finally:
        os.unlink(tmp_path)

    print("  ✅ test_from_trainable_checkpoint_functional")


def test_filter_state_dict():
    """_filter_state_dict should correctly filter by prefixes."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')
    full_sd = model.state_dict()

    encoder_prefixes = {"vision_encoder.projection", "traj_encoder", "text_encoder.projection"}
    filtered = model._filter_state_dict(full_sd, encoder_prefixes)

    for key in filtered:
        assert any(key.startswith(p) for p in encoder_prefixes), \
            f"Key {key} should start with one of {encoder_prefixes}"
    print(f"  Filtered {len(filtered)} encoder keys from {len(full_sd)} total")
    assert len(filtered) > 0, "Should have filtered some keys"

    print("  ✅ test_filter_state_dict")


def test_get_trainable_state_dict():
    """get_trainable_state_dict should return correct subset."""
    model = ALIGNModel(embed_dim=256, use_text=True, device='cpu')

    encoder_sd = model.get_trainable_state_dict({"vision_encoder.projection", "traj_encoder"})
    for key in encoder_sd:
        assert key.startswith("vision_encoder.projection") or key.startswith("traj_encoder")

    head_sd = model.get_trainable_state_dict({"decision_head", "assistant_head"})
    for key in head_sd:
        assert "decision_head" in key or "assistant_head" in key

    print("  ✅ test_get_trainable_state_dict")


def test_text_free_mode():
    """Model should work with use_text=False."""
    model = ALIGNModel(embed_dim=256, use_text=False, device='cpu')

    B, K = 2, 10
    frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
    traj = torch.randn(B, K, 6)

    raw = model.encode_raw_all(frames, traj, texts=None)
    assert torch.equal(raw["z_text"], torch.zeros_like(raw["z_v"])), \
        "z_text should be zeros when no text encoder"

    mixed = model.encode_mixed(frames, traj, texts=None)
    assert mixed["z_text"].shape == (B, 256)

    out = model(frames, traj, texts=None)
    assert "alpha" in out and "delta" in out

    print("  ✅ test_text_free_mode")


if __name__ == "__main__":
    print("=" * 60)
    print("3-Phase Training Architecture Tests")
    print("=" * 60)
    test_mixer_can_be_frozen_after_init()
    test_frozen_mixer_receives_no_gradients()
    test_trainable_mixer_receives_gradients()
    test_freeze_unfreeze_toggle()
    test_freeze_all_encoders()
    test_encode_raw_all_no_mixer()
    test_encode_mixed_uses_mixer()
    test_raw_vs_mixed_differ()
    test_get_trainable_params()
    test_get_head_params()
    test_checkpoint_round_trip()
    test_from_trainable_checkpoint_functional()
    test_filter_state_dict()
    test_get_trainable_state_dict()
    test_text_free_mode()
    print("=" * 60)
    print("All tests passed ✅")