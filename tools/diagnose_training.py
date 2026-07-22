#!/usr/bin/env python3
"""Diagnostic script to verify ALIGN v3 training pipeline.

Run this to check if all the pieces are wired up correctly:
  - Memory bank output shape
  - Head input/output shape
  - Loss value (should be ~1.0 for untrained model)

Usage:
    python tools/diagnose_training.py
"""
import sys
import torch
sys.path.insert(0, '/home/ucluser/ALIGN')

from models.align_intention import ALIGNIntentionModel


def diagnose():
    """Build a model with the user's config and run one forward pass."""
    print("=" * 60)
    print("ALIGN v3 Training Diagnostic")
    print("=" * 60)

    # User's config
    config = {
        "state_dim": 256,
        "mamba_output_dim": 0,  # no Mamba to avoid CUDA
        "action_dim": 7,
        "chunk_size": 10,
        "num_cameras": 2,
        "head_type": "diffusion",
        "compressed_dim": 8,  # user used 8 (was 4 in earlier runs)
        "use_intent_tokens": True,
        "num_intent_tokens": 2,
        "intent_dim": 512,
        "use_memory_bank": True,
        "memory_bank_len": 16,
    }
    print(f"\nConfig: {config}")

    # Build model
    model = ALIGNIntentionModel(**config)
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")

    # Build head with correct pool_out_dim
    pool_out_dim = 2 * 256 * 8  # V * P * compressed_dim = 4096
    print(f"\nBuilding head with pool_out_dim={pool_out_dim}")
    model._build_head_and_bank(pool_out_dim)
    print(f"After build:")
    print(f"  intention_head: {type(model.intention_head).__name__}")
    print(f"  pool_out_dim: {model.pool_out_dim}")
    print(f"  memory_module: {type(model.memory_module).__name__}")

    # ============ Test 1: Memory bank output shape ============
    print("\n" + "=" * 60)
    print("Test 1: Memory bank output shape")
    print("=" * 60)
    B = 2
    N_tok, comp_dim = 512, 8
    state_dim = 256
    intent_dim = 512

    z_v_current = torch.randn(B, N_tok * comp_dim)  # (B, pool_out_dim) 2D
    z_s_current = torch.randn(B, state_dim)  # (B, state_dim) 2D
    intent_emb = torch.randn(B, 2, intent_dim)  # (B, N, intent_dim) 3D

    model.memory_module.reset(batch_size=B, device=torch.device('cpu'))
    z_v_fused, z_s_fused, intent_fused = model.memory_module(
        z_v_current, z_s_current, intent_emb
    )
    print(f"z_v_fused: {z_v_fused.shape}  (should be (B, {pool_out_dim}))")
    print(f"z_s_fused: {z_s_fused.shape}  (should be (B, {state_dim}))")
    print(f"intent_fused: {intent_fused.shape}  (should be (B, N, {intent_dim}))")

    expected_cond_dim = pool_out_dim + state_dim + 2 * intent_dim
    print(f"\nExpected cond_dim: {expected_cond_dim}")

    # ============ Test 2: Head with 3D input ============
    print("\n" + "=" * 60)
    print("Test 2: Head with 3D input (after unsqueeze)")
    print("=" * 60)
    z_v_3d = z_v_fused.unsqueeze(1)  # (B, 1, pool_out_dim)
    z_s_3d = z_s_fused.unsqueeze(1)  # (B, 1, state_dim)
    print(f"z_v_3d: {z_v_3d.shape}")
    print(f"z_s_3d: {z_s_3d.shape}")
    print(f"intent_fused: {intent_fused.shape}")

    cond = model.intention_head(z_v_3d, z_s_3d, intent_fused)
    print(f"\ncond: {cond.shape}  (should be (B, 1, {expected_cond_dim}))")
    print(f"cond finite: {torch.isfinite(cond).all().item()}")

    # ============ Test 3: Loss value ============
    print("\n" + "=" * 60)
    print("Test 3: Loss value (should be ~1.0 for untrained)")
    print("=" * 60)
    target = torch.randn(B, 10, 7)  # (B, K, action_dim)
    loss = model.intention_head.loss(target, cond)
    print(f"target: {target.shape}")
    print(f"loss: {loss.item()}")
    print(f"loss finite: {torch.isfinite(loss).item()}")

    # ============ Test 4: Sample value ============
    print("\n" + "=" * 60)
    print("Test 4: Sample value")
    print("=" * 60)
    sample = model.intention_head.sample(cond, num_steps=10)
    print(f"sample: {sample.shape}")
    print(f"sample finite: {torch.isfinite(sample).all().item()}")
    print(f"sample abs.mean: {sample.abs().mean().item()}")

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    checks = [
        ("Memory bank output", z_v_fused.shape == (B, pool_out_dim)),
        ("Head cond shape", cond.shape == (B, 1, expected_cond_dim)),
        ("Loss is finite", torch.isfinite(loss).item()),
        ("Loss is positive", loss.item() > 0.0),
        ("Sample is finite", torch.isfinite(sample).all().item()),
    ]
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    if all(ok for _, ok in checks):
        print("\n  All checks passed! Training should work.")
    else:
        print("\n  Some checks FAILED. See output above.")


if __name__ == "__main__":
    diagnose()
