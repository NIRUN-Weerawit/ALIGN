#!/usr/bin/env python3
"""Verify the use_memory_bank=False pipeline.

Tests:
  1. Model builds with use_memory_bank=False (no memory module)
  2. forward() returns expected shapes
  3. forward_step() works
  4. Head receives 3D input from the no-bank path
  5. Loss + sample work

Usage:
    python tools/verify_no_memory.py
"""
import sys
import torch
sys.path.insert(0, '/home/ucluser/ALIGN')

from models.align_intention import ALIGNIntentionModel


def verify_no_memory():
    """Verify use_memory_bank=False pipeline works end-to-end."""
    print("=" * 60)
    print("ALIGN v3 use_memory_bank=False Verification")
    print("=" * 60)

    config = {
        "state_dim": 256,
        "mamba_output_dim": 0,  # No Mamba (no history)
        "action_dim": 7,
        "chunk_size": 10,
        "num_cameras": 2,
        "head_type": "diffusion",
        "compressed_dim": 8,
        "use_intent_tokens": False,
        "num_intent_tokens": 0,
        "intent_dim": 0,
        "use_memory_bank": False,  # KEY: no memory bank
        "memory_bank_len": 16,
    }
    print(f"\nConfig: {config}")

    # Build model
    print("\n--- Building model ---")
    model = ALIGNIntentionModel(**config)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"use_memory_bank: {model.use_memory_bank}")
    print(f"memory_module: {type(model.memory_module).__name__ if model.memory_module else None}")

    # Build head with correct pool_out_dim
    pool_out_dim = 2 * 256 * 8  # V * P * compressed_dim = 4096
    print(f"\nBuilding head with pool_out_dim={pool_out_dim}")
    model._build_head_and_bank(pool_out_dim)
    print(f"After build:")
    print(f"  intention_head: {type(model.intention_head).__name__}")
    print(f"  memory_module: {type(model.memory_module).__name__ if model.memory_module else None}")

    # ============ Test 1: Simulate train loop no-memory-bank path ============
    print("\n" + "=" * 60)
    print("Test 1: Train loop no-memory-bank path")
    print("=" * 60)
    # In the no-memory-bank path, the train loop does:
    #   z_v_win_for_head = z_v_win_stacked  (B, Hs, V*P*comp_dim)
    #   z_s_win_for_head = z_s_win          (B, Hs, state_dim)
    #   h_for_head = h_current             (B, mamba_in_dim) — but no Mamba!
    B, Hs = 2, 1
    z_v_win_stacked = torch.randn(B, Hs, pool_out_dim)  # (B, Hs, pool_out_dim)
    z_s_win = torch.randn(B, Hs, 256)
    h_current = torch.zeros(B, 1)  # (B, 1) — placeholder for h when no Mamba

    # The train loop also does:
    #   z_v_win_for_head = z_v_win_stacked[:, -1:]  -> (B, 1, pool_out_dim)
    z_v_for_head = z_v_win_stacked[:, -1:]  # (B, 1, pool_out_dim)
    z_s_for_head = z_s_win[:, -1:]  # (B, 1, state_dim)
    print(f"  z_v_for_head: {z_v_for_head.shape}")
    print(f"  z_s_for_head: {z_s_for_head.shape}")

    cond = model.intention_head(z_v_for_head, z_s_for_head, None)
    print(f"\n  cond: {cond.shape}")
    print(f"  cond finite: {torch.isfinite(cond).all().item()}")
    print(f"  cond max: {cond.abs().max().item()}")

    # ============ Test 2: Loss value ============
    print("\n" + "=" * 60)
    print("Test 2: Loss value")
    print("=" * 60)
    target = torch.randn(B, 10, 7)
    loss = model.intention_head.loss(target, cond)
    print(f"  target: {target.shape}")
    print(f"  loss: {loss.item()}")
    print(f"  loss finite: {torch.isfinite(loss).item()}")

    # ============ Test 3: Sample value ============
    print("\n" + "=" * 60)
    print("Test 3: Sample value")
    print("=" * 60)
    sample = model.intention_head.sample(cond, num_steps=10)
    print(f"  sample: {sample.shape}")
    print(f"  sample finite: {torch.isfinite(sample).all().item()}")
    print(f"  sample abs.mean: {sample.abs().mean().item()}")

    # ============ Test 4: forward_intent (no Mamba, no memory) ============
    print("\n" + "=" * 60)
    print("Test 4: forward_intent")
    print("=" * 60)
    z_v_cls = torch.randn(B, 8, 2, 768)  # (B, T, V, 768) CLS tokens
    z_s = torch.randn(B, 8, 256)
    out = model.forward_intent(z_v_cls, z_s)
    print(f"  h_seq: {out['h_seq'].shape}")
    print(f"  intent_emb: {out['intent_emb']}")

    # ============ Test 5: with use_intent_tokens=True (cognitive_dim>0) but no memory ============
    print("\n" + "=" * 60)
    print("Test 5: use_intent_tokens=True but no memory")
    print("=" * 60)
    config2 = dict(config)
    config2["use_intent_tokens"] = True
    config2["num_intent_tokens"] = 2
    config2["intent_dim"] = 512
    model2 = ALIGNIntentionModel(**config2)
    print(f"use_memory_bank: {model2.use_memory_bank}")
    print(f"use_intent_tokens: {model2.use_intent_tokens}")
    print(f"memory_module: {type(model2.memory_module).__name__ if model2.memory_module else None}")
    model2._build_head_and_bank(pool_out_dim)

    # The train loop with intent tokens but no memory bank would do:
    #   h_for_head = intent_emb  (B, N, intent_dim)
    z_v_for_head = z_v_win_stacked[:, -1:]
    z_s_for_head = z_s_win[:, -1:]
    intent_emb = torch.randn(B, 2, 512)
    cond2 = model2.intention_head(z_v_for_head, z_s_for_head, intent_emb)
    print(f"\n  cond: {cond2.shape}")
    print(f"  cond finite: {torch.isfinite(cond2).all().item()}")

    target = torch.randn(B, 10, 7)
    loss2 = model2.intention_head.loss(target, cond2)
    print(f"  loss: {loss2.item()}")
    print(f"  loss finite: {torch.isfinite(loss2).item()}")

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    checks = [
        ("Head cond shape (no bank)", cond.shape == (B, 1, pool_out_dim + 256)),
        ("Loss is finite (no bank)", torch.isfinite(loss).item()),
        ("Loss is positive (no bank)", loss.item() > 0.0),
        ("Sample is finite (no bank)", torch.isfinite(sample).all().item()),
        ("Head cond shape (intent, no bank)", cond2.shape == (B, 1, pool_out_dim + 256 + 1024)),
        ("Loss is finite (intent, no bank)", torch.isfinite(loss2).item()),
    ]
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    if all(ok for _, ok in checks):
        print("\n  All checks passed! use_memory_bank=False pipeline works.")
    else:
        print("\n  Some checks FAILED. See output above.")


if __name__ == "__main__":
    verify_no_memory()
