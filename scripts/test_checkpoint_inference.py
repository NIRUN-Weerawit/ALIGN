#!/usr/bin/env python3
"""Test if a V4 checkpoint can be loaded and run inference with the current code."""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_intention import ALIGNIntentionModel
from models.intention_head import DiffusionPolicyHead


def test_checkpoint(checkpoint_path: str, device_str: str = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 1. Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"Config keys: {list(cfg.keys())}")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # 2. Build model with the checkpoint's config
    print(f"\nBuilding model...")
    model = ALIGNIntentionModel(
        state_dim=cfg.get("state_dim", 256),
        mamba_output_dim=cfg.get("mamba_output_dim", 512),
        action_dim=cfg.get("action_dim", 7),
        chunk_size=cfg.get("chunk_size", 10),
        history_size=cfg.get("history_size", 1),
        num_cameras=cfg.get("num_cameras", 2),
        use_patch_tokens=cfg.get("use_patch_tokens", True),
        mamba_d_state=cfg.get("mamba_d_state", 16),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        mamba_expand=cfg.get("mamba_expand", 2),
        head_type=cfg.get("head_type", "diffusion"),
        head_d_model=cfg.get("head_d_model", 384),
        head_nhead=cfg.get("head_nhead", 4),
        head_num_layers=cfg.get("head_num_layers", 2),
        head_dim_ff=cfg.get("head_dim_ff", 1024),
        use_text=cfg.get("use_text", False),
        text_dim=cfg.get("text_dim", 256),
        compressed_dim=cfg.get("compressed_dim", 4),
        raw_dim=cfg.get("raw_dim", 768),
        use_intent_tokens=cfg.get("use_intent_tokens", True),
        num_intent_tokens=cfg.get("num_intent_tokens", 2),
        intent_dim=cfg.get("intent_dim", 512),
        use_memory_bank=cfg.get("use_memory_bank", True),
        memory_bank_len=cfg.get("memory_bank_len", 16),
    ).to(device)

    # 3. Build head/bank (need pool_out_dim from probing)
    print(f"Probing vision output shape...")
    # Simulate probing: DINOv2 on 224x224 with V cameras
    # ViT-B/14: 16x16=256 patches + 1 CLS = 257 per camera
    num_cam = cfg.get("num_cameras", 2)
    comp_dim = cfg.get("compressed_dim", 4)
    N_tok_actual = num_cam * 257  # patches + CLS per camera
    pool_out_dim = (N_tok_actual - num_cam) * comp_dim
    print(f"  N_tok_actual={N_tok_actual}, pool_out_dim={pool_out_dim}")
    model._build_head_and_bank(pool_out_dim)

    # 4. Load state dict
    print(f"\nLoading state dict...")
    sd = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(sd, strict=True)
        print("  Strict load: OK")
    except RuntimeError as e:
        print(f"  Strict load FAILED: {e}")
        print(f"\n  Trying non-strict load...")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Missing keys:   {len(missing)}")
        if missing:
            for k in missing[:10]:
                print(f"    - {k}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            for k in unexpected[:10]:
                print(f"    - {k}")

    model.eval()
    print(f"\nModel loaded successfully!")

    # 5. Run a forward pass with synthetic data
    print(f"\nRunning forward pass with synthetic data...")
    B, K, V, H, W = 1, cfg.get("chunk_size", 10), num_cam, 224, 224
    dummy_frames = torch.randint(0, 256, (B, K, V, H, W, 3), dtype=torch.uint8, device=device)
    dummy_state = torch.randn(B, K, 7, device=device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            out = model(dummy_frames, dummy_state)
            print(f"  out keys: {list(out.keys())}")
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
                else:
                    print(f"    {k}: {v}")

            h_current = out["h_seq"][:, -1]
            intent_emb = out.get("intent_emb", None)
            print(f"  h_current: {h_current.shape}")
            print(f"  intent_emb: {intent_emb.shape if intent_emb is not None else None}")

            # Memory bank step
            if getattr(model, "use_memory_bank", False) and intent_emb is not None:
                model.memory_module.reset(batch_size=B, device=device)
                z_v_current = out["z_v_pooled_seq"][:, -1]
                z_s_current = out["z_s_seq"][:, -1]
                z_v_fused, z_s_fused, intent_fused = model.memory_module(
                    z_v_current, z_s_current, intent_emb,
                )
                print(f"  Memory bank: z_v_fused={z_v_fused.shape}, z_s_fused={z_s_fused.shape}, intent_fused={intent_fused.shape}")
                h_for_head = intent_fused
            else:
                h_for_head = intent_emb if intent_emb is not None else h_current

            # Head forward
            if model.head_type == "diffusion":
                actions = model.sample_actions(
                    out["z_v_pooled_seq"], out["z_s_seq"], h_for_head,
                )
            else:
                actions = model.predict_actions(
                    out["z_v_pooled_seq"], out["z_s_seq"], h_for_head,
                )
            print(f"  actions: {actions.shape}")
            print(f"  action mean: {actions.float().mean().item():.6f}")
            print(f"  action std:  {actions.float().std().item():.6f}")
            print(f"  action finite: {torch.isfinite(actions).all().item()}")

    print(f"\n✅ Inference test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    test_checkpoint(args.checkpoint, args.device)
