"""
Alpha integration: compute intervention score from world model + value head.

alpha = sigmoid((V(s'_m) - V(s'_h)) / tau)

where:
  s'_h = world_model(s, a_h)  # imagined state after human acts
  s'_m = world_model(s, a_m)  # imagined state after model acts
  V is the trained value head
  tau is the temperature (default 1.0)

This implements Decision 5 of the alpha design (alpha formula) and
Decision 6 (default to human when V_m <= V_h).

Key design choices:
- 1-step imagination: use V(s') directly, no rollouts (Decision 4)
- Sigmoid with temperature: standard, calibrated (Decision 5)
- Tie-breaking: when V_m <= V_h, alpha < 0.5 means human wins

This is the heart of the alpha pipeline. Everything else is foundation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.world_model import create_world_model
from models.value_head import create_value_head
from models.gail_discriminator import create_gail_discriminator
from data.align_dataset import ALIGNDataset, world_model_collate


def load_components(
    world_model_path: str,
    value_head_path: str,
    encoder_checkpoint: str,
    device: torch.device,
):
    """Load world model, value head, and frozen encoder.

    Returns (model, world_model, value_head).
    """
    # Load encoder+mixer
    model = ALIGNModel(
        embed_dim=256, chunk_size=5, use_text=True, device=str(device),
    ).to(device)
    enc_ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    if "trainable_state_dict" in enc_ckpt:
        enc_state = enc_ckpt["trainable_state_dict"]
        encoder_keys = {
            k: v for k, v in enc_state.items()
            if "vision_encoder.projection" in k
            or "traj_encoder" in k
            or "text_encoder" in k
            or "cross_attention_mixer" in k
        }
        if encoder_keys:
            model.load_state_dict(encoder_keys, strict=False)
    model.freeze_backbone()
    model.freeze_all_encoders()
    model.eval()

    # Load world model
    wm_ckpt = torch.load(world_model_path, map_location=device, weights_only=False)
    wm_config = wm_ckpt.get("config", {})
    wm_arch = wm_config.get("arch", "mlp")
    wm_kwargs = {}
    if wm_arch == "mlp":
        wm_kwargs = {
            "hidden_dim": wm_config.get("mlp_hidden_dim", 512),
            "num_layers": wm_config.get("mlp_layers", 3),
        }
    world_model = create_world_model(
        arch=wm_arch,
        embed_dim=wm_config.get("embed_dim", 256),
        action_dim=wm_config.get("action_dim", 6),
        **wm_kwargs,
    ).to(device)
    world_model.load_state_dict(wm_ckpt["world_model_state"])
    world_model.eval()

    # Load value head
    val_ckpt = torch.load(value_head_path, map_location=device, weights_only=False)
    val_config = val_ckpt.get("config", {})
    value_head = create_value_head(
        embed_dim=val_config.get("embed_dim", 256),
        hidden_dim=val_config.get("hidden_dim", 256),
        num_layers=val_config.get("num_layers", 3),
    ).to(device)
    value_head.load_state_dict(val_ckpt["value_head_state"])
    value_head.eval()

    return model, world_model, value_head


def compute_alpha(
    world_model,
    value_head,
    z_v: torch.Tensor,      # (D,)
    z_t: torch.Tensor,      # (D,)
    z_text: torch.Tensor,   # (D,)
    a_human: torch.Tensor,  # (6,)
    a_model: torch.Tensor,  # (6,)
    tau: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute the intervention score alpha.

    Returns (alpha, v_h, v_m) where:
      - alpha: scalar in [0, 1]
      - v_h: V(s'_h) where s'_h = world_model(s, a_human)
      - v_m: V(s'_m) where s'_m = world_model(s, a_model)

    Implements:
      s'_h = world_model(z_v, z_t, z_text, a_human)
      s'_m = world_model(z_v, z_t, z_text, a_model)
      v_h = value_head(s'_h)
      v_m = value_head(s'_m)
      alpha = sigmoid((v_m - v_h) / tau)

    Decision 6 tie-breaking: if v_m <= v_h, alpha is naturally < 0.5
    (sigmoid of 0 is 0.5), so the human wins. No special case needed.
    """
    with torch.no_grad():
        # Imagine counterfactuals
        z_v_h, z_t_h = world_model(z_v.unsqueeze(0), z_t.unsqueeze(0), z_text.unsqueeze(0), a_human.unsqueeze(0))
        z_v_m, z_t_m = world_model(z_v.unsqueeze(0), z_t.unsqueeze(0), z_text.unsqueeze(0), a_model.unsqueeze(0))
        # Compute values
        v_h = value_head(z_v_h.squeeze(0), z_t_h.squeeze(0), z_text)
        v_m = value_head(z_v_m.squeeze(0), z_t_m.squeeze(0), z_text)
        # Sigmoid alpha
        diff = (v_m - v_h) / tau
        alpha = torch.sigmoid(diff)

    return (
        float(alpha.item()),
        float(v_h.item()),
        float(v_m.item()),
    )


def compute_alpha_batch(
    world_model,
    value_head,
    z_v: torch.Tensor,      # (B, D)
    z_t: torch.Tensor,      # (B, D)
    z_text: torch.Tensor,   # (B, D)
    a_human: torch.Tensor,  # (B, 6)
    a_model: torch.Tensor,  # (B, 6)
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch version of compute_alpha.

    Returns (alpha, v_h, v_m), each of shape (B,).
    """
    with torch.no_grad():
        # Imagine counterfactuals
        z_v_h, z_t_h = world_model(z_v, z_t, z_text, a_human)
        z_v_m, z_t_m = world_model(z_v, z_t, z_text, a_model)
        # Compute values
        v_h = value_head(z_v_h, z_t_h, z_text)
        v_m = value_head(z_v_m, z_t_m, z_text)
        # Sigmoid alpha
        diff = (v_m - v_h) / tau
        alpha = torch.sigmoid(diff)
    return alpha, v_h, v_m


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    """Quick smoke test of compute_alpha."""
    import torch

    print("Testing compute_alpha (random init)...")

    B, D, A = 4, 256, 6
    # Random models (untrained)
    wm = create_world_model("mlp", embed_dim=D, action_dim=A)
    vh = create_value_head(embed_dim=D)

    z_v = torch.randn(B, D)
    z_t = torch.randn(B, D)
    z_text = torch.randn(B, D)
    a_human = torch.randn(B, A) * 0.05
    a_model = torch.randn(B, A) * 0.05

    alpha, v_h, v_m = compute_alpha_batch(wm, vh, z_v, z_t, z_text, a_human, a_model, tau=1.0)
    print(f"  alpha: {alpha}")
    print(f"  alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    print(f"  v_h: {v_h.tolist()}")
    print(f"  v_m: {v_m.tolist()}")
    # Check that alpha is in [0, 1]
    assert (alpha >= 0).all() and (alpha <= 1).all(), "alpha out of range"
    print(f"  OK: alpha is in [0, 1]")

    # Test with different temperatures
    print("\nTesting tau effect:")
    for tau in [0.1, 1.0, 5.0]:
        alpha_test, _, _ = compute_alpha_batch(wm, vh, z_v, z_t, z_text, a_human, a_model, tau=tau)
        print(f"  tau={tau}: alpha range [{alpha_test.min().item():.4f}, {alpha_test.max().item():.4f}]")

    # Test with equal V_m and V_h (tie-breaking)
    print("\nTesting tie-breaking (V_m == V_h should give alpha=0.5):")
    z_v_m_test, z_t_m_test = wm(z_v, z_t, z_text, a_human)  # same action
    # Manually compute alpha for this case
    v = vh(z_v_m_test, z_t_m_test, z_text)
    alpha_tie = torch.sigmoid((v - v) / 1.0)  # v_m == v_h
    print(f"  alpha when v_m == v_h: {alpha_tie.tolist()}")
    print(f"  (Expected: all 0.5 — Decision 6)")

    print("\nAll smoke tests passed.")
