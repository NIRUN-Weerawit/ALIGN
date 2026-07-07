#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PoseDeltaToAction — learned inverse-dynamics connector model (bounded).

Maps EEF pose deltas (m, rad) → OSC actions (normalized), with a
per-dim bounded output:

    out = tanh(net(x)) * range + mid
    range = (action_max - action_min) / 2
    mid   = (action_max + action_min) / 2

so out ∈ [action_min[d], action_max[d]] for each output dim d.
The bounded output is hard-architectural: no input, no matter how
out-of-distribution, can produce an output outside the per-dim range.

The per-dim asymmetric range preserves the actual LIBERO action
distribution (e.g. rotation X ∈ [-0.1875, +0.1971]) — unlike
`tanh × symmetric_bound` which would clip the positive side.

See training/train_pose_to_action.py for the training script.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PoseDeltaToAction(nn.Module):
    """Maps EEF pose delta (m, rad) → OSC action (normalized), bounded.

    Input:  (B, 6) — [dx, dy, dz, dax, day, daz] in meters and radians
    Output: (B, 6) — [ax, ay, az, arx, ary, arz] in OSC action space
            Hard-bounded to [action_min[d], action_max[d]] per dim.

    The mapping is the inverse dynamics of the OSC controller. For a
    simple controller with a fixed Jacobian, this reduces to per-axis
    scaling. For real controllers with configuration-dependent Jacobians,
    the learned model captures the non-linear coupling.

    Args:
        pose_dim: input dim (default 6)
        action_dim: output dim (default 6)
        hidden_dim: MLP hidden width (default 128)
        action_min: (action_dim,) sequence of per-dim min output.
        action_max: (action_dim,) sequence of per-dim max output.

    Output for sample i and dim d is guaranteed to be in
    [action_min[d], action_max[d]].

    Note: `tanh` saturation near ±1 is a real issue with MSE loss. If
    the model's pre-tanh output is consistently large, gradients vanish
    in the saturated region. See training/train_pose_to_action.py for
    the loss choice.
    """

    def __init__(
        self,
        pose_dim: int = 6,
        action_dim: int = 6,
        hidden_dim: int = 128,
        action_min: Sequence[float] = (-1.0,) * 6,
        action_max: Sequence[float] = (+1.0,) * 6,
    ):
        super().__init__()
        if len(action_min) != action_dim or len(action_max) != action_dim:
            raise ValueError(
                f"action_min and action_max must each have length {action_dim}, "
                f"got {len(action_min)} and {len(action_max)}"
            )
        self.pose_dim = pose_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Per-dim bounds as buffers so they move with .to(device) and
        # are saved in state_dict alongside the network weights.
        am = torch.tensor(list(action_min), dtype=torch.float32)
        ax = torch.tensor(list(action_max), dtype=torch.float32)
        if (ax <= am).any():
            raise ValueError(
                f"action_max must be strictly greater than action_min per dim, "
                f"got min={am.tolist()}, max={ax.tolist()}"
            )
        self.register_buffer("action_min", am)
        self.register_buffer("action_max", ax)
        # Pre-compute the affine parameters (constant after construction)
        self.register_buffer("action_range", (ax - am) / 2.0)
        self.register_buffer("action_mid",   (ax + am) / 2.0)

    def forward(self, pose_delta: torch.Tensor) -> torch.Tensor:
        h = self.net(pose_delta)
        return torch.tanh(h) * self.action_range + self.action_mid

    @property
    def output_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_min, self.action_max

    def extra_repr(self) -> str:
        return (
            f"pose_dim={self.pose_dim}, action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim}, bounded=per-dim"
        )


# ================================================================
# Smoke test
# ================================================================

if __name__ == "__main__":
    """Quick smoke test of bounded PoseDeltaToAction."""
    import torch

    print("Testing PoseDeltaToAction (bounded) — symmetric bounds...")
    model = PoseDeltaToAction(
        action_min=[-1, -1, -1, -0.2, -0.4, -0.4],
        action_max=[ 1,  1,  1,  0.2,  0.4,  0.4],
    )
    x = torch.randn(4, 6) * 0.01
    out = model(x)
    print(f"  in shape:  {tuple(x.shape)}  out shape: {tuple(out.shape)}")
    print(f"  out range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    assert (out >= model.action_min).all() and (out <= model.action_max).all()
    print(f"  output_bounds: min={model.action_min.tolist()}  max={model.action_max.tolist()}")

    print("\nTesting — asymmetric bounds (real LIBERO ranges)...")
    model_a = PoseDeltaToAction(
        action_min=[-0.9375, -0.9375, -0.9375, -0.1875, -0.3675, -0.36],
        action_max=[ 0.9375,  0.9375,  0.9375,  0.1971,  0.3364,  0.375],
    )
    out_a = model_a(x)
    for d in range(6):
        lo = model_a.action_min[d].item()
        hi = model_a.action_max[d].item()
        assert out_a[:, d].min().item() >= lo - 1e-6
        assert out_a[:, d].max().item() <= hi + 1e-6
    print(f"  d=3 (rotation X) preserved asymmetric range: "
          f"[{model_a.action_min[3]:.4f}, {model_a.action_max[3]:.4f}]")
    print(f"  all 6 dims in their asymmetric range ✓")

    print("\nTesting — extreme extrapolation (10x training input)...")
    x_extreme = torch.randn(100, 6) * 0.1
    out_extreme = model_a(x_extreme)
    print(f"  in range:  [{x_extreme.min():.4f}, {x_extreme.max():.4f}]")
    print(f"  out range: [{out_extreme.min():.4f}, {out_extreme.max():.4f}]")
    for d in range(6):
        lo, hi = model_a.action_min[d].item(), model_a.action_max[d].item()
        assert out_extreme[:, d].min().item() >= lo - 1e-6
        assert out_extreme[:, d].max().item() <= hi + 1e-6
    print(f"  bounded by construction ✓ (no extrapolation past bounds)")

    print("\nTesting — default symmetric ±1...")
    model_d = PoseDeltaToAction()
    x_default = torch.randn(8, 6) * 5.0
    out_default = model_d(x_default)
    print(f"  in range:  [{x_default.min():.4f}, {x_default.max():.4f}]")
    print(f"  out range: [{out_default.min():.4f}, {out_default.max():.4f}]")
    assert (out_default >= -1.0).all() and (out_default <= 1.0).all()
    print(f"  default ±1 bounds hold ✓")

    print("\nTesting — invalid bounds rejected...")
    try:
        PoseDeltaToAction(action_min=[1, 1, 1, 1, 1, 1], action_max=[-1, -1, -1, -1, -1, -1])
    except ValueError as e:
        print(f"  rejected with: {e} ✓")
    try:
        PoseDeltaToAction(action_min=[-1] * 5, action_max=[1] * 5)
    except ValueError as e:
        print(f"  rejected with: {e} ✓")

    print("\nAll smoke tests passed.")
