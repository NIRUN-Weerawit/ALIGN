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
    """Maps EEF pose delta + current pose → OSC action (bounded).

    Input:  (B, pose_dim) — cat(pose_delta, current_pose)
            pose_delta: [dx, dy, dz, dax, day, daz] in meters and radians
            current_pose: [x, y, z, ax, ay, az] current EEF pose
    Output: (B, action_dim) — [ax, ay, az, arx, ary, arz] in OSC action space
            Hard-bounded to [action_min[d], action_max[d]] per dim.

    The current pose is included because the OSC Jacobian is
    configuration-dependent — the same pose delta requires different
    actions depending on the arm's current position. The EEF pose is
    a proxy for the arm configuration.

    Args:
        pose_dim: total input dim (default 12 = 6 delta + 6 current)
        action_dim: output dim (default 6)
        hidden_dim: MLP hidden width (default 128)
        action_min: (action_dim,) sequence of per-dim min output.
        action_max: (action_dim,) sequence of per-dim max output.
    """

    def __init__(
        self,
        pose_dim: int = 12,  # 6 delta + 6 current pose
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

    def forward(self, pose_delta: torch.Tensor, current_pose: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            pose_delta: (B, 6) — EEF pose delta in (m, rad)
            current_pose: (B, 6) — current EEF pose in (m, rad).
                          If None, zeros are used (backward compat).
        Returns:
            (B, action_dim) bounded OSC action
        """
        if current_pose is not None:
            x = torch.cat([pose_delta, current_pose], dim=-1)
        else:
            # Backward compat: pad with zeros for current pose
            B = pose_delta.shape[0]
            if pose_delta.shape[-1] == self.pose_dim:
                x = pose_delta  # already concatenated
            else:
                zeros = torch.zeros(B, self.pose_dim - pose_delta.shape[-1],
                                    device=pose_delta.device, dtype=pose_delta.dtype)
                x = torch.cat([pose_delta, zeros], dim=-1)
        h = self.net(x)
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
        pose_dim=12,
        action_min=[-1, -1, -1, -0.2, -0.4, -0.4],
        action_max=[ 1,  1,  1,  0.2,  0.4,  0.4],
    )
    x_delta = torch.randn(4, 6) * 0.01
    x_pose = torch.randn(4, 6) * 0.1
    out = model(x_delta, x_pose)
    print(f"  in shape:  delta={tuple(x_delta.shape)} pose={tuple(x_pose.shape)}  out shape: {tuple(out.shape)}")
    print(f"  out range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    assert (out >= model.action_min).all() and (out <= model.action_max).all()
    print(f"  output_bounds: min={model.action_min.tolist()}  max={model.action_max.tolist()}")

    print("\nTesting — asymmetric bounds (real LIBERO ranges)...")
    model_a = PoseDeltaToAction(
        pose_dim=12,
        action_min=[-0.9375, -0.9375, -0.9375, -0.1875, -0.3675, -0.36],
        action_max=[ 0.9375,  0.9375,  0.9375,  0.1971,  0.3364,  0.375],
    )
    out_a = model_a(x_delta, x_pose)
    for d in range(6):
        lo = model_a.action_min[d].item()
        hi = model_a.action_max[d].item()
        assert out_a[:, d].min().item() >= lo - 1e-6
        assert out_a[:, d].max().item() <= hi + 1e-6
    print(f"  d=3 (rotation X) preserved asymmetric range: "
          f"[{model_a.action_min[3]:.4f}, {model_a.action_max[3]:.4f}]")
    print(f"  all 6 dims in their asymmetric range ✓")

    print("\nTesting — extreme extrapolation (10x training input)...")
    x_extreme_delta = torch.randn(100, 6) * 0.1
    x_extreme_pose = torch.randn(100, 6) * 0.5
    out_extreme = model_a(x_extreme_delta, x_extreme_pose)
    print(f"  in range:  delta=[{x_extreme_delta.min():.4f}, {x_extreme_delta.max():.4f}]  pose=[{x_extreme_pose.min():.4f}, {x_extreme_pose.max():.4f}]")
    print(f"  out range: [{out_extreme.min():.4f}, {out_extreme.max():.4f}]")
    for d in range(6):
        lo, hi = model_a.action_min[d].item(), model_a.action_max[d].item()
        assert out_extreme[:, d].min().item() >= lo - 1e-6
        assert out_extreme[:, d].max().item() <= hi + 1e-6
    print(f"  bounded by construction ✓ (no extrapolation past bounds)")

    print("\nTesting — default symmetric ±1...")
    model_d = PoseDeltaToAction()
    x_def_delta = torch.randn(8, 6) * 5.0
    x_def_pose = torch.randn(8, 6) * 0.5
    out_default = model_d(x_def_delta, x_def_pose)
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
