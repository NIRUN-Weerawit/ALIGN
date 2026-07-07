#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PoseDeltaToAction — learned inverse-dynamics connector model.

Maps EEF pose deltas (m, rad) → OSC actions (normalized), replacing
fixed per-axis scales. Captures non-linear Jacobian coupling and
configuration-dependent dynamics that fixed scales miss.

See training/train_pose_to_action.py for the training script.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseDeltaToAction(nn.Module):
    """Maps EEF pose delta (m, rad) → OSC action (normalized).

    Input:  (B, 6) — [dx, dy, dz, dax, day, daz] in meters and radians
    Output: (B, 6) — [ax, ay, az, arx, ary, arz] in OSC action space

    The mapping is the inverse dynamics of the OSC controller. For a
    simple controller with a fixed Jacobian, this reduces to per-axis
    scaling. For real controllers with configuration-dependent Jacobians,
    the learned model captures the non-linear coupling.
    """

    def __init__(self, pose_dim: int = 6, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.pose_dim = pose_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, pose_delta: torch.Tensor) -> torch.Tensor:
        return self.net(pose_delta)