#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN v3: Intention estimation model with Mamba.

This is a new model class that uses the new information encoding:
  - State-Conditioned Attention Pool (z_t queries z_v patches)
  - Mamba recurrence (maintains h(t) across timesteps)
  - IntentionTransformerHead (K past + 1 h → K future actions)

Compared to ALIGNModel (v1/v2):
  - No text encoder (text is removed)
  - No cross-attention mixer (replaced by Mamba)
  - No GatedCrossAttention, ModLN, LanguageConditionedVisualAttention
  - All heads consume (z_v_pooled, z_t, h) instead of (z_v, z_t, z_text)

Per timestep t:
  frames(t) (B, [V,] H, W, 3)     robot_state(t) (B, 7)
    ↓ VisionEncoder                ↓ StateEncoder
  z_v_patches (B, [V,] P, D)        z_t (B, state_dim)
    ↓                                ↓
    └─ State-Conditioned Pool ──────┘
       z_t queries z_v_patches
              ↓
       z_v_pooled (B, V*vision_dim)
              ↓
       mamba_in = concat[z_v_pooled, z_t]  (B, V*vision_dim + state_dim)
              ↓
          Mamba (recurrent)
              ↓
          h(t) (B, mamba_output_dim)

The IntentionTransformerHead consumes a window of K past [z_v_pooled, z_t]
plus the latest h(t), and outputs K future actions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.align_model import VisionEncoder, RobotStateEncoder
from models.intention_encoder import IntentionEncoder, PerCameraStateConditionedPool
from models.intention_head import IntentionTransformerHead, MambaActionHead


class ALIGNIntentionModel(nn.Module):
    """ALIGN v3 intention model with Mamba.

    Args:
        vision_dim:       per-patch dim (e.g., 256)
        state_dim:        robot state dim (e.g., 256)
        mamba_output_dim: Mamba hidden state dim (e.g., 512)
        action_dim:       action output dim (e.g., 6)
        chunk_size:       K — number of past steps / future actions (default 10)
        num_cameras:      number of cameras (default 1)
        use_patch_tokens: use DINOv2 patch tokens (default True)
        mamba_d_state:    SSM state dim (default 16)
        mamba_d_conv:     local conv width (default 4)
        mamba_expand:     Mamba block expand (default 2)
        head_type:        'transformer' or 'mamba' or 'hybrid' (default 'mamba')
        head_d_model:     head transformer dim (default 384, only for transformer)
        head_nhead:       head attention heads (default 4, only for transformer)
        head_num_layers:  head transformer layers (default 2, only for transformer)
        head_dim_ff:      head FFN dim (default 1024, only for transformer)
        use_text:         enable text encoder + text-conditioned head (default False)
        text_dim:         text encoder output dim (default 256)
    """
    def __init__(
        self,
        vision_dim: int = 256,
        state_dim: int = 256,
        mamba_output_dim: int = 512,
        action_dim: int = 6,
        chunk_size: int = 10,
        num_cameras: int = 1,
        use_patch_tokens: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        head_type: str = "mamba",
        head_d_model: int = 384,
        head_nhead: int = 4,
        head_num_layers: int = 2,
        head_dim_ff: int = 1024,
        use_text: bool = False,
        text_dim: int = 256,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.mamba_output_dim = mamba_output_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras
        self.use_patch_tokens = use_patch_tokens
        self.use_text = use_text
        self.text_dim = text_dim
        # Pool output dim: num_cameras * vision_dim (per-camera pools concatenated)
        self.pool_out_dim = num_cameras * vision_dim

        # Vision encoder (DINOv2 with patch tokens)
        self.vision_encoder = VisionEncoder(
            embed_dim=vision_dim,
            num_cameras=num_cameras,
            use_patch_tokens=use_patch_tokens,
        )
        # State encoder (one-step 7-D → state_dim)
        self.state_encoder = RobotStateEncoder(
            input_dim=7,
            state_dim=state_dim,
        )
        # Intention encoder (state-conditioned pool + Mamba)
        # If mamba_output_dim=0, skip the Mamba history component entirely.
        self.use_history = mamba_output_dim > 0
        # Pool is always present (per-camera state-conditioned attention pool)
        self.pool = PerCameraStateConditionedPool(
            vision_dim=vision_dim, state_dim=state_dim,
            num_cameras=num_cameras,
        )
        if self.use_history:
            self.intention_encoder = IntentionEncoder(
                vision_dim=vision_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim,
                num_cameras=num_cameras,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
        else:
            self.intention_encoder = None
        # Intention head: transformer OR mamba
        self.head_type = head_type
        if head_type == "transformer":
            self.intention_head = IntentionTransformerHead(
                pool_out_dim=self.pool_out_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim,
                text_dim=text_dim if use_text else 0,
                action_dim=action_dim,
                chunk_size=chunk_size,
                vision_dim=vision_dim,  # for backwards compat
                d_model=head_d_model,
                nhead=head_nhead,
                num_layers=head_num_layers,
                dim_feedforward=head_dim_ff,
            )
        elif head_type == "mamba":
            self.intention_head = MambaActionHead(
                pool_out_dim=self.pool_out_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim,
                text_dim=text_dim if use_text else 0,
                action_dim=action_dim,
                chunk_size=chunk_size,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
        elif head_type == "hybrid":
            # For now, hybrid = Mamba (TODO: add hybrid head)
            self.intention_head = MambaActionHead(
                pool_out_dim=self.pool_out_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim,
                text_dim=text_dim if use_text else 0,
                action_dim=action_dim,
                chunk_size=chunk_size,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Text encoder (optional — only built if use_text=True)
        if use_text:
            from models.align_model import TextEncoder
            self.text_encoder = TextEncoder(embed_dim=text_dim)
        else:
            self.text_encoder = None

        # Trainable prefixes (for freezing encoders)
        self._trainable_prefixes = {
            "intention_encoder", "intention_head",
        }
        self._encoder_prefixes = {
            "vision_encoder.backbone", "state_encoder",
        }

    # ----------------------------------------------------------------
    # Vision helpers
    # ----------------------------------------------------------------
    def _vision_forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Run vision encoder and return patch tokens (B, [V,] P, vision_dim).

        Args:
            frames: (B, H, W, 3) or (B, V, H, W, 3)
        Returns:
            (B, P, vision_dim) or (B, V, P, vision_dim) — patch tokens
        """
        return self.vision_encoder(frames)

    def _pool_patches(self, z_v_patches: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Apply state-conditioned attention pool.

        Args:
            z_v_patches: (B, P, vision_dim) or (B, V, P, vision_dim)
            z_t: (B, state_dim)
        Returns:
            (B, V*vision_dim) — pooled vector
        """
        if z_v_patches.ndim == 3:
            z_v_patches = z_v_patches.unsqueeze(1)
        return self.pool(z_v_patches, z_t)

    # ----------------------------------------------------------------
    # Batched training forward
    # ----------------------------------------------------------------
    def forward(self, frames_seq: torch.Tensor, state_seq: torch.Tensor
                ) -> dict:
        """Batched T-step encoding (training).

        Args:
            frames_seq: (B, T, H, W, 3) or (B, T, V, H, W, 3)
            state_seq:  (B, T, 7) — T robot states
        Returns:
            dict with:
              z_v_pooled_seq: (B, T, pool_out_dim)
              z_t_seq:         (B, T, state_dim)
              h_seq:           (B, T, mamba_output_dim)
        """
        B, T = frames_seq.shape[:2]

        # Encode each frame
        z_v_patches_seq = []
        for t in range(T):
            z_v_t = self._vision_forward(frames_seq[:, t])  # (B, [V,] P, vision_dim)
            z_v_patches_seq.append(z_v_t)
        # Stack into (B, T, [V,] P, vision_dim)
        if z_v_patches_seq[0].ndim == 4:
            # Multi-cam: (B, V, P, vision_dim)
            z_v_patches_seq = torch.stack(z_v_patches_seq, dim=1)  # (B, T, V, P, vision_dim)
        else:
            # Single-cam: (B, P, vision_dim)
            z_v_patches_seq = torch.stack(z_v_patches_seq, dim=1)  # (B, T, P, vision_dim)
            z_v_patches_seq = z_v_patches_seq.unsqueeze(2)         # (B, T, 1, P, vision_dim)

        # Encode states (batched)
        z_t_seq = self.state_encoder(state_seq)  # (B, T, state_dim)

        # Forward through intention encoder (if history is enabled)
        if self.use_history:
            h_seq = self.intention_encoder(z_v_patches_seq, z_t_seq)  # (B, T, mamba_output_dim)
        else:
            # No history: dummy h_seq (zeros of the right shape, but the head may not use it)
            h_seq = torch.zeros(z_t_seq.shape[0], z_t_seq.shape[1], 1, device=z_t_seq.device)

        # Get pooled vision per timestep (B, T, pool_out_dim)
        z_v_pooled_seq = []
        for t in range(T):
            z_v_t = z_v_patches_seq[:, t]  # (B, V, P, vision_dim)
            z_t_t = z_t_seq[:, t]          # (B, state_dim)
            z_v_pooled_t = self._pool_patches(z_v_t, z_t_t)
            z_v_pooled_seq.append(z_v_pooled_t)
        z_v_pooled_seq = torch.stack(z_v_pooled_seq, dim=1)

        return {
            "z_v_pooled_seq": z_v_pooled_seq,
            "z_t_seq": z_t_seq,
            "h_seq": h_seq,
        }

    # ----------------------------------------------------------------
    # Single-step inference forward
    # ----------------------------------------------------------------
    def encode_step(self, frames: torch.Tensor, robot_state: torch.Tensor,
                    h_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor,
                               torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One step of encoding (inference).

        Args:
            frames: (B, H, W, 3) or (B, V, H, W, 3)
            robot_state: (B, 7)
            h_states: (conv_state, ssm_state) from previous step, or None

        Returns:
            z_v_pooled: (B, pool_out_dim)
            z_t:         (B, state_dim)
            h_new:       (B, mamba_output_dim)
            h_states_new: (conv_state, ssm_state) for next step
        """
        # Vision
        z_v_patches = self._vision_forward(frames)  # (B, [V,] P, vision_dim)
        # Ensure (B, V, P, vision_dim)
        if z_v_patches.ndim == 3:
            z_v_patches = z_v_patches.unsqueeze(1)
        # State
        z_t = self.state_encoder(robot_state)  # (B, state_dim)
        # Pool
        z_v_pooled = self._pool_patches(z_v_patches, z_t)
        # Mamba step (only if history is enabled)
        if self.use_history:
            h_new, h_states_new = self.intention_encoder.forward_step(
                z_v_patches, z_t, h_states,
            )
        else:
            h_new = torch.zeros(z_t.shape[0], 1, device=z_t.device)
            h_states_new = h_states  # pass through (None or empty)
        return z_v_pooled, z_t, h_new, h_states_new

    # ----------------------------------------------------------------
    # Predict actions from window
    # ----------------------------------------------------------------
    def predict_actions(self, z_v_pooled_window: torch.Tensor,
                        z_t_window: torch.Tensor,
                        h_current: torch.Tensor,
                        z_text: torch.Tensor = None) -> torch.Tensor:
        """Predict K future actions from K past states + 1 h.

        Args:
            z_v_pooled_window: (B, K, pool_out_dim)
            z_t_window:        (B, K, state_dim)
            h_current:         (B, mamba_output_dim)
            z_text:            (B, text_dim) — task text embedding (or None)
        Returns:
            actions: (B, K, action_dim)
        """
        return self.intention_head(
            z_v_pooled_window, z_t_window, h_current, z_text=z_text,
        )

    # ----------------------------------------------------------------
    # Encoder freeze helpers
    # ----------------------------------------------------------------
    def freeze_encoders(self):
        """Freeze vision + state encoders (only train intention + head)."""
        for p in self.vision_encoder.backbone.parameters():
            p.requires_grad = False
        for p in self.state_encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
