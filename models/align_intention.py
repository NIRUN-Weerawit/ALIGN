#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN v3/v4: Intention estimation model with Mamba + optional intent tokens + memory bank.

V3 (default): Mamba recurrence + head → K future actions
V4 (opt-in):  Intent tokens + Perceptual-Cognitive Memory Bank

Per timestep t:
  frames(t) (B, [V,] H, W, 3)     robot_state(t) (B, 7)
    ↓ VisionEncoder                ↓ StateEncoder
  z_v_patches (B, [V,] P, D)        z_t (B, state_dim)
    ↓                                ↓
    └─ VisionPatchEncoder ──────────┘
       SE compress + state modulate
              ↓
       z_v_mod (B, VP, comp_dim) → flatten → mamba_in (B, VP*comp_dim + state_dim)
              ↓
          Mamba (recurrent)
              ↓
          h(t) (B, mamba_output_dim)

V4: [z0..zT, INTENT_1..INTENT_N] → Mamba → intent_emb (B, N, intent_dim)
    + PerceptualCognitiveMemoryModule for retrieval + gate fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.align_model import VisionEncoder, RobotStateEncoder
from models.intention_encoder import IntentionEncoder, PerCameraStateConditionedPool
from models.intention_head import (
    IntentionTransformerHead, MambaActionHead,
    DiffusionActionHead, FlowMatchingActionHead, DiffusionPolicyHead,
)
from models.memory_bank import PerceptualCognitiveMemoryModule


class ALIGNIntentionModel(nn.Module):
    """ALIGN v3/v4 intention model with Mamba.

    V3 (default): Mamba + head → K future actions
    V4 (opt-in):  Intent tokens + Perceptual-Cognitive Memory Bank

    Args:
        vision_dim:       per-patch dim (e.g., 256) — legacy, maps to compressed_dim
        state_dim:        robot state dim (e.g., 256)
        mamba_output_dim: Mamba hidden state dim (e.g., 512)
        action_dim:       action output dim (e.g., 6)
        chunk_size:       future action prediction length (default 10)
        history_size:     past frames for Mamba window (default 20)
        num_cameras:      number of cameras (default 1)
        use_patch_tokens: use DINOv2 patch tokens (default True)
        mamba_d_state:    SSM state dim (default 16)
        mamba_d_conv:     local conv width (default 4)
        mamba_expand:     Mamba block expand (default 2)
        head_type:        'transformer' or 'mamba' or 'hybrid' or 'flow' (default 'mamba')
        head_d_model:     head transformer dim (default 384)
        head_nhead:       head attention heads (default 4)
        head_num_layers:  head transformer layers (default 2)
        head_dim_ff:      head FFN dim (default 1024)
        use_text:         enable text encoder (default False)
        text_dim:         text encoder output dim (default 256)
        compressed_dim:   per-patch channel dim after SE compression (default 16)
        pool_num_queries: deprecated (kept for backward compat)
        # V4 args:
        use_intent_tokens: enable learnable intent tokens (default False)
        num_intent_tokens: number of intent tokens N (default 2)
        intent_dim:        output dim of each intent token (default 512)
        use_memory_bank:   enable Perceptual-Cognitive Memory Bank (default False)
        memory_bank_len:   max paired entries in bank (default 16)
    """
    def __init__(
        self,
        vision_dim: int = 256,
        state_dim: int = 256,
        mamba_output_dim: int = 512,
        action_dim: int = 6,
        chunk_size: int = 10,
        history_size: int = 20,
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
        compressed_dim: int = 16,
        pool_num_queries: int = 8,
        # V4 args
        use_intent_tokens: bool = False,
        num_intent_tokens: int = 2,
        intent_dim: int = 512,
        use_memory_bank: bool = False,
        memory_bank_len: int = 16,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.mamba_output_dim = mamba_output_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.history_size = history_size
        self.num_cameras = num_cameras
        self.use_patch_tokens = use_patch_tokens
        self.use_text = use_text
        self.text_dim = text_dim
        self.compressed_dim = compressed_dim
        # V4 flags
        self.use_intent_tokens = use_intent_tokens
        self.num_intent_tokens = num_intent_tokens
        self.intent_dim = intent_dim
        self.use_memory_bank = use_memory_bank
        self.memory_bank_len = memory_bank_len

        # Pool output dim: total VP tokens * compressed_dim
        # New architecture: pool_out_dim = mean across all patches (B, compressed_dim).
        # This keeps the head's per-step input small (16 default) instead of
        # the huge 8192 (2 cams * 256 patches * 16) we'd get with full patches.
        # Per-patch information is preserved in the Mamba state (h_seq) which
        # is built from the full (B, T, V*P, compressed_dim) sequence.
        self.pool_out_dim = compressed_dim

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
        # Intention encoder (patch encoder + Mamba)
        self.use_history = mamba_output_dim > 0
        self.pool = PerCameraStateConditionedPool(
            compressed_dim=compressed_dim, state_dim=state_dim,
            num_cameras=num_cameras,
        )
        if self.use_history:
            self.intention_encoder = IntentionEncoder(
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim,
                num_cameras=num_cameras,
                compressed_dim=compressed_dim,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                use_intent_tokens=use_intent_tokens,
                num_intent_tokens=num_intent_tokens,
                intent_dim=intent_dim,
            )
        else:
            self.intention_encoder = None

        # Intention head
        self.head_type = head_type
        if head_type == "transformer":
            self.intention_head = IntentionTransformerHead(
                pool_out_dim=self.pool_out_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim if use_intent_tokens else mamba_output_dim,
                text_dim=text_dim if use_text else 0,
                action_dim=action_dim,
                chunk_size=chunk_size,
                vision_dim=vision_dim,
                d_model=head_d_model,
                nhead=head_nhead,
                num_layers=head_num_layers,
                dim_feedforward=head_dim_ff,
            )
        elif head_type == "mamba":
            self.intention_head = MambaActionHead(
                pool_out_dim=self.pool_out_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim if use_intent_tokens else mamba_output_dim,
                text_dim=text_dim if use_text else 0,
                action_dim=action_dim,
                chunk_size=chunk_size,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
        elif head_type == "hybrid":
            self.intention_head = MambaActionHead(
                pool_out_dim=self.pool_out_dim,
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim if use_intent_tokens else mamba_output_dim,
                text_dim=text_dim if use_text else 0,
                action_dim=action_dim,
                chunk_size=chunk_size,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
        elif head_type == "diffusion":
            cond_dim = (
                self.pool_out_dim
                + state_dim
                + (text_dim if use_text else 0)
                + (mamba_output_dim if mamba_output_dim > 0 else 0)
            )
            self.intention_head = DiffusionActionHead(
                cond_dim=cond_dim,
                action_dim=action_dim,
                hidden_dim=head_d_model,
                num_inference_steps=20,
                time_dim=128,
                chunk_size=chunk_size,
            )
        elif head_type == "diffusion_policy":
            cond_dim = (
                self.pool_out_dim
                + state_dim
                + (text_dim if use_text else 0)
                + (mamba_output_dim if mamba_output_dim > 0 else 0)
            )
            self.intention_head = DiffusionPolicyHead(
                cond_dim=cond_dim,
                action_dim=action_dim,
                hidden_dim=head_d_model,
                num_inference_steps=10,
                time_dim=64,
                chunk_size=chunk_size,
                use_history=mamba_output_dim > 0,
            )
        elif head_type == "flow":
            cond_dim = (
                self.pool_out_dim
                + state_dim
                + (text_dim if use_text else 0)
                + (mamba_output_dim if mamba_output_dim > 0 else 0)
            )
            self.intention_head = FlowMatchingActionHead(
                cond_dim=cond_dim,
                action_dim=action_dim,
                hidden_dim=head_d_model,
                num_inference_steps=10,
                time_dim=64,
                chunk_size=chunk_size,
                use_history=mamba_output_dim > 0,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Text encoder (optional)
        if use_text:
            from models.align_model import TextEncoder
            self.text_encoder = TextEncoder(embed_dim=text_dim)
        else:
            self.text_encoder = None

        # V4: Perceptual-Cognitive Memory Bank
        if use_memory_bank:
            self.memory_module = PerceptualCognitiveMemoryModule(
                perceptual_dim=self.pool_out_dim,
                cognitive_dim=intent_dim if use_intent_tokens else mamba_output_dim,
                bank_len=memory_bank_len,
                num_heads=4,
            )
        else:
            self.memory_module = None

        # Trainable prefixes
        self._trainable_prefixes = {
            "intention_encoder", "intention_head",
        }
        self._encoder_prefixes = {
            "vision_encoder.backbone", "state_encoder",
        }

    # ----------------------------------------------------------------
    # Vision helpers
    # ----------------------------------------------------------------
    def _V_from_input(self, frames: torch.Tensor) -> int:
        return frames.shape[1] if frames.ndim == 5 else 1

    def _vision_forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(frames)

    def _pool_patches(self, z_v_patches: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Encode raw DINOv2 patches via VisionPatchEncoder (new architecture).

        z_v_patches: (B, V*P, 768) -- raw 768-D features from VisionEncoder
                    after cross-camera attention. V*P is treated as N_tokens.
        z_t: (B, state_dim)
        Returns: (B, compressed_dim) — mean-pooled across all VP positions.
                The per-patch info is preserved in the Mamba state (h_seq);
                here we just need a compact per-step summary for the head.
        """
        out = self.pool(z_v_patches, z_t)           # (B, V*P, compressed_dim)
        return out.mean(dim=1)                       # (B, compressed_dim)

    # ----------------------------------------------------------------
    # Batched training forward
    # ----------------------------------------------------------------
    def forward(self, frames_seq: torch.Tensor, state_seq: torch.Tensor
                ) -> dict:
        """Batched T-step encoding (training).

        V3: returns {z_v_pooled_seq, z_t_seq, h_seq}
        V4 (use_intent_tokens): also returns intent_emb
        V4 (use_memory_bank): memory module called externally (not here)

        Args:
            frames_seq: (B, T, H, W, 3) or (B, T, V, H, W, 3)
            state_seq:  (B, T, 7)
        Returns:
            dict with:
              z_v_pooled_seq: (B, T, pool_out_dim)
              z_t_seq:         (B, T, state_dim)
              h_seq:           (B, T, mamba_output_dim) or (B, T, mamba_in_dim)
              intent_emb:      (B, N, intent_dim) or None
        """
        B, T = frames_seq.shape[:2]

        # Encode each frame
        # New architecture: VisionEncoder returns (B, V*P, 768) raw DINOv2 features
        # after cross-camera attention. We flatten V*P into a single token dim
        # so IntentionEncoder can process all VP tokens uniformly.
        z_v_patches_seq = []
        for t in range(T):
            z_v_t = self._vision_forward(frames_seq[:, t])
            # z_v_t shape: (B, V*P, 768) from multi-cam, or (B, P, 768) from single
            if z_v_t.ndim == 3:
                # Reshape to (B, V*P, 768) — flatten cameras into the patch axis
                # The IntentionEncoder expects (B, T, N_tokens, raw_dim) where
                # N_tokens = V * P (e.g. 2 * 256 = 512 for 2-cam).
                # We keep it as (B, V*P, 768) — IntentionEncoder treats this as
                # N_tokens = V*P tokens.
                pass  # shape already (B, V*P, 768)
            z_v_patches_seq.append(z_v_t)
        z_v_patches_seq = torch.stack(z_v_patches_seq, dim=1)  # (B, T, V*P, 768)

        # Encode states (batched)
        z_t_seq = self.state_encoder(state_seq)  # (B, T, state_dim)

        # Forward through intention encoder
        intent_emb = None
        if self.use_history:
            result = self.intention_encoder(z_v_patches_seq, z_t_seq)
            if self.use_intent_tokens:
                h_seq, intent_emb = result
            else:
                h_seq = result
        else:
            h_seq = torch.zeros(z_t_seq.shape[0], z_t_seq.shape[1], 1, device=z_t_seq.device)

        # Get pooled vision per timestep
        # z_v_patches_seq is (B, T, V*P, 768) — raw DINOv2 patches
        # We apply _pool_patches (VisionPatchEncoder) to each timestep:
        #   (B, V*P, 768) -> SE compress (16) -> state modulate -> (B, V*P, 16)
        # Then flatten to (B, V*P*16) = (B, pool_out_dim) for the head.
        z_v_pooled_seq = []
        for t in range(T):
            z_v_t = z_v_patches_seq[:, t]
            z_t_t = z_t_seq[:, t]
            z_v_pooled_t = self._pool_patches(z_v_t, z_t_t)
            z_v_pooled_seq.append(z_v_pooled_t)
        z_v_pooled_seq = torch.stack(z_v_pooled_seq, dim=1)  # (B, T, pool_out_dim)

        return {
            "z_v_pooled_seq": z_v_pooled_seq,
            "z_t_seq": z_t_seq,
            "h_seq": h_seq,
            "intent_emb": intent_emb,
        }

    # ----------------------------------------------------------------
    # Single-step inference forward
    # ----------------------------------------------------------------
    def encode_step(self, frames: torch.Tensor, robot_state: torch.Tensor,
                    h_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    produce_intent: bool = False
                    ):
        """One step of encoding (inference).

        V3: returns (z_v_pooled, z_t, h_new, h_states_new)
        V4 (use_intent_tokens and produce_intent): also returns intent_emb

        Args:
            frames: (B, H, W, 3) or (B, V, H, W, 3)
            robot_state: (B, 7)
            h_states: (conv_state, ssm_state) from previous step, or None
            produce_intent: if True, run intent tokens after history step

        Returns:
            (z_v_pooled, z_t, h_new, h_states_new) or
            (z_v_pooled, z_t, h_new, h_states_new, intent_emb)
        """
        z_v_patches = self._vision_forward(frames)
        # z_v_patches shape: (B, V*P, 768) — flattened camera-patch tokens
        z_t = self.state_encoder(robot_state)
        z_v_pooled = self._pool_patches(z_v_patches, z_t)

        if self.use_history:
            result = self.intention_encoder.forward_step(
                z_v_patches, z_t, h_states, produce_intent=produce_intent,
            )
            if self.use_intent_tokens and produce_intent:
                h_new, h_states_new, intent_emb = result
                return z_v_pooled, z_t, h_new, h_states_new, intent_emb
            else:
                h_new, h_states_new = result
        else:
            h_new = torch.zeros(z_t.shape[0], 1, device=z_t.device)
            h_states_new = h_states

        return z_v_pooled, z_t, h_new, h_states_new

    # ----------------------------------------------------------------
    # Predict actions from window
    # ----------------------------------------------------------------
    def predict_actions(self, z_v_pooled_window: torch.Tensor,
                        z_t_window: torch.Tensor,
                        h_current: torch.Tensor,
                        z_text: torch.Tensor = None) -> torch.Tensor:
        """Predict K future actions from K past states + conditioning.

        For V4 with intent tokens, h_current is (B, N, intent_dim).
        For V3, h_current is (B, mamba_output_dim).

        Args:
            z_v_pooled_window: (B, K, pool_out_dim)
            z_t_window:        (B, K, state_dim)
            h_current:         (B, mamba_output_dim) or (B, N, intent_dim)
            z_text:            (B, text_dim) or None
        Returns:
            actions: (B, K, action_dim)
        """
        return self.intention_head(
            z_v_pooled_window, z_t_window, h_current, z_text=z_text,
        )

    @torch.no_grad()
    def sample_actions(self, z_v_pooled_window: torch.Tensor,
                       z_t_window: torch.Tensor,
                       h_current: torch.Tensor,
                       z_text: torch.Tensor = None,
                       num_steps: int = None) -> torch.Tensor:
        if isinstance(self.intention_head, (DiffusionActionHead, FlowMatchingActionHead, DiffusionPolicyHead)):
            cond = self.intention_head(
                z_v_pooled_window, z_t_window, h_current, z_text=z_text,
            )
            return self.intention_head.sample(cond, num_steps=num_steps)
        else:
            return self.intention_head(
                z_v_pooled_window, z_t_window, h_current, z_text=z_text,
            )

    # ----------------------------------------------------------------
    # Encoder freeze helpers
    # ----------------------------------------------------------------
    def freeze_encoders(self):
        for p in self.vision_encoder.backbone.parameters():
            p.requires_grad = False
        for p in self.state_encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
