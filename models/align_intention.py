#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN v3/v4: Intention estimation model with Mamba + optional intent tokens + memory bank.

V3 (default): Mamba recurrence + head → K future actions
V4 (opt-in):  Intent tokens + Perceptual-Cognitive Memory Bank

Per timestep t:
  frames(t) (B, [V,] H, W, 3)     robot_state(t) (B, 7)
    ↓ VisionEncoder                ↓ StateEncoder
  z_v_patches (B, [V,] P, D)        z_s (B, state_dim)
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
from models.intention_encoder import IntentionEncoder
from models.intention_head import (
    IntentionTransformerHead, MambaActionHead, DiffusionPolicyHead,
)
from models.memory_bank import PerceptualCognitiveMemoryModule


class ALIGNIntentionModel(nn.Module):
    """ALIGN v3/v4 intention model with Mamba.

    V3 (default): Mamba + head → K future actions
    V4 (opt-in):  Intent tokens + Perceptual-Cognitive Memory Bank

    Head construction is deferred to the first forward pass so that
    pool_out_dim (which depends on the actual DINOv2 patch count) is
    known before building the head.
    """
    def __init__(
        self,
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
        raw_dim: int = 768,
        # V4 args
        use_intent_tokens: bool = False,
        num_intent_tokens: int = 2,
        intent_dim: int = 512,
        use_memory_bank: bool = False,
        memory_bank_len: int = 16,
    ):
        super().__init__()
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
        self.raw_dim = raw_dim
        # V4 flags
        self.use_intent_tokens = use_intent_tokens
        self.num_intent_tokens = num_intent_tokens
        self.intent_dim = intent_dim
        self.use_memory_bank = use_memory_bank
        self.memory_bank_len = memory_bank_len
        self.head_type = head_type
        self.head_d_model = head_d_model
        self.head_nhead = head_nhead
        self.head_num_layers = head_num_layers
        self.head_dim_ff = head_dim_ff

        # Pool output dim: computed dynamically from first forward pass
        self.pool_out_dim: Optional[int] = None
        self._built = False

        # Vision encoder (DINOv2 with patch tokens, outputs raw 768-D features)
        self.vision_encoder = VisionEncoder(
            embed_dim=raw_dim,  # DINOv2 ViT-B/14 output dim, hardcoded
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

        if self.use_history:
            self.intention_encoder = IntentionEncoder(
                state_dim=state_dim,
                mamba_output_dim=mamba_output_dim,
                num_cameras=num_cameras,
                compressed_dim=compressed_dim,
                raw_dim=raw_dim,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                use_intent_tokens=use_intent_tokens,
                num_intent_tokens=num_intent_tokens,
                intent_dim=intent_dim,
            )
        else:
            self.intention_encoder = None

        # Head and memory bank: built lazily on first forward
        self.intention_head: Optional[nn.Module] = None
        self.memory_module: Optional[nn.Module] = None

        # Text encoder (optional)
        if use_text:
            from models.align_model import TextEncoder
            self.text_encoder = TextEncoder(embed_dim=text_dim)
        else:
            self.text_encoder = None

        # Trainable prefixes
        self._trainable_prefixes = {
            "intention_encoder", "intention_head",
        }
        self._encoder_prefixes = {
            "vision_encoder.backbone", "state_encoder",
        }

    def _build_head_and_bank(self, pool_out_dim: int):
        """Build head and memory bank once pool_out_dim is known."""
        if self._built:
            return
        self.pool_out_dim = pool_out_dim
        self._built = True

        # Determine device from existing parameters
        device = next(self.vision_encoder.parameters()).device

        # Build head
        if self.head_type == "transformer":
            self.intention_head = IntentionTransformerHead(
                pool_out_dim=pool_out_dim,
                state_dim=self.state_dim,
                intent_dim=self.intent_dim * self.num_intent_tokens if self.use_intent_tokens else 0,
                action_dim=self.action_dim,
                chunk_size=self.chunk_size,
                num_intent_tokens=self.num_intent_tokens,
                d_model=self.head_d_model,
                nhead=self.head_nhead,
                num_layers=self.head_num_layers,
                dim_feedforward=self.head_dim_ff,
            )
        elif self.head_type in ("mamba", "hybrid"):
            self.intention_head = MambaActionHead(
                pool_out_dim=pool_out_dim,
                state_dim=self.state_dim,
                intent_dim=self.intent_dim * self.num_intent_tokens if self.use_intent_tokens else 0,
                action_dim=self.action_dim,
                chunk_size=self.chunk_size,
                mamba_d_state=self.mamba_d_state,
                mamba_d_conv=self.mamba_d_conv,
                mamba_expand=self.mamba_expand,
                use_intent=self.use_intent_tokens,
            )
        elif self.head_type == "diffusion":
            cond_dim = pool_out_dim + self.state_dim + (self.intent_dim * self.num_intent_tokens if self.use_intent_tokens else 0)
            self.intention_head = DiffusionPolicyHead(
                cond_dim=cond_dim,
                action_dim=self.action_dim,
                hidden_dim=self.head_d_model,
                num_inference_steps=10,
                time_dim=64,
                chunk_size=self.chunk_size,
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

        # Move head to the same device as the rest of the model
        self.intention_head = self.intention_head.to(device)

        # Build memory bank (3-stream: perceptual, cognitive, state)
        if self.use_memory_bank:
            self.memory_module = PerceptualCognitiveMemoryModule(
                perceptual_dim=pool_out_dim,
                cognitive_dim=self.intent_dim*self.num_intent_tokens if self.use_intent_tokens else self.mamba_output_dim,
                state_dim=self.state_dim,
                bank_len=self.memory_bank_len,
                num_heads=2,
            ).to(device)

    # ----------------------------------------------------------------
    # Vision helpers
    # ----------------------------------------------------------------
    def _V_from_input(self, frames: torch.Tensor) -> int:
        return frames.shape[1] if frames.ndim == 5 else 1

    def _vision_forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(frames)

    # ----------------------------------------------------------------
    # Batched training forward
    # ----------------------------------------------------------------
    def forward_intent(self, z_v_cls_seq: torch.Tensor, z_s_seq: torch.Tensor) -> dict:
        """Batched T-step encoding (training) with intent tokens.

        Args:
            z_v_cls_seq: (B, T, V, raw_dim=768) — CLS tokens from DINOv2
            z_s_seq:     (B, T, state_dim)
        Returns:
            dict with:
              h_seq:           (B, T, mamba_in_dim)
              intent_emb:      (B, N, intent_dim)

        NOTE: Does NOT trigger the lazy head build. The head should be built
        explicitly by the caller (e.g., train loop) with the correct pool_out_dim
        that matches the memory bank storage shape.
        """
        # Forward through intention encoder
        intent_emb = None
        if self.use_history:
            result = self.intention_encoder(z_v_cls_seq, z_s_seq)
            if self.use_intent_tokens:
                h_seq, intent_emb = result
            else:
                h_seq = result
        else:
            h_seq = torch.zeros(z_s_seq.shape[0], z_s_seq.shape[1], 1, device=z_s_seq.device)

        return {
            "h_seq": h_seq,
            "intent_emb": intent_emb,
        }
        
    def forward(self, frames_seq: torch.Tensor, state_seq: torch.Tensor
                ) -> dict:
        """Batched T-step encoding (training).

        V3: returns {z_v_pooled_seq, z_s_seq, h_seq}
        V4 (use_intent_tokens): also returns intent_emb

        Args:
            frames_seq: (B, T, H, W, 3) or (B, T, V, H, W, 3)
            state_seq:  (B, T, 7)
        Returns:
            dict with:
              z_v_pooled_seq: (B, T, pool_out_dim)
              z_s_seq:         (B, T, state_dim)
              h_seq:           (B, T, mamba_output_dim) or (B, T, mamba_in_dim)
              intent_emb:      (B, N, intent_dim) or None
        """
        B, T = frames_seq.shape[:2]

        # Batch DINOv2: reshape all frames into one big tensor
        if frames_seq.ndim == 6:
            # (B, T, V, H, W, 3) — multi-camera
            V = frames_seq.shape[2]
            z_v_all = self._vision_forward(
                frames_seq.reshape(B * T * V, *frames_seq.shape[3:])
            )  # (B*T*V, P+1, raw_dim=768)
        else:
            # (B, T, H, W, 3) — single camera
            V = 1
            z_v_all = self._vision_forward(
                frames_seq.reshape(B * T, *frames_seq.shape[2:])
            )  # (B*T, P+1, raw_dim=768)

        # Split CLS tokens from patch tokens
        z_v_CLS_all = z_v_all[:, -1]  # (B*T*V, raw_dim=768)
        z_v_CLS_all = z_v_CLS_all.reshape(B, T, V, -1)  # (B, T, V, raw_dim=768)

        z_v_all = z_v_all[:, :-1]  # (B*T*V, P, raw_dim=768)
        _, P, raw_dim = z_v_all.shape
        z_v_all = z_v_all.reshape(B, T, V * P, raw_dim)  # (B, T, V*P, raw_dim=768)

        # Encode states (batched)
        z_s_seq = self.state_encoder(state_seq)  # (B, T, state_dim)

        # Encode patches for head consumption
        if self.use_history:
            z_v_mod_seq = self.intention_encoder.encode_patches(z_v_all, z_s_seq)  # (B, T, V*P, comp_dim)
            B, T, N_tok, comp_dim = z_v_mod_seq.shape
            pool_out_dim = N_tok * comp_dim
            z_v_pooled_seq = z_v_mod_seq.reshape(B, T, pool_out_dim)  # (B, T, pool_out_dim)
        else:
            pool_out_dim = 1
            z_v_pooled_seq = torch.zeros(B, T, 1, device=frames_seq.device)

        # Build head and bank on first forward (now we know pool_out_dim)
        self._build_head_and_bank(pool_out_dim)

        # Forward through intention encoder (uses CLS tokens)
        if self.use_history:
            intent = self.forward_intent(z_v_CLS_all, z_s_seq)
            h_seq = intent["h_seq"]
            intent_emb = intent.get("intent_emb", None)
        else:
            h_seq = torch.zeros(B, T, 1, device=frames_seq.device)
            intent_emb = None

        return {
            "z_v_pooled_seq": z_v_pooled_seq,
            "z_s_seq": z_s_seq,
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

        V3: returns (z_v_pooled, z_s, h_new, h_states_new)
        V4 (use_intent_tokens and produce_intent): also returns intent_emb

        Args:
            frames: (B, H, W, 3) or (B, V, H, W, 3)
            robot_state: (B, 7)
            h_states: (conv_state, ssm_state) from previous step, or None
            produce_intent: if True, run intent tokens after history step

        Returns:
            (z_v_pooled, z_s, h_new, h_states_new) or
            (z_v_pooled, z_s, h_new, h_states_new, intent_emb)
        """
        z_v_all = self._vision_forward(frames)  # (B, V*(P+1), 768) — patches + 1 CLS per camera
        z_s = self.state_encoder(robot_state)
        V = self.num_cameras
        # VisionEncoder output layout: [cam0_patches..., cam1_patches..., cam0_CLS, cam1_CLS, ...]
        # Total tokens = V * (P + 1) where each camera has P patches + 1 CLS at the end.
        total_tokens = z_v_all.shape[1]
        P_plus_1 = total_tokens // V  # P + 1 per camera
        P = P_plus_1 - 1
        # Reshape to (B, V, P+1, 768) so we can split into patches and CLS per camera
        z_v_all_reshaped = z_v_all.reshape(z_v_all.shape[0], V, P_plus_1, 768)
        # CLS is the last position per camera: (B, V, 768)
        z_v_cls = z_v_all_reshaped[:, :, -1, :]  # (B, V, 768)
        # Patches are all positions except the last per camera: (B, V, P, 768)
        z_v_patches = z_v_all_reshaped[:, :, :-1, :].reshape(z_v_all.shape[0], V * P, 768)
        z_v_pooled = z_v_patches

        # Build head on first call
        if not self._built:
            B, N_tok, comp_dim = z_v_pooled.shape
            self._build_head_and_bank(N_tok * comp_dim)

        if self.use_history:
            result = self.intention_encoder.forward_step(
                z_v_cls, z_s, h_states, produce_intent=produce_intent,
            )
            if self.use_intent_tokens and produce_intent:
                h_new, h_states_new, intent_emb = result
                return z_v_pooled, z_s, h_new, h_states_new, intent_emb
            else:
                h_new, h_states_new = result
        else:
            h_new = torch.zeros(z_s.shape[0], 1, device=z_s.device)
            h_states_new = h_states

        return z_v_pooled, z_s, h_new, h_states_new

    # ----------------------------------------------------------------
    # Predict actions from window
    # ----------------------------------------------------------------
    def predict_actions(self, z_v_pooled_window: torch.Tensor,
                        z_s_window: torch.Tensor,
                        intent_emb: torch.Tensor = None) -> torch.Tensor:
        """Predict K future actions from K past states + intent tokens.

        Args:
            z_v_pooled_window: (B, K, pool_out_dim)
            z_s_window:        (B, K, state_dim)
            intent_emb:        (B, N, intent_dim) or None
        Returns:
            actions: (B, K, action_dim)
        """
        return self.intention_head(
            z_v_pooled_window, z_s_window, intent_emb=intent_emb,
        )

    @torch.no_grad()
    def sample_actions(self, z_v_pooled_window: torch.Tensor,
                       z_s_window: torch.Tensor,
                       intent_emb: torch.Tensor = None,
                       num_steps: int = None) -> torch.Tensor:
        if isinstance(self.intention_head, DiffusionPolicyHead):
            cond = self.intention_head(
                z_v_pooled_window, z_s_window, intent_emb=intent_emb,
            )
            return self.intention_head.sample(cond, num_steps=num_steps)
        else:
            return self.intention_head(
                z_v_pooled_window, z_s_window, intent_emb=intent_emb,
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
