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
    FlowMatchingPolicyHead,
)
from models.memory_bank import PerceptualCognitiveMemoryModule, EpisodicMemoryModule


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
        history_size: int = 1,
        num_cameras: int = 1,
        use_patch_tokens: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        head_type: str = "diffusion",
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
        memory_mode: str = "episodic",
        memory_detach_writes: bool = True,
        memory_write_fused: bool = False,
        memory_patch_retrieval: bool = False,
        diffusion_train_steps: int = 100,
        diffusion_loss_repeats: int = 4,
        visual_token_attention: bool = False,
        diffusion_clip_sample: bool = True,
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
        if use_intent_tokens and mamba_output_dim <= 0:
            raise ValueError("Intent tokens require the Mamba history encoder")
        self.num_intent_tokens = num_intent_tokens
        self.intent_dim = intent_dim
        self.use_memory_bank = use_memory_bank
        self.memory_bank_len = memory_bank_len
        self.memory_mode = memory_mode
        self.memory_detach_writes = memory_detach_writes
        self.memory_write_fused = memory_write_fused
        self.memory_patch_retrieval = memory_patch_retrieval
        self.diffusion_train_steps = diffusion_train_steps
        self.diffusion_loss_repeats = diffusion_loss_repeats
        self.visual_token_attention = visual_token_attention
        self.diffusion_clip_sample = diffusion_clip_sample
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
        # History controls the observation window. The Mamba encoder exists
        # only with intent tokens; the disabled-token ablation omits it entirely.
        self.use_history = mamba_output_dim > 0

        # Vision patch encoder: always needed (SE compress + state modulate)
        # Used for head input regardless of whether Mamba history is enabled
        from models.intention_encoder import VisionPatchEncoder
        self.vision_patch_encoder = VisionPatchEncoder(
            compressed_dim=compressed_dim, state_dim=state_dim,
            num_cameras=num_cameras, raw_dim=raw_dim, se_reduction=8,
        )

        if self.use_history and self.use_intent_tokens:
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
                num_train_timesteps=self.diffusion_train_steps,
                loss_repeats=self.diffusion_loss_repeats,
                clip_denoised=self.diffusion_clip_sample,
                time_dim=64,
                chunk_size=self.chunk_size,
            )
        elif self.head_type == "flow_matching":
            cond_dim = pool_out_dim + self.state_dim + (self.intent_dim * self.num_intent_tokens if self.use_intent_tokens else 0)
            self.intention_head = FlowMatchingPolicyHead(
                cond_dim=cond_dim,
                action_dim=self.action_dim,
                hidden_dim=self.head_d_model,
                num_inference_steps=10,
                time_dim=64,
                chunk_size=self.chunk_size,
                solver="euler",
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

        if self.visual_token_attention and hasattr(self.intention_head,"unet"):
            self.intention_head.unet.configure_visual_attention(pool_out_dim,self.compressed_dim)

        # Move head to the same device as the rest of the model
        self.intention_head = self.intention_head.to(device)

        # Build memory bank (2-stream: perceptual, state; cognitive optional)
        if self.use_memory_bank:
            cognitive_dim = self.intent_dim * self.num_intent_tokens if self.use_intent_tokens else 0
            memory_class = EpisodicMemoryModule if self.memory_mode == "episodic" else PerceptualCognitiveMemoryModule
            memory_kwargs = dict(detach_writes=self.memory_detach_writes,write_fused=self.memory_write_fused,
                                 patch_dim=self.compressed_dim if self.memory_patch_retrieval else None) if self.memory_mode == "episodic" else {}
            self.memory_module = memory_class(
                perceptual_dim=pool_out_dim,
                cognitive_dim=cognitive_dim,
                state_dim=self.state_dim,
                bank_len=self.memory_bank_len,
                num_heads=2, **memory_kwargs,
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
        if self.intention_encoder is not None:
            result = self.intention_encoder(z_v_cls_seq, z_s_seq)
            if self.use_intent_tokens:
                h_seq, intent_emb = result
            else:
                h_seq = result
        else:
            h_seq = z_s_seq.new_zeros(z_s_seq.shape[0], z_s_seq.shape[1], 1)

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

        # Encode patches for head consumption (always — even without history)
        # We need the real pool_out_dim for head and memory bank construction
        # vision_patch_encoder expects (B, VP, raw_dim) and (B, state_dim) per timestep
        # So we flatten B*T together, process, then reshape back
        B, T, N_tok, raw_dim = z_v_all.shape
        z_v_mod_seq = self.vision_patch_encoder(
            z_v_all.reshape(B * T, N_tok, raw_dim),
            z_s_seq.reshape(B * T, -1),
        )  # (B*T, VP, comp_dim)
        z_v_mod_seq = z_v_mod_seq.reshape(B, T, N_tok, -1)  # (B, T, VP, comp_dim)
        pool_out_dim = N_tok * z_v_mod_seq.shape[-1]
        z_v_pooled_seq = z_v_mod_seq.reshape(B, T, pool_out_dim)  # (B, T, pool_out_dim)

        # Build head and bank on first forward (now we know pool_out_dim)
        self._build_head_and_bank(pool_out_dim)

        # Forward through intention encoder (uses CLS tokens)
        if self.use_history:
            intent = self.forward_intent(z_v_CLS_all, z_s_seq)
            h_seq = intent["h_seq"]
            intent_emb = intent.get("intent_emb", None)
        else:
            # No history: each timestep is independent, no Mamba recurrence
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
            produce_intent: if True, read tokens on a fork after the observation

        Returns:
            (z_v_pooled, z_s, h_new, h_states_new) or
            (z_v_pooled, z_s, h_new, h_states_new, intent_emb)
        """
        # Match batched training: encode raw cameras individually, then restore
        # camera-major patches/CLS. Cached features already have that layout.
        if frames.ndim == 5:
            batch, cameras = frames.shape[:2]
            z_v_all = self._vision_forward(frames.flatten(0, 1))
            z_v_all = z_v_all.reshape(batch, cameras * z_v_all.shape[1], -1)
        else:
            z_v_all = self._vision_forward(frames)
        z_s = self.state_encoder(robot_state)
        V = self.num_cameras
        # Per-camera training/cache layout: [cam0_patches..., cam0_CLS,
        # cam1_patches..., cam1_CLS, ...].
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
        z_v_mod = self.vision_patch_encoder(z_v_patches, z_s)
        z_v_pooled = z_v_mod.flatten(1)

        # Build head on first call
        if not self._built:
            self._build_head_and_bank(z_v_pooled.shape[-1])

        if self.intention_encoder is not None:
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
    def encode_patch_sequence(self,patches,states,chunk_size=16):
        """Bound long-episode feature activations; preserve gradients to encoders."""
        from torch.utils.checkpoint import checkpoint
        outputs = []
        with torch.autocast(device_type=patches.device.type,dtype=torch.bfloat16,
                            enabled=patches.device.type=="cuda"):
            for start in range(0,len(patches),chunk_size):
                p,s = patches[start:start+chunk_size],states[start:start+chunk_size]
                if self.training and torch.is_grad_enabled():
                    outputs.append(checkpoint(self.vision_patch_encoder,p,s,use_reentrant=False))
                else:
                    outputs.append(self.vision_patch_encoder(p,s))
        return torch.cat(outputs)

    def condition_actions(self, z_v_window: torch.Tensor,
                          z_s_window: torch.Tensor,
                          intent_emb: Optional[torch.Tensor] = None,
                          observed_mask: Optional[torch.Tensor] = None,
                          timestamp: Optional[torch.Tensor] = None):
        """Use the same memory conditioning in training and deployment.

        Visual/state retrieval remains active without cognitive intent tokens.
        observed_mask excludes padded observations from memory storage.
        """
        if not self.use_memory_bank:
            return z_v_window, z_s_window, intent_emb
        z_v, z_s, intent = self.memory_module(
            z_v_window[:, -1], z_s_window[:, -1], intent_emb,
            observed_mask=observed_mask,
            **({"timestamp":timestamp} if self.memory_mode == "episodic" else {}),
        )
        return z_v.unsqueeze(1), z_s.unsqueeze(1), intent

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

    # ----------------------------------------------------------------
    # Probe / interpretability forward
    # ----------------------------------------------------------------
    @torch.no_grad()
    def forward_with_probe(
        self,
        frames_seq: torch.Tensor,
        state_seq: torch.Tensor,
    ) -> dict:
        """One-shot forward pass that returns intent_emb + surrounding context.

        Used by tools/probe_intent_clustering.py to inspect the structure of the
        learned intent representations on held-out episodes without affecting
        training.

        This method re-uses the same forward path as `forward()` so the probe
        matches training-time behavior exactly. No gradients are computed.

        Args:
            frames_seq: (B, T, V, H, W, 3) raw frames  OR
                        (B, T, V*P+CLS, raw_dim=768) pre-computed DINOv2 features.
                        Detected by ndim / last-dim value; see _vision_forward dispatch.
            state_seq:  (B, T, 7) raw robot state

        Returns:
            dict with keys:
              intent_emb:         (B, N, intent_dim)  or None if use_intent_tokens=False
              z_v_pooled_seq:     (B, T, pool_out_dim)
              z_s_seq:            (B, T, state_dim)
              h_seq:              (B, T, mamba_output_dim)  or (B, T, 1) when no history
              z_v_CLS_seq:        (B, T, V, raw_dim)
        """
        B, T = frames_seq.shape[:2]

        # ---- Vision encoding (handles both raw frames and pre-computed features) ----
        if frames_seq.ndim == 6:
            V = frames_seq.shape[2]
            z_v_all = self._vision_forward(
                frames_seq.reshape(B * T * V, *frames_seq.shape[3:])
            )  # (B*T*V, P+1, raw_dim=768)
        else:
            V = 1
            z_v_all = self._vision_forward(
                frames_seq.reshape(B * T, *frames_seq.shape[2:])
            )  # (B*T, P+1, raw_dim=768)

        # Split CLS from patches
        z_v_CLS_all = z_v_all[:, -1]  # (B*T*V, raw_dim)
        z_v_CLS_all = z_v_CLS_all.reshape(B, T, V, -1)  # (B, T, V, raw_dim)

        z_v_patches = z_v_all[:, :-1]  # (B*T*V, P, raw_dim)
        _, P, raw_dim = z_v_patches.shape
        z_v_patches = z_v_patches.reshape(B, T, V * P, raw_dim)

        # ---- State encoding ----
        z_s_seq = self.state_encoder(state_seq)  # (B, T, state_dim)

        # ---- Patch encoding for head consumption ----
        z_v_mod_seq = self.vision_patch_encoder(
            z_v_patches.reshape(B * T, -1, raw_dim),
            z_s_seq.reshape(B * T, -1),
        )  # (B*T, V*P, comp_dim)
        z_v_mod_seq = z_v_mod_seq.reshape(B, T, -1, z_v_mod_seq.shape[-1])
        pool_out_dim = z_v_mod_seq.shape[2] * z_v_mod_seq.shape[3]
        z_v_pooled_seq = z_v_mod_seq.reshape(B, T, pool_out_dim)

        # Build head / memory bank on first call (same as training forward)
        self._build_head_and_bank(pool_out_dim)

        # ---- Intention encoder (Mamba + intent tokens) ----
        intent_emb = None
        h_seq = None
        if self.use_history:
            intent = self.forward_intent(z_v_CLS_all, z_s_seq)
            h_seq = intent["h_seq"]
            intent_emb = intent.get("intent_emb", None)
        else:
            h_seq = torch.zeros(B, T, 1, device=frames_seq.device)

        return {
            "intent_emb": intent_emb,
            "z_v_pooled_seq": z_v_pooled_seq,
            "z_s_seq": z_s_seq,
            "h_seq": h_seq,
            "z_v_CLS_seq": z_v_CLS_all,
        }

    @torch.no_grad()
    def sample_actions(self, z_v_pooled_window: torch.Tensor,
                       z_s_window: torch.Tensor,
                       intent_emb: torch.Tensor = None,
                       num_steps: int = None) -> torch.Tensor:
        if isinstance(self.intention_head, (DiffusionPolicyHead, FlowMatchingPolicyHead)):
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
