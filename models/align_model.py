#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN model architecture — shared backbone + dual heads.

Architecture:
    DINOv2 ViT-B (frozen) → projection → z_v (256d)
    Transformer (trained) → projection → z_t (256d)
    CLIP text (frozen) → projection → z_text (256d)
        │                              │
        └────────── CrossAttnMixer ─────┘
                   │          │
         ┌─────────┴─────────┐
         │                   │
    Decision Head (α)   Assistant Head (Δposes)

Usage:
    from models.align_model import ALIGNModel
    model = ALIGNModel()
    vision = model.encode_raw_vision(frames)        # (B, 256) — no mixer
    z_v, z_t, z_text = model.encode_mixed(...)      # through mixer
    alpha = model.decision_head(z_v, z_t, z_text)
    delta = model.assistant_head(z_v, z_t, z_text, noisy_pose)
"""

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


# ================================================================
# Vision Encoder
# ================================================================

class VisionEncoder(nn.Module):
    """Frozen DINOv2 ViT-B + trainable projection head."""

    def __init__(self, backbone: str = "dinov2_vitb14", embed_dim: int = 256):
        super().__init__()
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
        except Exception:
            raise ImportError(
                "DINOv2 not installed. Run: pip install dinov2"
            )
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode camera frame(s).

        Args:
            x: (B, H, W, 3) uint8 RGB images.

        Returns:
            (B, embed_dim) vision embeddings.
        """
        B, H, W, C = x.shape
        if C != 3:
            raise ValueError(f"Expected HWC RGB images, got shape {x.shape}")
        if H != 224 or W != 224:
            x = F.interpolate(
                x.permute(0, 3, 1, 2).float(), size=(224, 224), mode="bilinear"
            )
        else:
            x = x.permute(0, 3, 1, 2).float()
        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            features = self.backbone(x)  # (B, 768)
        return self.projection(features)  # (B, 256)


# ================================================================
# Trajectory Encoder
# ================================================================

class TrajectoryEncoder(nn.Module):
    """Transformer-based trajectory encoder with temporal pooling."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode trajectory window.

        Args:
            x: (B, K, D) trajectory window of poses.

        Returns:
            (B, embed_dim) trajectory embedding (mean-pooled).
        """
        x = self.input_proj(x)  # (B, K, d_model)
        x = self.transformer(x)  # (B, K, d_model)
        x = x.mean(dim=1)  # (B, d_model) mean pooling
        return self.projection(x)  # (B, 256)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode trajectory and return per-token embeddings (no pooling).

        Args:
            x: (B, K, D) trajectory window.

        Returns:
            (B, K, embed_dim) per-token embeddings.
        """
        x = self.input_proj(x)  # (B, K, d_model)
        x = self.transformer(x)  # (B, K, d_model)
        return self.projection(x)  # (B, K, 256)


# ================================================================
# Text Encoder
# ================================================================

class TextEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 text tower + trainable projection head."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self._use_open_clip = False
        try:
            import open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._use_open_clip = True
        except ImportError:
            try:
                import clip
            except ImportError:
                raise ImportError(
                    "Neither open_clip nor clip installed. "
                    "Run: pip install open-clip-torch"
                )
            self.model, _ = clip.load("ViT-B/32", device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        if self._use_open_clip:
            import open_clip
            tokens = self._tokenizer(texts).to(next(self.projection.parameters()).device)
        else:
            import clip
            tokens = clip.tokenize(texts, truncate=True).to(next(self.projection.parameters()).device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()
        return self.projection(features)


# ================================================================
# Decision Head
# ================================================================

class DecisionHead(nn.Module):
    """MLP predicting α ∈ [0, 1] from shared embeddings + alignment scores.

    Input (all computable at inference):
        z_v (256) + z_t (256) + z_text (256) + cos_vt(1) + cos_vl(1) + cos_tl(1)
    No external inputs (e.g. object distance) — the model learns
    "near vs. far" from visual features. This is critical for real
    deployment where per-object distance is rarely available.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        input_dim = latent_dim * 3 + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        z_text: torch.Tensor,
    ) -> torch.Tensor:
        B = z_v.shape[0]
        z_v_n = F.normalize(z_v, dim=-1)
        z_t_n = F.normalize(z_t, dim=-1)
        z_text_n = F.normalize(z_text, dim=-1)
        cos_vt = (z_v_n * z_t_n).sum(dim=-1, keepdim=True)
        cos_vl = (z_v_n * z_text_n).sum(dim=-1, keepdim=True)
        cos_tl = (z_t_n * z_text_n).sum(dim=-1, keepdim=True)
        x = torch.cat([z_v, z_t, z_text, cos_vt, cos_vl, cos_tl], dim=-1)
        return self.mlp(x)


# ================================================================
# Assistant Head
# ================================================================

class AssistantHead(nn.Module):
    """MLP predicting chunk of K corrective Δposes from shared embeddings."""

    def __init__(self, latent_dim: int = 256, chunk_size: int = 5, action_dim: int = 6):
        super().__init__()
        input_dim = latent_dim * 3 + action_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, chunk_size * action_dim),
        )

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        z_text: torch.Tensor,
        noisy_pose: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_v, z_t, z_text, noisy_pose], dim=-1)
        out = self.mlp(x)
        return out.reshape(-1, self.chunk_size, self.action_dim)


# ================================================================
# Factory for cross-attention mixer
# ================================================================

def _create_mixer(embed_dim: int = 256, mixer_dim: int = 512,
                  num_blocks: int = 2, nhead: int = 8,
                  max_traj_len: int = 64) -> nn.Module:
    """Import and create CrossAttentionMixer (lazy import)."""
    from models.cross_attention_mixer import CrossAttentionMixer
    return CrossAttentionMixer(
        enc_dim=embed_dim,
        mixer_dim=mixer_dim,
        num_blocks=num_blocks,
        nhead=nhead,
        max_traj_len=max_traj_len,
    )


# ================================================================
# Full ALIGN Model
# ================================================================

TRAINABLE_MODULES = {
    "vision_encoder.projection",   # vision_proj
    "traj_encoder",                # full trajectory encoder
    "text_encoder.projection",     # text_proj
    "cross_attention_mixer",       # mixer (always present)
    "decision_head",               # α predictor
    "assistant_head",              # Δpose predictor
}

ENCODER_MODULES = {
    "vision_encoder.projection",
    "traj_encoder",
    "text_encoder.projection",
    "cross_attention_mixer",
}

FROZEN_BACKBONE_MODULES = {
    "vision_encoder.backbone",     # DINOv2
    "text_encoder.model",          # CLIP text
}


class ALIGNModel(nn.Module):
    """Complete ALIGN shared autonomy model.

    Combines three encoders, a cross-attention mixer, and two heads.
    Cross-attention mixer is always present (identity-initialized).
    Text encoder is optional (``use_text=False`` for text-free mode).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        traj_input_dim: int = 6,
        traj_d_model: int = 128,
        traj_nhead: int = 4,
        traj_num_layers: int = 3,
        chunk_size: int = 5,
        action_dim: int = 6,
        use_text: bool = True,
        device: Optional[str] = None,
        mixer_dim: int = 512,
        num_mixer_blocks: int = 2,
        mixer_nhead: int = 8,
        max_traj_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Encoders
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.traj_encoder = TrajectoryEncoder(
            input_dim=traj_input_dim,
            d_model=traj_d_model,
            nhead=traj_nhead,
            num_layers=traj_num_layers,
            embed_dim=embed_dim,
        )
        self.text_encoder = TextEncoder(embed_dim=embed_dim) if use_text else None

        # Cross-attention mixer (always present, identity-initialized)
        self.cross_attention_mixer = _create_mixer(
            embed_dim=embed_dim,
            mixer_dim=mixer_dim,
            num_blocks=num_mixer_blocks,
            nhead=mixer_nhead,
            max_traj_len=max_traj_len,
        )

        # Heads
        self.decision_head = DecisionHead(latent_dim=embed_dim)
        self.assistant_head = AssistantHead(
            latent_dim=embed_dim, chunk_size=chunk_size, action_dim=action_dim
        )

        # Register which modules are trainable vs frozen
        self._trainable_prefixes = {"vision_encoder.projection", "traj_encoder",
                                     "text_encoder.projection", "cross_attention_mixer",
                                     "decision_head", "assistant_head"}
        self._encoder_prefixes = {"vision_encoder.projection", "traj_encoder",
                                   "text_encoder.projection", "cross_attention_mixer"}
        self._backbone_prefixes = {"vision_encoder.backbone", "text_encoder.model"}

    # ── Phase 1a helpers: raw encoder outputs (no mixer) ─────────

    def encode_raw_vision(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames, return raw vision embedding (no mixer)."""
        return self.vision_encoder(frames)

    def encode_raw_trajectory(self, poses: torch.Tensor) -> torch.Tensor:
        """Encode trajectory, return raw mean-pooled trajectory embedding (no mixer)."""
        return self.traj_encoder(poses)

    def encode_raw_trajectory_tokens(self, poses: torch.Tensor) -> torch.Tensor:
        """Encode trajectory, return per-token embeddings (no mixer, no pooling)."""
        return self.traj_encoder.encode_tokens(poses)

    def encode_raw_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        return self.text_encoder(texts)

    def encode_raw_all(self, frames: torch.Tensor,
                       traj: torch.Tensor,
                       texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Encode all three modalities, return raw (pre-mixer) embeddings.

        Phase 1a uses this: InfoNCE on raw encoder outputs with mixer frozen.
        """
        z_v = self.encode_raw_vision(frames)
        z_t = self.encode_raw_trajectory(traj)
        z_text = self.encode_raw_text(texts)
        if z_text is None:
            z_text = torch.zeros_like(z_v)
        return {"z_v": z_v, "z_t": z_t, "z_text": z_text}

    # ── Phase 1b helper: full encode-through-mixer ─────────────

    def encode_mixed(self, frames: torch.Tensor,
                     traj: torch.Tensor,
                     texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Encode all three modalities and pass through cross-attention mixer.

        Phase 1b uses this: InfoNCE on mixer outputs, mixer unfrozen.

        Returns:
            Dict with 'z_v', 'z_t', 'z_text' (all (B, 256)) from mixer output.
            'z_t' is mean-pooled for head consumption.
        """
        z_v = self.encode_raw_vision(frames)
        z_t_tokens = self.encode_raw_trajectory_tokens(traj)  # (B, K, 256)
        z_text = self.encode_raw_text(texts)
        if z_text is None:
            z_text = torch.zeros_like(z_v)

        # Through mixer
        z_v, z_t_tokens, z_text = self.cross_attention_mixer(z_v, z_t_tokens, z_text)

        # Mean-pool trajectory tokens for head/InfoNCE consumption
        z_t = z_t_tokens.mean(dim=1)

        return {"z_v": z_v, "z_t": z_t, "z_text": z_text,
                "z_t_tokens": z_t_tokens}

    # ── Standard encode (for compatibility and Phase 2) ────────

    def encode_vision(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(frames)

    def encode_trajectory(self, poses: torch.Tensor) -> torch.Tensor:
        return self.traj_encoder(poses)

    def encode_trajectory_tokens(self, poses: torch.Tensor) -> torch.Tensor:
        return self.traj_encoder.encode_tokens(poses)

    def encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        return self.text_encoder(texts)

    def forward(
        self,
        frames: torch.Tensor,
        traj: torch.Tensor,
        texts: Optional[List[str]] = None,
        compute_decision: bool = True,
        compute_assistant: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass (encoders + mixer + heads).

        Args:
            frames: (B, H, W, 3) RGB images.
            traj: (B, K, 6) trajectory window.
            texts: Optional list of task descriptions.
            compute_decision: Whether to compute α.
            compute_assistant: Whether to compute Δposes.

        Returns:
            Dict with 'alpha', 'delta' keys as present.
        """
        mixed = self.encode_mixed(frames, traj, texts)
        z_v, z_t = mixed["z_v"], mixed["z_t"]
        z_text = mixed["z_text"]

        result: Dict[str, torch.Tensor] = {}
        if compute_decision:
            result["alpha"] = self.decision_head(z_v, z_t, z_text)
        if compute_assistant:
            # Use last pose from trajectory window as current noisy pose
            result["delta"] = self.assistant_head(z_v, z_t, z_text, traj[:, -1])
        return result

    # ── Freeze helpers ─────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze vision and text backbones (DINOv2 + CLIP).

        Called at the start of every training phase.
        The backbones are never trained.
        """
        for p in self.vision_encoder.backbone.parameters():
            p.requires_grad = False
        if self.text_encoder is not None:
            for p in self.text_encoder.model.parameters():
                p.requires_grad = False

    def freeze_all_encoders(self):
        """Freeze ALL encoders (vision_proj, traj_encoder, text_proj, mixer).

        Called in Phase 2 (head training) so gradients only flow
        into decision_head and assistant_head.
        """
        for name, module in self.named_modules():
            # Check if this module is an encoder (not backbone, not head)
            is_encoder = any(prefix in name for prefix in
                             ["vision_encoder.projection", "traj_encoder",
                              "text_encoder.projection", "cross_attention_mixer"])
            if is_encoder:
                for p in module.parameters():
                    p.requires_grad = False

    def freeze_mixer(self):
        """Freeze only the cross-attention mixer.

        Called in Phase 1a (encoder pretrain) so InfoNCE gradient
        only flows through raw encoders, not the mixer.
        """
        if self.cross_attention_mixer is not None:
            for p in self.cross_attention_mixer.parameters():
                p.requires_grad = False
            self.cross_attention_mixer.eval()

    def unfreeze_mixer(self):
        """Unfreeze the cross-attention mixer.

        Called in Phase 1b (mixer warm-up). The mixer's identity init
        means it starts near-pass-through, so unfreezing is safe.
        """
        if self.cross_attention_mixer is not None:
            for p in self.cross_attention_mixer.parameters():
                p.requires_grad = True
            self.cross_attention_mixer.train()

    def get_trainable_params(self, include_heads: bool = True) -> List[nn.Parameter]:
        """Return all trainable parameters.

        Args:
            include_heads: If False, exclude decision_head and assistant_head.
                           Used in Phase 1 (encoder + mixer only).
        """
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if not include_heads and ("decision_head" in name or "assistant_head" in name):
                    continue
                params.append(param)
        return params

    def get_head_params(self) -> List[nn.Parameter]:
        """Return only head parameters (decision_head + assistant_head)."""
        return list(self.decision_head.parameters()) + list(self.assistant_head.parameters())

    # ── Checkpoint helpers ─────────────────────────────────────

    @staticmethod
    def _filter_state_dict(state_dict: Dict[str, torch.Tensor],
                           keep_prefixes: set) -> Dict[str, torch.Tensor]:
        """Filter state dict to keep only params whose name starts with any prefix."""
        filtered = {}
        for key, val in state_dict.items():
            if any(key.startswith(p) for p in keep_prefixes):
                filtered[key] = val
        return filtered

    def get_trainable_state_dict(self, module_prefixes: set) -> Dict[str, torch.Tensor]:
        """Get state dict for trainable params matching given module prefixes."""
        full_sd = self.state_dict()
        return self._filter_state_dict(full_sd, module_prefixes)

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor],
                                   strict: bool = False):
        """Load state dict (may be a subset of full model)."""
        current = self.state_dict()
        # Only update keys that exist in both
        for key, val in state_dict.items():
            if key in current:
                current[key] = val
        self.load_state_dict(current, strict=False)

    def save_pretrain_checkpoint(self, path: str, epoch: int, loss: float,
                                  phase: str, optimizer_state: dict, config: dict):
        """Save a Phase 1 checkpoint (trainable params only, ~14MB)."""
        if phase == "encoder":
            prefixes = {"vision_encoder.projection", "traj_encoder",
                        "text_encoder.projection"}
        else:  # full pretrain (includes mixer)
            prefixes = {"vision_encoder.projection", "traj_encoder",
                        "text_encoder.projection", "cross_attention_mixer"}

        torch.save({
            "format_version": 2,
            "phase": phase,
            "trainable_state_dict": self.get_trainable_state_dict(prefixes),
            "optimizer_state_dict": optimizer_state,
            "backbone_refs": {"vision": "dinov2_vitb14", "text": "ViT-B-32"},
            "config": config,
            "epoch": epoch,
            "loss": loss,
        }, path)

    def save_heads_checkpoint(self, path: str, epoch: int, loss: float,
                               optimizer_state: dict, config: dict):
        """Save a Phase 2 checkpoint (head params only, ~0.4MB)."""
        torch.save({
            "format_version": 2,
            "phase": "heads",
            "trainable_state_dict": self.get_trainable_state_dict(
                {"decision_head", "assistant_head"}),
            "optimizer_state_dict": optimizer_state,
            "config": config,
            "epoch": epoch,
            "loss": loss,
        }, path)

    @classmethod
    def from_trainable_checkpoint(cls, path: str,
                                   device: str = "cpu") -> "ALIGNModel":
        """Load a full model from trainable-only checkpoints.

        DINOv2 and CLIP are loaded by name via torch.hub (frozen, cached).
        Only trainable params (proj + traj_encoder + mixer + heads) come
        from the checkpoint.
        """
        ckpt = torch.load(path, map_location=device)
        config = ckpt.get("config", {})
        model = cls(
            embed_dim=config.get("embed_dim", 256),
            chunk_size=config.get("chunk_size", 5),
            use_text=True,
            device=device,
            mixer_dim=config.get("mixer_dim", 512),
            num_mixer_blocks=config.get("num_mixer_blocks", 2),
        ).to(device)
        model.freeze_backbone()
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
        return model


# ================================================================
# Factory
# ================================================================

def create_align_model(
    embed_dim: int = 256,
    chunk_size: int = 5,
    use_text: bool = True,
) -> ALIGNModel:
    """Create a default ALIGN model for training."""
    return ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        use_text=use_text,
    )