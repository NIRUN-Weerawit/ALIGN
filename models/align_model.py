#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN model architecture — shared backbone + dual heads.

Architecture:
    DINOv2 ViT-B (frozen) → projection → z_v (256d)
    Transformer (trained) → projection → z_t (256d)
    CLIP text (frozen) → projection → z_text (256d)
        │                              │
        └────────── shared ─────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Decision Head (α)   Assistant Head (Δposes)

Usage:
    from models.align_model import ALIGNModel
    model = ALIGNModel(use_text=True)
    vision = model.encode_vision(frames)        # (B, 256)
    traj = model.encode_trajectory(poses)       # (B, K, 6) → (B, 256)
    text = model.encode_text(descriptions)      # (B, 256)
    alpha = model.decision_head(vision, traj, text, cos_sims)
    # NOTE: decision head now takes ONLY z_v, z_t, z_text — no distances
    # Distance is learned implicitly from visual features (DINOv2 captures
    # object scale/depth cues). This is required for real deployment where
    # per-object distance sensors are unreliable or absent.
    delta = model.assistant_head(vision, traj, text, noisy_pose)
"""

from typing import Optional, Tuple, Dict

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
        # Lazy import — DINOv2 may not be available without `pip install dinov2`
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
            x: (B, H, W, 3) uint8 RGB images. Will be normalized internally.

        Returns:
            (B, embed_dim) vision embeddings.
        """
        B, H, W, C = x.shape
        if C != 3:
            raise ValueError(f"Expected HWC RGB images, got shape {x.shape}")
        # Resize to 224×224 for DINOv2
        if H != 224 or W != 224:
            x = F.interpolate(
                x.permute(0, 3, 1, 2).float(), size=(224, 224), mode="bilinear"
            )
        else:
            x = x.permute(0, 3, 1, 2).float()
        # Normalize to [0, 1] then ImageNet mean/std
        x = x / 255.0
        # DINOv2 expects BCHW float with ImageNet normalization
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
            (B, embed_dim) trajectory embedding.
        """
        x = self.input_proj(x)  # (B, K, d_model)
        x = self.transformer(x)  # (B, K, d_model)
        x = x.mean(dim=1)  # (B, d_model) mean pooling over time
        return self.projection(x)  # (B, 256)


# ================================================================
# Text Encoder
# ================================================================

class TextEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 text tower + trainable projection head.

    Uses open_clip (LAION) which is actively maintained and installable via pip.
    Fallback to OpenAI's clip package if open_clip is unavailable.
    """

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

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode text descriptions.

        Args:
            texts: List of strings, length B.

        Returns:
            (B, embed_dim) text embeddings.
        """
        if self._use_open_clip:
            import open_clip
            tokens = self._tokenizer(texts).to(next(self.projection.parameters()).device)
        else:
            import clip
            tokens = clip.tokenize(texts, truncate=True).to(next(self.projection.parameters()).device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()  # (B, 512)
        return self.projection(features)  # (B, 256)


# ================================================================
# Decision Head
# ================================================================

class DecisionHead(nn.Module):
    """MLP predicting α ∈ [0, 1] from shared embeddings + alignment scores.

    Input is everything computable from the encoders at inference time:
        z_v, z_t, z_text (256d each) + cos_vt, cos_vl, cos_tl (1d each).
    No external inputs (e.g. object distance) — the model must learn
    "near vs. far" from the visual features themselves. This is critical
    for real deployment where per-object distance is rarely available.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Input: z_v (256) + z_t (256) + z_text (256) + cos_vt(1) + cos_vl(1) + cos_tl(1)
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
        """Predict assist confidence α.

        Args:
            z_v: (B, D) vision embeddings.
            z_t: (B, D) trajectory embeddings.
            z_text: (B, D) text embeddings.

        Returns:
            (B, 1) α ∈ [0, 1].
        """
        B = z_v.shape[0]
        z_v_n = F.normalize(z_v, dim=-1)
        z_t_n = F.normalize(z_t, dim=-1)
        z_text_n = F.normalize(z_text, dim=-1)
        cos_vt = (z_v_n * z_t_n).sum(dim=-1, keepdim=True)  # (B, 1)
        cos_vl = (z_v_n * z_text_n).sum(dim=-1, keepdim=True)  # (B, 1)
        cos_tl = (z_t_n * z_text_n).sum(dim=-1, keepdim=True)  # (B, 1)
        x = torch.cat([z_v, z_t, z_text, cos_vt, cos_vl, cos_tl], dim=-1)
        return self.mlp(x)


# ================================================================
# Assistant Head
# ================================================================

class AssistantHead(nn.Module):
    """MLP predicting chunk of K corrective Δposes from shared embeddings."""

    def __init__(self, latent_dim: int = 256, chunk_size: int = 5, action_dim: int = 6):
        super().__init__()
        input_dim = latent_dim * 3 + action_dim  # z_v + z_t + z_text + current_pose
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
        """Predict future corrective Δposes.

        Args:
            z_v: (B, D) vision embeddings.
            z_t: (B, D) trajectory embeddings.
            z_text: (B, D) text embeddings.
            noisy_pose: (B, action_dim) current noisy EEF pose.

        Returns:
            (B, chunk_size, action_dim) predicted corrective Δposes.
        """
        x = torch.cat([z_v, z_t, z_text, noisy_pose], dim=-1)
        out = self.mlp(x)  # (B, chunk_size * action_dim)
        return out.reshape(-1, self.chunk_size, self.action_dim)


# ================================================================
# Full ALIGN Model
# ================================================================

class ALIGNModel(nn.Module):
    """Complete ALIGN shared autonomy model.

    Combines three encoders and two heads. Text encoder is optional.
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
        use_cross_attention: bool = False,
        mixer_dim: int = 512,
        num_mixer_blocks: int = 2,
        mixer_nhead: int = 8,
        max_traj_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text
        self.use_cross_attention = use_cross_attention
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.traj_encoder = TrajectoryEncoder(
            input_dim=traj_input_dim,
            d_model=traj_d_model,
            nhead=traj_nhead,
            num_layers=traj_num_layers,
            embed_dim=embed_dim,
        )
        self.text_encoder = TextEncoder(embed_dim=embed_dim) if use_text else None

        # Cross-attention mixer (opt-in, default off)
        if use_cross_attention:
            from models.cross_attention_mixer import CrossAttentionMixer
            self.cross_attention_mixer = CrossAttentionMixer(
                enc_dim=embed_dim,
                mixer_dim=mixer_dim,
                num_blocks=num_mixer_blocks,
                nhead=mixer_nhead,
                max_traj_len=max_traj_len,
            )
        else:
            self.cross_attention_mixer = None

        self.decision_head = DecisionHead(latent_dim=embed_dim)
        self.assistant_head = AssistantHead(
            latent_dim=embed_dim, chunk_size=chunk_size, action_dim=action_dim
        )

    def encode_vision(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(frames)

    def encode_trajectory(self, poses: torch.Tensor) -> torch.Tensor:
        return self.traj_encoder(poses)

    def encode_trajectory_tokens(self, poses: torch.Tensor) -> torch.Tensor:
        """Return per-token trajectory embeddings (B, K, D) — for the mixer.

        Standard encode_trajectory mean-pools over K. The mixer needs the
        full K tokens for cross-attention (one per frame).
        """
        x = self.traj_encoder.input_proj(poses)  # (B, K, d_model)
        x = self.traj_encoder.transformer(x)    # (B, K, d_model)
        return self.traj_encoder.projection(x)   # (B, K, embed_dim)

    def encode_text(self, texts: list[str]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        return self.text_encoder(texts)

    def forward(
        self,
        frames: torch.Tensor,
        traj: torch.Tensor,
        texts: Optional[list[str]] = None,
        compute_decision: bool = True,
        compute_assistant: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            frames: (B, H, W, 3) RGB images.
            traj: (B, K, 6) trajectory window.
            texts: Optional list of task descriptions.
            compute_decision: Whether to compute α.
            compute_assistant: Whether to compute Δposes.

        Returns:
            Dict with 'alpha', 'delta' keys as present.
        """
        z_v = self.encode_vision(frames)
        if self.cross_attention_mixer is not None:
            z_t = self.encode_trajectory_tokens(traj)  # (B, K, D) for cross-attn
        else:
            z_t = self.encode_trajectory(traj)  # (B, D) mean-pooled
        z_text = self.encode_text(texts) if self.use_text and texts is not None else None
        if z_text is None:
            # No text — use zero embedding as neutral fallback
            z_text = torch.zeros_like(z_v)

        # Cross-attention mixer: enriches each modality with cross-modal info
        # Identity init means pre-trained encoder features are preserved at start.
        if self.cross_attention_mixer is not None:
            z_v, z_t, z_text = self.cross_attention_mixer(z_v, z_t, z_text)

        # Heads expect z_t as (B, 256) — mean-pool the K trajectory tokens
        if z_t.dim() == 3:
            z_t_for_head = z_t.mean(dim=1)  # (B, K, 256) → (B, 256)
        else:
            z_t_for_head = z_t

        result: Dict[str, torch.Tensor] = {}
        if compute_decision:
            result["alpha"] = self.decision_head(z_v, z_t_for_head, z_text)
        if compute_assistant:
            result["delta"] = self.assistant_head(z_v, z_t_for_head, z_text, traj[:, -1])
        return result

    def freeze_backbone(self):
        """Freeze vision and text backbones (called after contrastive pretraining)."""
        self.vision_encoder.backbone.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.model.requires_grad_(False)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Returns list of trainable parameters."""
        params = []
        params.extend(self.vision_encoder.projection.parameters())
        params.extend(self.traj_encoder.parameters())
        if self.text_encoder is not None:
            params.extend(self.text_encoder.projection.parameters())
        params.extend(self.decision_head.parameters())
        params.extend(self.assistant_head.parameters())
        return params


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
