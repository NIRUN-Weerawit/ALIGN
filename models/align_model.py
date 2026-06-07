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
    alpha = model.decision_head(vision, traj, text, cos_sims, distances)
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
    """Frozen CLIP ViT-B/32 text tower + trainable projection head."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
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
        try:
            import clip
        except ImportError:
            raise ImportError("CLIP not installed")
        tokens = clip.tokenize(texts, truncate=True).to(next(self.projection.parameters()).device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()  # (B, 512)
        return self.projection(features)  # (B, 256)


# ================================================================
# Decision Head
# ================================================================

class DecisionHead(nn.Module):
    """MLP predicting α ∈ [0, 1] from shared embeddings + alignment scores."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Input: z_v (256) + z_t (256) + z_text (256) + cos_vt(1) + cos_vl(1) + cos_tl(1) + dist(3)
        input_dim = latent_dim * 3 + 3 + 3
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
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Predict assist confidence α.

        Args:
            z_v: (B, D) vision embeddings.
            z_t: (B, D) trajectory embeddings.
            z_text: (B, D) text embeddings.
            distances: (B, 3) distance features.

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
        x = torch.cat([z_v, z_t, z_text, cos_vt, cos_vl, cos_tl, distances], dim=-1)
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text
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

        self.decision_head = DecisionHead(latent_dim=embed_dim)
        self.assistant_head = AssistantHead(
            latent_dim=embed_dim, chunk_size=chunk_size, action_dim=action_dim
        )

    def encode_vision(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(frames)

    def encode_trajectory(self, poses: torch.Tensor) -> torch.Tensor:
        return self.traj_encoder(poses)

    def encode_text(self, texts: list[str]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        return self.text_encoder(texts)

    def forward(
        self,
        frames: torch.Tensor,
        traj: torch.Tensor,
        texts: Optional[list[str]] = None,
        distances: Optional[torch.Tensor] = None,
        compute_decision: bool = True,
        compute_assistant: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            frames: (B, H, W, 3) RGB images.
            traj: (B, K, 6) trajectory window.
            texts: Optional list of task descriptions.
            distances: (B, 3) distance features. Required for Decision head.
            compute_decision: Whether to compute α.
            compute_assistant: Whether to compute Δposes.

        Returns:
            Dict with 'alpha', 'delta' keys as present.
        """
        z_v = self.encode_vision(frames)
        z_t = self.encode_trajectory(traj)
        z_text = self.encode_text(texts) if self.use_text and texts is not None else None
        if z_text is None:
            # No text — use zero embedding as neutral fallback
            z_text = torch.zeros_like(z_v)

        result: Dict[str, torch.Tensor] = {}
        if compute_decision:
            if distances is None:
                distances = torch.zeros(z_v.shape[0], 3, device=z_v.device)
            result["alpha"] = self.decision_head(z_v, z_t, z_text, distances)
        if compute_assistant:
            result["delta"] = self.assistant_head(z_v, z_t, z_text, traj[:, -1])
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
