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
    """Frozen DINOv2 ViT-B + trainable projection head.

    Supports single-camera (4D: (B, H, W, 3)) and multi-camera (5D:
    (B, V, H, W, 3)) inputs.

    v2 (default): uses DINOv2 patch tokens — 256 patches per image,
    projected to ``embed_dim``. Returns:
        single camera: (B, num_patches, embed_dim)
        multi camera:  (B, V * num_patches, embed_dim)
    The per-patch features preserve spatial information that the older
    CLS-only head collapsed away.

    v1 (use_patch_tokens=False): keeps the original CLS-token behavior.
    Returns (B, embed_dim) regardless of camera count (multi-camera
    views are fused via a learnable linear layer back to embed_dim).
    """

    def __init__(self, backbone: str = "dinov2_vitb14", embed_dim: int = 256,
                 num_cameras: int = 1, use_patch_tokens: bool = True):
        super().__init__()
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
        except Exception:
            raise ImportError(
                "DINOv2 not installed. Run: pip install dinov2"
            )
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim
        # v2: use patch tokens by default; v1 falls back to CLS
        self.use_patch_tokens = use_patch_tokens

        # Per-camera projection: DINOv2 feature (768) → embed_dim (256)
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Multi-camera fusion (v1 only): concatenate V * embed_dim → embed_dim
        if num_cameras > 1 and not use_patch_tokens:
            self.fusion = nn.Sequential(
                nn.Linear(num_cameras * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode camera frame(s).

        Args:
            x: Single-camera: (B, H, W, 3) uint8 RGB.
               Multi-camera: (B, V, H, W, 3) uint8 RGB, V = num_cameras.

        Returns:
            v2 (use_patch_tokens=True):
                single camera: (B, num_patches, embed_dim)
                multi camera:  (B, V * num_patches, embed_dim)
            v1 (use_patch_tokens=False): (B, embed_dim) (fused if multi-cam)
        """
        # Detect multi-camera input
        if x.ndim == 5:
            B, V, H, W, C = x.shape
            assert V == self.num_cameras, (
                f"VisionEncoder expects {self.num_cameras} cameras, got {V}"
            )
            # Reshape to (B*V, H, W, 3) for batched DINOv2
            x = x.reshape(B * V, H, W, C)
        else:
            B, H, W, C = x.shape
            V = 1

        # Standard DINOv2 preprocessing
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
            if self.use_patch_tokens:
                # v2: get all patch tokens (no CLS) — preserves spatial info
                features_dict = self.backbone.forward_features(x)
                patch_features = features_dict["x_norm_patchtokens"]
                # (B*V, num_patches, 768)
                features = self.projection(patch_features)  # (B*V, num_patches, embed_dim)
                # Reshape to (B, V * num_patches, embed_dim) — multi-cam stacks
                # patches along the sequence dim; single-cam is unchanged.
                P = features.size(1)
                features = features.reshape(B, V * P, self.embed_dim)
            else:
                # v1: CLS token only — collapses spatial info
                features = self.backbone(x)  # (B*V, 768)
                features = self.projection(features)  # (B*V, embed_dim)
                if V > 1:
                    # Fuse multi-camera features
                    features = features.reshape(B, V * self.embed_dim)
                    features = self.fusion(features)  # (B, embed_dim)

        return features


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
# Robot State Encoder (v2)
# ================================================================

class RobotStateEncoder(nn.Module):
    """One-step robot state encoder.

    Input layout (B, 7): [pos(3), orientation(3), gripper(1)]
    Output: (B, state_dim)

    v2 replaces the v1 ``TrajectoryEncoder`` (which took (B, K, 6)
    windows and used a transformer). The new model conditions on a
    single current state vector rather than a window of past poses —
    the temporal context lives elsewhere in the architecture.
    """

    def __init__(
        self,
        input_dim: int = 7,
        state_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a one-step robot state vector.

        Args:
            x: (B, input_dim) — [pos(3), orientation(3), gripper(1)].

        Returns:
            (B, state_dim) state embedding.
        """
        return self.mlp(x)  # (B, state_dim)


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

class FuturePredictionHeadMLP(nn.Module):
    """MLP-based future prediction head.

    Input:
        z_v_window: (B, K, embed_dim) — per-step vision embeddings
                    (in the simple case, all K are the same current frame)
        z_t_window: (B, K, embed_dim) — per-step trajectory tokens
        z_text:    (B, embed_dim) — broadcasted
    Output:
        (predicted_z_v, predicted_z_t) — each (B, K, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        K: int = 10,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        # Input: K * (3 * embed_dim) — flattened (z_v, z_t, z_text) at K timesteps
        # Output: K * (2 * embed_dim) — predicted (z_v, z_t) at K timesteps
        input_dim = K * 3 * embed_dim
        output_dim = K * 2 * embed_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_v_window: torch.Tensor,  # (B, K, embed_dim)
        z_t_window: torch.Tensor,  # (B, K, embed_dim)
        z_text: torch.Tensor,      # (B, embed_dim) — broadcast over K
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict K future (z_v, z_t) embeddings from K past ones.

        Returns:
            (predicted_z_v, predicted_z_t), each of shape (B, K, embed_dim)
        """
        B, K, D = z_v_window.shape
        # Concatenate z_v and z_text at each timestep, z_t and z_text likewise
        z_text_expanded = z_text.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)
        # Concatenate features: (B, K, 3*D) — (z_v, z_t, z_text) at each step
        features = torch.cat([z_v_window, z_t_window, z_text_expanded], dim=-1)
        # Flatten: (B, K * 3 * D)
        flat = features.reshape(B, -1)
        # Predict: (B, K * 2 * D)
        out = self.mlp(flat)
        # Reshape to (B, K, 2, D) and split into z_v, z_t
        out = out.reshape(B, K, 2, D)
        predicted_z_v = out[:, :, 0, :]  # (B, K, D)
        predicted_z_t = out[:, :, 1, :]  # (B, K, D)
        return predicted_z_v, predicted_z_t


class FuturePredictionHeadTransformer(nn.Module):
    """Transformer-based future prediction head.

    Input: (B, K, 3*D) — concatenated (z_v, z_t, z_text) at K past timesteps
           plus learned positional encoding
    Output: (B, K, 2*D) — predicted (z_v, z_t) at K future timesteps

    Uses self-attention to capture temporal dependencies in the input
    window. The output for each position is a parallel prediction of
    the next-step embedding.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        K: int = 10,
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_timesteps: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.d_model = d_model
        # Input projection: 3 * embed_dim -> d_model
        self.input_proj = nn.Linear(3 * embed_dim, d_model)
        # Output projection: d_model -> 2 * embed_dim
        self.output_proj = nn.Linear(d_model, 2 * embed_dim)
        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_timesteps, d_model) * 0.02)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        z_v_window: torch.Tensor,  # (B, K, embed_dim)
        z_t_window: torch.Tensor,  # (B, K, embed_dim)
        z_text: torch.Tensor,      # (B, embed_dim) — broadcast over K
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict K future (z_v, z_t) embeddings from K past ones.

        Returns:
            (predicted_z_v, predicted_z_t), each of shape (B, K, embed_dim)
        """
        B, K, D = z_v_window.shape
        # Concatenate features per timestep
        z_text_expanded = z_text.unsqueeze(1).expand(-1, K, -1)
        features = torch.cat([z_v_window, z_t_window, z_text_expanded], dim=-1)
        # Project to d_model and add positional encoding
        x = self.input_proj(features)  # (B, K, d_model)
        x = x + self.pos_encoding[:K].unsqueeze(0)  # (1, K, d_model) broadcasts
        # Self-attention
        x = self.transformer(x)  # (B, K, d_model)
        # Project to output embeddings
        out = self.output_proj(x)  # (B, K, 2*embed_dim)
        # Split into z_v, z_t predictions
        predicted_z_v = out[:, :, :D]
        predicted_z_t = out[:, :, D:]
        return predicted_z_v, predicted_z_t



# ================================================================
# Assistant Head
# ================================================================

class AssistantHead(nn.Module):
    """MLP predicting chunk of K pose-relative GOALS from shared embeddings.

    Input layout: cat([z_v, z_t, z_text, current_action], dim=-1)
      - z_v: vision embedding (256D)
      - z_t: pose trajectory embedding (256D) — past K poses
      - z_text: text embedding (256D)

    Output: (B, K, 6) — K POSE-RELATIVE GOALS (delta from current noisy pose).
      goal[k] = (where the EEF should be at step k+1) - (current noisy pose)

    This is a planning-oriented quantity (vs. the older recovery-correction
    formulation, which only existed when there was error). Combined with α at
    inference time via:
        a_model = goal[0]                                   # model's proposed action
        final_action = (1 - α) * current_action + α * a_model  # α-weighted blend

    Args:
        latent_dim: per-modality embedding dim (default 256).
        chunk_size: number of goals to predict.
        action_dim: 6 (OSC_POSE).
        hidden_dim: hidden layer width (default 256).
        num_hidden_layers: number of hidden layers (default 2 → 256→128).
        dropout: dropout rate applied between hidden layers (default 0).
    """

    def __init__(self, latent_dim: int = 256, chunk_size: int = 5, action_dim: int = 6,
                 hidden_dim: int = 256, num_hidden_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        input_dim = latent_dim * 3
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        # Build a configurable MLP: input → hidden → hidden → ... → output
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, chunk_size * action_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        z_text: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_v, z_t, z_text], dim=-1)
        out = self.mlp(x)
        return out.reshape(-1, self.chunk_size, self.action_dim)


class AssistantHeadTransformer(nn.Module):
    """Transformer predicting a single next action from a window of K past embeddings.

    Input layout (per timestep): cat([z_v_k, z_t_k, z_text_broadcast], dim=-1)
      - z_v_window:    (B, K, embed_dim)  — vision embeddings for K past timesteps
      - z_t_window:    (B, K, embed_dim)  — pose trajectory embeddings for K past timesteps
      - z_text:        (B, embed_dim)     — text embedding (broadcast over K)

    Output: (B, action_dim) — single next ACTION in OSC units (m, rad → OSC scale).
            The transformer attends over the K past timesteps and emits one
            action prediction per query.

    Unlike AssistantHead (which sees a flat 3*embed_dim input), this variant
    sees K past timesteps and uses self-attention to model temporal patterns
    in the trajectory. The K input slots provide context, but only ONE
    action is produced.

    Args:
        embed_dim: per-modality embedding dim (default 256).
        action_dim: 6 (OSC_POSE).
        d_model: transformer hidden dim (default 384).
        nhead: number of attention heads (default 4).
        num_layers: number of transformer encoder layers (default 2).
        dim_feedforward: FFN hidden dim (default 1024).
        dropout: dropout rate (default 0.1).
        pool: "last" | "mean" — how to aggregate the K output tokens into
            a single action prediction. "last" uses the final timestep
            (default); "mean" averages all K positions.
    """

    def __init__(self, embed_dim: int = 256, chunk_size: int = 5, action_dim: int = 6,
                 d_model: int = 384, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 pool: str = "last"):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.pool = pool
        # Project per-timestep features (3*embed_dim) to d_model
        self.input_proj = nn.Linear(3 * embed_dim, d_model)
        # Learned positional encoding for the K input timesteps
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output head: (B, d_model) -> (B, action_dim)
        self.output_proj = nn.Linear(d_model, action_dim)

    def forward(
        self,
        z_v_window: torch.Tensor,  # (B, K, embed_dim)
        z_t_window: torch.Tensor,  # (B, K, embed_dim)
        z_text: torch.Tensor,       # (B, embed_dim)
    ) -> torch.Tensor:
        B, K = z_v_window.shape[:2]
        # Broadcast text over the K window
        z_text_expanded = z_text.unsqueeze(1).expand(-1, K, -1)  # (B, K, embed_dim)
        # Concatenate per-timestep features
        features = torch.cat([z_v_window, z_t_window, z_text_expanded], dim=-1)  # (B, K, 3D)
        # Project to d_model + add positional encoding
        x = self.input_proj(features) + self.pos_encoding[:K].unsqueeze(0)  # (B, K, d_model)
        # Self-attention over the K timesteps
        x = self.transformer(x)  # (B, K, d_model)
        # Pool the K output tokens into a single vector
        if self.pool == "last":
            pooled = x[:, -1]  # (B, d_model)
        elif self.pool == "mean":
            pooled = x.mean(dim=1)  # (B, d_model)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")
        # Output head
        out = self.output_proj(pooled)  # (B, action_dim)
        return out


# ================================================================
# Factory for cross-attention mixer
# ================================================================

def _create_mixer(embed_dim: int = 256, mixer_dim: int = 512,
                  num_blocks: int = 2, nhead: int = 8,
                  max_traj_len: int = 64,
                  # v2 per-modality dims (default to embed_dim for v1 callers)
                  vision_dim: Optional[int] = None,
                  state_dim: Optional[int] = None,
                  text_dim: Optional[int] = None) -> nn.Module:
    """Import and create CrossAttentionMixer (lazy import).

    v2 passes per-modality dims; v1 callers can pass ``embed_dim`` only
    and it will be used for all three modalities inside the mixer.
    """
    from models.cross_attention_mixer import CrossAttentionMixer
    return CrossAttentionMixer(
        enc_dim=embed_dim,
        vision_dim=vision_dim if vision_dim is not None else embed_dim,
        state_dim=state_dim if state_dim is not None else embed_dim,
        text_dim=text_dim if text_dim is not None else embed_dim,
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
    "state_encoder",               # v2: state encoder (one-step)
    "traj_encoder",                # v1: trajectory encoder (kept for back-compat)
    "text_encoder.projection",     # text_proj
    "cross_attention_mixer",       # mixer (always present)
    "decision_head",               # α predictor
    "assistant_head",              # Δpose predictor
}

ENCODER_MODULES = {
    "vision_encoder.projection",
    "state_encoder",
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

    v2 architecture (default):
        - VisionEncoder uses DINOv2 patch tokens → (B, P, vision_dim)
        - RobotStateEncoder encodes one-step state (B, 7) → (B, state_dim)
        - TextEncoder (B, 512) → (B, text_dim)
        - CrossAttentionMixer mixes per-modality dims, returns per-modality dims
        - Heads consume the mixed embeddings

    v1 backward compat (set ``use_patch_tokens=False``, ``embed_dim=256``):
        - All dims collapse to 256 — old checkpoints load unchanged.
    """

    def __init__(
        self,
        # ── Backward-compat single dim (v1) ────────────────────
        # If the caller passes embed_dim (v1), it is used for ALL three
        # per-modality dims unless they are explicitly overridden.
        embed_dim: Optional[int] = None,
        # ── v2 per-modality dims ───────────────────────────────
        vision_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        # ── Vision / state architecture flags ──────────────────
        use_patch_tokens: bool = True,
        state_input_dim: int = 7,
        # ── Legacy trajectory-encoder params (ignored in v2) ────
        traj_input_dim: int = 6,
        traj_d_model: int = 128,
        traj_nhead: int = 4,
        traj_num_layers: int = 3,
        # ── Standard head / model params ────────────────────────
        chunk_size: int = 5,
        action_dim: int = 6,
        use_text: bool = True,
        device: Optional[str] = None,
        mixer_dim: int = 512,
        num_mixer_blocks: int = 2,
        mixer_nhead: int = 8,
        max_traj_len: int = 64,
        decision_K: int = 10,
        decision_arch: str = "mlp",
        # Multi-camera vision encoder
        num_cameras: int = 1,
        # MLP head params
        mlp_hidden_dim: int = 512,
        mlp_num_layers: int = 3,
        # Transformer head params
        num_layers: int = 2,
        d_model: int = 384,
        nhead: int = 4,
        dropout: float = 0.0,
        dim_feedforward: int = 1024,
        # Assistant head params
        assistant_hidden: int = 256,
        assistant_layers: int = 2,
        assistant_dropout: float = 0.0,
        # Assistant architecture: "mlp" (default, original) or "transformer"
        assistant_arch: str = "mlp",
        # Transformer assistant params
        assistant_d_model: int = 384,
        assistant_nhead: int = 4,
        assistant_num_layers: int = 2,
        assistant_dim_ff: int = 1024,
    ):
        super().__init__()

        # Resolve per-modality dims.
        #   - If embed_dim is given (v1 backward compat), use it for all three
        #     modalities unless a per-modality dim is also given.
        #   - Otherwise fall back to v2 defaults: vision=512, state=256, text=256.
        if embed_dim is not None:
            # v1 caller: route embed_dim into the missing per-modality dims
            if vision_dim is None:
                vision_dim = embed_dim
            if state_dim is None:
                state_dim = embed_dim
            if text_dim is None:
                text_dim = embed_dim
        else:
            # v2 defaults
            if vision_dim is None:
                vision_dim = 512
            if state_dim is None:
                state_dim = 256
            if text_dim is None:
                text_dim = 256

        # Keep the legacy attribute for any external code that reads it.
        self.embed_dim = embed_dim if embed_dim is not None else vision_dim
        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.use_patch_tokens = use_patch_tokens
        self.use_text = use_text
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.decision_K = decision_K
        self.decision_arch = decision_arch

        # Encoders
        self.num_cameras = num_cameras
        self.vision_encoder = VisionEncoder(
            embed_dim=vision_dim, num_cameras=num_cameras,
            use_patch_tokens=use_patch_tokens,
        )
        # v2: state encoder (one-step). v1 TrajectoryEncoder is also
        # constructed (marked as `state_encoder_legacy`) so old checkpoints
        # that have `traj_encoder.*` keys can still be loaded — those
        # parameters are simply ignored at inference.
        self.state_encoder = RobotStateEncoder(
            input_dim=state_input_dim,
            state_dim=state_dim,
            hidden_dim=state_dim,
            dropout=0.1,
        )
        # Keep the legacy TrajectoryEncoder around for backward compat
        # (so old checkpoints load without missing-key errors). Marked as
        # not-trained-by-default since the new pipeline uses the state
        # encoder. The legacy module is reachable via ``traj_encoder``.
        self.traj_encoder = TrajectoryEncoder(
            input_dim=traj_input_dim,
            d_model=traj_d_model,
            nhead=traj_nhead,
            num_layers=traj_num_layers,
            embed_dim=embed_dim if embed_dim is not None else vision_dim,
        )
        self.text_encoder = TextEncoder(embed_dim=text_dim) if use_text else None

        # Cross-attention mixer (always present, identity-initialized)
        self.cross_attention_mixer = _create_mixer(
            embed_dim=embed_dim if embed_dim is not None else vision_dim,
            vision_dim=vision_dim,
            state_dim=state_dim,
            text_dim=text_dim,
            mixer_dim=mixer_dim,
            num_blocks=num_mixer_blocks,
            nhead=mixer_nhead,
            max_traj_len=max_traj_len,
        )

        # Heads
        # The heads still use a single latent dim (vision_dim, which is
        # the largest). State and text are projected to that dim inside
        # the heads if needed (see ``_align_for_head`` helper).
        head_dim = vision_dim
        # Decision Head = Future Prediction Head.
        # Predicts K future embeddings from K past ones. The prediction
        # error becomes the alpha signal at inference time.
        if decision_arch == "mlp":
            self.decision_head = FuturePredictionHeadMLP(
                embed_dim=head_dim, K=decision_K,
                hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers,
            )
        elif decision_arch == "transformer":
            self.decision_head = FuturePredictionHeadTransformer(
                embed_dim=head_dim, K=decision_K,
                d_model=d_model, nhead=nhead,
                num_layers=num_layers, dropout=dropout,
                dim_feedforward=dim_feedforward,
            )
        else:
            raise ValueError(
                f"Unknown decision_arch: {decision_arch} (expected 'mlp' or 'transformer')"
            )
        # Assistant head: "mlp" (default, original) or "transformer"
        self.assistant_arch = assistant_arch
        if assistant_arch == "mlp":
            self.assistant_head = AssistantHead(
                latent_dim=head_dim, chunk_size=chunk_size, action_dim=action_dim,
                hidden_dim=assistant_hidden,
                num_hidden_layers=assistant_layers,
                dropout=assistant_dropout,
            )
        elif assistant_arch == "transformer":
            self.assistant_head = AssistantHeadTransformer(
                embed_dim=head_dim, chunk_size=chunk_size, action_dim=action_dim,
                d_model=assistant_d_model,
                nhead=assistant_nhead,
                num_layers=assistant_num_layers,
                dim_feedforward=assistant_dim_ff,
                dropout=assistant_dropout,
            )
        else:
            raise ValueError(
                f"Unknown assistant_arch: {assistant_arch} (expected 'mlp' or 'transformer')"
            )

        # Register which modules are trainable vs frozen
        # v2: state_encoder replaces traj_encoder. We list BOTH prefixes
        # so old checkpoints that save "traj_encoder.*" still find a
        # matching module, AND the new "state_encoder.*" is also trainable.
        self._trainable_prefixes = {"vision_encoder.projection", "state_encoder",
                                     "traj_encoder", "text_encoder.projection",
                                     "cross_attention_mixer",
                                     "decision_head", "assistant_head"}
        self._encoder_prefixes = {"vision_encoder.projection", "state_encoder",
                                   "traj_encoder", "text_encoder.projection",
                                   "cross_attention_mixer"}
        self._backbone_prefixes = {"vision_encoder.backbone", "text_encoder.model"}

        # Build head input adapters so heads see a common dim.
        # The mixer outputs per-modality dims (vision_dim, state_dim, text_dim).
        # The heads want a single latent_dim = head_dim. For the v2 default
        # (vision_dim == head_dim) these are identity-shaped no-ops.
        self._head_state_proj = nn.Linear(state_dim, head_dim, bias=False) if state_dim != head_dim else None
        self._head_text_proj = nn.Linear(text_dim, head_dim, bias=False) if text_dim != head_dim else None

    # ── Phase 1a helpers: raw encoder outputs (no mixer) ─────────

    def encode_raw_vision(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames, return raw vision embedding (no mixer).

        v2 returns patch tokens (no mean-pooling). Callers that want a
        single vector per frame should use ``encode_raw_vision_pooled``
        or mean-pool over the patch dim.

        Supports:
          - (B, H, W, 3) — single camera, single timestep
                          → (B, num_patches, vision_dim)
          - (B, V, H, W, 3) — multi-camera, single timestep
                          → (B, V * num_patches, vision_dim)
          - (B, K, H, W, 3) — single camera, K timesteps
                          → (B, K, num_patches, vision_dim)
          - (B, K, V, H, W, 3) — multi-camera, K timesteps
                          → (B, K, V * num_patches, vision_dim)
        """
        return self.vision_encoder(frames)

    def encode_raw_vision_pooled(self, frames: torch.Tensor) -> torch.Tensor:
        """Mean-pool patch tokens into a single vector per frame.

        v2 only. Returns (B, vision_dim) for single-step input or
        (B, K, vision_dim) for windowed input.
        """
        z = self.encode_raw_vision(frames)
        if z.ndim == 3:
            # (B, P, D) or (B, K, D) — pool over the middle dim
            return z.mean(dim=1)
        elif z.ndim == 4:
            # (B, K, P, D) — pool over the patch dim
            return z.mean(dim=2)
        return z

    def encode_raw_vision_window(
        self, frames_window: torch.Tensor
    ) -> torch.Tensor:
        """Encode a window of K frames, return per-timestep patch embeddings.

        Args:
            frames_window: (B, K, H, W, 3) for single-camera or
                           (B, K, V, H, W, 3) for multi-camera.

        Returns:
            v2: (B, K, num_patches, vision_dim) — per-timestep patch tokens.
            v1 (use_patch_tokens=False): (B, K, vision_dim) — single CLS token
                  per timestep, fused across cameras.

        For v2 the caller can mean-pool over the patch dim to recover a
        per-timestep summary embedding.
        """
        if frames_window.ndim == 5:
            B, K, H, W, C = frames_window.shape
            frames_flat = frames_window.reshape(B * K, H, W, C)
            z_v_flat = self.vision_encoder(frames_flat)  # (B*K, P, D) or (B*K, D)
            if self.use_patch_tokens:
                P = z_v_flat.size(1)
                D = z_v_flat.size(2)
                return z_v_flat.reshape(B, K, P, D)
            return z_v_flat.reshape(B, K, -1)
        elif frames_window.ndim == 6:
            # Multi-camera window: (B, K, V, H, W, 3)
            B, K, V, H, W, C = frames_window.shape
            frames_flat = frames_window.reshape(B * K, V, H, W, C)
            z_v_flat = self.vision_encoder(frames_flat)  # (B*K, V*P, D) or (B*K, D)
            if self.use_patch_tokens:
                VP = z_v_flat.size(1)
                D = z_v_flat.size(2)
                return z_v_flat.reshape(B, K, VP, D)
            return z_v_flat.reshape(B, K, -1)
        else:
            raise ValueError(
                f"frames_window must be 5D (B,K,H,W,3) or 6D (B,K,V,H,W,3), "
                f"got {frames_window.ndim}D"
            )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a one-step robot state vector (v2).

        Args:
            state: (B, 7) — [pos(3), orientation(3), gripper(1)].

        Returns:
            (B, state_dim) state embedding.
        """
        return self.state_encoder(state)

    def encode_raw_trajectory(self, poses: torch.Tensor) -> torch.Tensor:
        """[v1 only] Encode trajectory, return raw mean-pooled trajectory embedding (no mixer).

        In v2 this is a backward-compat wrapper around the legacy
        ``TrajectoryEncoder``. New code should use ``encode_state``.
        """
        return self.traj_encoder(poses)

    def encode_raw_trajectory_tokens(self, state: torch.Tensor) -> torch.Tensor:
        """[v2] Encode a one-step state — alias for ``encode_state``.

        Kept as a backward-compat name. Old callers that passed
        ``(B, K, 6)`` trajectory windows can still call this method;
        the result is now a (B, state_dim) state embedding (no K dim).
        """
        # Backward compat: if input is (B, K, D), take the last step
        if state.ndim == 3:
            state = state[:, -1, :]
        return self.state_encoder(state)

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
        z_t = self.encode_raw_trajectory_tokens(traj)
        z_text = self.encode_raw_text(texts)
        if z_text is None:
            # Match the modality dim of z_v (vision is the largest in v2)
            z_text = torch.zeros(z_v.shape[0], self.text_dim, device=z_v.device,
                                  dtype=z_v.dtype)
        return {"z_v": z_v, "z_t": z_t, "z_text": z_text}

    # ── Phase 1b helper: full encode-through-mixer ─────────────

    def encode_mixed(self, frames: torch.Tensor,
                     traj: torch.Tensor,
                     texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Encode all three modalities and pass through cross-attention mixer.

        Phase 1b uses this: InfoNCE on mixer outputs, mixer unfrozen.

        v2 changes the input contract: ``traj`` is now a one-step state
        ``(B, 7)`` (forwarded to ``encode_state``) instead of a window
        ``(B, K, 6)``. Old callers passing ``(B, K, 6)`` still work — the
        last step is used as the current state.

        Returns:
            Dict with:
              'z_v'        — (B, P, vision_dim) v2 patch tokens, or
                              (B, vision_dim) v1 CLS token.
              'z_v_pooled' — (B, vision_dim) v2 mean of patches (for back-compat).
              'z_t'        — (B, state_dim) state embedding (one-step).
              'z_text'     — (B, text_dim) text embedding.
              'z_t_tokens' — (B, 1, state_dim) state token in mixer format
                              (kept for back-compat with v1 callers that
                              expect a K-dim window).
        """
        z_v = self.encode_raw_vision(frames)            # v2: (B, P, vision_dim)
        z_state = self.encode_raw_trajectory_tokens(traj)  # v2: (B, state_dim)
        z_text = self.encode_raw_text(texts)
        if z_text is None:
            z_text = torch.zeros(z_v.shape[0], self.text_dim,
                                  device=z_v.device, dtype=z_v.dtype)

        # Mean-pool patch tokens for the v1-shaped "z_v_pooled" output.
        # v1 callers expect a single (B, D) vector here.
        if z_v.ndim == 3:
            z_v_pooled = z_v.mean(dim=1)  # (B, vision_dim)
        else:
            z_v_pooled = z_v

        # The mixer expects z_t as (B, K, D). We have a one-step state
        # (B, state_dim). Add a K=1 dim — the position encoding slot for
        # the current frame is well-defined (position 0 = now).
        z_state_for_mixer = z_state.unsqueeze(1)  # (B, 1, state_dim)

        # Through mixer. The mixer's per-modality input/output projections
        # handle the (possibly different) per-modality dims.
        z_v_out, z_state_out, z_text_out = self.cross_attention_mixer(
            z_v, z_state_for_mixer, z_text
        )

        # Squeeze back to (B, state_dim) for the one-step state output.
        z_state_out = z_state_out.squeeze(1)  # (B, state_dim)

        # Mean-pool the mixer's vision output for back-compat
        if z_v_out.ndim == 3:
            z_v_pooled_out = z_v_out.mean(dim=1)
        else:
            z_v_pooled_out = z_v_out

        return {
            "z_v": z_v_out,
            "z_v_pooled": z_v_pooled_out,
            "z_t": z_state_out,
            "z_text": z_text_out,
            # K=1 window form, kept for back-compat with old training code
            "z_t_tokens": z_state_out.unsqueeze(1),
            # Also keep the original raw pooled vision for downstream use
            "z_v_pooled_pre_mixer": z_v_pooled,
        }

    # ── Standard encode (for compatibility and Phase 2) ────────

    def encode_vision(self, frames: torch.Tensor) -> torch.Tensor:
        """[v2] Returns patch tokens (B, P, vision_dim).

        For v1 (use_patch_tokens=False): returns (B, vision_dim).
        """
        return self.vision_encoder(frames)

    def encode_trajectory(self, poses: torch.Tensor) -> torch.Tensor:
        """[v1 legacy] Mean-pooled trajectory embedding. Use ``encode_state`` in v2."""
        return self.traj_encoder(poses)

    def encode_trajectory_tokens(self, poses: torch.Tensor) -> torch.Tensor:
        """[v1 legacy] Per-token trajectory embeddings. Use ``encode_state`` in v2."""
        return self.traj_encoder.encode_tokens(poses)

    def encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        return self.text_encoder(texts)

    # ── Phase 2 helper: future prediction → α ─────────────

    def predict_future(
        self,
        z_v_window: torch.Tensor,  # (B, K, embed_dim)
        z_t_window: torch.Tensor,  # (B, K, embed_dim)
        z_text: torch.Tensor,      # (B, embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Decision (Future Prediction) head.

        Returns:
            (predicted_z_v, predicted_z_t) — each (B, K, embed_dim)
        """
        return self.decision_head(z_v_window, z_t_window, z_text)

    @staticmethod
    def compute_alpha_from_predictions(
        predicted_z_v: torch.Tensor,  # (B, K, embed_dim)
        predicted_z_t: torch.Tensor,  # (B, K, embed_dim)
        actual_z_v: torch.Tensor,     # (B, K, embed_dim)
        actual_z_t: torch.Tensor,     # (B, K, embed_dim)
        aggregation: str = "weighted_mean",
        decay: float = 0.7,
    ) -> torch.Tensor:
        """Compute α from the cosine-similarity loss between predicted and actual.

        Cosine loss is bounded in [0, 2]; α = 1 - cos_loss / 2 maps to [0, 1].
        The model helps fully when the prediction matches (α ≈ 1), and
        lays low when it doesn't (α ≈ 0).

        Args:
            predicted_z_v, predicted_z_t: (B, K, embed_dim) — model predictions
            actual_z_v, actual_z_t: (B, K, embed_dim) — ground truth embeddings
            aggregation: "weighted_mean", "last_step_only", or "mean"
            decay: weight decay for "weighted_mean" (most recent has highest weight)

        Returns:
            α: (B,) — intervention strength
        """
        # Cosine similarity along the last dim, then convert to "error"
        cos_v = F.cosine_similarity(predicted_z_v, actual_z_v, dim=-1)  # (B, K)
        cos_t = F.cosine_similarity(predicted_z_t, actual_z_t, dim=-1)  # (B, K)
        cos_error = ((1 - cos_v) + (1 - cos_t)) / 2  # (B, K), in [0, 2]

        if aggregation == "last_step_only":
            cos_error = cos_error[:, -1]
        elif aggregation == "mean":
            cos_error = cos_error.mean(dim=-1)
        elif aggregation == "weighted_mean":
            K = cos_error.shape[1]
            # weights = [decay^(K-1-i) for i in range(K)]  (most recent = i=K-1 has weight 1)
            weights = torch.tensor(
                [decay ** (K - 1 - i) for i in range(K)],
                device=cos_error.device,
                dtype=cos_error.dtype,
            )
            cos_error = (cos_error * weights).sum(dim=-1) / weights.sum()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Map to [0, 1]
        alpha = 1.0 - cos_error / 2.0
        return alpha

    @staticmethod
    def future_prediction_loss(
        predicted_z_v: torch.Tensor,  # (B, K, embed_dim)
        predicted_z_t: torch.Tensor,  # (B, K, embed_dim)
        target_z_v: torch.Tensor,     # (B, K, embed_dim) — detached
        target_z_t: torch.Tensor,     # (B, K, embed_dim) — detached
        decay: float = 1.0,           # exponential decay weight on older steps
    ) -> torch.Tensor:
        """Cosine-similarity loss for future prediction.

        Bounded in [0, 2] per step. Returns the weighted mean loss over the
        K window, where older steps are weighted by `decay^(K-1-i)`. With
        decay=1.0, all steps are weighted equally (simple mean). With
        decay=0.7, the most recent step is weighted 1.0 and the oldest step
        is weighted 0.7^(K-1).

        Targets are detached (stop-gradient) so this loss doesn't try to
        reshape the encoder.
        """
        cos_v = F.cosine_similarity(predicted_z_v, target_z_v.detach(), dim=-1)  # (B, K)
        cos_t = F.cosine_similarity(predicted_z_t, target_z_t.detach(), dim=-1)  # (B, K)
        per_step = ((1 - cos_v) + (1 - cos_t)) / 2  # (B, K) in [0, 1]
        if decay < 1.0:
            K = per_step.shape[1]
            # weight = decay^(K-1-i) — most recent step gets the highest weight
            weights = torch.tensor(
                [decay ** (K - 1 - i) for i in range(K)],
                device=per_step.device, dtype=per_step.dtype,
            )
            weights = weights / weights.sum()  # normalize
            loss = (per_step * weights).sum()
        else:
            loss = per_step.mean()
        return loss

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
            traj: (B, 7) one-step state vector (v2) or (B, K, 6)
                  trajectory window (v1 backward compat — last step is used).
            texts: Optional list of task descriptions.
            compute_decision: Whether to compute α.
            compute_assistant: Whether to compute Δposes.

        Returns:
            Dict with 'alpha', 'delta' keys as present.
        """
        mixed = self.encode_mixed(frames, traj, texts)
        # v2: z_v is (B, P, vision_dim); z_v_pooled is (B, vision_dim).
        #     z_t is (B, state_dim); z_t_tokens is (B, 1, state_dim).
        z_v_pooled = mixed["z_v_pooled"]
        z_t = mixed["z_t"]  # (B, state_dim)
        z_text = mixed["z_text"]
        z_t_tokens = mixed["z_t_tokens"]  # (B, 1, head_dim)

        # Project state and text to the head's common dim if needed.
        # For the v2 default (state_dim=256, text_dim=256, head_dim=vision_dim=512)
        # this is a (256, 512) linear projection. When all dims are equal
        # (e.g. embed_dim=256 path), the projections are skipped and the
        # tensors are used as-is.
        if self._head_state_proj is not None:
            z_t_for_head = self._head_state_proj(z_t)
            z_t_tokens_for_head = self._head_state_proj(z_t_tokens) if z_t_tokens.ndim == 2 else self._head_state_proj(z_t_tokens.view(-1, self.state_dim)).view(z_t_tokens.shape[0], z_t_tokens.shape[1], -1)
        else:
            z_t_for_head = z_t
            z_t_tokens_for_head = z_t_tokens
        if self._head_text_proj is not None:
            z_text_for_head = self._head_text_proj(z_text)
        else:
            z_text_for_head = z_text

        result: Dict[str, torch.Tensor] = {}
        if compute_decision:
            # The Decision head takes a window of past embeddings and predicts
            # K future ones. In v2 the window is K=1 (the current one-step
            # state), and the head still predicts K=decision_K future states.
            B = z_v_pooled.shape[0]
            K_input = z_t_tokens_for_head.shape[1]
            z_v_window = z_v_pooled.unsqueeze(1).expand(-1, K_input, -1)  # (B, K, head_dim)
            predicted_z_v, predicted_z_t = self.decision_head(
                z_v_window, z_t_tokens_for_head, z_text_for_head
            )
            result["predicted_z_v"] = predicted_z_v
            result["predicted_z_t"] = predicted_z_t
            # For backward-compat: also compute a per-batch "alpha" estimate
            # assuming the predictions match the inputs (degenerate case).
            # In practice, callers should use compute_alpha_from_predictions()
            # with the actual future embeddings.
            result["alpha"] = torch.ones(B, 1, device=z_v_pooled.device)
        if compute_assistant:
            # Call the appropriate assistant head based on architecture.
            # Both MLP and transformer heads take (z_v, z_t, z_text).
            if self.assistant_arch == "transformer":
                # Transformer needs (B, K, D) windows. Replicate the current
                # embeddings K times to form a fake "window" of K identical
                # timesteps. In practice, callers should pass a real window
                # (see encode_mixed_windowed for the proper API).
                K = self.assistant_head.chunk_size
                z_v_window = z_v_pooled.unsqueeze(1).expand(-1, K, -1)
                z_t_window = z_t_for_head.unsqueeze(1).expand(-1, K, -1)
                result["delta"] = self.assistant_head(
                    z_v_window, z_t_window, z_text_for_head
                )
            else:
                result["delta"] = self.assistant_head(
                    z_v_pooled, z_t_for_head, z_text_for_head
                )
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
        """Freeze ALL encoders (vision_proj, state_encoder, traj_encoder,
        text_proj, mixer).

        Called in Phase 2 (head training) so gradients only flow
        into decision_head and assistant_head.
        """
        for name, module in self.named_modules():
            # Check if this module is an encoder (not backbone, not head)
            is_encoder = any(prefix in name for prefix in
                             ["vision_encoder.projection", "state_encoder",
                              "traj_encoder",
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
            prefixes = {"vision_encoder.projection", "state_encoder",
                        "traj_encoder", "text_encoder.projection"}
        else:  # full pretrain (includes mixer)
            prefixes = {"vision_encoder.projection", "state_encoder",
                        "traj_encoder", "text_encoder.projection",
                        "cross_attention_mixer"}

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