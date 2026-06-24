"""
World model for ALIGN.

Action-conditioned single-step transition model: f(s, a) -> s'.

Used for counterfactual imagination in the alpha pipeline:
  - Imagine: s'_h = f(s, a_h)  (what happens if the human acts)
  - Imagine: s'_m = f(s, a_m)  (what happens if the model acts)
  - Compare V(s'_m) vs V(s'_h) to derive alpha

This is a SEPARATE component from the existing FuturePredictionHead:
  - FuturePredictionHead: predicts K parallel future embeddings (no action)
  - WorldModel: predicts 1 next embedding from current state + action

Both can coexist in the model; the FuturePredictionHead continues to
serve the current alpha-decision signal, while the WorldModel is the
foundation for the new counterfactual alpha computation.

Implementation mirrors FuturePredictionHeadMLP / FuturePredictionHeadTransformer
but with:
  - Action input (6D OSC_POSE delta) concatenated to (z_v, z_t, z_text)
  - Single-step output (z_v', z_t') instead of K parallel
  - MSE loss on real embedding values (not cosine) for accurate dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WorldModelMLP(nn.Module):
    """MLP-based action-conditioned transition model with temporal window.

    f(s_{t-K:t}, a_t) -> s_{t+1}

    Accepts a window of K past embeddings to provide temporal context,
    similar to how the decision head sees K past timesteps.

    Input layout (per sample):
      - z_v_window: (B, K, embed_dim) K past vision embeddings
      - z_t_window: (B, K, embed_dim) K past trajectory embeddings
      - z_text:     (B, embed_dim) text embedding (constant for the task)
      - action:     (B, action_dim) the action to apply (6D OSC_POSE delta)

    Output:
      - z_v_prime: (B, embed_dim) predicted next vision embedding
      - z_t_prime: (B, embed_dim) predicted next trajectory embedding

    Args:
        embed_dim: per-modality embedding dim (default 256).
        action_dim: 6 (OSC_POSE delta in meters / axis-angle).
        window_size: number of past timesteps to condition on (default 5).
        hidden_dim: MLP hidden layer width.
        num_layers: total number of linear layers (>=2).
        dropout: dropout between hidden layers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        action_dim: int = 6,
        window_size: int = 5,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.window_size = window_size
        # Input: K * (z_v + z_t) + z_text + action
        input_dim = window_size * 2 * embed_dim + embed_dim + action_dim
        output_dim = 2 * embed_dim

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, output_dim)
        nn.init.normal_(final.weight, std=0.02)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_v_window: torch.Tensor,  # (B, K, embed_dim)
        z_t_window: torch.Tensor,  # (B, K, embed_dim)
        z_text: torch.Tensor,      # (B, embed_dim)
        action: torch.Tensor,      # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state embeddings from a window of past states + action.

        Returns:
            (z_v_prime, z_t_prime), each of shape (B, embed_dim)
        """
        B, K, D = z_v_window.shape
        # Flatten window: (B, K * 2 * D)
        window_flat = torch.cat([z_v_window, z_t_window], dim=-1).reshape(B, -1)
        # Concatenate with text and action: (B, K*2*D + D + A)
        x = torch.cat([window_flat, z_text, action], dim=-1)
        out = self.mlp(x)
        z_v_prime = out[:, :self.embed_dim]
        z_t_prime = out[:, self.embed_dim:]
        return z_v_prime, z_t_prime


class WorldModelTransformer(nn.Module):
    """Transformer-based action-conditioned transition model.

    Same input/output contract as WorldModelMLP, but uses a transformer
    encoder for richer representation. The action is concatenated to
    each input, then projected to d_model.

    For single-step prediction, this is over-parameterized vs MLP,
    but allows for future extensions (e.g., action-sequence models).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        action_dim: int = 6,
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.d_model = d_model
        # Input projection: 3*embed_dim + action_dim -> d_model
        self.input_proj = nn.Linear(3 * embed_dim + action_dim, d_model)
        # Output projection: d_model -> 2*embed_dim
        self.output_proj = nn.Linear(d_model, 2 * embed_dim)
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
        z_v: torch.Tensor,    # (B, embed_dim)
        z_t: torch.Tensor,    # (B, embed_dim)
        z_text: torch.Tensor, # (B, embed_dim)
        action: torch.Tensor, # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state embeddings.

        Returns:
            (z_v_prime, z_t_prime), each of shape (B, embed_dim)
        """
        B, D = z_v.shape
        # Stack the three state vectors + action into a sequence of 1 token
        # (single-step: one "transition" token, but we can also think of
        # this as a sequence of length 1)
        x = torch.cat([z_v, z_t, z_text, action], dim=-1)  # (B, 3D + A)
        x = x.unsqueeze(1)  # (B, 1, 3D + A)
        # Project
        x = self.input_proj(x)  # (B, 1, d_model)
        # Transformer
        x = self.transformer(x)  # (B, 1, d_model)
        # Project to output
        out = self.output_proj(x)  # (B, 1, 2*embed_dim)
        out = out.squeeze(1)  # (B, 2*embed_dim)
        # Split
        z_v_prime = out[:, :D]
        z_t_prime = out[:, D:]
        return z_v_prime, z_t_prime


def create_world_model(
    arch: str = "mlp",
    embed_dim: int = 256,
    action_dim: int = 6,
    **kwargs,
) -> nn.Module:
    """Factory for world model.

    Args:
        arch: 'mlp' or 'transformer'.
        embed_dim: per-modality embedding dim.
        action_dim: action dim (default 6 for OSC_POSE).
        **kwargs: passed to the underlying class.
    """
    if arch == "mlp":
        return WorldModelMLP(
            embed_dim=embed_dim, action_dim=action_dim, **kwargs
        )
    elif arch == "transformer":
        return WorldModelTransformer(
            embed_dim=embed_dim, action_dim=action_dim, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown arch: {arch} (expected 'mlp' or 'transformer')"
        )


# ============================================================
# Loss
# ============================================================

def world_model_loss(
    z_v_pred: torch.Tensor,
    z_v_true: torch.Tensor,
    z_t_pred: torch.Tensor,
    z_t_true: torch.Tensor,
) -> torch.Tensor:
    """MSE on actual embedding values (not cosine).

    Returns scalar loss = mean(MSE(z_v) + MSE(z_t)).

    Why MSE not cosine: we need imagined states to have the right scale.
    V(s) is meaningless if imagined s' is in a different scale than
    real s'. Cosine only captures direction, not magnitude.

    Targets should be detached (stop-gradient) so this loss doesn't
    reshape the encoder.
    """
    loss_v = F.mse_loss(z_v_pred, z_v_true.detach())
    loss_t = F.mse_loss(z_t_pred, z_t_true.detach())
    return (loss_v + loss_t) / 2


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    """Quick smoke test of the WorldModel class."""
    import torch

    print("Testing WorldModelMLP with window...")
    B, D, A, K = 4, 256, 6, 5
    wm_mlp = WorldModelMLP(embed_dim=D, action_dim=A, window_size=K)
    z_v_window = torch.randn(B, K, D)
    z_t_window = torch.randn(B, K, D)
    z_text = torch.randn(B, D)
    action = torch.randn(B, A)
    z_v_p, z_t_p = wm_mlp(z_v_window, z_t_window, z_text, action)
    assert z_v_p.shape == (B, D), f"z_v_p shape: {z_v_p.shape}"
    assert z_t_p.shape == (B, D), f"z_t_p shape: {z_t_p.shape}"
    print(f"  Output shapes OK: {z_v_p.shape}, {z_t_p.shape}")
    loss = world_model_loss(z_v_p, z_v_window[:, -1], z_t_p, z_t_window[:, -1])
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print(f"  Backward OK, params have gradients")

    print("\nTesting WorldModelTransformer...")
    wm_tx = WorldModelTransformer(embed_dim=D, action_dim=A)
    z_v_p, z_t_p = wm_tx(z_v_window[:, -1], z_t_window[:, -1], z_text, action)
    assert z_v_p.shape == (B, D)
    assert z_t_p.shape == (B, D)
    print(f"  Output shapes OK: {z_v_p.shape}, {z_t_p.shape}")
    loss = world_model_loss(z_v_p, z_v_window[:, -1], z_t_p, z_t_window[:, -1])
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print(f"  Backward OK")

    print("\nTesting create_world_model factory...")
    wm_factory = create_world_model("mlp", embed_dim=D, action_dim=A, window_size=K)
    z_v_p, z_t_p = wm_factory(z_v_window, z_t_window, z_text, action)
    print(f"  Factory created: {type(wm_factory).__name__}")

    print("\nAll smoke tests passed.")
