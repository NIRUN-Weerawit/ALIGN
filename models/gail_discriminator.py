"""
GAIL discriminator for ALIGN.

A binary classifier D(s, a) trained to distinguish expert (s, a) pairs
from rollout (s, a) pairs. The discriminator is used as a learned
reward signal for the alpha pipeline:

    r(s, a) = -log(1 - D(s, a))           # GAIL reward (Ho & Ermon, 2016)

Higher reward = more "expert-like". The reward is then used to train
the value head V(s) which drives the alpha intervention score.

Stage 1 of the alpha pipeline (per docs/ALPHA_INTERVENTION_DESIGN.md):

    Stage 1: Train GAIL discriminator D(s, a)
      - Expert data:  real LIBERO demonstrations
      - Rollout data: random actions OR world model rollouts
      - Loss: BCE(D(expert) -> 1) + BCE(D(rollout) -> 0)

This is the SEPARATE Stage-1 component, distinct from the world model
(Stage 0, see models/world_model.py) which models dynamics only, and
from the value head (Stage 2, future work) which regresses on the
GAIL reward.

Implementation mirrors WorldModelMLP / WorldModelTransformer:
  - Same input contract: (z_v, z_s, z_sext, action) shape (B, 3*D + 6)
  - Output: a SINGLE logit per sample (apply sigmoid for probability)
  - Loss: BCEWithLogitsLoss on expert vs rollout labels
  - Numerical stability: log() of small sigmoid values done via softplus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# Models
# =============================================================

class GAILDiscriminatorMLP(nn.Module):
    """MLP-based GAIL discriminator D(s, a) -> logit.

    Binary classifier: expert (s, a) vs rollout (s, a).

    Input layout (per sample):
      - z_v:    (B, embed_dim) current vision embedding
      - z_s:    (B, embed_dim) current trajectory embedding
      - z_sext: (B, embed_dim) text embedding (constant for the task)
      - action: (B, action_dim) the action to apply (6D OSC_POSE delta)

    Output:
      - logits: (B,) raw discriminator logit. P(expert) = sigmoid(logits).
        Output is a logit (not a probability) so we can use
        BCEWithLogitsLoss directly without numerical issues.

    Args:
        embed_dim: per-modality embedding dim (default 256).
        action_dim: 6 (OSC_POSE delta in meters / axis-angle).
        hidden_dim: MLP hidden layer width.
        num_layers: total number of linear layers (>=2).
        dropout: dropout between hidden layers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        action_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.arch = "mlp"

        # Input: 3*embed_dim + action_dim
        # Output: 1 logit per sample
        input_dim = 3 * embed_dim + action_dim
        output_dim = 1

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Final layer: a single logit. Small init so initial D ≈ 0.5
        # (no early signal helps the discriminator learn a non-trivial
        # boundary vs always predicting "expert" / "rollout").
        final = nn.Linear(in_dim, output_dim)
        nn.init.normal_(final.weight, std=0.02)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_v: torch.Tensor,    # (B, embed_dim)
        z_s: torch.Tensor,    # (B, embed_dim)
        z_sext: torch.Tensor, # (B, embed_dim)
        action: torch.Tensor, # (B, action_dim)
    ) -> torch.Tensor:
        """Compute discriminator logits.

        Returns:
            logits: (B,) — raw discriminator output. P(expert) = sigmoid(logits).
        """
        # Concatenate all inputs: (B, 3*embed_dim + action_dim)
        x = torch.cat([z_v, z_s, z_sext, action], dim=-1)
        # MLP: (B, 1) -> squeeze to (B,)
        out = self.mlp(x).squeeze(-1)
        return out

    def predict_proba(
        self,
        z_v: torch.Tensor,
        z_s: torch.Tensor,
        z_sext: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: P(expert | s, a) ∈ (0, 1) (NOT for loss use).

        For training, use the logit output + BCEWithLogitsLoss directly.
        For reward computation, use `compute_reward(logits)` below.
        """
        logits = self.forward(z_v, z_s, z_sext, action)
        return torch.sigmoid(logits)


class GAILDiscriminatorTransformer(nn.Module):
    """Transformer-based GAIL discriminator D(s, a) -> logit.

    Same input/output contract as GAILDiscriminatorMLP, but uses a
    transformer encoder. Mirrors WorldModelTransformer's structure
    for consistency.

    For single-step (s, a) classification, this is over-parameterized
    vs MLP, but allows for future extensions (e.g., action-sequence
    discriminators).
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
        self.arch = "transformer"

        # Input projection: 3*embed_dim + action_dim -> d_model
        self.input_proj = nn.Linear(3 * embed_dim + action_dim, d_model)
        # Output projection: d_model -> 1 logit
        self.output_proj = nn.Linear(d_model, 1)
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
        z_s: torch.Tensor,    # (B, embed_dim)
        z_sext: torch.Tensor, # (B, embed_dim)
        action: torch.Tensor, # (B, action_dim)
    ) -> torch.Tensor:
        """Compute discriminator logits.

        Returns:
            logits: (B,) — raw discriminator output.
        """
        B = z_v.shape[0]
        # Stack the three state vectors + action into a single token
        x = torch.cat([z_v, z_s, z_sext, action], dim=-1)  # (B, 3D + A)
        x = x.unsqueeze(1)  # (B, 1, 3D + A)
        # Project
        x = self.input_proj(x)  # (B, 1, d_model)
        # Transformer
        x = self.transformer(x)  # (B, 1, d_model)
        # Project to single logit
        out = self.output_proj(x)  # (B, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (B,)
        return out

    def predict_proba(
        self,
        z_v: torch.Tensor,
        z_s: torch.Tensor,
        z_sext: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: P(expert | s, a) ∈ (0, 1)."""
        logits = self.forward(z_v, z_s, z_sext, action)
        return torch.sigmoid(logits)


# =============================================================
# Factory
# =============================================================

def create_gail_discriminator(
    arch: str = "mlp",
    embed_dim: int = 256,
    action_dim: int = 6,
    **kwargs,
) -> nn.Module:
    """Factory for GAIL discriminator.

    Args:
        arch: 'mlp' or 'transformer'.
        embed_dim: per-modality embedding dim.
        action_dim: action dim (default 6 for OSC_POSE).
        **kwargs: passed to the underlying class.
    """
    if arch == "mlp":
        return GAILDiscriminatorMLP(
            embed_dim=embed_dim, action_dim=action_dim, **kwargs
        )
    elif arch == "transformer":
        return GAILDiscriminatorTransformer(
            embed_dim=embed_dim, action_dim=action_dim, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown arch: {arch} (expected 'mlp' or 'transformer')"
        )


# =============================================================
# Loss
# =============================================================

def gail_loss(
    expert_logits: torch.Tensor,    # (B,) — D(s_expert, a_expert)
    rollout_logits: torch.Tensor,   # (B,) — D(s_rollout, a_rollout)
) -> tuple:
    """GAIL discriminator loss.

    Standard BCE-with-logits:
        D(expert) -> 1,  D(rollout) -> 0.

    BCEWithLogitsLoss is numerically stable (does the sigmoid
    internally) and decoupled from any external threshold.

    Returns:
        (total_loss, expert_acc, rollout_acc) — each a 0-d tensor.
        Accuracies are computed as `sigmoid(logit) > 0.5`.
    """
    B = expert_logits.shape[0]
    expert_labels = torch.ones_like(expert_logits)
    rollout_labels = torch.zeros_like(rollout_logits)

    loss_expert = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
    loss_rollout = F.binary_cross_entropy_with_logits(rollout_logits, rollout_labels)
    total_loss = (loss_expert + loss_rollout) / 2

    with torch.no_grad():
        # Accuracy: sigmoid(logit) > 0.5 for the correct class
        expert_acc = (torch.sigmoid(expert_logits) > 0.5).float().mean()
        rollout_acc = (torch.sigmoid(rollout_logits) <= 0.5).float().mean()

    return total_loss, expert_acc, rollout_acc


# =============================================================
# Reward
# =============================================================

def compute_reward(logits: torch.Tensor) -> torch.Tensor:
    """GAIL reward r(s, a) = -log(1 - sigmoid(logits)).

    Numerically-stable form using softplus:
        -log(1 - sigmoid(x)) = softplus(x) = log(1 + exp(x))

    Properties:
      - reward >= 0
      - reward is monotonically increasing in the discriminator's
        confidence that (s, a) is expert-like
      - reward -> 0 as D -> 0  (rollout-like)
      - reward -> +inf as D -> 1  (expert-like)

    Args:
        logits: (B,) raw discriminator logits.

    Returns:
        reward: (B,) non-negative GAIL reward.
    """
    # softplus(x) = log(1 + exp(x)) = -log(sigmoid(-x)) = -log(1 - sigmoid(x))
    # Use softplus directly (numerically stable for both large + and large - x).
    return F.softplus(logits)


# =============================================================
# Smoke test
# =============================================================

if __name__ == "__main__":
    """Quick smoke test of the GAIL discriminator."""
    import torch

    print("Testing GAILDiscriminatorMLP...")
    B, D, A = 4, 256, 6
    disc_mlp = GAILDiscriminatorMLP(embed_dim=D, action_dim=A)
    z_v = torch.randn(B, D)
    z_s = torch.randn(B, D)
    z_sext = torch.randn(B, D)
    action = torch.randn(B, A)
    logits = disc_mlp(z_v, z_s, z_sext, action)
    assert logits.shape == (B,), f"logits shape: {logits.shape}"
    print(f"  Output shape OK: {logits.shape}")

    # predict_proba
    proba = disc_mlp.predict_proba(z_v, z_s, z_sext, action)
    assert proba.shape == (B,)
    assert (proba >= 0).all() and (proba <= 1).all()
    print(f"  predict_proba in [0, 1]: min={proba.min().item():.3f}, max={proba.max().item():.3f}")

    # Loss
    expert_logits = logits          # synthetic
    rollout_logits = torch.randn(B)
    loss, exp_acc, rol_acc = gail_loss(expert_logits, rollout_logits)
    print(f"  Loss: {loss.item():.4f}  exp_acc={exp_acc.item():.3f}  rol_acc={rol_acc.item():.3f}")
    loss.backward()
    print(f"  Backward OK")

    # Reward
    reward = compute_reward(logits)
    assert (reward >= 0).all(), f"reward must be >= 0, got min={reward.min().item()}"
    print(f"  Reward >= 0: min={reward.min().item():.3f}, max={reward.max().item():.3f}")

    print("\nTesting GAILDiscriminatorTransformer...")
    disc_tx = GAILDiscriminatorTransformer(embed_dim=D, action_dim=A)
    logits = disc_tx(z_v, z_s, z_sext, action)
    assert logits.shape == (B,), f"logits shape: {logits.shape}"
    print(f"  Output shape OK: {logits.shape}")
    loss, _, _ = gail_loss(logits, torch.randn(B))
    loss.backward()
    print(f"  Backward OK")

    print("\nTesting create_gail_discriminator factory...")
    disc_factory = create_gail_discriminator("mlp", embed_dim=D, action_dim=A)
    logits = disc_factory(z_v, z_s, z_sext, action)
    print(f"  Factory created: {type(disc_factory).__name__}")

    print("\nAll smoke tests passed.")
