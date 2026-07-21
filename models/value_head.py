"""
Value head for the alpha pipeline.

Estimates V(s) = E[sum_k gamma^k r(s_k, a_k) | s_0 = s]
where r(s, a) comes from the trained GAIL discriminator.

Used for counterfactual alpha:
  alpha = sigmoid((V(s'_m) - V(s'_h)) / tau)

where s'_m = world_model(s, a_m) and s'_h = world_model(s, a_h).

V is trained with TD(λ) return on real expert trajectories:
  G_t^lambda = (1-lambda) * sum_n lambda^{n-1} * G_t^{(n)} + lambda^{T-t-1} * G_t^{(T-t)}
  where G_t^{(n)} = r_t + gamma * r_{t+1} + ... + gamma^n * V(s_{t+n})

Loss: MSE(V(s_t), G_t^lambda.detach())

This is intentionally simple: just a small MLP that maps (z_v, z_s, z_sext)
to a scalar. The complexity is in the training (TD(λ) returns), not the
architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHeadMLP(nn.Module):
    """Estimates V(s) = expected cumulative reward from state s.

    Input:  (z_v, z_s, z_sext) shape (B, 3*D) — concatenated
    Output: scalar V(s) shape (B,)

    The head is intentionally simple: a small MLP. The value function
    needs to capture the relationship between state and expected return,
    not the action — that's the world model's job.

    Args:
        embed_dim: per-modality embedding dim (default 256).
        hidden_dim: hidden layer width.
        num_layers: total number of linear layers (>=2).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = 3 * embed_dim
        output_dim = 1  # scalar V

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Final layer: small init so V starts near 0
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
    ) -> torch.Tensor:
        """Compute V(s) for each state.

        Returns:
            V: (B,) scalar value per state
        """
        x = torch.cat([z_v, z_s, z_sext], dim=-1)
        out = self.mlp(x)  # (B, 1)
        return out.squeeze(-1)  # (B,)


def value_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for V(s).

    Targets should be detached (stop-gradient) so the loss doesn't try
    to make V(s) match itself.

    Args:
        v_pred: (B,) predicted values
        v_target: (B,) target values (TD(λ) return)
    """
    return F.mse_loss(v_pred, v_target.detach())


def compute_td_lambda_return(
    rewards: torch.Tensor,   # (T,) per-step rewards
    values: torch.Tensor,     # (T+1,) values at each state (includes terminal)
    gamma: float = 0.99,
    lam: float = 0.7,
) -> torch.Tensor:
    """Compute TD(λ) return for each timestep.

    G_t^lambda = (1-lambda) * sum_{n=1}^{T-t-1} lambda^{n-1} * G_t^{(n)} + lambda^{T-t-1} * G_t^{(T-t)}

    where G_t^{(n)} = r_t + gamma*r_{t+1} + ... + gamma^n * V(s_{t+n})

    Args:
        rewards: (T,) per-step rewards
        values: (T+1,) values at each state s_0, s_1, ..., s_T (terminal)
        gamma: discount factor
        lam: trace decay

    Returns:
        returns: (T,) TD(λ) return at each timestep
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)

    # Compute the lambda-blend weights
    # For each n-step return, weight = (1-lambda) * lambda^{n-1}
    # Cap at T-t (the rest is the full return with weight lambda^{T-t-1})
    for t in range(T):
        g = 0.0
        weight = 1.0
        for n in range(1, T - t + 1):
            # n-step return: sum_k=t^{t+n-1} gamma^{k-t} * r_k + gamma^n * V(s_{t+n})
            n_step = 0.0
            for k in range(t, min(t + n, T)):
                n_step += (gamma ** (k - t)) * rewards[k]
            n_step += (gamma ** n) * values[min(t + n, T)]
            if n < T - t:
                w = (1 - lam) * (lam ** (n - 1))
            else:
                w = lam ** (T - t - 1)  # Last term
            g += w * n_step
        returns[t] = g

    return returns


def create_value_head(
    embed_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
) -> ValueHeadMLP:
    """Factory for value head. Always uses MLP for now."""
    return ValueHeadMLP(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    """Quick smoke test of the value head."""
    import torch

    print("Testing ValueHeadMLP...")
    B, D = 4, 256
    vh = ValueHeadMLP(embed_dim=D)
    z_v = torch.randn(B, D)
    z_s = torch.randn(B, D)
    z_sext = torch.randn(B, D)
    v = vh(z_v, z_s, z_sext)
    assert v.shape == (B,), f"v shape: {v.shape}"
    print(f"  V shape: {v.shape}")
    print(f"  V values: {v.tolist()}")
    # Test loss
    target = torch.randn(B)
    loss = value_loss(v, target)
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print(f"  Backward OK")
    # Test TD(λ)
    T = 10
    rewards = torch.randn(T) * 0.1  # small rewards
    values = torch.randn(T + 1)
    returns = compute_td_lambda_return(rewards, values, gamma=0.99, lam=0.7)
    print(f"  TD(λ) returns shape: {returns.shape}")
    print(f"  Returns range: [{returns.min().item():.4f}, {returns.max().item():.4f}]")

    print("\nAll smoke tests passed.")
