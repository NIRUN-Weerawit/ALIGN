#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task Identification Head for ALIGN.

Predicts the LIBERO task the user is performing from (z_v, z_t) — the
shared vision+trajectory embedding produced by the cross-attention mixer.

Output is a (B, K+1) logit vector over K known tasks plus one OOD class.
The softmax of the logits does double duty:

  1. Confidence → gating signal α
        α = (1 - H) * (1 - p_ood) * safety_gate

  2. Task-aware language conditioning for the assistant head
        z_text = p @ E_clip   # E_clip is a (K+1, 256) buffer of CLIP text
                              # embeddings, precomputed at construction

Both come from the same softmax, so the model's confidence and the
language signal it feeds the assistant head are guaranteed to be
coherent: peaked on a hypothesis → strong task signal + high α.

Architecture
------------
    z_v (B, 256) ─┐
                   ├── cat → (B, 512) → Linear(512, 256) → ReLU
    z_t (B, 256) ─┘                              → Linear(256, 256) → ReLU
                                                  → Linear(256, K+1)
                                                  → logits (B, K+1)

Three linear layers, ~200K parameters. The OOD logit is the (K+1)-th
output of the same final linear layer; it competes with the K known-
class logits through the same softmax.

OOD training
------------
The OOD logit is trained against two sources of OOD samples:

  - **Held-out real tasks.** LIBERO tasks explicitly excluded from the K
    training classes. These teach the head to recognize genuinely novel
    tasks.
  - **Augmented in-distribution frames.** Clean frames corrupted with
    Gaussian noise / color jitter / random masking. These teach the head
    to recognize camera drift, lighting changes, and partial occlusion.

Loss:
    L_ce  = cross_entropy(logits[:, :-1], task_labels)        # K known
    L_ood = bce_with_logits(logits[:, -1],  ood_labels)        # OOD logit
    L     = L_ce + L_ood

References
----------
See docs/TASK_IDENTIFICATION_HEAD.md for the full design rationale.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Head
# ================================================================

class TaskHead(nn.Module):
    """Predicts the LIBERO task the user is performing.

    Inputs:
        z_v:  (B, 256) vision embedding (post-mixer)
        z_t:  (B, 256) trajectory embedding (post-mixer)

    Outputs:
        logits: (B, K+1) where the last logit is the OOD class.
                Apply softmax(dim=-1) to get a probability distribution.

    The head is a small 3-layer MLP. No attention, no recurrence.
    The trajectory window's sub-phase information is already encoded in
    z_t; the head only needs to identify which task those sub-phases
    belong to.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_known_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_known_classes = num_known_classes
        # K+1: K known + 1 OOD. The OOD class is the last logit, learned
        # through the same final linear layer (so it competes with known
        # classes in the same softmax).
        self.num_classes = num_known_classes + 1
        self.hidden_dim = hidden_dim

        layers: List[nn.Module] = [
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, self.num_classes),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_v: torch.Tensor,    # (B, embed_dim)
        z_t: torch.Tensor,    # (B, embed_dim)
    ) -> torch.Tensor:
        """Return (B, K+1) logits. Apply softmax(dim=-1) for probabilities."""
        x = torch.cat([z_v, z_t], dim=-1)
        return self.mlp(x)


# ================================================================
# Loss
# ================================================================

def task_head_loss(
    logits: torch.Tensor,        # (B, K+1)
    task_labels: torch.Tensor,   # (B,) int64 in [0, K)
    ood_labels: torch.Tensor,    # (B,) float32 in {0, 1}
    lambda_ood: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined cross-entropy + OOD binary cross-entropy.

    Both losses are on the same scale (lambda_ood = 1.0 by default) and
    both operate on logits from the same linear layer, so the OOD class
    is forced to compete with the K known classes for the model's mass.

    Returns:
        (total_loss, components_dict) where components_dict has
        {"task_ce": ..., "ood_bce": ...} as detached floats (for logging).
    """
    known_logits = logits[:, :-1]                # (B, K)
    ood_logit = logits[:, -1]                    # (B,)

    task_ce = F.cross_entropy(known_logits, task_labels)
    ood_bce = F.binary_cross_entropy_with_logits(ood_logit, ood_labels.float())

    total = task_ce + lambda_ood * ood_bce

    return total, {
        "task_ce": float(task_ce.detach()),
        "ood_bce": float(ood_bce.detach()),
    }


# ================================================================
# Bundle: head + CLIP-text basis + inference-time α
# ================================================================

class TaskHeadBundle(nn.Module):
    """TaskHead + precomputed CLIP text-embedding basis.

    This bundle is the complete inference-time unit. It contains:

      - `head`:    the TaskHead MLP, trained.
      - `E_clip`:  (K+1, 256) buffer of CLIP text embeddings, frozen.
                   Row i (i < K) is the embedding of the i-th known task
                   description. Row K (the OOD anchor) is the mean of
                   all K task embeddings, which represents a generic /
                   unknown task.

    At inference, the same softmax `p = softmax(logits)` does two jobs:

      1. α (the gating signal) from entropy + OOD probability
      2. z_text (the language conditioning for the assistant head) from
         `p @ E_clip`

    Both are coherent because they come from the same distribution.
    """

    def __init__(
        self,
        head: TaskHead,
        e_clip: torch.Tensor,    # (K+1, embed_dim) — precomputed CLIP basis
    ):
        super().__init__()
        self.head = head
        # register_buffer so E_clip moves with .to(device), gets saved in
        # state_dict, and is *not* registered as a learnable parameter.
        self.register_buffer("E_clip", e_clip)

    # ── forward ──────────────────────────────────────────────

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return all derived quantities from the head.

        Returns:
            {
              "logits":   (B, K+1) raw logits,
              "p":        (B, K+1) softmax probabilities,
              "entropy":  (B,)    normalized entropy H/H_max in [0, 1],
              "p_ood":    (B,)    OOD probability p[:, -1],
              "alpha":    (B,)    (1 - H) * (1 - p_ood),
              "z_text":   (B, D)  task-aware language embedding, p @ E_clip,
            }
        """
        logits = self.head(z_v, z_t)                    # (B, K+1)
        p = F.softmax(logits, dim=-1)                   # (B, K+1)

        # Normalized entropy: H in [0, 1], 0 = peaked, 1 = uniform.
        K = p.shape[-1]
        log_p = torch.log(p.clamp_min(1e-12))
        H = -(p * log_p).sum(dim=-1)                    # (B,) raw entropy
        H_max = float(torch.log(torch.tensor(K, dtype=H.dtype, device=H.device)))
        H_norm = H / H_max

        p_ood = p[:, -1]                                # (B,)
        alpha = (1.0 - H_norm) * (1.0 - p_ood)          # (B,)

        # Project task belief into the assistant head's language slot.
        # p @ E_clip is shape (B, D) where D = E_clip.shape[-1].
        z_text = p @ self.E_clip                        # (B, D)

        return {
            "logits": logits,
            "p": p,
            "entropy": H_norm,
            "p_ood": p_ood,
            "alpha": alpha,
            "z_text": z_text,
        }

    # ── convenience: build E_clip from a list of task strings ──

    @staticmethod
    def build_e_clip(
        task_texts: List[str],
        text_encoder,             # ALIGN's TextEncoder
        ood_anchor: str = "unknown task",
    ) -> torch.Tensor:
        """Precompute the (K+1, embed_dim) CLIP basis.

        Args:
            task_texts: K known task descriptions, e.g.
                        ["pick up the red mug",
                         "put the red mug on the left plate", ...]
            text_encoder: ALIGN's TextEncoder. Called once with
                          `text_encoder(task_texts + [ood_anchor])`,
                          producing a (K+1, embed_dim) tensor.
            ood_anchor: a string whose CLIP embedding will be used as
                        the OOD row. Defaults to "unknown task"; can be
                        set to anything, but the simplest convention is
                        to use the mean of the K known embeddings
                        (see `build_e_clip_mean` below).

        Returns:
            E_clip: (K+1, embed_dim) tensor suitable for register_buffer.
        """
        with torch.no_grad():
            E = text_encoder(task_texts + [ood_anchor])     # (K+1, D)
        return E.float()

    @staticmethod
    def build_e_clip_mean(
        task_texts: List[str],
        text_encoder,
    ) -> torch.Tensor:
        """Like `build_e_clip`, but the OOD row is the mean of the K
        known task embeddings rather than a separate text string.

        This is the recommended option — the OOD anchor is then
        guaranteed to be the geometric "average task" in CLIP space,
        which is the most semantically neutral possible embedding.
        """
        with torch.no_grad():
            E_known = text_encoder(task_texts)              # (K, D)
            ood_anchor = E_known.mean(dim=0, keepdim=True)  # (1, D)
            E = torch.cat([E_known, ood_anchor], dim=0)     # (K+1, D)
        return E.float()


# ================================================================
# Factory
# ================================================================

def create_task_head(
    num_known_classes: int,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.0,
) -> TaskHead:
    """Factory for TaskHead. Mirrors `create_value_head` / `create_world_model` style."""
    return TaskHead(
        embed_dim=embed_dim,
        num_known_classes=num_known_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


# ================================================================
# Smoke test
# ================================================================

if __name__ == "__main__":
    """Quick smoke test of TaskHead + TaskHeadBundle."""
    import torch

    print("Testing TaskHead...")
    B, D, K = 8, 256, 10
    head = TaskHead(embed_dim=D, num_known_classes=K)
    z_v = torch.randn(B, D)
    z_t = torch.randn(B, D)
    logits = head(z_v, z_t)
    assert logits.shape == (B, K + 1), f"logits shape: {logits.shape}"
    print(f"  logits shape: {logits.shape}")
    print(f"  logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

    print("\nTesting task_head_loss...")
    task_labels = torch.randint(0, K, (B,))
    ood_labels = torch.zeros(B)
    ood_labels[0:2] = 1.0  # 25% OOD
    loss, comps = task_head_loss(logits, task_labels, ood_labels)
    print(f"  loss: {loss.item():.4f}")
    print(f"  components: {comps}")
    loss.backward()
    print(f"  backward OK")

    print("\nTesting TaskHeadBundle (E_clip)...")
    E_clip = torch.randn(K + 1, D)
    bundle = TaskHeadBundle(head, E_clip)
    out = bundle(z_v, z_t)
    for k, v in out.items():
        print(f"  {k:8s}: {tuple(v.shape)}")
    assert out["z_text"].shape == (B, D)
    assert torch.all(out["alpha"] >= 0) and torch.all(out["alpha"] <= 1)
    print(f"  alpha range: [{out['alpha'].min().item():.3f}, {out['alpha'].max().item():.3f}]")

    print("\nAll smoke tests passed.")
