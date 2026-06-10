#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-way contrastive loss for ALIGN pretraining.

InfoNCE loss for three modality pairs:
    L = mean(L_vt + L_vl + L_tl) / 3

where each L_xy = (cross_entropy(logits_xy) + cross_entropy(logits_yx)) / 2

Positive pairs come from same episode (within temporal window for v↔t).
Negative pairs come from different episodes in the batch.

All embeddings are L2-normalized before computing similarity.
Temperature is fixed (not a learned parameter).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss3Way(nn.Module):
    """3-way contrastive loss: InfoNCE on (vision, trajectory, text) triples.

    Inputs must be raw (un-normalized) embeddings — normalization is performed
    internally to ensure correct cosine similarity.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature  # float, NOT nn.Parameter

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        z_text: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute 3-way contrastive loss.

        Args:
            z_v: (B, D) vision embeddings (raw, will be L2-normalized).
            z_t: (B, D) trajectory embeddings (raw, will be L2-normalized).
            z_text: (B, D) text embeddings (raw, will be L2-normalized).

        Returns:
            dict with 'loss', 'loss_vt', 'loss_vl', 'loss_tl', 'avg_cos_vt',
            'avg_cos_vl', 'avg_cos_tl' where all cos values are in [-1, 1].
        """
        # L2 normalize all embeddings — cosine similarity requires this
        z_v = F.normalize(z_v, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        z_text = F.normalize(z_text, dim=-1)

        loss_vt, stats_vt = _pairwise_info_nce(z_v, z_t, self.temperature)
        loss_vl, stats_vl = _pairwise_info_nce(z_v, z_text, self.temperature)
        loss_tl, stats_tl = _pairwise_info_nce(z_t, z_text, self.temperature)

        loss = (loss_vt + loss_vl + loss_tl) / 3.0

        return {
            "loss": loss,
            "loss_vt": loss_vt,
            "loss_vl": loss_vl,
            "loss_tl": loss_tl,
            "avg_cos_vt": stats_vt["avg_cos"],
            "avg_cos_vl": stats_vl["avg_cos"],
            "avg_cos_tl": stats_tl["avg_cos"],
        }


def _pairwise_info_nce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    """Standard InfoNCE loss between two modality batches.

    Args:
        z_a: (B, D) L2-normalized modality A embeddings.
        z_b: (B, D) L2-normalized modality B embeddings.
        temperature: Scalar temperature (higher = softer distribution).

    Returns:
        (loss, stats) where stats has 'avg_cos' (mean cosine similarity, in [-1, 1]).
    """
    B = z_a.shape[0]

    # Logits: (B, B) similarity matrix — these are proper cosine similarities
    # because z_a and z_b are already L2-normalized
    logits = (z_a @ z_b.T) / temperature
    labels = torch.arange(B, device=z_a.device)

    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)

    # True cosine similarity (same as dot product since both are normalized)
    avg_cos = (z_a * z_b).sum(dim=-1).mean()

    return (loss_a2b + loss_b2a) / 2.0, {"avg_cos": avg_cos}


def compute_contrastive_loss(
    z_v: torch.Tensor,
    z_t: torch.Tensor,
    z_text: torch.Tensor,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Standalone loss function wrapper.

    Normalizes embeddings and computes 3-way InfoNCE.
    """
    criterion = ContrastiveLoss3Way(temperature=temperature)
    return criterion(z_v, z_t, z_text)