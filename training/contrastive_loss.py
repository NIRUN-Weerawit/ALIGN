#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-way contrastive loss for ALIGN pretraining.

InfoNCE loss for three modality pairs:
    L = mean(L_vt + L_vl + L_tl) / 3

where each L_xy = (cross_entropy(logits_xy) + cross_entropy(logits_yx)) / 2

Positive pairs come from same episode (within temporal window for v↔t).
Negative pairs come from different episodes in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss3Way(nn.Module):
    """3-way contrastive loss: InfoNCE on (vision, trajectory, text) triples."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        z_text: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute 3-way contrastive loss.

        Args:
            z_v: (B, D) L2-normalized vision embeddings.
            z_t: (B, D) L2-normalized trajectory embeddings.
            z_text: (B, D) L2-normalized text embeddings.

        Returns:
            dict with 'loss', 'loss_vt', 'loss_vl', 'loss_tl', 'avg_cos_vt',
            'avg_cos_vl', 'avg_cos_tl'.
        """
        tau = self.temperature.abs()  # ensure positive

        loss_vt, stats_vt = _pairwise_info_nce(z_v, z_t, tau)
        loss_vl, stats_vl = _pairwise_info_nce(z_v, z_text, tau)
        loss_tl, stats_tl = _pairwise_info_nce(z_t, z_text, tau)

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
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Standard InfoNCE loss between two modality batches.

    Args:
        z_a: (B, D) first modality embeddings.
        z_b: (B, D) second modality embeddings.
        temperature: Scalar temperature parameter.

    Returns:
        (loss, stats) where stats has 'avg_cos' (mean cosine similarity).
    """
    B = z_a.shape[0]

    # Logits: (B, B) similarity matrix
    logits = (z_a @ z_b.T) / temperature
    labels = torch.arange(B, device=z_a.device)

    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)

    # Average cosine similarity along the diagonal (positive pairs)
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
    z_v = F.normalize(z_v, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    z_text = F.normalize(z_text, dim=-1)

    criterion = ContrastiveLoss3Way(temperature=temperature)
    return criterion(z_v, z_t, z_text)
