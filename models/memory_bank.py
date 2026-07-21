#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perceptual-Cognitive-State Memory Bank for ALIGN V4.

Architecture:
  - Triplet bank entries: (z_v_pooled, intent_emb, z_s) stored together
  - Tri-stream retrieval: perceptual stream queries past z_v_pooled,
    cognitive stream queries past intent_emb, state stream queries past z_s
  - Gate fusion: learned gate blends retrieved context with current
  - Token-merge consolidation: when bank is full, merge most similar
    adjacent pair (using perceptual similarity) and average all three fields

Reference: MemoryVLA (Shi et al., ICLR 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ================================================================
# Memory Retrieval (cross-attention over bank entries)
# ================================================================

class MemoryRetrieval(nn.Module):
    """Cross-attention from current token → memory bank.

    Args:
        dim: feature dimension of the stream (perceptual or cognitive)
        num_heads: attention heads (default 4)
        dropout: attention dropout
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.retrieval_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Two-layer FFN (like a Transformer decoder layer without self-attn)
        # Input is concat of query and attn_out -> 2*dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, bank_kv: torch.Tensor,
                bank_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retrieve from memory bank.

        Args:
            query: (B, dim) — current step representation
            bank_kv: (B, L, dim) — bank entries (keys and values)
            bank_mask: (B, L) bool mask, True = valid entry, False = padding
        Returns:
            (B, dim) — retrieved context with residual
        """
        B = query.shape[0]
        q = query.unsqueeze(1)  # (B, 1, dim)

        # Handle empty bank: return query directly
        if bank_kv.shape[1] == 0:
            return query

        # Attention mask: True = attend, False = don't attend
        # nn.MultiheadAttention expects key_padding_mask where True = masked
        if bank_mask is not None:
            # bank_mask: True = valid → invert for key_padding_mask
            attn_mask = ~bank_mask  # (B, L), True = padding (masked out)
        else:
            attn_mask = None

        # Use math SDPA backend for stability (same fix as align_model.py)
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                attn_out, _ = self.retrieval_attn(
                    q, bank_kv, bank_kv,
                    key_padding_mask=attn_mask,
                )
        except ImportError:
            attn_out, _ = self.retrieval_attn(
                q, bank_kv, bank_kv,
                key_padding_mask=attn_mask,
            )

        # FFN with residual
        ffn_in = torch.cat([q, attn_out], dim=-1)  # (B, 1, 2*dim)
        ffn_out = self.ffn(ffn_in)  # (B, 1, dim)
        out = self.out_norm(ffn_out.squeeze(1) + query)  # (B, dim)
        return out


# ================================================================
# Memory Gate Fusion
# ================================================================

class MemoryGateFusion(nn.Module):
    """Learned gate: blends current representation with retrieved context.

    g = sigmoid(MLP(concat[current, retrieved]))
    fused = g * retrieved + (1 - g) * current

    When bank is empty, g → 0 so fused ≈ current.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Initialize gate bias so g starts near 0 (trust current, not retrieved)
        # This is important: early in training, the bank is unreliable
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, current: torch.Tensor,
                retrieved: torch.Tensor) -> torch.Tensor:
        """Gate-fuse current and retrieved.

        Args:
            current: (B, dim) — current step representation
            retrieved: (B, dim) — retrieved from bank
        Returns:
            (B, dim) — fused representation
        """
        g = torch.sigmoid(
            self.gate_mlp(torch.cat([current, retrieved], dim=-1))
        )
        return g * retrieved + (1 - g) * current


# ================================================================
# Perceptual-Cognitive Memory Module (top-level)
# ================================================================

class PerceptualCognitiveMemoryModule(nn.Module):
    """Dual-stream episodic memory bank.

    Stores paired entries (z_v_pooled, intent_emb) and provides
    retrieval + gate fusion for both streams.

    Args:
        perceptual_dim: dim of z_v_pooled (V * vision_dim)
        cognitive_dim: dim of cognitive features (intent_emb * num_intent_tokens)
        bank_len: max paired entries (L, default 16)
        num_heads: attention heads for retrieval (default 4)
    """
    def __init__(self,  perceptual_dim: int,
                        cognitive_dim: int,
                        state_dim: int, 
                        bank_len: int = 16, 
                        num_heads: int = 4):
        super().__init__()
        self.perceptual_dim = perceptual_dim
        self.cognitive_dim = cognitive_dim
        self.state_dim = state_dim
        self.bank_len = bank_len

        # Retrieval modules (one per stream)
        self.perceptual_retrieval   = MemoryRetrieval(perceptual_dim, num_heads)
        self.cognitive_retrieval    = MemoryRetrieval(cognitive_dim, num_heads)
        self.state_retrieval        = MemoryRetrieval(state_dim, num_heads)

        # Gate fusion modules (one per stream)
        self.perceptual_gate        = MemoryGateFusion(perceptual_dim)
        self.cognitive_gate         = MemoryGateFusion(cognitive_dim)
        self.state_gate             = MemoryGateFusion(state_dim)

        # Sinusoidal timestep positional encoding (shared, not per-stream)
        # Max timestep: we support up to 1024 steps
        max_dim = max(perceptual_dim, cognitive_dim, state_dim)
        self.register_buffer(
            "_timestep_pe",
            self._make_timestep_pe(1024, max_dim),
            persistent=False,
        )

        # Stateful bank buffers (not nn.Parameter — reset per segment)
        self.perceptual_bank: Optional[torch.Tensor] = None  # (B, L, perceptual_dim)
        self.cognitive_bank: Optional[torch.Tensor] = None   # (B, L, cognitive_dim)
        self.state_bank: Optional[torch.Tensor] = None       # (B, L, state_dim)
        self._bank_mask: Optional[torch.Tensor] = None        # (B, L) bool
        self._count: int = 0

    @staticmethod
    def _make_timestep_pe(max_len: int, dim: int) -> torch.Tensor:
        """Sinusoidal positional encoding for timesteps."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, dim)

    def _add_timestep_pe(self, bank: torch.Tensor, count: int) -> torch.Tensor:
        """Add timestep PE to bank entries (returns a copy, does not modify in-place).

        Args:
            bank: (B, L, dim)
            count: number of valid entries (0..L)
        Returns:
            (B, L, dim) with PE added to first `count` entries
        """
        B, L, D = bank.shape
        if count == 0:
            return bank
        # PE for positions 0..count-1
        pe = self._timestep_pe[:count, :D].unsqueeze(0).expand(B, -1, -1)  # (B, count, D)
        bank_with_pe = bank.clone()
        bank_with_pe[:, :count] = bank[:, :count] + pe
        return bank_with_pe

    def reset(self, batch_size: int, device: torch.device):
        """Clear bank for a new segment.

        Args:
            batch_size: number of parallel segments
            device: torch device
        """
        self.perceptual_bank = torch.zeros(
            batch_size, 0, self.perceptual_dim, device=device,
        )
        self.cognitive_bank = torch.zeros(
            batch_size, 0, self.cognitive_dim, device=device,
        )
        self.state_bank = torch.zeros(
            batch_size, 0, self.state_dim, device=device,
        )
        self._bank_mask = None
        self._count = 0

    def store_perceptual_only(self, z_v_pooled: torch.Tensor,
                                    z_s: torch.Tensor):
        """Store only the perceptual and state fields (warmup phase, no intent_emb yet).

        Args:
            z_v_pooled: (B, perceptual_dim)
            z_s:        (B, state_dim) or None
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device
        p_new = z_v_pooled.unsqueeze(1)  # (B, 1, perceptual_dim)
        # Dummy cognitive entry (zeros) — will be overwritten in active phase
        c_new = torch.zeros(B, 1, self.cognitive_dim, device=device)
        # State entry 
        s_new = z_s.unsqueeze(1)  # (B, 1, state_dim)


        if self.perceptual_bank is None or self.perceptual_bank.shape[1] == 0:
            self.perceptual_bank = p_new
            self.cognitive_bank = c_new
            self.state_bank = s_new
            self._count = 1
            self._bank_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            return

        p_combined = torch.cat([self.perceptual_bank, p_new], dim=1)
        c_combined = torch.cat([self.cognitive_bank, c_new], dim=1)
        s_combined = torch.cat([self.state_bank, s_new], dim=1)
        new_count = self._count + 1

        if new_count > self.bank_len:
            p_combined, c_combined, s_combined, new_count = self._token_merge(
                p_combined, c_combined, s_combined, new_count,
            )

        self.perceptual_bank = p_combined
        self.cognitive_bank = c_combined
        self.state_bank = s_combined
        self._count = new_count
        self._bank_mask = torch.ones(B, new_count, dtype=torch.bool, device=device)

    def forward(self, z_v_pooled: torch.Tensor,
                      z_s: torch.Tensor, 
                      intent_emb: torch.Tensor,
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve, fuse, and store.

        Args:
            z_v_pooled: (B, perceptual_dim)     — current pooled vision
            intent_emb: (B, N, cognitive_dim)   — current intent tokens
            z_s:        (B, state_dim)          — current robot state
        Returns:
            z_v_pooled_fused: (B, perceptual_dim)
            intent_emb_fused: (B, N, cognitive_dim)
            z_s_fused:        (B, state_dim)
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device

        # --- 1. Retrieve from banks ---
        # Add timestep PE to bank keys/values
        p_bank_pe = self._add_timestep_pe(
            self.perceptual_bank, self._count
        ) if self.perceptual_bank is not None and self._count > 0 else (
            self.perceptual_bank if self.perceptual_bank is not None
            else torch.zeros(B, 0, self.perceptual_dim, device=device)
        )
        c_bank_pe = self._add_timestep_pe(
            self.cognitive_bank, self._count
        ) if self.cognitive_bank is not None and self._count > 0 else (
            self.cognitive_bank if self.cognitive_bank is not None
            else torch.zeros(B, 0, self.cognitive_dim, device=device)
        )
        s_bank_pe = self._add_timestep_pe(
            self.state_bank, self._count
        ) if self.state_bank is not None and self._count > 0 else (
            self.state_bank if self.state_bank is not None
            else torch.zeros(B, 0, self.state_dim, device=device)
        )

        # Perceptual retrieval: z_v_pooled queries perceptual bank
        p_retrieved = self.perceptual_retrieval(
            z_v_pooled, p_bank_pe, bank_mask=self._bank_mask,
        )  # (B, perceptual_dim)

        # Cognitive retrieval: intent_emb queries cognitive bank
        # Reshape N intent tokens to a single vector for query
        intent_query = intent_emb.reshape(intent_emb.shape[0], -1)   # [B, cognitive_dim]
        assert intent_query.shape[1] == self.cognitive_dim, f"Intent query should have a single token with {self.cognitive_dim} dimension after reshaping."
        intent_query = intent_query.unsqueeze(1)                   # [B, 1, cognitive_dim]       
        c_retrieved = self.cognitive_retrieval(
            intent_query, c_bank_pe, bank_mask=self._bank_mask,
        )  # (B, cognitive_dim)

        # State retrieval: z_s queries state bank
        s_retrieved = self.state_retrieval(
            z_s, s_bank_pe, bank_mask=self._bank_mask,
        )  # (B, state_dim)

        # --- 2. Gate fusion ---
        z_v_pooled_fused    = self.perceptual_gate(z_v_pooled, p_retrieved)
        intent_emb_fused    = self.cognitive_gate(intent_query, c_retrieved)
        z_s_fused           = self.state_gate(z_s, s_retrieved)
        # print(f"Memory bank: z_v_pooled_fused shape: {z_v_pooled_fused.shape}, intent_emb_fused shape: {intent_emb_fused.shape}, z_s_fused shape: {z_s_fused.shape}")
        
        # --- 3. Store current triplet into bank ---
        self._store(z_v_pooled_fused, z_s_fused, intent_query)
        
        # Expand retrieved context back to N tokens
        intent_emb_fused = intent_emb_fused.reshape(B, intent_emb.shape[1], -1) # (B, N, intent_dim)
        assert intent_emb_fused.shape == intent_emb.shape, f"Intent embedding shape mismatch after fusion: expected {intent_emb.shape}, got {intent_emb_fused.shape}"
        
        return z_v_pooled_fused, z_s_fused, intent_emb_fused

    def _store(self, z_v_pooled: torch.Tensor,
                     z_s: torch.Tensor, 
                     intent_query: torch.Tensor):
        """Store triplet entry. Consolidate if bank is full.

        Args:
            z_v_pooled:     (B, perceptual_dim)
            intent_query:   (B, cognitive_dim)    — pooled intent for storage
            z_s:            (B, state_dim)        — robot state
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device

        # Append new entries
        p_new = z_v_pooled.unsqueeze(1)     # (B, 1, perceptual_dim)
        c_new = intent_query.unsqueeze(1)   # (B, 1, cognitive_dim)
        s_new = z_s.unsqueeze(1)            # (B, 1, state_dim)


        if self.perceptual_bank is None or self.perceptual_bank.shape[1] == 0:
            # First entry
            self.perceptual_bank = p_new
            self.cognitive_bank = c_new
            self.state_bank = s_new
            self._count = 1
            self._bank_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            return

        # Concatenate new entry
        p_combined = torch.cat([self.perceptual_bank, p_new], dim=1)  # (B, L+1, D_p)
        c_combined = torch.cat([self.cognitive_bank, c_new], dim=1)  # (B, L+1, D_c)
        s_combined = torch.cat([self.state_bank, s_new], dim=1)  # (B, L+1, D_s)
        new_count = self._count + 1

        # Consolidate if over capacity
        if new_count > self.bank_len:
            p_combined, c_combined, s_combined, new_count = self._token_merge(
                p_combined, c_combined, s_combined, new_count,
            )

        self.perceptual_bank = p_combined
        self.cognitive_bank = c_combined
        self.state_bank = s_combined
        self._count = new_count
        self._bank_mask = torch.ones(B, new_count, dtype=torch.bool, device=device)

    def _token_merge(self,  p_bank: torch.Tensor, 
                            c_bank: torch.Tensor,
                            s_bank: torch.Tensor,
                            count: int
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Token-merge consolidation: merge most similar adjacent pair.

        Uses perceptual vectors for similarity metric. All three fields
        (perceptual, cognitive, state) are averaged together as a unit.

        Args:
            p_bank: (B, L+1, perceptual_dim)
            c_bank: (B, L+1, cognitive_dim)
            s_bank: (B, L+1, state_dim)
            count: number of valid entries (L+1)
        Returns:
            p_merged: (B, L, perceptual_dim)
            c_merged: (B, L, cognitive_dim)
            s_merged: (B, L, state_dim)
            new_count: L
        """
        B = p_bank.shape[0]
        # Only consider valid entries (first `count` entries)
        p_valid = p_bank[:, :count]  # (B, L+1, D_p)

        # Normalize for cosine similarity
        p_norm = F.normalize(p_valid, dim=-1)  # (B, L+1, D_p)

        # Cosine similarity between adjacent pairs
        sim = (p_norm[:, :-1] * p_norm[:, 1:]).sum(dim=-1)  # (B, L)
        # Average across batch to find globally most similar pair
        sim_mean = sim.mean(dim=0)  # (L,)
        merge_idx = sim_mean.argmax().item()  # merge the MOST similar (highest cos)

        # Merge pair (merge_idx, merge_idx+1) by averaging all three fields
        p_merged_vec = (p_bank[:, merge_idx] + p_bank[:, merge_idx + 1]) / 2.0
        c_merged_vec = (c_bank[:, merge_idx] + c_bank[:, merge_idx + 1]) / 2.0
        s_merged_vec = (s_bank[:, merge_idx] + s_bank[:, merge_idx + 1]) / 2.0

        # Reconstruct: keep entries before merge_idx, merged entry, entries after
        p_out = torch.cat([
            p_bank[:, :merge_idx],
            p_merged_vec.unsqueeze(1),
            p_bank[:, merge_idx + 2:],
        ], dim=1)  # (B, L, D_p)
        c_out = torch.cat([
            c_bank[:, :merge_idx],
            c_merged_vec.unsqueeze(1),
            c_bank[:, merge_idx + 2:],
        ], dim=1)  # (B, L, D_c)
        s_out = torch.cat([
            s_bank[:, :merge_idx],
            s_merged_vec.unsqueeze(1),
            s_bank[:, merge_idx + 2:],
        ], dim=1)  # (B, L, D_s)

        return p_out, c_out, s_out, self.bank_len
