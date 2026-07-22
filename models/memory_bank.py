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
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
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
        self._count: Optional[torch.Tensor] = None

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

        For circular buffer, the count can exceed bank_len. We use modular PE
        so positions in the circular buffer get a consistent PE that depends
        only on their actual position in the circular buffer (0..bank_len-1).

        Args:
            bank: (B, L, dim)
            count: number of valid entries (0..N, can exceed L)
        Returns:
            (B, L, dim) with PE added to all bank entries
        """
        B, L, D = bank.shape
        if count == 0:
            return bank
        # Use bank positions 0..L-1 for PE (not the count, which can exceed L)
        # This ensures PE is consistent across overwrites
        pe = self._timestep_pe[:L, :D].unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        bank_with_pe = bank + pe
        return bank_with_pe

    def reset(self, batch_size: int, device: torch.device):
        """Clear bank for a new segment.

        Args:
            batch_size: number of parallel segments
            device: torch device
        """
        self.perceptual_bank = torch.zeros(
            batch_size, self.bank_len, self.perceptual_dim, device=device,
        )
        self.cognitive_bank = torch.zeros(
            batch_size, self.bank_len, self.cognitive_dim, device=device,
        )
        self.state_bank = torch.zeros(
            batch_size, self.bank_len, self.state_dim, device=device,
        )
        self._count = torch.zeros(batch_size, dtype=torch.long, device=device)

    def store_perceptual_only(self, z_v_pooled: torch.Tensor,
                                    z_s: torch.Tensor,
                                    valid_mask: Optional[torch.Tensor] = None):
        """Store only the perceptual and state fields (warmup phase, no intent_emb yet).

        Args:
            z_v_pooled: (B, perceptual_dim)
            z_s:        (B, state_dim)
            valid_mask: (B,) bool — which samples to store
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device
        # Dummy cognitive entry (zeros) — will be overwritten in active phase
        c_new = torch.zeros(B, self.cognitive_dim, device=device)

        for b in range(B):
            if valid_mask is not None and not valid_mask[b]:
                continue
            idx = self._count[b] % self.bank_len
            self.perceptual_bank[b, idx] = z_v_pooled[b]
            self.cognitive_bank[b, idx] = c_new[b]
            self.state_bank[b, idx] = z_s[b]
            self._count[b] += 1

    def forward(self, z_v_pooled: torch.Tensor,
                      z_s: torch.Tensor, 
                      intent_emb: torch.Tensor,
                      valid_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve, fuse, and store.

        Args:
            z_v_pooled: (B, perceptual_dim)     — current pooled vision
            intent_emb: (B, N, cognitive_dim)   — current intent tokens
            z_s:        (B, state_dim)          — current robot state
            valid_mask: (B,) bool — which samples are valid (not padding)
        Returns:
            z_v_pooled_fused: (B, perceptual_dim)
            intent_emb_fused: (B, N, cognitive_dim)
            z_s_fused:        (B, state_dim)
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device

        # Build mask: which bank slots are valid (have been written to)
        bank_mask = torch.arange(self.bank_len, device=device).unsqueeze(0).expand(B, -1) < self._count.unsqueeze(1)  # (B, L)
        max_count = self._count.max().item()

        # --- 1. Retrieve from banks ---
        # Add timestep PE to bank keys/values
        p_bank_pe = self._add_timestep_pe(self.perceptual_bank, max_count)
        c_bank_pe = self._add_timestep_pe(self.cognitive_bank, max_count)
        s_bank_pe = self._add_timestep_pe(self.state_bank, max_count)

        # Perceptual retrieval: z_v_pooled queries perceptual bank
        p_retrieved = self.perceptual_retrieval(
            z_v_pooled, p_bank_pe, bank_mask=bank_mask,
        )  # (B, perceptual_dim)

        # Cognitive retrieval: intent_emb queries cognitive bank
        intent_query = intent_emb.reshape(intent_emb.shape[0], -1)   # (B, cognitive_dim)
        c_retrieved = self.cognitive_retrieval(
            intent_query, c_bank_pe, bank_mask=bank_mask,
        )  # (B, cognitive_dim)

        # State retrieval: z_s queries state bank
        s_retrieved = self.state_retrieval(
            z_s, s_bank_pe, bank_mask=bank_mask,
        )  # (B, state_dim)

        # --- 2. Gate fusion ---
        z_v_pooled_fused    = self.perceptual_gate(z_v_pooled, p_retrieved)
        intent_emb_fused    = self.cognitive_gate(intent_query, c_retrieved)
        z_s_fused           = self.state_gate(z_s, s_retrieved)
        
        # --- 3. Store current triplet into bank (circular buffer) ---
        self._store(z_v_pooled_fused, z_s_fused, intent_query, valid_mask)
        
        # Expand retrieved context back to N tokens
        intent_emb_fused = intent_emb_fused.reshape(B, intent_emb.shape[1], -1)  # (B, N, intent_dim)
        assert intent_emb_fused.shape == intent_emb.shape, \
            f"Intent embedding shape mismatch after fusion: expected {intent_emb.shape}, got {intent_emb_fused.shape}"
        
        return z_v_pooled_fused, z_s_fused, intent_emb_fused

    def _store(self, z_v_pooled: torch.Tensor,
                     z_s: torch.Tensor, 
                     intent_query: torch.Tensor,
                     valid_mask: torch.Tensor):
        """Store triplet entry with consolidation.

        When the bank is full, run _token_merge first to make room by
        merging the most similar pair. The bank stays fixed at bank_len
        size; counts never exceed bank_len.

        Args:
            z_v_pooled:     (B, perceptual_dim)
            intent_query:   (B, cognitive_dim)    — pooled intent for storage
            z_s:            (B, state_dim)        — robot state
            valid_mask:     (B,) bool — which samples to store
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device

        for b in range(B):
            if valid_mask is not None and not valid_mask[b]:
                continue
            # If bank is full, consolidate first to free a slot
            if self._count[b] >= self.bank_len:
                # _token_merge reduces count by 1
                (self.perceptual_bank[b:b+1], self.cognitive_bank[b:b+1],
                 self.state_bank[b:b+1], new_count) = self._token_merge(
                    self.perceptual_bank[b:b+1], self.cognitive_bank[b:b+1],
                    self.state_bank[b:b+1], int(self._count[b].item()),
                )
                self._count[b] = new_count
            # Now add the new entry
            idx = int(self._count[b].item())
            self.perceptual_bank[b, idx] = z_v_pooled[b]
            self.cognitive_bank[b, idx] = intent_query[b]
            self.state_bank[b, idx] = z_s[b]
            self._count[b] += 1

    def _token_merge(self,  p_bank: torch.Tensor, 
                            c_bank: torch.Tensor,
                            s_bank: torch.Tensor,
                            count: int
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Token-merge consolidation: merge most similar adjacent pair.

        Uses perceptual vectors for similarity metric. All three fields
        (perceptual, cognitive, state) are averaged together as a unit.

        Args:
            p_bank: (B, L, perceptual_dim)
            c_bank: (B, L, cognitive_dim)
            s_bank: (B, L, state_dim)
            count: number of valid entries (0..L)
        Returns:
            p_merged: (B, L, perceptual_dim)
            c_merged: (B, L, cognitive_dim)
            s_merged: (B, L, state_dim)
            new_count: L (unchanged)
        """
        B, L, D_p = p_bank.shape
        # Only consider valid entries (first `count` entries)
        if count < 2:
            return p_bank, c_bank, s_bank, count
        p_valid = p_bank[:, :count]  # (B, count, D_p)

        # Normalize for cosine similarity
        p_norm = F.normalize(p_valid, dim=-1)  # (B, count, D_p)

        # Cosine similarity between adjacent pairs (only valid positions)
        # Pad with -inf to disable attention to invalid pairs
        sim = (p_norm[:, :-1] * p_norm[:, 1:]).sum(dim=-1)  # (B, count-1)
        # Mask out pairs that include invalid positions
        valid_pair = torch.arange(count-1, device=p_bank.device) < (count - 1)
        valid_pair = valid_pair.unsqueeze(0).expand(B, -1)  # (B, count-1)
        sim = sim.masked_fill(~valid_pair, -1.0)
        # Average across batch to find globally most similar pair
        sim_mean = sim.mean(dim=0)  # (count-1,)
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
        ], dim=1)  # (B, count-1, D_p)
        c_out = torch.cat([
            c_bank[:, :merge_idx],
            c_merged_vec.unsqueeze(1),
            c_bank[:, merge_idx + 2:],
        ], dim=1)  # (B, count-1, D_c)
        s_out = torch.cat([
            s_bank[:, :merge_idx],
            s_merged_vec.unsqueeze(1),
            s_bank[:, merge_idx + 2:],
        ], dim=1)  # (B, count-1, D_s)

        # Pad to bank_len so the buffer stays fixed size
        p_padded = F.pad(p_out, (0, 0, 0, L - p_out.shape[1]))
        c_padded = F.pad(c_out, (0, 0, 0, L - c_out.shape[1]))
        s_padded = F.pad(s_out, (0, 0, 0, L - s_out.shape[1]))
        return p_padded, c_padded, s_padded, count - 1

