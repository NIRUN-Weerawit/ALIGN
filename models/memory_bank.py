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
        self.context_only = False
        self.value_preserving = False

    def forward(self, query: torch.Tensor, bank_kv: torch.Tensor,
                bank_mask: Optional[torch.Tensor] = None,
                bank_values: Optional[torch.Tensor] = None,
                attention_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retrieve from memory bank.

        Args:
            query: (B, dim) — current step representation
            bank_kv: (B, L, dim) — bank entries (keys and values)
            bank_mask: (B, L) bool mask, True = valid entry, False = padding
        Returns:
            (B, dim) — retrieved context with residual
        """
        B = query.shape[0]
        if attention_bias is not None and not self.value_preserving:
            raise ValueError('Attention bias requires raw-value retrieval')
        q = query.unsqueeze(1) if query.ndim == 2 else query

        # Handle empty bank: return query directly
        if bank_kv.shape[1] == 0:
            return query

        # Handle bank with all masked entries (no valid entries to attend to).
        # This happens when the bank is empty (no entries stored yet) and we still
        # want to query it. Without this check, the attention softmax would be
        # all-masked → NaN, which propagates through the FFN.
        if bank_mask is not None and not bank_mask.any():
            return query

        if bank_mask is not None:
            nonempty = bank_mask.any(dim=1)
            if not nonempty.all():
                # An empty sample must not enter an all-masked softmax.
                out = query.clone()
                out[nonempty] = self.forward(
                    query[nonempty], bank_kv[nonempty], bank_mask[nonempty],
                    None if bank_values is None else bank_values[nonempty],
                    None if attention_bias is None else attention_bias[nonempty],
                )
                return out

        # Attention mask: True = attend, False = don't attend
        # nn.MultiheadAttention expects key_padding_mask where True = masked
        if bank_mask is not None:
            # bank_mask: True = valid → invert for key_padding_mask
            attn_mask = ~bank_mask  # (B, L), True = padding (masked out)
        else:
            attn_mask = None

        if self.value_preserving:
            # Learn selection in Q/K space, but return raw encoded values in the
            # representation already understood by a warm-started action head.
            dim=self.retrieval_attn.embed_dim
            heads=self.retrieval_attn.num_heads
            weight,bias=self.retrieval_attn.in_proj_weight,self.retrieval_attn.in_proj_bias
            q_proj=F.linear(q,weight[:dim],None if bias is None else bias[:dim])
            k_proj=F.linear(bank_kv,weight[dim:2*dim],None if bias is None else bias[dim:2*dim])
            q_proj=q_proj.reshape(B,q.shape[1],heads,dim//heads).transpose(1,2)
            k_proj=k_proj.reshape(B,bank_kv.shape[1],heads,dim//heads).transpose(1,2)
            with torch.autocast(device_type=query.device.type,enabled=False):
                scores=q_proj.float()@k_proj.float().transpose(-1,-2)/(dim//heads)**.5
                if attention_bias is not None:scores=scores+attention_bias.float()[:,None,None,:]
                if attn_mask is not None:scores=scores.masked_fill(attn_mask[:,None,None],float('-inf'))
                weights=scores.softmax(-1)
                weights=F.dropout(weights,p=self.retrieval_attn.dropout,training=self.training)
                values=bank_kv if bank_values is None else bank_values
                out=weights.mean(1)@values.float()
            out=out.to(query.dtype)
            return out.squeeze(1) if query.ndim==2 else out

        # Use math SDPA backend for stability (same fix as align_model.py)
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                attn_out, _ = self.retrieval_attn(
                    q, bank_kv, bank_kv if bank_values is None else bank_values,
                    key_padding_mask=attn_mask, need_weights=False,
                )
        except ImportError:
            attn_out, _ = self.retrieval_attn(
                q, bank_kv, bank_kv if bank_values is None else bank_values,
                key_padding_mask=attn_mask, need_weights=False,
            )

        # FFN with residual
        # Optional context branch keeps the current query only in attention
        # selection. Current features are already preserved by the fusion gate.
        residual = attn_out if self.context_only else q
        ffn_in = torch.cat([residual, attn_out], dim=-1)  # (B, 1, 2*dim)
        ffn_out = self.ffn(ffn_in)  # (B, 1, dim)
        out = self.out_norm(ffn_out + residual)
        return out.squeeze(1) if query.ndim == 2 else out


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
        self.cognitive_dim = max(cognitive_dim, 1)  # minimum 1 to avoid empty MultiheadAttention
        self.state_dim = state_dim
        self.bank_len = bank_len
        self._has_cognitive = cognitive_dim > 0

        # Retrieval modules (one per stream). Cognitive is only created if there
        # are intent tokens (cognitive_dim > 0), otherwise nn.MultiheadAttention
        # would fail with embed_dim=0.
        self.perceptual_retrieval   = MemoryRetrieval(perceptual_dim, num_heads)
        if cognitive_dim > 0:
            self.cognitive_retrieval    = MemoryRetrieval(cognitive_dim, num_heads)
        else:
            self.cognitive_retrieval    = None
        self.state_retrieval        = MemoryRetrieval(state_dim, num_heads)

        # Gate fusion modules (one per stream)
        self.perceptual_gate        = MemoryGateFusion(perceptual_dim)
        if cognitive_dim > 0:
            self.cognitive_gate         = MemoryGateFusion(cognitive_dim)
        else:
            self.cognitive_gate         = None
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
                                    z_s: torch.Tensor):
        """Store only the perceptual and state fields (warmup phase, no intent_emb yet).

        Always stores all samples in the batch. The past observations are
        always valid (we observed them); whether the target is available
        is a separate concern handled by the loss.

        Args:
            z_v_pooled: (B, perceptual_dim)
            z_s:        (B, state_dim)
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device
        # Dummy cognitive entry (zeros) — will be overwritten in active phase
        c_new = torch.zeros(B, self.cognitive_dim, device=device)

        for b in range(B):
            if self._count[b] >= self.bank_len:
                # Consolidate first to free a slot
                (self.perceptual_bank[b:b+1], self.cognitive_bank[b:b+1],
                 self.state_bank[b:b+1], new_count) = self._token_merge(
                    self.perceptual_bank[b:b+1], self.cognitive_bank[b:b+1],
                    self.state_bank[b:b+1], int(self._count[b].item()),
                )
                self._count[b] = new_count
            idx = int(self._count[b].item())
            self.perceptual_bank[b, idx] = z_v_pooled[b]
            self.cognitive_bank[b, idx] = c_new[b]
            self.state_bank[b, idx] = z_s[b]
            self._count[b] += 1

    def forward(self, z_v_pooled: torch.Tensor,
                      z_s: torch.Tensor, 
                      intent_emb: Optional[torch.Tensor] = None,
                      observed_mask: Optional[torch.Tensor] = None,
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

        # Cognitive retrieval: intent_emb queries cognitive bank (if enabled)
        if self._has_cognitive and intent_emb is not None:
            intent_query = intent_emb.reshape(intent_emb.shape[0], -1)   # (B, cognitive_dim)
            c_retrieved = self.cognitive_retrieval(
                intent_query, c_bank_pe, bank_mask=bank_mask,
            )  # (B, cognitive_dim)
            intent_emb_fused = self.cognitive_gate(intent_query, c_retrieved)
            # Expand retrieved context back to N tokens
            intent_emb_fused = intent_emb_fused.reshape(B, intent_emb.shape[1], -1)  # (B, N, intent_dim)
        else:
            intent_emb_fused = intent_emb  # pass through unchanged
            c_retrieved = None

        # State retrieval: z_s queries state bank
        s_retrieved = self.state_retrieval(
            z_s, s_bank_pe, bank_mask=bank_mask,
        )  # (B, state_dim)

        # --- 2. Gate fusion ---
        z_v_pooled_fused    = self.perceptual_gate(z_v_pooled, p_retrieved)
        z_s_fused           = self.state_gate(z_s, s_retrieved)
        
        # --- 3. Store current triplet into bank (with consolidation when full) ---
        if self._has_cognitive and intent_emb is not None:
            intent_query = intent_emb.reshape(intent_emb.shape[0], -1)
            self._store(z_v_pooled_fused, z_s_fused, intent_query, observed_mask)
        else:
            self._store(z_v_pooled_fused, z_s_fused, None, observed_mask)
        
        return z_v_pooled_fused, z_s_fused, intent_emb_fused

    def _store(self, z_v_pooled: torch.Tensor,
                     z_s: torch.Tensor, 
                     intent_query: Optional[torch.Tensor] = None,
                     observed_mask: Optional[torch.Tensor] = None):
        """Store triplet entry with consolidation.

        When the bank is full, run _token_merge first to make room by
        merging the most similar pair. The bank stays fixed at bank_len
        size; counts never exceed bank_len. observed_mask prevents storing
        replicated padding for samples whose observed segment has ended.

        Args:
            z_v_pooled:     (B, perceptual_dim)
            intent_query:   (B, cognitive_dim) or None — pooled intent for storage
            z_s:            (B, state_dim)        — robot state
        """
        B = z_v_pooled.shape[0]
        device = z_v_pooled.device

        for b in range(B):
            if observed_mask is not None and not bool(observed_mask[b]):
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
            if intent_query is not None:
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


class EpisodicMemoryModule(PerceptualCognitiveMemoryModule):
    """Causal raw/detached memory with real timestamps and key-only age encoding.

    Optional patch retrieval keeps spatial tokens instead of one flattened token.
    Bank buffers are episode-local state, never checkpoint parameters.
    """
    def __init__(self, perceptual_dim, cognitive_dim, state_dim, bank_len=16,
                 num_heads=2, detach_writes=True, write_fused=False, patch_dim=None, context_only=False, patch_temporal=False, value_preserving=False, mask_missing_fields=False, perceptual_recency_scale=0.):
        if not __import__('math').isfinite(perceptual_recency_scale) or perceptual_recency_scale<0 or (perceptual_recency_scale>0 and not value_preserving):
            raise ValueError('Perceptual recency scale must be finite, nonnegative, and requires raw-value retrieval')
        if mask_missing_fields and write_fused:
            raise ValueError('Missing-field masks require raw observation writes')
        if value_preserving and context_only:
            raise ValueError("Value-preserving retrieval replaces the FFN context branch; do not combine modes")
        if value_preserving and patch_dim is not None and not patch_temporal:
            raise ValueError("Value-preserving patch retrieval requires temporal spatial alignment")
        if bank_len < 2:
            raise ValueError('Episodic memory capacity must be at least two for consolidation')
        super().__init__(perceptual_dim,cognitive_dim,state_dim,bank_len,num_heads)
        self.detach_writes, self.write_fused = detach_writes,write_fused
        self.mask_missing_fields = mask_missing_fields
        self.perceptual_recency_scale = perceptual_recency_scale
        if patch_temporal and patch_dim is None:
            raise ValueError("Temporal patch retrieval requires patch-preserving memory")
        self.patch_dim = patch_dim
        self.patch_temporal = patch_temporal
        if patch_dim is not None:
            if perceptual_dim % patch_dim or patch_dim % num_heads:
                raise ValueError('Patch width must divide perceptual width and attention heads')
            self.perceptual_retrieval = MemoryRetrieval(patch_dim,num_heads)
            self.perceptual_gate = MemoryGateFusion(patch_dim)
        for retrieval in [self.perceptual_retrieval,self.state_retrieval,self.cognitive_retrieval]:
            if retrieval is not None:
                retrieval.context_only = context_only
                retrieval.value_preserving = value_preserving
                if value_preserving:
                    retrieval.ffn.requires_grad_(False)
                    retrieval.out_norm.requires_grad_(False)
                    retrieval.retrieval_attn.out_proj.requires_grad_(False)
        # Begin with balanced current/history fusion, rather than suppressing memory.
        for gate in [self.perceptual_gate,self.state_gate,self.cognitive_gate]:
            if gate is not None:
                nn.init.constant_(gate.gate_mlp[-1].bias,0.)
        self.timestamps = self._next_timestep = None
        self.perceptual_times = self.state_times = self.cognitive_times = None

    def reset(self,batch_size,device):
        super().reset(batch_size,device)
        if self.patch_dim is not None:
            self.perceptual_bank = self.perceptual_bank.reshape(batch_size,self.bank_len,-1,self.patch_dim)
        self.timestamps = torch.full((batch_size,self.bank_len),-1.,device=device)
        self._next_timestep = torch.zeros(batch_size,device=device)
        if self.mask_missing_fields:
            for name,bank in [('perceptual',self.perceptual_bank),('state',self.state_bank),('cognitive',self.cognitive_bank)]:
                setattr(self,name+'_times',torch.full(bank.shape[:-1],-1.,device=device))

    @staticmethod
    def age_encoding(age,dim):
        frequency = torch.exp(torch.arange(0,dim,2,device=age.device).float()*(-__import__('math').log(10000.)/dim))
        phase = age.float().unsqueeze(-1)*frequency
        out = age.new_zeros(*age.shape,dim,dtype=torch.float32)
        out[...,0::2] = phase.sin()
        out[...,1::2] = phase[..., :dim//2].cos()
        return out

    def _retrieve(self,module,query,bank,mask,age):
        values = bank.clone()
        def recency_bias(ages):
            return {'attention_bias':-self.perceptual_recency_scale*ages} if self.perceptual_recency_scale>0 and module is self.perceptual_retrieval else {}
        if bank.ndim==4:
            B,L,N,D = bank.shape
            age_features = self.age_encoding(age,D).to(values.dtype)
            if age.ndim==2:age_features=age_features.unsqueeze(2)
            patch_mask=mask if mask.ndim==3 else mask[:,:,None].expand(B,L,N)
            if self.patch_temporal:
                # Each fixed camera/grid slot searches its own temporal history.
                # This preserves spatial identity without mixing L*N tokens in
                # a very narrow attention space; work scales as N*L, not N*N*L.
                temporal_values = values.permute(0,2,1,3).reshape(B*N,L,D)
                temporal_keys = (values + age_features).permute(0,2,1,3).reshape(B*N,L,D)
                temporal_mask = patch_mask.permute(0,2,1).reshape(B*N,L)
                temporal_age=(age if age.ndim==3 else age[:,:,None].expand(B,L,N)).permute(0,2,1).reshape(B*N,L)
                result = module(query.reshape(B*N,D),temporal_keys,temporal_mask,temporal_values,**recency_bias(temporal_age)).reshape(B,N,D)
                return torch.where(patch_mask.any(1)[:,:,None],result,query)
            keys = values + age_features
            # Spatial position remains separate from the recorded observation age.
            spatial = self.age_encoding(torch.arange(N,device=bank.device),D).to(values.dtype)
            keys = keys + spatial[None,None]
            q = query + spatial[None]
            result = module(q,keys.reshape(B,L*N,D),patch_mask.reshape(B,L*N),values.reshape(B,L*N,D))
            return torch.where(patch_mask.flatten(1).any(1)[:,None,None],result,query)
        keys = values + self.age_encoding(age,bank.shape[-1]).to(values.dtype)
        return module(query,keys,mask,values,**recency_bias(age))

    def _write(self,p,s,c,timestamp,observed_mask):
        B = p.shape[0]
        mask = torch.ones(B,device=p.device,dtype=torch.bool) if observed_mask is None else observed_mask
        p = p.reshape(B,*self.perceptual_bank.shape[2:])
        c = p.new_zeros(B,self.cognitive_dim) if c is None else c.reshape(B,-1)
        if self.detach_writes:
            p,s,c = p.detach(),s.detach(),c.detach()
        # Each episode chooses its own adjacent merge, without device-to-host
        # synchronization for every row and observation.
        full = (self._count == self.bank_len) & mask
        normalized = F.normalize(self.perceptual_bank.flatten(2),dim=-1)
        pair = (normalized[:,:-1]*normalized[:,1:]).sum(-1).argmax(1)
        positions = torch.arange(self.bank_len,device=p.device)[None].expand(B,-1)
        shifted_indices = (positions + (positions > pair[:,None])).clamp_max(self.bank_len-1)
        count = self._count - full.long()

        def merge_and_write(bank,value,field_times=None):
            extra = (1,) * (bank.ndim-2)
            def index(column):
                return column.reshape(B,column.shape[1],*extra).expand(B,column.shape[1],*bank.shape[2:])
            shifted = bank.gather(1,index(shifted_indices))
            average = (bank.gather(1,index(pair[:,None])) +
                       bank.gather(1,index(pair[:,None]+1))) / 2
            new_times=None
            if field_times is not None:
                time_extra=(1,)*(field_times.ndim-2)
                def time_index(pos):return pos.reshape(B,pos.shape[1],*time_extra).expand(B,pos.shape[1],*field_times.shape[2:])
                first=field_times.gather(1,time_index(pair[:,None]))
                second=field_times.gather(1,time_index(pair[:,None]+1))
                v1,v2=first>=0,second>=0
                denominator=(v1.float()+v2.float()).clamp_min(1)
                average=(bank.gather(1,index(pair[:,None]))*v1[...,None]+
                         bank.gather(1,index(pair[:,None]+1))*v2[...,None])/denominator[...,None].to(bank.dtype)
                merged_time=torch.where(v1|v2,(first.clamp_min(0)*v1+second.clamp_min(0)*v2)/denominator,-1.)
                shifted_time=field_times.gather(1,time_index(shifted_indices))
                retained_time=torch.where((positions==pair[:,None]).reshape(B,self.bank_len,*time_extra),merged_time,shifted_time)
                retained_time=torch.where(full.reshape(B,1,*time_extra),retained_time,field_times)
                value_valid=value.ne(0).any(-1)
                value_time=timestamp.reshape(B,*time_extra).expand_as(value_valid)
                value_time=torch.where(value_valid,value_time,-1.)
                new_times=retained_time.scatter(1,time_index(count.clamp_max(self.bank_len-1)[:,None]),value_time.unsqueeze(1))
                new_times=torch.where(mask.reshape(B,1,*time_extra),new_times,retained_time)
            merged = torch.where((positions == pair[:,None]).reshape(B,self.bank_len,*extra),average,shifted)
            retained = torch.where(full.reshape(B,1,*extra),merged,bank)
            added = retained.scatter(1,index(count.clamp_max(self.bank_len-1)[:,None]),value.to(dtype=bank.dtype).unsqueeze(1))
            output=torch.where(mask.reshape(B,1,*extra),added,retained)
            return output if field_times is None else (output,new_times)

        if self.mask_missing_fields:
            for name,value in [('perceptual',p),('state',s),('cognitive',c)]:
                bank,times=merge_and_write(getattr(self,name+'_bank'),value,getattr(self,name+'_times'))
                setattr(self,name+'_bank',bank);setattr(self,name+'_times',times)
        else:
            self.perceptual_bank = merge_and_write(self.perceptual_bank,p)
            self.state_bank = merge_and_write(self.state_bank,s)
            self.cognitive_bank = merge_and_write(self.cognitive_bank,c)
        self.timestamps = merge_and_write(self.timestamps,timestamp)
        self._count = torch.where(mask,count+1,self._count)
        self._next_timestep = torch.where(mask,timestamp+1,self._next_timestep)

    def _resolve_timestamp(self,timestamp,observed_mask):
        if timestamp is None:
            return self._next_timestep.clone()
        timestamp = timestamp.to(self._next_timestep.device,dtype=torch.float32)
        if timestamp.shape != self._next_timestep.shape:
            raise ValueError('Memory timestamps must have one value per episode row')
        observed = torch.ones_like(timestamp,dtype=torch.bool) if observed_mask is None else observed_mask.bool()
        invalid = ~torch.isfinite(timestamp) | (timestamp < self._next_timestep)
        if bool((observed & invalid).any()):
            raise ValueError('Memory observations must have finite, strictly increasing timestamps; reset at episode boundaries')
        return timestamp

    def observe_only(self,p,s,c=None,observed_mask=None,timestamp=None):
        if self.write_fused:
            return self.forward(p,s,c,observed_mask,timestamp)
        timestamp = self._resolve_timestamp(timestamp,observed_mask)
        self._write(p,s,c,timestamp,observed_mask)

    def forward(self,p,s,c=None,observed_mask=None,timestamp=None):
        B = p.shape[0]
        timestamp = self._resolve_timestamp(timestamp,observed_mask)
        mask = torch.arange(self.bank_len,device=p.device)[None] < self._count[:,None]
        age = (timestamp[:,None]-self.timestamps).clamp_min(0)
        def stream_context(name):
            if not self.mask_missing_fields:return mask,age
            times=getattr(self,name+'_times')
            present=mask if times.ndim==2 else mask[:,:,None]
            time_now=timestamp.reshape(B,*([1]*(times.ndim-1)))
            return present & (times>=0),(time_now-times).clamp_min(0)
        pmask,page=stream_context('perceptual')
        smask,sage=stream_context('state')
        query = p.reshape(B,-1,self.patch_dim) if self.patch_dim is not None else p
        pr = self._retrieve(self.perceptual_retrieval,query,self.perceptual_bank,pmask,page)
        pf = self.perceptual_gate(query,pr)
        if self.mask_missing_fields:
            present=pmask.any(1).unsqueeze(-1)
            pf=torch.where(present,pf,query)
        pf=pf.reshape(B,-1)
        nonempty = mask.any(1)
        pf = torch.where(nonempty[:,None],pf,p)
        sr = self._retrieve(self.state_retrieval,s,self.state_bank,smask,sage)
        sf = self.state_gate(s,sr)
        sf = torch.where(smask.any(1)[:,None],sf,s)
        cf = c
        if self._has_cognitive and c is not None:
            cq = c.reshape(B,-1)
            cmask,cage=stream_context('cognitive')
            cr = self._retrieve(self.cognitive_retrieval,cq,self.cognitive_bank,cmask,cage)
            cf = self.cognitive_gate(cq,cr).reshape_as(c)
            cf = torch.where(cmask.any(1).reshape(B,*([1]*(c.ndim-1))),cf,c)
        self._write(pf if self.write_fused else p,sf if self.write_fused else s,
                    cf if self.write_fused else c,timestamp,observed_mask)
        return pf,sf,cf
