#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLS/state Mamba intention encoder and patch features for the action head.

Mamba consumes flattened per-camera CLS tokens plus the encoded robot state.
Learned intent tokens read out its causal observation state. The separate patch
encoder compresses/modulates patches while preserving spatial positions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ================================================================
# SE Visual Compressor -- raw DINOv2 768 -> compact per-patch dim
# Preserves all VP token positions. Channels compressed, not positions.
# ================================================================

class SEVisualCompressor(nn.Module):
    """SE-bottleneck + channel compression per patch position.

    Squeeze-excitation produces adaptive per-channel weights BEFORE
    projection, so DINOv2 dims are reweighted based on scene content.

    Args:
        raw_dim:      DINOv2 output dim (768 for ViT-B)
        compressed_dim: target per-patch dim after compression (default 16)
        reduction:    SE bottleneck ratio, 768//reduction = hidden dims (default 8)

    Input:  (B, N_tokens, raw_dim=768)   -- e.g. VP_tokens = V x 256 patches
    Output: (B, N_tokens, compressed_dim=16)  -- SAME position count, channels reduced
    """

    def __init__(self, raw_dim: int = 768, compressed_dim: int = 16, reduction: int = 8):
        super().__init__()
        se_hidden = max(1, raw_dim // reduction)
        # SE squeeze-excitation: learn adaptive channel importance per-image before projection
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excitation = nn.Sequential(
            nn.Linear(raw_dim, se_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(se_hidden, raw_dim),
            nn.Sigmoid(),              # channel weights in [0, 1]
        )
        # Projection: actual dimension reduction -- 768 -> comp_dim
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, compressed_dim),
            nn.LayerNorm(compressed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress raw DINOv2 patches to compact per-patch dim.

        Args:
            x: (B, N_tokens, raw_dim) where N_tokens = V * P (e.g. 2*256=512)
               or (B, raw_dim) for v1 CLS mode.
        Returns:
            (B, N_tokens, compressed_dim) for v2 patch mode
            (B, compressed_dim) for v1 CLS mode
        """
        is_2d = x.ndim == 2
        if is_2d:
            x = x.unsqueeze(1)         # (B, 1, raw_dim) for v1 CLS compat

        # Squeeze: global mean across token positions -> per-channel stats
        # x: (B, N_tokens, raw_dim=768) -> transpose -> (B, raw_dim, N_tokens)
        # -> adaptive_avg_pool1d -> (B, raw_dim, 1) -> squeeze -> (B, raw_dim)
        squeezed = self.se_squeeze(x.transpose(1, 2))
        squeezed = squeezed.squeeze(-1)                # (B, raw_dim)

        # Excitation: adaptive per-channel weights based on scene content
        weights = self.se_excitation(squeezed)         # (B, raw_dim), sigmoided

        # Scale: suppress noisy DINOv2 channels BEFORE compression
        # x is (B, N_tokens, raw_dim), weights is (B, raw_dim)
        # Broadcast: weights.unsqueeze(1) -> (B, 1, raw_dim)
        x_se = x * weights.unsqueeze(1)                # (B, N_tokens, raw_dim)

        # Project down to compressed_dim
        y = self.projection(x_se)                     # (B, N_tokens, compressed_dim)

        if is_2d:
            return y.squeeze(1)                       # (B, compressed_dim)
        return y                                       # (B, N_tokens, compressed_dim)


# ================================================================
# State-Conditional Cross-Attn Modulator -- z_s modulates patches per position
# NOT pooling. Each patch token gets its own modulation from state context.
# ================================================================

class StateConditionalCrossAttn(nn.Module):
    """Per-patch cross-attention where VP token positions each get a query
    derived from z_s robot state, attending to the actual patch KV features.

    Unlike pooling (which collapses P->1 by averaging), this modulates each
    position independently through unique per-position queries while all
    derive from the same underlying z_s embedding via learned projections.

    Args:
        compressed_dim:  per-token dim after SE compression (default 16)
        state_dim:       robot state emb dim (default 256)
        num_heads:       cross-attention heads (default 4)

    Input:
        z_v_comp: (B, N_tokens, compressed_dim)  -- from SEVisualCompressor
        z_s:      (B, state_dim)                   -- robot state embedding

    Output:
        (B, N_tokens, compressed_dim)  -- same position count, values modulated
    """

    def __init__(self, compressed_dim: int = 16, state_dim: int = 256, num_heads: int = 4):
        super().__init__()
        # Per-position query from z_s. Different random weights per position -> queries
        # diverge during training and attend to distinct spatial regions.
        self.q_proj = nn.Linear(state_dim, compressed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=compressed_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(compressed_dim)
        # Start at identity so modulator learns adaptively from gradient signals
        self.attn_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_v_comp: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        B, N_pos, D = z_v_comp.shape

        # One query per position, all derived from same z_s via projection weights
        q = self.q_proj(z_s).unsqueeze(1).expand(-1, N_pos, -1)  # (B, N_pos, D)
        k = v = z_v_comp                                         # patches as KV

        attn_out, _ = self.cross_attn(q, k, v)                   # (B, N_pos, D)

        out = z_v_comp + self.attn_scale * attn_out              # residual modulation
        return self.norm(out)


# ================================================================
# Vision Patch Encoder -- raw vision -> SE compress -> state modulate
# All VP token positions preserved throughout the pipeline.
# ================================================================

class VisionPatchEncoder(nn.Module):
    """End-to-end patch-level visual encoder without any pooling step.

    Pipeline:
      Raw DINOv2 patches (B, VP, 768)
        -> SEVisualCompressor   (B, VP, compressed_dim)
        -> StateConditionalCrossAttn + z_s  (B, VP, compressed_dim)

    Args:
        compressed_dim:  per-patch dim after compression (default 16)
        state_dim:       robot state emb dim (default 256)
        num_cameras:     number of cameras (default 1)
        raw_dim:         DINOv2 output dim per patch (768 for ViT-B)
        se_reduction:    SE bottleneck ratio (default 8)
        num_heads:       cross-attention heads in modulator (default 4)
    """

    def __init__(self, compressed_dim: int = 16, state_dim: int = 256,
                 num_cameras: int = 1, raw_dim: int = 768, se_reduction: int = 8,
                 num_heads: int = 4):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.state_dim = state_dim
        self.num_cameras = num_cameras
        # VP tokens = V cameras x 16x16 grid = V * 256
        self.total_tokens = num_cameras * 256

        self.se_compressor = SEVisualCompressor(
            raw_dim=raw_dim, compressed_dim=compressed_dim, reduction=se_reduction)
        self.state_modulator = StateConditionalCrossAttn(
            compressed_dim=compressed_dim, state_dim=state_dim, num_heads=num_heads)

    def forward(self, v_patches_raw: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """Compress channels then modulate via state. All VP positions preserved.

        Args:
            v_patches_raw: (B, VP_tokens, raw_dim=768) -- from VisionEncoder
            z_s:           (B, state_dim)               -- robot state embedding
        Returns:
            (B, VP_tokens, compressed_dim) -- modulated patches, positions preserved
        """
        z_v_comp = self.se_compressor(v_patches_raw)         # (B, VP, comp_dim)
        z_v_mod  = self.state_modulator(z_v_comp, z_s)       # (B, VP, comp_dim)
        return z_v_mod


# ================================================================
# Intention Encoder (Mamba-based) -- full patch sequence to Mamba
# ================================================================

class IntentionEncoder(nn.Module):
    """Mamba intention encoder with full VP token sequences as input.

    Per timestep:
      v_patches(t) = VisionEncoder output  (B, VP_tokens, raw_dim=768)
      z_s(t)       = StateEncoder state     (B, state_dim=256)
          -> VisionPatchEncoder(v_patches, z_s) -> modulated patches  (B, VP, comp_dim)
          -> concat with z_s -> mamba_in    (B, VP*comp_dim + state_dim)
          -> Mamba recurrence                (B, mamba_output_dim=512)

    V4: When use_intent_tokens=True, learnable intent tokens are appended
    to the Mamba input sequence. The Mamba processes them with full history
    context via the SSM state, producing intent_emb alongside h_seq.

    Args:
        state_dim:         robot state dim (default 256)
        mamba_output_dim:  Mamba hidden output dim (default 512)
        num_cameras:       cameras count (default 1, so VP = 256 per step)
        compressed_dim:    per-patch SE-compressed dim (default 16)
        mamba_d_state:     SSM state dim for Mamba block (default 16)
        mamba_d_conv:      local conv width in Mamba (default 4)
        mamba_expand:      Mamba expand factor (default 2)
        raw_dim:           DINOv2 per-patch dim (768 for ViT-B)
        se_reduction:      SE bottleneck ratio (default 8)
        use_intent_tokens: enable learnable intent tokens (V4, default False)
        num_intent_tokens: number of intent tokens N (default 2)
        intent_dim:        output dim of each intent token (default 512)
    """

    def __init__(self, state_dim: int = 256, mamba_output_dim: int = 512,
                 num_cameras: int = 1, compressed_dim: int = 16,
                 mamba_d_state: int = 16, mamba_d_conv: int = 4,
                 mamba_expand: int = 2, raw_dim: int = 768,
                 se_reduction: int = 8,
                 use_intent_tokens: bool = False,
                 num_intent_tokens: int = 2,
                 intent_dim: int = 512):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm not installed. Run: pip install mamba-ssm causal-conv1d")

        self.state_dim = state_dim
        self.mamba_output_dim = mamba_output_dim
        self.num_cameras = num_cameras
        self.compressed_dim = compressed_dim
        self.raw_dim = raw_dim
        self.total_tokens = num_cameras

        # V4 intent token config
        self.use_intent_tokens = use_intent_tokens
        self.num_intent_tokens = num_intent_tokens
        self.intent_dim = intent_dim

        # Patch encoder: raw vision -> SE compress -> state modulation (no pooling)
        self.vision_patch_encoder = VisionPatchEncoder(
            compressed_dim=compressed_dim, state_dim=state_dim,
            num_cameras=num_cameras, raw_dim=raw_dim, se_reduction=se_reduction)

        # Mamba input: CLS tokens (V * raw_dim) + state embedding
        self.mamba_in_dim = self.total_tokens * raw_dim + state_dim

        # self.pool_out_dim = self.total_tokens * compressed_dim
        # self.pool = self.vision_patch_encoder

        self.mamba = Mamba(
            d_model=self.mamba_in_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        # Legacy projection: Mamba output -> mamba_output_dim
        self.mamba_to_hidden = nn.Linear(self.mamba_in_dim, mamba_output_dim)

        # V4: learnable intent tokens (appended to Mamba input sequence)
        if use_intent_tokens:
            self.intent_tokens = nn.Parameter(
                torch.randn(1, num_intent_tokens, self.mamba_in_dim) * 0.02
            )
            self.intent_proj = nn.Linear(self.mamba_in_dim, intent_dim)

    def pool_patches(self, z_v_patches: torch.Tensor, z_s: torch.Tensor
                     ) -> torch.Tensor:
        """Legacy interface -- encode patches via VisionPatchEncoder.

        Args:
            z_v_patches: (B, P_or_VP_tokens, raw_dim=768)
            z_s:         (B, state_dim)
        Returns:
            (B, compressed_dim) -- mean-pooled across VP positions
        """
        out = self.vision_patch_encoder(z_v_patches, z_s)  # (B, VP, comp_dim)
        return out.mean(dim=1)                               # (B, comp_dim) — mean pool
    
    def encode_patches(self, z_v_patches: torch.Tensor, z_s: torch.Tensor
                       ) -> torch.Tensor:
        """Encode patches via VisionPatchEncoder without pooling.
        
        Args:
            z_v_patches: (B, T, P_or_VP_tokens, raw_dim=768)
            z_s:         (B, T, state_dim)
        Returns:
            (B, T, VP_tokens, compressed_dim) -- all VP positions preserved
        """
        B, T = z_v_patches.shape[:2]
        # Encode each timestep: raw -> SE compress -> state modulate (no pooling)
        z_v_mod_seq = []
        for t in range(T):
            out_t = self.vision_patch_encoder(z_v_patches[:, t], z_s[:, t])
            z_v_mod_seq.append(out_t)
        
        return torch.stack(z_v_mod_seq, dim=1)  # (B, T, VP, comp_dim)
        
    def forward(self, z_v_cls_seq: torch.Tensor, z_s_seq: torch.Tensor
                ) -> torch.Tensor:
        """Batched T-step Mamba forward with CLS tokens.

        V3: returns h_seq (B, T, mamba_output_dim)
        V4 (use_intent_tokens=True): returns (h_seq, intent_emb)
            h_seq: (B, T, mamba_in_dim) — history outputs
            intent_emb: (B, N, intent_dim) — intent token outputs

        Args:
            z_v_cls_seq: (B, T, V, raw_dim=768) — CLS tokens from DINOv2
            z_s_seq:     (B, T, state_dim)
        Returns:
            h_seq or (h_seq, intent_emb)
        """
        # Flatten CLS tokens into feature dimension -> Mamba input
        B, T, V, D = z_v_cls_seq.shape
        mamba_in = torch.cat([
            z_v_cls_seq.reshape(B, T, V * D),
            z_s_seq,
        ], dim=-1)  # (B, T, mamba_in_dim)
        assert mamba_in.shape[-1] == self.mamba_in_dim, \
            f"mamba_in last dim {mamba_in.shape[-1]} != expected {self.mamba_in_dim}"
        if self.use_intent_tokens:
            # Append intent tokens to the input sequence
            intent_tokens = self.intent_tokens.expand(B, -1, -1)  # (B, N, mamba_in_dim)
            mamba_in = torch.cat([mamba_in, intent_tokens], dim=1)  # (B, T+N, mamba_in_dim)

            h_full = self.mamba(mamba_in)  # (B, T+N, mamba_in_dim)
            # print(f"h_full shape: {h_full.shape} it should be {(B, T + self.num_intent_tokens, self.mamba_in_dim)}")
            # Split: history outputs + intent outputs
            h_seq = h_full[:, :T, :]        # (B, T, mamba_in_dim)
            h_intent = h_full[:, T:, :]     # (B, N, mamba_in_dim)
            intent_emb = self.intent_proj(h_intent)  # (B, N, intent_dim)

            return h_seq, intent_emb
        else:
            # Legacy V3 path
            h_seq = self.mamba(mamba_in)                       # (B, T, mamba_in_dim)
            h_seq = self.mamba_to_hidden(h_seq)                # (B, T, mamba_output_dim)
            return h_seq

    def forward_sequence(self, z_v_cls_seq: torch.Tensor, z_s_seq: torch.Tensor,
                         readout_start: int = 0, chunk_size: int = 16):
        """Differentiable observation recurrence with causal intent readouts.

        Project/convolve observations once. Each learned-token readout forks the
        observation state, so tokens never become observation history. Checkpoint
        bounded chunks during training; no state is detached between chunks.
        Returns (B, T, N, intent_dim), with zeros before readout_start.
        Uses the existing Mamba1 weights and appended-token readout semantics.
        """
        if not self.use_intent_tokens:
            raise ValueError("Causal intent readouts require intent tokens")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        batch, length = z_s_seq.shape[:2]
        if not 0 <= readout_start <= length:
            raise ValueError("readout_start must be within the sequence")
        m = self.mamba
        inputs = torch.cat([z_v_cls_seq.flatten(2), z_s_seq], dim=-1)
        x_raw, _ = m.in_proj(inputs).chunk(2, dim=-1)
        x = F.silu(m.conv1d(x_raw.transpose(1, 2))[..., :length]).transpose(1, 2)
        dt_raw, b, _ = torch.split(m.x_proj(x), [m.dt_rank, m.d_state, m.d_state], dim=-1)
        # Match selective_scan: delta bias/softplus and recurrence in FP32.
        dt = F.softplus(F.linear(dt_raw, m.dt_proj.weight).float() + m.dt_proj.bias.float())
        a = -torch.exp(m.A_log.float())
        conv_windows = F.pad(x_raw.transpose(1, 2), (m.d_conv - 1, 0)).unfold(2, m.d_conv, 1)
        conv_windows = conv_windows.permute(0, 2, 1, 3)
        state = torch.zeros(batch, m.d_inner, m.d_state, device=x.device, dtype=torch.float32)
        readouts = []

        def scan_chunk(state_in, x_chunk, dt_chunk, b_chunk, conv_chunk, emit):
            state_out = state_in
            prefixes = []
            for t in range(x_chunk.shape[1]):
                delta = dt_chunk[:, t, :, None]
                state_out = state_out * torch.exp(delta * a)
                state_out = state_out + x_chunk[:, t, :, None].float() * delta * b_chunk[:, t, None, :].float()
                if emit:
                    prefixes.append(state_out)
            if not emit:
                empty = x_chunk.new_empty(batch, 0, self.num_intent_tokens, self.intent_dim)
                return state_out, empty
            count = len(prefixes)
            fork_ssm = torch.stack(prefixes, dim=1).flatten(0, 1)
            fork_conv = conv_chunk.flatten(0, 1)
            token_outputs = []
            # Token projections are shared across all causal prefix readouts.
            token_x, token_z = m.in_proj(self.intent_tokens).chunk(2, dim=-1)
            for n in range(self.num_intent_tokens):
                tx = token_x[:, n].expand(batch * count, -1)
                tz = token_z[:, n].expand(batch * count, -1)
                fork_conv = torch.cat([fork_conv[..., 1:], tx[..., None]], dim=-1)
                tx = (fork_conv * m.conv1d.weight.squeeze(1)).sum(-1)
                if m.conv1d.bias is not None:
                    tx = tx + m.conv1d.bias
                tx = F.silu(tx).to(x_chunk.dtype)
                td, tb, tc = torch.split(m.x_proj(tx), [m.dt_rank, m.d_state, m.d_state], dim=-1)
                td = F.softplus(F.linear(td, m.dt_proj.weight).float() + m.dt_proj.bias.float())
                fork_ssm = fork_ssm * torch.exp(td[..., None] * a)
                fork_ssm = fork_ssm + tx[..., None].float() * td[..., None] * tb[:, None].float()
                y = (fork_ssm * tc[:, None].float()).sum(-1) + m.D.float() * tx.float()
                y = (y * F.silu(tz.float())).to(x_chunk.dtype)
                token_outputs.append(m.out_proj(y))
            intent = self.intent_proj(torch.stack(token_outputs, dim=1))
            return state_out, intent.reshape(batch, count, self.num_intent_tokens, self.intent_dim)

        # Separate warmup from readout chunks, avoiding unnecessary token work.
        boundaries = list(range(0, readout_start, chunk_size))
        boundaries += list(range(readout_start, length, chunk_size))
        boundaries += [length]
        for start, end in zip(boundaries, boundaries[1:]):
            if start == end:
                continue
            emit = start >= readout_start
            args = (state, x[:, start:end], dt[:, start:end], b[:, start:end],
                    conv_windows[:, start:end], emit)
            if self.training and torch.is_grad_enabled():
                state, intent = checkpoint(scan_chunk, *args, use_reentrant=False)
            else:
                state, intent = scan_chunk(*args)
            if emit:
                readouts.append(intent)
        zeros = x.new_zeros(batch, readout_start, self.num_intent_tokens, self.intent_dim)
        return torch.cat([zeros] + readouts, dim=1)

    @torch.no_grad()
    def forward_step(self, z_v_cls: torch.Tensor, z_s: torch.Tensor,
                     h_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     produce_intent: bool = False
                     ):
        """Recurrent one-step inference.

        V3: returns (h_new, h_states)
        V4 (use_intent_tokens and produce_intent=True): returns (h_new, h_states, intent_emb)

        Args:
            z_v_cls:    (B, V, raw_dim=768) — CLS tokens from DINOv2 (one per camera)
            z_s:        (B, state_dim)
            h_states:   (conv_state, ssm_state) from prev step, or None
            produce_intent: if True, read tokens on a fork after the observation

        Returns:
            (h_new, h_states) or (h_new, h_states, intent_emb)

        NOTE: Following the design (CLS tokens for intention_encoder, patches for head),
        this method takes CLS tokens directly. It does NOT use VisionPatchEncoder
        (which is only used for the head's input via the memory bank).
        """
        B = z_v_cls.shape[0]

        if h_states is None:
            h_states = self.mamba.allocate_inference_cache(
                batch_size=B, max_seqlen=1)
        conv_state, ssm_state = h_states

        # Flatten CLS tokens and concat with z_s (same as forward())
        B_tok, V, D = z_v_cls.shape
        mamba_in = torch.cat([z_v_cls.reshape(B_tok, V * D), z_s], dim=-1)

        mamba_out, conv_state, ssm_state = self.mamba.step(
            mamba_in.unsqueeze(1),  # (B, 1, mamba_in_dim)
            conv_state, ssm_state,
        )

        h_new = self.mamba_to_hidden(mamba_out.squeeze(1))  # (B, mamba_output_dim)

        if self.use_intent_tokens and produce_intent:
            # Readout must not modify the carried observation-only cache.
            intent_conv, intent_ssm = conv_state.clone(), ssm_state.clone()
            intent_out = []
            intent_tokens = self.intent_tokens.expand(B, -1, -1)  # (B, N, mamba_in_dim)
            for i in range(self.num_intent_tokens):
                tok = intent_tokens[:, i:i+1, :]
                out, intent_conv, intent_ssm = self.mamba.step(tok, intent_conv, intent_ssm)
                intent_out.append(out)
            intent_out = torch.cat(intent_out, dim=1)  # (B, N, mamba_in_dim)
            intent_emb = self.intent_proj(intent_out)  # (B, N, intent_dim)
            return h_new, (conv_state, ssm_state), intent_emb

        return h_new, (conv_state, ssm_state)

    def allocate_state(self, batch_size: int, device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate Mamba inference cache before first step."""
        return tuple(state.to(device) for state in self.mamba.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=1))
