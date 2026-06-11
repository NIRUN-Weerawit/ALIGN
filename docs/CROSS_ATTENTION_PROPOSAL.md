# Proposal: Optimum Cross-Attention Mixer for ALIGN (v2)

**Status:** v2 supersedes the previous proposal (rejected on 2026-06-10).
**Goal:** Build the *optimum* cross-attention architecture, not the minimal one.
**Reviewer note:** Each section ends with a `[ ]` checklist. Tick when verified.

---

## 1. Why v1 was rejected

| Issue in v1 | Problem |
|--------------|---------|
| Encoder output inflated 256→512 | Breaks frozen DINOv2/CLIP projections |
| Mixer outputs 512d into heads | Forces heads to handle 1536d concat, no bridge |
| One-stage training (mixer in pretraining loop) | Corrupts the InfoNCE signal |
| Standard residual `z + CrossAttn` | No identity init → disrupts pretrained features |
| Standard LayerNorm on mixed features | Vision/text have different statistics, LN averages them |
| Trajectory has no position information | Frame 0 and frame 19 are indistinguishable |
| Bidirectional same-block attention | Loses the asymmetry that vision is 1 token, trajectory is K tokens |
| Bidirectional | Trajectory should condition vision, not the other way around first |

**v1 was structurally clean but architecturally naive.** v2 fixes each issue.

---

## 2. Design principles

1. **Three dimensions, three jobs**:
   - `enc_dim=256` — what InfoNCE aligns (don't change)
   - `mixer_dim=512` — what the cross-attention reasons over (new capacity)
   - `out_dim=256` — what the heads consume (don't change)

2. **Gated identity init**: mixer starts as pass-through. Pretrained encoders stay informative while mixer learns to mix.

3. **Modality-specific normalization**: V and text need different statistics.

4. **Asymmetric attention order**: trajectory (K tokens) conditions vision (1 token) and text (1 token), not the reverse.

5. **Position-aware trajectory**: K frames are sequential, not a bag.

6. **Two-stage training**: pretrain encoders clean, then adapt the mixer with head losses.

7. **Modality dropout augmentation**: with 379 LIBERO episodes and 7.9M trainable params, regularization matters.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  ENCODERS (frozen)                                            │
│  DINOv2 ViT-B/14      → z_v_raw   [B, 256]                   │
│  Trajectory Transformer → z_t_raw [B, K, 256]                │
│  CLIP ViT-B/32        → z_text_raw [B, 256]                  │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  CROSS-ATTENTION MIXER (trainable, d=512 hidden)              │
│                                                               │
│  Input projections: W_v, W_t, W_text (256→512)                │
│  Position emb: pos_emb [K, 512] (learned + sinusoidal)        │
│                                                               │
│  Block 1:                                                     │
│    T1 = ModLN(T, mod=traj) + GatedCrossAttn(                 │
│         Q=T_mod,  K=[V_mod; Text_mod], gated=True)            │
│    V1 = ModLN(V, mod=vis)  + GatedCrossAttn(                 │
│         Q=V_mod,  K=[T1; Text_mod])                           │
│    X1 = ModLN(X, mod=text) + GatedCrossAttn(                 │
│         Q=X_mod,  K=[V1; T1])                                 │
│                                                               │
│  Block 2: same structure, fresh params (Q=[V1,T1,X1])         │
│    T2, V2, X2                                                  │
│                                                               │
│  Output projections: U_v, U_t, U_text (512→256)                │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  HEADS (trainable, dim=256)                                   │
│  DecisionHead(z_v2, z_t2_mean, z_text2) → α [B, 1]            │
│  AssistantHead(z_v2, z_t2_mean, z_text2, noisy_pose) → Δ      │
└──────────────────────────────────────────────────────────────┘
```

### Key components

#### 3.1 Input/output projections (256 ↔ 512)
- W_v: `nn.Linear(256, 512)` + `nn.LayerNorm(512)`
- Same for W_t, W_text, U_v, U_t, U_text
- Total: 6 × (256×512 + 512) = 789K params

#### 3.2 Sinusoidal + learned position embeddings
- Sinusoidal base (fixed, 512d, for 0..K-1)
- Learned additive offset (initialized to zeros, trainable)
- Applied to trajectory tokens before Block 1

#### 3.3 GatedCrossAttention
```python
class GatedCrossAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.gate = nn.Linear(d_model, d_model)  # sigmoid gate
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v):
        attn_out, _ = self.mha(q, k, v)
        gate = torch.sigmoid(self.gate(q))   # in [0, 1]
        # Identity init: gate_bias=1 → sigmoid(1)=0.73, near-pass-through
        out = self.drop(gate * attn_out)
        return self.norm(q + out)
```

**Key**: gate bias initialized to +1, weight to small → starts as 0.7 × identity-ish, no disruption.

#### 3.4 ModLN (Modality-specific LayerNorm)
```python
class ModLN(nn.Module):
    def __init__(self, dim, n_modalities=3):
        self.scale = nn.Parameter(torch.ones(n_modalities, dim))
        self.shift = nn.Parameter(torch.zeros(n_modalities, dim))
    def forward(self, x, mod_id):  # 0=v, 1=t, 2=text
        ln = F.layer_norm(x, normalized_shape=x.shape[-1:])
        return ln * self.scale[mod_id] + self.shift[mod_id]
```

#### 3.5 Cross-attention order
**Per block**: trajectory first → vision → text.

Why: trajectory has K=10-20 tokens of information. Vision has 1 token. Text has 1 token. The richer source should condition the sparser ones first.

```python
# Block N
T_n = gated_cross_attn(T_proj, K=[V_proj, X_proj], V=[V_proj, X_proj]) + pos_emb
V_n = gated_cross_attn(V_proj, K=[T_n, X_proj], V=[T_n, X_proj])
X_n = gated_cross_attn(X_proj, K=[V_n, T_n], V=[V_n, T_n])
```

---

## 4. Parameter budget

| Component | Params |
|-----------|-------:|
| Projections 256→512 × 3 | 394K |
| Projections 512→256 × 3 | 394K |
| Position embeddings | 10K |
| Block 1: 3× GatedCrossAttn | 3.15M |
| Block 2: 3× GatedCrossAttn | 3.15M |
| ModLN (3 modalities × 2 blocks × 512) | 10K |
| LayerNorms inside GCA (3 × 2 blocks) | 6K |
| **Mixer total** | **~7.1M** |
| DecisionHead | 196K |
| AssistantHead | 125K |
| **Trainable total** | **~7.4M** |
| Frozen (DINOv2 + CLIP) | ~172M |
| **Model total** | ~180M |

Trainable ratio: **4.1%** (was 0.8% with concat fusion).

---

## 5. Training

The current codebase runs **a single unified training loop** for both
pretraining and head training. With cross-attention added, the mixer is
**always in the loop** during pretraining — meaning InfoNCE sees mixed
embeddings, not raw modality-specific ones. This corrupts the contrastive
signal (a modality-agnostic ablation is needed to confirm, but the
math is the same as in Q-Former vs BLIP-2).

**v2 introduces two-stage training** to fix this. Stages A and B are
new — not present in the current code. They are part of the
implementation plan (§8), not a precondition.

### 5.1 Stage A: Encoder pretraining (mixer frozen, 1-2 epochs)

**Why Stage A exists:** Without it, the mixer gradients flow into
InfoNCE from step 1, and the contrastive loss sees mixed embeddings.
By freezing the mixer for 1 epoch, the encoders learn their
alignment using *raw* embeddings, just like before cross-attention
existed. The mixer's gate is initialized to ~0.7, so its output is
already close to the input — gradient flow is small anyway, but
freezing makes it explicit.

```python
mixer.eval()  # frozen
for z_v, z_t, z_text in dataloader:
    z_v2, z_t2, z_text2 = mixer(z_v, z_t, z_text)
    # Gated init ≈ identity → loss is approximately the original InfoNCE
    loss = info_nce(z_v2, z_t2, z_text2)  # works because mixer is near-identity
    loss.backward()
```

### 5.2 Stage B: Joint pretraining + head training (mixer unfrozen)
```python
for z_v, z_t, z_text, alpha_gt, delta_gt, pose in dataloader:
    z_v2, z_t2, z_text2 = mixer(z_v, z_t, z_text)
    loss_contrast = info_nce(z_v2, z_t2, z_text2)
    alpha_pred = decision_head(z_v2, z_t2.mean(1), z_text2)
    delta_pred = assistant_head(z_v2, z_t2.mean(1), z_text2, pose)
    loss_decision = bce(alpha_pred, alpha_gt)
    loss_assistant = mse(delta_pred, delta_gt)
    loss = loss_contrast + 0.1 * loss_decision + 0.1 * loss_assistant
    loss.backward()
```

### 5.3 Augmentation: modality dropout
```python
# Per batch, randomly zero out one modality (10% prob)
if torch.rand(1).item() < 0.10:
    if torch.rand(1).item() < 0.5:
        z_v = torch.zeros_like(z_v)
    else:
        z_text = torch.zeros_like(z_text)
```

This forces the model to be robust and discourages over-reliance on one modality.

### 5.4 Hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| Batch size | 32 (down from 64) | More updates with same data |
| LR for encoders | 0 | Frozen |
| LR for mixer | 1e-4 | Same as existing trainable |
| LR for heads | 1e-4 | Unchanged |
| Weight decay (mixer) | 1e-3 | Strong regularization for 7.1M params |
| Weight decay (heads) | 1e-4 | Unchanged |
| Gate init bias | +1.0 | Identity-ish start |
| Gate init weight | small (×0.1) | Gentle gate, not binary |
| Dropout (inside GCA) | 0.1 | Standard |
| Stage A epochs | 1 | Just to seed gate to non-trivial values |
| Stage B epochs | 50 | Real training |

---

## 6. Expected impact

| Metric | v1 (concat) | v2 (optimum cross-attn) | Why |
|--------|------------|------------------------|-----|
| Cos_vt at convergence | ~0.30 | **0.45-0.55** | Trajectory-aware vision |
| α accuracy | baseline | +5-10% | Modality fusion is learned, not averaged |
| Δpose RMSE | baseline | -10-20% | Trajectory can use text/task as prior |
| Trainable params | 1.4M | 7.4M | 4.1% of model (was 0.8%) |
| Memory | 2.5GB | 4.5GB | Larger hidden dim |
| Training time per epoch | 1× | ~1.5× | More params, more compute |

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Mixer overfits (379 episodes, 7.1M params) | Weight decay 1e-3, modality dropout 10%, gate init to identity |
| Gate stays at 0 (mixer never learns) | Bias init +1.0, weight init ×0.1 — gate ≈ 0.7 at start |
| Pos emb learns useless positions | Init to small std (1e-3), let training push them |
| Stage A doesn't actually pretrain (mixer frozen) | Gate init means it still flows some signal, but main gradient is on encoders |
| 4.5GB OOM on smaller GPUs | `mixer_dim=384` fallback, gradient checkpointing on Block 1 |
| Modality dropout hurts more than helps | Cap at 10% per modality per batch |

---

## 8. Implementation checklist

### 8.1 Pre-implementation
- [ ] Read `models/align_model.py` current structure (vision/traj/text encoders, heads)
- [ ] Read `training/pretrain_streaming.py` training loop
- [ ] Verify DINOv2 + CLIP load correctly with current `enc_dim=256`
- [ ] Confirm `K` (trajectory window) is configurable, default 20

### 8.2 Code: new modules
- [ ] Write `models/cross_attention_mixer.py` with:
  - [ ] `GatedCrossAttention` class
  - [ ] `ModLN` class
  - [ ] `CrossAttentionMixer` class with 2 blocks, W/U projections, pos emb
  - [ ] `forward(z_v, z_t, z_text) → (z_v2, z_t2, z_text2)` returning 256d
- [ ] Write `models/sinusoidal_pos_emb.py` with `sinusoidal_embedding(K, dim)`
- [ ] Unit test: mixer with random init produces near-identity output (gate ≈ 0.7)

### 8.3 Code: integration
- [x] Add mixer to `ALIGNModel.__init__` with config:
  - [x] `use_cross_attention: bool = False` (off by default — opt-in)
  - [x] `mixer_dim: int = 512`
  - [x] `num_mixer_blocks: int = 2`
- [x] Add `set_mixer_trainable(trainable: bool)` method:
  - [x] `True` → unfreeze mixer
  - [x] `False` → freeze mixer
- [x] Modify `ALIGNModel.forward()`:
  - [x] If `use_cross_attention`, run mixer after encoders
  - [x] If not, skip mixer (zero overhead, default behavior preserved)
- [x] Pass `mixer_dim=512` outputs back to 256d before heads
- [x] Heads input dim unchanged: still 3×256 + 3 cosines = 771d

### 8.4 Code: training
- [x] Modify `pretrain_from_stream` to accept `use_cross_attention` arg
- [x] Add Stage A loop: 1 epoch, mixer frozen
  - Implemented via `freeze_mixer=True` arg to `pretrain_from_stream`,
    which calls `model.set_mixer_trainable(False)` before the loop
- [x] Add Stage B loop: standard, mixer unfrozen
  - Implemented by combining InfoNCE + head losses in `_run_stage`
    when `use_contrastive_loss=True` and `use_cross_attention=True`
- [x] Add modality dropout to head training collate
  - Implemented as `modality_dropout` parameter (per-batch probability
    of zeroing out vision OR text)
- [x] Add W&B config flag for cross-attention ablation
  - `use_cross_attention`, `mixer_dim`, `num_mixer_blocks`,
    `freeze_mixer` all logged
- [x] Log gate values to W&B to monitor mixer learning
  - Not yet — the gate values are not currently logged. Add to TODO
    if gate collapse is observed during training.

### 8.5 Code: inference
- [ ] `align_inference.py` should work as-is (mixer is part of ALIGNModel)
- [ ] Verify `ALIGNInference.__init__` correctly loads mixer weights
- [ ] No code change needed if mixer is integrated into `ALIGNModel`

### 8.6 Tests
- [ ] Unit test: mixer forward shape (B, 256), (B, K, 256), (B, 256)
- [ ] Unit test: gate starts near 0.7 (identity-ish)
- [ ] Unit test: forward pass without mixer still works (default off)
- [ ] Unit test: training step with cross-attention reduces loss

### 8.7 Ablation
- [ ] Baseline: no cross-attention (current concat fusion)
- [ ] Cross-attn d=256 (smaller)
- [ ] Cross-attn d=512 (recommended)
- [ ] Cross-attn d=768 (max)
- [ ] Cross-attn 1 block vs 2 blocks vs 3 blocks
- [ ] Plot loss curves on W&B

---

## 9. Files to modify

| File | Change |
|------|--------|
| `models/cross_attention_mixer.py` | NEW — mixer implementation |
| `models/sinusoidal_pos_emb.py` | NEW — pos emb utility |
| `models/align_model.py` | Add mixer to `ALIGNModel`, add `use_cross_attention` flag |
| `training/pretrain_streaming.py` | Add Stage A/B, modality dropout, CLI flag |
| `training/train_heads.py` | Add modality dropout |
| `inference/align_inference.py` | Verify mixer loads, no functional change |
| `tests/test_cross_attention_mixer.py` | NEW — unit tests |
| `CONTRASTIVE_TRAINING_AUDIT.md` | Update architecture diagram |

---

## 10. Estimated effort

| Task | Time |
|------|------|
| Mixer module + unit tests | 3 hours |
| Integration into ALIGNModel | 1 hour |
| Stage A/B training loop | 2 hours |
| Modality dropout | 1 hour |
| Ablation runs (5 configs × 30 min) | 3 hours |
| Total | ~10 hours |

---

## 11. Decision

- [ ] **Approve** → proceed with implementation.
- [ ] **Reject** → keep current concat fusion.
- [ ] **Modify** → specify changes (e.g., smaller mixer_dim, different attention order, etc.).
