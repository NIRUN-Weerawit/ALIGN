# Proposed Training Architecture — ALIGN (Final)

## Model Architecture

```
DINOv2 ViT-B/14 ─→ vision_proj ─→ z_v (256) ─┐
                                                │
Trajectory Transformer ─→ traj_proj ─→ z_t (256)─┤─→ CrossAttentionMixer ─→ z'_v, z'_t, z'_text
                                                │       (identity init, gates≈0.7)
CLIP ViT-B/32 ─→ text_proj ─→ z_text (256) ────┘
                                                         │
                                                  ┌──────┴──────┐
                                                  │             │
                                             DecisionHead   AssistantHead
                                              (α ∈ [0,1])   (Δposes, K=5)
```

The mixer enriches each modality with cross-modal context. Identity init means at startup, ~70% of input passes through unchanged — safe to unfreeze from the start, but freezing during Phase 1a is defensive best practice.

---

## Training: 3 Sub-Phases, 2 Losses, 3 Epoch Knobs

```
Phase 1a: Encoder Pretrain (InfoNCE only)
──────────────────────────────────────────
Loss:       InfoNCE(z_v, z_t, z_text) — on RAW encoder outputs
Train:      vision_proj + traj_encoder + text_proj
Mixer:      FROZEN (passes features through via identity init)
Frozen:     DINOv2 backbone, CLIP text model
Epochs:     --epochs-pretrain-encoder (default 40)
Output:     encoder_best.pt — encoder params only (~5.4MB)

Phase 1b: Mixer Warm-Up (InfoNCE only)
───────────────────────────────────────
Loss:       InfoNCE(z'_v, z'_t, z'_text) — on MIXER outputs
Train:      encoders + mixer (everything unfrozen except DINOv2/CLIP)
Mixer:      Starts from identity init → gates gradually open via gradient
Epochs:     --epochs-pretrain-mixer (default 10)
Output:     pretrain_best.pt — full checkpoint (~14MB)
            Loads from encoder_best.pt

Phase 2: Head Training (BCE + MSE)
───────────────────────────────────
Loss:       BCE(α_pred, α_target) + 0.5 × MSE(Δpred, Δtarget)
Train:      decision_head + assistant_head ONLY
Frozen:     EVERYTHING else (encoders, mixer, DINOv2, CLIP)
Epochs:     --epochs-heads (default 30)
Output:     heads_best.pt — head params only (~0.4MB)
            Loads from pretrain_best.pt
```

---

## Why 3 Sub-Phases?

| Sub-phase | Why it exists | What it prevents |
|-----------|---------------|------------------|
| **1a (encoder only)** | InfoNCE on raw embeddings teaches a structured latent space before the mixer touches it | Mixer learning cross-modal associations from half-baked encoder features |
| **1b (mixer warm-up)** | Gradual cross-modal learning. Identity init ≈ no-op for first few epochs of 1b, so encoders barely feel the mixer's gradient | Abrupt cross-modal coupling from unfreezing mixer on under-trained encoders |
| **2 (heads only)** | Heads learn task predictions on a fixed, stable embedding space | InfoNCE gradients interfering with BCE/MSE, loss scale imbalance |

Phase 1a + 1b could be a single phase (both use InfoNCE, both train encoders), but splitting them makes it explicit that the mixer joins after the encoders are stable.

**Default epoch split:** 40 + 10 + 30 = 80 total epochs. The mixer only gets 10 epochs because identity init means it converges quickly (it starts near-optimal).

---

## Checkpoint Format

### encoder_best.pt (~5.4MB)
```python
{
    "format_version": 2,
    "phase": "encoder_pretrain",
    "trainable_state_dict": {
        "vision_encoder.projection.0.weight": ...,
        "vision_encoder.projection.1.weight": ...,
        "traj_encoder.input_proj.weight": ...,
        "traj_encoder.transformer.layers.*": ...,
        "text_encoder.projection.0.weight": ...,
        ...
    },
    "backbone_refs": {"vision": "dinov2_vitb14", "text": "ViT-B-32"},
    "config": {"embed_dim": 256, "traj_window": 20, ...},
    "epoch": 40, "loss": 2.34,
}
```

### pretrain_best.pt (~14MB) — loaded from encoder_best.pt
```python
{
    "format_version": 2,
    "phase": "full_pretrain",
    "trainable_state_dict": {
        # encoder params + mixer params
        "vision_encoder.projection.*": ...,
        "traj_encoder.*": ...,
        "text_encoder.projection.*": ...,
        "cross_attention_mixer.*": ...,
    },
    "backbone_refs": {...},
    "config": {...},
    "epoch": 50, "loss": 1.89,
}
```

### heads_best.pt (~0.4MB) — loaded from pretrain_best.pt
```python
{
    "format_version": 2,
    "phase": "heads",
    "trainable_state_dict": {
        "decision_head.*": ...,
        "assistant_head.*": ...,
    },
    "pretrain_checkpoint": "path/to/pretrain_best.pt",
    "config": {"chunk_size": 5, ...},
    "epoch": 30, "loss": 0.12,
}
```

---

## What Gets Removed vs Current Code

| Removed | Why |
|---------|-----|
| `--freeze-mixer` | Implicit: Phase 1a freezes, Phase 1b unfreezes |
| `--stage-a-epochs` | Replaced by concrete `--epochs-pretrain-encoder` |
| `--stage-b-head-loss-weight` | No combined loss in Phase 2 |
| `--modality-dropout` | Adds complexity, not validated |
| `--epochs-decision` | Single joint head loss |
| `--epochs-assistant` | Single joint head loss |
| `--epochs-joint` | Single joint head loss |
| 3-stage `_run_stage` | One `train_heads_joint()` function |
| `clip_grad_norm_(model.parameters())` | Uses optimizer param group |
| `freeze_backbone()` (traj_encoder leak) | `freeze_all_encoders()` is explicit |
| 965MB checkpoint | ~15MB total (5.4 + 14 + 0.4 are saved separately, inference loads ~15MB) |
| HDF5 pipeline divergence | Streaming is the primary path |

---

## CLI

```bash
# Full training (Phase 1a → 1b → 2)
python training/pretrain_streaming.py \
    --dataset nvidia/LIBERO_LeRobot_v3 \
    --output-dir ./checkpoints/streaming \
    --epochs-pretrain-encoder 40 \
    --epochs-pretrain-mixer 10 \
    --epochs-heads 30

# Phase 1a only (encoder pretrain)
python training/pretrain_streaming.py \
    --dataset nvidia/LIBERO_LeRobot_v3 \
    --stages encoder \
    --epochs-pretrain-encoder 40

# Resume from Phase 1a → continue to 1b + 2
python training/pretrain_streaming.py \
    --dataset nvidia/LIBERO_LeRobot_v3 \
    --stages all \
    --encoder-checkpoint ./checkpoints/streaming/encoder/best.pt \
    --epochs-pretrain-mixer 10 \
    --epochs-heads 30

# Phase 2 only (heads from pretrained)
python training/pretrain_streaming.py \
    --dataset nvidia/LIBERO_LeRobot_v3 \
    --stages heads \
    --pretrained ./checkpoints/streaming/pretrain/best.pt \
    --epochs-heads 30
```

---

## Implementation Order

1. **`ALIGNModel` changes:**
   - `freeze_all_encoders()` — freezes vision_proj, traj_encoder, text_proj, mixer, DINOv2, CLIP
   - `freeze_backbone()` unchanged (still freezes DINOv2 + CLIP only)
   - `_get_trainable_state_dict()` + `_load_trainable_state_dict()` — selective save/load by prefix
   - Mixer is always initialized (remove `use_cross_attention` flag)

2. **`pretrain_from_stream()` changes:**
   - Accept `--epochs-pretrain-encoder`, `--epochs-pretrain-mixer`
   - Phase 1a: freeze mixer, train encoders, InfoNCE on raw outputs
   - Phase 1b: unfreeze mixer, InfoNCE on mixer outputs
   - Save `encoder_best.pt` after Phase 1a, `pretrain_best.pt` after Phase 1b
   - Trainable-only checkpoint format

3. **`train_heads_from_stream()` changes:**
   - Replace 3-stage `_run_stage` with single `train_heads_joint()`
   - Single optimizer for both heads
   - Call `freeze_all_encoders()` — no training leakage
   - Fix gradient clip scope
   - Save `heads_best.pt`

4. **CLI cleanup:**
   - Remove 10+ stale flags
   - Add `--stages {encoder, pretrain, heads, all}`