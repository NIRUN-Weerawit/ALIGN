# Training Pipeline Logic Audit — Contradictions, Complexity, and Structural Issues

## Overview: Two Parallel Pipelines Doing the Same Thing Differently

The codebase has **two independent training pipelines** that accomplish the same goal but with different code paths, different bugs, and incompatible features:

| Aspect | HDF5 Pipeline (`train_full_pipeline.py` + `pretrain.py` + `train_heads.py`) | Streaming Pipeline (`pretrain_streaming.py`) |
|--------|-------|---------|
| Data source | Pre-converted .h5 files on disk | LeRobot v3 streaming (Hub or local) |
| Noise injection | Pre-computed offline (3 variants × episodes × 3 noise levels) | On-the-fly per batch (1 noise config cycles) |
| α_target | Position-only (OUTDATED) | Position + orientation (UPDATED) |
| Δpose targets | Correct chunk computation (`t+i` varies) | Single-step fallback (`actions_window` likely always None) |
| Distances input to DecisionHead | Read from dataset | Hardcoded zeros |
| Epoch splits | decision=10, assistant=20, joint=20 (DEFAULTS) | decision=epochs//3, assistant=2*epochs//3, joint=epochs//3 |
| Cross-attention mixer | NOT SUPPORTED | Supported (Stage A/B) |

**Consequence:** Any bug fix applied to one pipeline (e.g. orientation-aware α, proper Δpose targets) must be manually ported to the other. Currently they diverge.

---

## Structural Contradiction 1: Encoder Pretraining + Head Training are NOT Separate

The name "encoder/head separate training" implies encoders are trained first, then heads are trained separately on frozen encoders. But **the HDF5 pipeline does not cleanly separate them**:

```
┌─────────────────────────────────────────────┐
│                 pretrain.py                  │
│  Trains:                                     │
│    - vision projection head (usually frozen) │
│    - trajectory encoder (all params)         │
│    - text projection head (usually frozen)   │
│  Freezes:                                    │
│    - DINOv2 backbone                         │
│    - CLIP text model                         │
│  Not trained at all:                         │
│    - Decision head                           │
│    - Assistant head                          │
└─────────────────────────────────────────────┘
                      │ checkpoint.pt (includes DINOv2+CLIP — 965MB)
                      ▼
┌─────────────────────────────────────────────┐
│               train_heads.py                 │
│  Loads full checkpoint, then freezes:        │
│    - DINOv2 backbone (already frozen)        │
│    - CLIP text model (already frozen)        │
│  But trajectory encoder is NOT re-frozen!    │
│  ├── Stage 1: train Decision head only       │
│  ├── Stage 2: train Assistant head only      │
│  └── Stage 3: joint fine-tune both heads     │
│  Trainer uses: clip_grad_norm_(model.parameters())│ 
│  So trajectory encoder CAN receive gradients │
│  during head training (contradiction)         │
└─────────────────────────────────────────────┘
```

**Bugs:**
- `train_epoch()` calls `clip_grad_norm_(model.parameters())` but the optimizer only contains head parameters — so the gradient norm calculation iterates all model parameters including frozen ones (which have grad=None), but the clip is on the optimizer's param group. Actually, the clip is on `model.parameters()` not optimizer params, so it clips ALL params including frozen ones with None gradients. **This wastes cycles and the norm computation is dominated by DINOv2's zero-grad entries.** The clip should be `optimizer.param_groups[0]["params"]`.
- The trajectory encoder is **never explicitly frozen** during head training. With `freeze_backbone()` only freezing vision backbone and text model, the trajectory encoder + all projection heads remain trainable through the full head training process. This means head training is actually fine-tuning the trajectory encoder too — which isn't documented or intended.

---

## Structural Contradiction 2: 3-Stage Head Training is Over-Engineered

The head training splits into 3 sequential stages, each with its own optimizer:

```
Stage 1 (decision):    opt = AdamW(decision_head.params)      — 10 epochs
Stage 2 (assistant):   opt = AdamW(assistant_head.params)     — 20 epochs  
Stage 3 (joint):       opt = AdamW([decision+assistant]) @ lr*0.5 — 10 epochs
```

**Problems:**
- During Stage 1, the assistant head receives **zero gradient signal for 10 epochs**. If the decision head ever diverges, the assistant head's weights are stale by the time it trains.
- During Stage 2, the decision head receives **zero gradient signal for 20 epochs**. If the assistant head diverges, same problem.
- The epoch split (decision=10, assistant=20, joint=10) means the assistant head gets **2× more training than the decision head** — an arbitrary ratio with no justification.
- **Each stage iterates over the entire dataset**, so 3 stages = 3 full passes. If the validation loss for Stage 1 "best" is saved as `decision_best.pt`, but Stage 2 loads from the LIVE model (not from the best), Stage 2 starts from whatever state the model happened to be in at epoch 10. **The best decision head checkpoint is silently discarded.**
- The streaming pipeline's `_run_stage` saves checkpoints per stage — but Stage 2 loads `model` from the end of Stage 1, not from `decision_best.pt`. Same issue.

**Simpler alternative:** Train both heads jointly from the start with a combined loss `L = BCE(α_pred, α_target) + 0.5 * MSE(Δpred, Δtarget)`. This eliminates 2/3 of the hyperparameters and avoids the stale-weight problem. The staged approach adds complexity without evidence it helps.

---

## Structural Contradiction 3: Stage A/B Cross-Attention Mixer Adds Unnecessary Complexity

The `use_cross_attention` flag introduces an entirely separate training regime:

### Without mixer (default):
```
pretrain_from_stream() → InfoNCE only → train_heads_from_stream() → BCE+MSE
```

### With mixer:
```
Stage A (opt): pretrain_from_stream(freeze_mixer=True) → InfoNCE on raw encoders
Stage B:       pretrain_from_stream(freeze_mixer=False) → InfoNCE on mixers
               train_heads_from_stream() → BCE+MSE + InfoNCE combined
```

**Problems:**
- Stage A defaults to 0 epochs — effectively dead code unless explicitly enabled
- When mixer IS enabled, head training uses a **combined loss** (`head_loss_weight * head_loss + InfoNCE_loss`) but this InfoNCE is computed on the **output of the mixer**, re-encoding the same batch through the encoders AGAIN with the same frames and trajectories. The InfoNCE gradient now flows through the mixer too, which partially undoes the Stage A pretraining.
- The `freeze_mixer` parameter name is misleading — `set_mixer_trainable(False)` which calls `requires_grad_(False)` but also calls `.eval()` which affects dropout/BatchNorm behavior. These have different semantics.
- The modality dropout feature (`modality_dropout > 0`) is only active when the mixer is present, creating yet another divergent code path.

---

## Structural Contradiction 4: Too Many Epoch Parameters (Hyperparameter Explosion)

Current epoch parameters (count them):

| Parameter | Default | Used by |
|-----------|---------|---------|
| `--epochs` (pretrain) | 50 | `pretrain.py`, `pretrain_from_stream()` |
| `--epochs-pretrain` | 50 | `run_full_pipeline()` |
| `--epochs-heads` | 30 | `run_full_pipeline()` |
| `--epochs-decision` | 10 | `train_heads.py` |
| `--epochs-assistant` | 20 | `train_heads.py` |
| `--epochs-joint` | 20 | `train_heads.py` |
| `stage_a_epochs` | 0 | `run_streaming_pipeline()` |
| `max_steps_per_epoch` | 2000 | streaming only |

**Total: 8 epoch-related parameters for what is essentially 2 training phases.**

Worse, the orchestration is inconsistent:

In `run_full_pipeline()`:
```python
"--epochs-decision", str(epochs_heads // 3),      # epochs_heads=30 → 10
"--epochs-assistant", str(2 * epochs_heads // 3),  # epochs_heads=30 → 20
"--epochs-joint", str(epochs_heads // 3),           # epochs_heads=30 → 10
```

But `train_heads.py` defaults:
```python
DEFAULT_EPOCHS_DECISION = 10
DEFAULT_EPOCHS_ASSISTANT = 20
DEFAULT_EPOCHS_JOINT = 20  # <-- 20, not 10
```

So if someone calls `train_heads.py` directly (not through the pipeline), the joint phase gets **20 epochs**, but when called through the pipeline it gets **10 epochs**. Different defaults = inconsistent behavior depending on entry point.

---

## Structural Contradiction 5: Data Loading for Contrastive Pretraining Samples Positive Pairs Wrong

From the previous audit (still unfixed): `align_dataset.py:181-188` samples frame and trajectory at **independent random timestamps** within an episode. This means the vision frame at t1 may be completely unrelated to the trajectory window at t2:K. For contrastive learning, positive pairs MUST be time-aligned (t1 == t2) so the model learns that a specific trajectory window corresponds to the same moment's visual observation.

---

## Structural Contradiction 6: HDF5 Noisy Dataset Expands Data ×3 Without Benefit

`create_noisy_hdf5()` generates **3 variants per episode** (light, medium, heavy noise). Each variant copies the frames (same images repeated 3×) and stores them as separate HDF5 groups. This means:
- 3× disk usage for identical image data
- Training sees the same frame 3 times per epoch with different noise levels
- No curriculum (ordering of light→heavy) — all levels mixed

The streaming pipeline handles this on-the-fly with noise rotation (`injector = injectors[step % len(injectors)]`), which is **more efficient but doesn't expose the model to multiple noise levels per sample**.

---

## Structural Contradiction 7: Checkpoint Size Waste

The pretrained checkpoint includes the full DINOv2 backbone (86M params) and CLIP text model — all frozen and never updated. The checkpoint is ~965MB. Only trainable parameters (projection heads + trajectory encoder) are actually necessary:

```
Trainable params breakdown:
  vision projection:   768→256 Linear + LayerNorm    ≈ 0.2M
  trajectory encoder:  input_proj + Transformer(3L×4H)  ≈ 0.5M
  text projection:     512→256 Linear + LayerNorm    ≈ 0.1M
  decision head:       (771→256→64→1 MLP)             ≈ 0.2M
  assistant head:      (774→256→128→30 MLP)           ≈ 0.2M
Total trainable: ~1.2M params (~5MB at float32)
```

Saving only trainable params would reduce checkpoint size by **~200×**.

---

## Summary: What Needs Simplification

| Issue | Fix |
|-------|-----|
| Two divergent pipelines | Merge HDF5 + streaming into one path; make data source a config option |
| Trajectory encoder unfrozen during head training | Add `model.traj_encoder.requires_grad_(False)` in `freeze_backbone()` or document intent |
| 3-stage head training | Replace with single joint loss (BCE + 0.5×MSE), one optimizer |
| 8 epoch parameters | Reduce to 2: `--epochs-pretrain`, `--epochs-heads` |
| Stale head weights between stages | Eliminate stages — joint training from start |
| Best checkpoint silently discarded | Load best checkpoint between stages or eliminate stages |
| Gradient clip on all params instead of optimizer group | `clip_grad_norm_(optimizer.param_groups[0]['params'], ...)` |
| Inconsistent defaults (head epochs) | Single source of truth for all defaults |
| α_target position-only in HDF5 pipeline | Port orientation-aware fix from streaming to HDF5 |
| Positive pairs misaligned | t1 = t2 for same-episode positives |
| Checkpoint 200× too large | Save only state_dict of trainable params + backbone ref |
| Stage A/B cross-attention dead code | Remove or simplify to one training pass |