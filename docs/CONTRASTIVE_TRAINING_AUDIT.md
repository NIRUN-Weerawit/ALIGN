# Contrastive Encoder Training Pipeline Audit

**Date:** 2026-06-10  
**Scope:** Full training pipeline for ALIGN contrastive encoders — pretraining, head training, data pipeline, loss, evaluation, streaming.

**Reviewed files:**
- `models/align_model.py` — Architecture (vision, trajectory, text encoders + heads)
- `training/contrastive_loss.py` — 3-way InfoNCE loss
- `training/pretrain.py` — HDF5-based contrastive pretraining
- `training/pretrain_streaming.py` — Streaming (LeRobot v3) pretraining + head training
- `training/train_heads.py` — HDF5-based decision/assistant head training
- `training/train_full_pipeline.py` — Full orchestration + synthetic noise
- `training/wandb_utils.py` — W&B wrapper
- `data/align_dataset.py` — HDF5 dataset + collate functions
- `data/open_dataset.py` — Robomimic/DROID/Bridge/LeRobot adapters
- `eval/eval_contrastive.py` — Evaluation of contrastive backbone
- `tests/test_contrastive_loss.py` — Unit tests for loss
- `configs/run_ktraj_sweep.py` — K_traj ablation sweep
- `configs/sweep_ktraj.yaml` — Sweep config
- `checkpoints/streaming/pretrain/best.pt` (965MB) + `streaming_training_log.jsonl`

---

## 1. Critical Issues (Behavioral Bugs)

### 1.1 Actual training run shows impossible cosine values > 1.0

**Evidence** from `checkpoints/streaming/pretrain/streaming_training_log.jsonl`:

```
epoch 1:  loss=23.16  cos_vt=25.35  cos_vl=11.68  cos_tl=-3.06
epoch 2:  loss=19.53  cos_vt=21.45  cos_vl=11.89  cos_tl=-5.33
epoch 3:  loss=11.66  cos_vt=17.65  cos_vl=11.41  cos_tl=-9.32
epoch 4:  loss=12.20  cos_vt=13.55  cos_vl=10.25  cos_tl=-10.12
epoch 5:  loss=10.16  cos_vt=11.22  cos_vl=11.90  cos_tl=-8.78
```

Cosine similarity is bounded to [-1, 1], but these values reach **25.35**. The values match `avg_cos * batch_size` (64 × ~0.4 ≈ 25.6), suggesting the metric accumulated over the batch without dividing.

**Code status:** The current `_pairwise_info_nce` in `contrastive_loss.py:98` computes `(z_a * z_b).sum(dim=-1).mean()` after L2 normalization, which is **correct**. The checkpoint was likely trained with an earlier buggy version of the metric or the image pipeline corruption (1.2) caused abnormal activations.

**Loss also abnormal:** Random InfoNCE baseline is ~ln(64) ≈ 4.16. Starting at 23.16 and ending at 10.16 after 5 epochs is suspiciously high — indicates either NaN/inf contamination, broken gradients, or corrupted inputs.

**Severity: HIGH** — The actual trained checkpoint's metrics cannot be trusted. The existing `best.pt` should be re-trained from scratch after fixing the bugs below.

### 1.2 Frame uint8 conversion in streaming training destroys image data

**Location:** `pretrain_streaming.py:169-171` (in `streaming_pretrain_collate`)

```python
if frame.dtype != torch.uint8:
    frame = frame.to(torch.uint8)
```

**Problem:** LeRobot returns float32 frames in range [0, 1]. Calling `.to(torch.uint8)` on floats in [0, 1] **truncates everything to 0 or 1** — producing near-black images. DINOv2's `VisionEncoder` then divides by 255.0, so the backbone sees all-zero input.

**Fix:** Replace with `(frame * 255).to(torch.uint8)` when frame is float in [0, 1].

**Severity: HIGH** — Renders vision encoder training completely broken when using streaming data sources.

### 1.3 Positive pair sampling in pretraining is temporally misaligned

**Location:** `align_dataset.py:181-188` (in `pretrain_collate`)

```python
t1 = np.random.randint(0, max_offset)  # frame timestamp
t2 = np.random.randint(0, max_offset)  # trajectory timestamp
```

**Problem:** The vision frame and the trajectory window are sampled at **independent random timestamps** within the same episode. The contrastive objective wants vision↔trajectory to align for the same moment — but these may be completely out of sync.

**Fix:** Use `t1 = t2` for positive pairs (temporally aligned). Draw negatives from different episodes or far-apart timestamps.

**Severity: HIGH** — Undermines the fundamental contrastive learning signal for vision↔trajectory alignment.

### 1.4 Streaming head training produces identical Δpose targets for all chunk steps

**Location:** `pretrain_streaming.py:598-599` (in `_run_stage`)

```python
for i in range(1, chunk_size + 1):
    delta_target[:, i - 1, :] = clean_6d - noisy_poses[:, :6]
```

**Problem:** The loop body is independent of `i` — every step of the chunk receives the same target (`clean_6d - noisy_poses[:, :6]`). The correct target should use `clean_poses[t+i]` for each future step.

**Compare** with the correct HDF5 version in `train_full_pipeline.py:232-235`:
```python
for i in range(1, chunk_size + 1):
    chunks[t, i - 1, :3] = smooth_poses[t + i, :3] - noisy_poses[t, :3]
```

**Fix:** Use `clean_poses[t+i] - noisy_poses[t]` where `clean_poses` varies per timestep `i`.

**Severity: HIGH** — Assistant head is trained with invalid targets that have no temporal structure.

---

## 2. Medium Severity Issues

### 2.1 torch.hub.load for DINOv2 is unstable in production

**Location:** `align_model.py:44`

```python
self.backbone = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
```

**Problem:** Contacts the network on instantiation. Can fail in air-gapped environments, return different versions across machines, or break when hub cache formats change.

**Fix:** Use `pip install dinov2` and import locally, or use HuggingFace transformers' `Dinov2Model`.

### 2.2 LeRobot get_safe_version monkey-patch is fragile

**Location:** `open_dataset.py:716-723`

```python
_ld.get_safe_version = lambda repo_id, revision: revision or "main"
```

**Problem:** Mutates `lerobot.datasets.lerobot_dataset` at module level. Affects all code that imports that module. Will silently break if lerobot fixes the underlying version-check bug in a future release.

**Fix:** Catch the version error locally in a try/except wrapper rather than monkey-patching a global function.

### 2.3 HDF5 file handle leak in ALIGNDataset

**Location:** `align_dataset.py:84`

```python
self._h5 = h5py.File(self.h5_path, "r")
```

**Problem:** No `close()` or `__del__` method. With multi-process DataLoaders, each worker gets a serialized copy holding its own handle. Open handles accumulate over long training runs.

**Fix:** Add a `close()` method and/or use a context manager pattern.

### 2.4 α_target ignores orientation error entirely

**Location:** `train_full_pipeline.py:186-187`

```python
pos_error = np.linalg.norm(noisy_poses[:, :3] - clean_poses[:, :3], axis=1)
alpha_target = np.clip(pos_error / d_max, 0.0, 1.0)
```

**Problem:** Only uses 3-DOF position error. A 30° orientation error produces α ≈ 0 if position is close — the assistant won't activate when rotational help is needed.

**Fix:** Include orientation error (axis-angle difference or quaternion distance) in the α computation.

### 2.5 Streaming head training hardcodes distances to zero
*(RESOLVED — distances removed from Decision head entirely)*

**Decision:** The Decision head's distance input was removed as an architectural
change. Distance to the target object is hard to obtain reliably in real
deployment (no per-object depth sensor in most setups), so the head must
learn "near vs far" implicitly from the visual features (DINOv2 captures
object scale and depth cues).

**Changes:**
- `models/align_model.py: DecisionHead.forward()` now takes only
  `(z_v, z_t, z_text)` — no `distances` arg
- Input dim reduced from 774 to 771
- All callers updated (train_heads.py, pretrain_streaming.py,
  inference/align_inference.py, data/align_dataset.py)
- `head_collate` no longer computes placeholder distance features
- Docstring updated to explain the deployment rationale

**Files changed:** `models/align_model.py`, `data/align_dataset.py`,
`training/train_heads.py`, `training/pretrain_streaming.py`,
`inference/align_inference.py`, `CONTRASTIVE_TRAINING_AUDIT.md`.

### 2.6 Streaming head training lacks temporal position sampling

**Location:** `pretrain_streaming.py:571-605`

**Problem:** `_run_stage` processes each streaming sample as one training point, without sampling different temporal positions within the episode. The HDF5 `head_collate()` randomly samples a position `t` for variety.

**Fix:** Add random temporal offset sampling in the streaming collate function.

---

## 3. Minor Issues & Code Smells

### 3.1 No ep_id uniqueness enforcement in pretrain batch

`pretrain_collate` returns `ep_ids` but never verifies that all samples in a batch come from different episodes. Same-episode samples become false negatives.

### 3.2 MultiDatasetStream lacks `__len__`

**Location:** `pretrain_streaming.py:52`

As an `IterableDataset` with no `__len__`, DataLoader's `drop_last=True` behavior is undefined and may affect prefetching.

### 3.3 Test coverage is incomplete

`test_contrastive_loss.py` covers the loss well but misses:
- Positive/negative pair sampling (`pretrain_collate`, `head_collate`)
- Full training step (forward + backward + clip)
- Checkpoint save/load correctness
- Streaming collate functions
- NaN/Inf stability across modalities

### 3.4 W&B init has repeated code pattern

Both `pretrain.py` and `pretrain_streaming.py` call `init_wandb()` with empty config then overwrite. Could use a lazy-init pattern.

### 3.5 Mean pooling discards trajectory sequence structure

**Location:** `align_model.py:125`

```python
x = x.mean(dim=1)  # mean pooling over time
```

Learnable attention pooling or a [CLS] token would preserve temporal ordering (important for "approach before grasp" tasks).

### 3.6 Large checkpoint includes frozen backbone

**965MB** for `best.pt` includes the full frozen DINOv2 ViT-B (~86M params). Saving only trainable parameters (~few MB) would be faster to load and more distributable.

---

## 4. Priority Summary Table

| Priority | Issue | File | Status |
|----------|-------|------|--------|
| **HIGH** | Cosine values > 1.0 + abnormally high loss in checkpoint | `training_log.jsonl` | ✅ Code fixed (L2 normalize, scale-invariant). Old `best.pt` must be re-trained. |
| **HIGH** | Frame uint8 conversion truncates images | `pretrain_streaming.py:170-181` | ✅ Fixed — multiply by 255 when float [0,1] before casting to uint8 |
| **HIGH** | Δpose targets always identical across chunk steps | `pretrain_streaming.py:594-625` | ✅ Fixed — real K-step action window from LeRobot `delta_timestamps` |
| **HIGH** | Positive pairs temporally misaligned | `align_dataset.py:178-188` | ✅ Fixed — `t1 = t2` (same anchor) for positive pairs |
| **MED** | torch.hub.load unstable | `align_model.py:44` | ❌ Open — accepted trade-off for now |
| **MED** | LeRobot monkey-patch fragile | `open_dataset.py:716-723` | ❌ Open — needs try/except wrapper |
| **MED** | HDF5 handle leak | `align_dataset.py:84` | ✅ Fixed — added `close()`, `__enter__`/`__exit__`, `__del__` |
| **MED** | α_target ignores orientation | `train_full_pipeline.py:186` | ❌ Open — needs axis-angle distance term |
| **MED** | dists hardcoded to zero in streaming head training | `pretrain_streaming.py:607` | ✅ **Removed entirely** — Decision head no longer takes distances |
| **LOW** | No ep_id uniqueness check | `align_dataset.py` | ❌ Open |
| **LOW** | Missing `__len__` on streaming dataset | `pretrain_streaming.py:52` | ❌ Open |
| **LOW** | Incomplete test coverage | `tests/` | ❌ Open |
| **LOW** | Large checkpoint includes frozen backbone | `checkpoints/` | ❌ Open |
| **LOW** | Mean pooling discards trajectory sequence | `align_model.py:125` | Reduced temporal awareness |

---

## 5. Performance & Architecture Observations

### Architecture
- **DINOv2 ViT-B/14** vision backbone (~86M params, frozen) is a reasonable choice. Strong visual features transfer well for robotic manipulation.
- **3-layer, 4-head, d=128 Transformer** for trajectory encoding at K=20 has O(K²) cost. A 1-second window at 20fps (200 tokens) is computationally fine. At K=50 (2.5s) cost would be ~6× higher.

### Checkpoint size
- **965MB** includes the full frozen DINOv2 backbone. Saving only trainable parameters (projection heads ~a few MB, trajectory encoder ~500KB) would drastically reduce size and load time.

### Trajectory pooling
- Mean pooling discards time information. Attention pooling or a learned [CLS] token would better capture temporal structure for sequential tasks.

### Training quality
- Loss went from 23.16 → 10.16 over 5 epochs (~56% reduction), but 10.16 is still ~2.4× above random baseline (ln(64) ≈ 4.16). Likely causes:
  - (a) Temperature τ=0.07 may be too low — near-one-hot logits saturate softmax and kill gradients
  - (b) Very limited training (5 epochs)
  - (c) The uint8 frame bug (1.2) was almost certainly active, corrupting vision input
  - (d) Starting loss of 23 (several times baseline) suggests NaN/inf or broken gradients at init

**Bottom line:** The training run that produced `best.pt` was fundamentally corrupted. Fix bugs 1.2, 1.3, 1.4 before re-training from scratch. Consider also tuning temperature and verifying gradient flow (no NaN, no dead ReLUs).