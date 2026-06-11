# ALIGN: Training Pipeline

**Assistive Latent Intention-Guided Network (ALIGN)**

## Overview

ALIGN uses a single dataset collected from human teleoperation, with **per-episode text annotations** providing task-level grounding. Training proceeds in four stages:

1. **Data Collection** — raw teleop episodes with camera + noisy poses + smooth reference + text
2. **Ground Truth Generation** — smooth trajectories + α_target from 3-way alignment
3. **3-Way Contrastive Pretraining** — align vision, trajectory, and text embeddings
4. **Joint Head Training** — train Decision + Assistant heads on frozen backbone

---

## Stage 1: Data Collection

### Environment Setup (Phase 0 — Isaac Sim Franka)

```
┌──────────────────────────────────────────┐
│  Table surface                           │
│   ┌────┐  ┌───┐  ┌─────┐                │
│   │mug │  │bowl│  │ can │  [3-5 objects] │
│   └────┘  └───┘  └─────┘                │
│                                          │
│              Franka Panda               │
│              (fixed mount)               │
│                   │                      │
│              ┌────┴────┐                 │
│              │ Wrist   │                 │
│              │ Camera  │  (egocentric)   │
│              │ 224×224 │                 │
│              └─────────┘                 │
└──────────────────────────────────────────┘
```

### Episode Recording

Each episode records at 30Hz:

```python
{
  "episode_id": 42,
  "objects_on_table": ["mug_red", "bowl_blue", "can_soup"],
  "target_object": "mug_red",
  "task_description": "pick up the red mug",     # ← NEW: per-episode text
  "frames": [
      {
          "t": 0.0,
          "camera_frame": <224×224×3 RGB>,       # wrist cam
          "noisy_pose": [0.35, 0.12, 0.45, ...],  # 6D EEF (pos+orn)
          "smooth_pose": [0.35, 0.12, 0.45, ...], # computed offline
          "gripper_state": 0.0,
          "object_poses": {
              "mug_red": [0.55, 0.10, 0.0, ...],
              "bowl_blue": [0.20, -0.15, 0.0, ...],
              "can_soup": [0.10, 0.25, 0.0, ...]
          }
      },
      ...
  ]
}
```

### Text Annotation Templates

To cover the range of descriptions the system will encounter at deployment:

```python
# Each episode generates multiple text variants for robustness:
text_variants = [
    # Specific (best)
    f"pick up the {object_color} {object_type}",
    f"grasp the {object_color} {object_type}",
    f"reach for the {object_type}",
    
    # Location-based (when color isn't unique)
    f"pick up the {object_type} on the {location}",
    
    # General (fallback — still functional, just less disambiguating)
    "pick and place the object",
    "grasp and move",
]
```

For training, we randomly select one variant per epoch — the model learns to handle varying specificity.

### Noise Injection

Human teleoperation is inherently noisy, but we also add synthetic noise to augment the training data:

| Noise Type | Parameters | Rationale |
|-----------|------------|-----------|
| Gaussian jitter | σ = 1-3cm pos, σ = 2-5° orn | Simulate tracking noise and imprecision |
| Hand tremor | 8-12 Hz sinusoidal oscillation at ±5mm | Simulate physiological tremor |
| Fatigue ramp | Noise amplitude grows 2× over episode | Simulate operator fatigue |
| Stick-slip jitter | Random jumps of 2-5mm at low probability | Simulate VR tracking glitches |

### Object Variety

| Object Set | Number | Examples |
|-----------|--------|---------|
| Training objects | 8-10 | Red mug, blue bowl, soup can, box, bottle, apple, sponge, cup, plate, ball |
| Held-out objects | 3-5 | Triangular prism, translucent cup, cloth, battery pack, rubber toy |
| Novel objects (test) | 5+ | Anything not in training or held-out — measures generalization |

### Data Targets

| Metric | Training | Test |
|--------|----------|------|
| Episodes per object | ~20 | ~10 |
| Total episodes | 200-250 | 50-60 |
| Total frames | ~30,000 (200 ep × 150 frames) | ~7,500 |
| Episodes per configuration | 3-5 (randomize object placement) | 3-5 |
| Operator variation | 2-3 operators | 1-2 operators |
| Text annotation cost | ~5 seconds per episode | Same |

---

## Stage 2: Ground Truth Generation

We compute the "smooth" trajectory from the collected raw data using a hybrid method:

### Transit Phase (far from object, d > 8cm)

```python
from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter to the full noisy trajectory
smooth_transit = savgol_filter(noisy_traj, window_length=11, polyorder=3)
```

### Approach Phase (near object, d < 8cm)

**Current (default):** Quintic polynomial interpolation from current hand pose → grasp pose.
Deterministic, smooth acceleration profile, zero-phase. Assumes straight-line approach —
covers 99% of cases in clutter-free tabletop scenes.

```python
# Take the final detected grasp pose as the goal
grasp_pose = noisy_poses[-1]  # or: detect_grasp_pose(object_type, camera_frame)
smooth_approach = quintic_interpolation(noisy_pose_current, grasp_pose)
```

**Future:** DMP-based approach phase. Record one expert approach trajectory per object type,
fit a Dynamic Movement Primitive to encode the approach style (arc, wrist rotation).
At deployment, the attractor landscape adapts to arbitrary start/goal poses. By construction,
DMPs produce human-like paths, converge deterministically, and execute instantly.
Bi-RRT + shortcut smoothing serves as collision-fallback when the DMP path hits obstacles.

```python
# Future pipeline:
dmp_path = dmp_plan(start_pose, grasp_pose, object_type)  # learned forcing term
smooth_approach = dmp_path if is_collision_free(dmp_path) else birrt_plan(start, grasp)
```

### Blending

```python
blend_weight = smooth_step(distance, d_start=0.10, d_end=0.05)
smooth_pose = (1 - blend_weight) * smooth_transit + blend_weight * smooth_approach
```

### Decision Head Ground Truth (α_target)

For each timestep, we compute the supervision signal for the Decision head:

```python
# Need: how much does the human deviate from the smooth trajectory?
need = clip(||noisy_pose - smooth_pose|| / D_MAX, 0, 1)
# D_MAX = 0.10m (normalized by max tolerable deviation)

# Capability: min of all 3 alignment scores (after contrastive pretraining)
capability = min(cos_sim_vt, cos_sim_vl, cos_sim_tl)
# All three must agree for the system to be confident

# α_target: multiplicative combination
α_target = need × capability

# Rationale: α should be high ONLY when BOTH:
#   (1) The human's trajectory needs correction (high need)
#   (2) ALL modalities agree on what's happening (high capability)
```

| Scenario | need | cos_vt | cos_vl | cos_tl | capability | α_target |
|----------|------|--------|--------|--------|------------|----------|
| Normal: reaching for mug, text="mug" | 0.8 | 0.9 | 0.9 | 0.9 | 0.9 | 0.72 |
| Wrong object: reaching for mug, text="bowl" | 0.8 | 0.9 | 0.3 | 0.2 | 0.2 | 0.16 |
| Novel object + general text | 0.8 | 0.3 | 0.4 | 0.3 | 0.3 | 0.24 |
| Two objects, same distance, text disambiguates | 0.7 | 0.8 | 0.8 | 0.85 | 0.8 | 0.56 |
| Mid-reach switch | 0.6 | drops | stays | drops | lows | low |
| Smooth, no correction needed | 0.1 | 0.9 | 0.9 | 0.9 | 0.9 | 0.09 |

### Assistant Head Ground Truth

```python
# Target: chunk of K consecutive smooth future poses
chunk_size = 5  # ~165ms at 30Hz
for t in range(len(episode) - chunk_size):
    Δpose_chunk_target = []
    for i in range(1, chunk_size + 1):
        Δpose_target = smooth_pose[t+i] - noisy_pose[t]
        Δpose_chunk_target.append(Δpose_target)
    # Shape: (K, 6) — K future corrections from current noisy_pose
```

---

## Stage 3: 3-Way Contrastive Pretraining

### Architecture

```python
# Modality 1: Vision (frozen backbone)
vision_encoder = DINOv2ViT("dinov2_vitb14", freeze_backbone=True)
vision_proj = nn.Sequential(
    nn.Linear(768, 256),
    nn.LayerNorm(256)
)

# Modality 2: Trajectory (fully trainable)
traj_encoder = TransformerEncoder(
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=512
)
traj_proj = nn.Sequential(
    nn.Linear(128, 256),
    nn.LayerNorm(256)
)

# Modality 3: Text (frozen backbone)  ← NEW
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_proj = nn.Sequential(
    nn.Linear(512, 256),
    nn.LayerNorm(256)
)

trainable_params = [
    vision_proj,
    traj_encoder,
    traj_proj,
    text_proj
]
```

### Positive / Negative Pair Construction (3-Way)

From one episode of N frames:

```
Vision ↔ Trajectory pairs:
  Positives: (frame[i], traj[j]) where |i - j| ≤ W_pos
  Negatives: all other (frame[i], traj[j]) in batch where |i - j| > W_pos

Vision ↔ Text pairs:
  Positives: (frame[i], text[ep]) — ALL frames within an episode
             match that episode's task description
  Negatives: (frame[i], text[other_ep]) — frames from one episode
             paired with text from a different episode

Trajectory ↔ Text pairs:
  Positives: (traj[i], text[ep]) — ALL trajectory windows within
             an episode match that episode's text
  Negatives: (traj[i], text[other_ep]) — trajectory from one
             episode vs. text from a different episode
```

### Batch Construction

```python
batch_size = 64
episodes_per_batch = 8
frames_per_episode = 8

# Each batch contains:
#   - 64 vision embeddings       (one per frame)
#   - 64 trajectory embeddings   (one per trajectory window)
#   - 8 text embeddings          (one per episode, broadcast to 64)
```

### 3-Way InfoNCE Loss

```python
def contrastive_loss_pair(z_a, z_b, temperature=0.07):
    """Standard InfoNCE between two modalities."""
    logits = (z_a @ z_b.T) / temperature
    labels = torch.arange(len(z_a))
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    return (loss_a2b + loss_b2a) / 2

def contrastive_loss_3way(z_v, z_t, z_text, temperature=0.07):
    """3-way InfoNCE: average of all pairwise losses."""
    loss_vt = contrastive_loss_pair(z_v, z_t, temperature)
    loss_vl = contrastive_loss_pair(z_v, z_text, temperature)
    loss_tl = contrastive_loss_pair(z_t, z_text, temperature)
    return (loss_vt + loss_vl + loss_tl) / 3
```

### Pretraining Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 64-128 |
| Temperature | 0.07 |
| K (trajectory window) | 10 frames (~330ms at 30Hz) |
| W_pos (positive window) | 5 frames (~165ms) |
| Embedding dim | 256 |
| Learning rate | 1e-4 (AdamW) |
| Weight decay | 1e-4 |
| Epochs | 50-100 (converges in 20-30) |
| GPU memory | ~9GB (batch=64) |

### Convergence Check

```python
# Within-episode alignment
z_v[i] · z_t[i] → 0.7-0.9  ✓
z_v[i] · z_text[ep] → 0.7-0.9  ✓
z_t[i] · z_text[ep] → 0.7-0.9  ✓

# Cross-episode alignment
z_v[ep_A] · z_t[ep_B] → 0.0-0.2  ✓
z_v[ep_A] · z_text[ep_B] → 0.0-0.2  ✓

# Text mismatch (e.g., "mug" text with bowl frame)
z_v[bowl_frame] · z_text["pick up the mug"] → <0.3  ✓

# Novel object + generic text
z_v[novel_object] · z_text["pick and place"] → <0.4  ✓
```

If these hold, the encoders are ready for head training.

---

## Stage 4: Joint Head Training

After contrastive pretraining, all three encoders and projection heads are frozen. Two lightweight MLP heads are trained:

### Decision Head

```python
decision_head = nn.Sequential(
    nn.Linear(256 + 256 + 256 + 1 + 1 + 1 + 3, 256),  # z_v + z_t + z_text + 3×cos_sim + 3×distance
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

loss_fn = F.binary_cross_entropy  # α_target ∈ [0, 1]
optimizer = AdamW(decision_head.parameters(), lr=1e-4)
```

### Assistant Head

```python
assistant_head = nn.Sequential(
    nn.Linear(256 + 256 + 256 + 6, 256),  # z_v + z_t + z_text + noisy_pose
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, K * 6)  # chunk of K future Δposes
)

loss_fn = F.mse_loss  # Δposes_target[i] = smooth[t+i] - noisy[t]
optimizer = AdamW(assistant_head.parameters(), lr=1e-4)
```

### Training Schedule

```python
# Epochs 1-10: Train Decision head only
for epoch in range(10):
    for batch in dataloader:
        with torch.no_grad():
            z_v, z_t, z_text = encoders(batch.frames, batch.trajectories, batch.texts)
        cos_vt = cos_sim(z_v, z_t)
        cos_vl = cos_sim(z_v, z_text)
        cos_tl = cos_sim(z_t, z_text)
        α_pred = decision_head(z_v, z_t, z_text, cos_vt, cos_vl, cos_tl, batch.distances)
        loss = BCE(α_pred, batch.α_target)
        loss.backward()
        optimizer.step()

# Epochs 11-30: Train Assistant head only
for epoch in range(10, 30):
    for batch in dataloader:
        with torch.no_grad():
            z_v, z_t, z_text = encoders(batch.frames, batch.trajectories, batch.texts)
        Δpose_pred = assistant_head(z_v, z_t, z_text, batch.noisy_pose)
        loss = MSE(Δpose_pred, batch.Δpose_target)
        loss.backward()
        optimizer.step()

# Epochs 31-50: Fine-tune both heads together (optional)
for epoch in range(30, 50):
    for batch in dataloader:
        with torch.no_grad():
            z_v, z_t, z_text = encoders(batch.frames, batch.trajectories, batch.texts)
        cos_vt = cos_sim(z_v, z_t)
        cos_vl = cos_sim(z_v, z_text)
        cos_tl = cos_sim(z_t, z_text)
        α_pred = decision_head(z_v, z_t, z_text, cos_vt, cos_vl, cos_tl, batch.distances)
        Δpose_pred = assistant_head(z_v, z_t, z_text, batch.noisy_pose)
        loss = BCE(α_pred, batch.α_target) + 0.5 * MSE(Δpose_pred, batch.Δpose_target)
        loss.backward()
        optimizer.step()
```

---

## Full Training Timeline

| Step | Duration | GPU Memory |
|------|----------|------------|
| Data collection (200 episodes) | 1-2 days | — |
| Text annotation | ~20 minutes | — |
| Ground truth generation | ~4 hours (offline) | — |
| 3-way contrastive pretraining | ~1-2 days | ~9GB |
| Decision head training | ~2 hours | ~4GB |
| Assistant head training | ~4 hours | ~4GB |
| Joint fine-tuning | ~4 hours | ~4GB |
| Evaluation + ablations | 1-2 days | ~4GB |
| **Total** | **~6-8 days** | |

## Key Design Choices

1. **One dataset for both heads**: The Decision head's α_target is derived from 3-way alignment (capability) + data (need). The Assistant head's Δpose_target is derived from smooth ground truth. Both come from the same recorded episodes — text annotations add ~20 minutes of labeling overhead.

2. **Frozen encoders during head training**: Keeps heads lightweight and ensures the embedding space is stable. Heads can be retrained without touching the backbone.

3. **Staged head training**: Training Decision first prevents the Assistant from dominating, and training Assistant second ensures dedicated capacity.

4. **Min-capability for safety**: Using `min(cos_vt, cos_vl, cos_tl)` as capability means all three modalities must agree for high α — creating a strong safety guard. If any modality is misaligned (text says "mug" but scene shows "bowl"), assistance is suppressed.

5. **One-time text embedding at inference**: The text encoder is only called once per task (not per frame), so the added cost is ~5ms one-time. The per-frame cost is unchanged from the vision-only version.

6. **Multiple text variants per episode**: Training with "pick up the red mug", "grasp the mug", and "pick and place" makes the system robust to varying description specificity at deployment.