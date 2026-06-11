# ALIGN: System Architecture

**Assistive Latent Intention-Guided Network (ALIGN)**

## Overview

ALIGN uses a **trilinear shared-encoder architecture** with three modalities (vision, motion, language) feeding two lightweight heads — one for deciding *when* to assist (Decision), and one for predicting *what* correction to apply (Assistant). A frozen CLIP text encoder provides task-context conditioning, enabling 3-way contrastive pretraining and semantically grounded assist gating.

```
Task Description ───▶┌──────────────────────────┐
("pick up the mug")  │  Text Encoder             │──▶ z_text (256d)
                     │  CLIP ViT-B/32 (frozen)   │
                     │  + Proj Head (512→256)   │
                     └──────────────────────────┘
                                 │
                     ┌──────────────────────────┐
Camera (224×224×3) ──▶│  Vision Encoder        │──▶ z_v (256d)
                     │  DINOv2 ViT-B (frozen)   │
                     │  + Proj Head (768→256)   │
                     └─────────────────────────-┘
                                    │
                       ┌────────────┴─────────────┐
                       │  Trajectory Encoder      │──▶ z_t (256d)
Noisy poses (K=10×6)─▶│  3-layer Transformer      │
                       │  + mean pooling          │
                       │  + Proj Head (128→256)   │
                       └────────────┬─────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
                    ▼                            ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ DECISION HEAD      │      │ ASSISTANT HEAD     │
          │ (3-layer MLP)      │      │ (3-layer MLP)      │
          │                    │      │                    │
          │ Input: [z_v, z_t,  │      │ Input: [z_v, z_t,  │
          │        z_text,     │      │        z_text,     │
          │        cos_sim_vt, │      │        noisy_pose] │
          │        cos_sim_vl, │      │                    │
          │        cos_sim_tl] │      │Output: Δposes[1..K]│
          │                    │      |   K×6: (K future   │
          │                    │      │  corrections chunk)│
          │ Output: α ∈ [0,1]  │      │                    │
          └────────┬───────────┘      └────────┬───────────┘
                   │                           │
                   └──────────────┬────────────┘
                                  ▼
                    final_pose = raw_human_pose + α · Δposes[0]
                    ───▶ IK solver → motor commands
```

## Component Details

### 1. Vision Encoder

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | DINOv2 ViT-B | Excellent general-purpose visual features, frozen for efficiency |
| Projection | Linear(768, 256) + LayerNorm | Adapts pretrained features to our task |
| Freeze status | Frozen (except proj. head) | Prevents catastrophic forgetting, enables fast iteration |
| Inference cost | ~25ms on Jetson Orin NX | Dominant latency cost, but acceptable |

### 2. Trajectory Encoder

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input shape | (K, 6) where K=10 | ~330ms window at 30Hz, enough to detect motion direction |
| Input features | (x, y, z, rx, ry, rz) | 6D EEF pose (position + axis-angle or quaternion) |
| Architecture | 3-layer TransformerEncoder | Captures temporal structure in motion |
| Hidden dim | 128 | Small enough for fast inference |
| Number of heads | 4 | Standard for this size |
| Pooling | Mean over time dimension | Aggregates temporal features into a single vector |
| Projection | Linear(128, 256) + LayerNorm | Maps to shared embedding space |
| Freeze status | Trainable | Needs to learn task-specific motion patterns |

### 3. Text Encoder (New)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | CLIP ViT-B/32 text tower | Natural language grounding, frozen |
| Input | Task description string (max 77 tokens) | e.g. "pick up the red mug", "place in the bin" |
| Projection | Linear(512, 256) + LayerNorm | Maps to shared embedding space |
| Freeze status | Frozen (except proj. head) | CLIP is pretrained on massive text; fine-tuning on ~200 episodes would destroy it |

The text encoder is called **once per task**, not per frame. The task description is fixed for an episode, so z_text is computed once and cached for the entire duration. Inference cost: ~5ms one-time.

**Three-tier task description** (order of preference):
1. Explicit: "grasp the blue bowl" — most specific, best performance
2. Implicit: "pick up the bowl" — works if only one bowl is visible
3. General: "pick and place" — reduces to vision-only (still functional)

### 4. 3-Way Contrastive Alignment (New)

Instead of a single cos_sim(z_v, z_t), we compute **three alignment scores**:

```python
cos_sim_vt = cos_sim(z_v, z_t)     # vision ↔ trajectory  (original)
cos_sim_vl = cos_sim(z_v, z_text)  # vision ↔ language     (NEW)
cos_sim_tl = cos_sim(z_t, z_text)  # trajectory ↔ language (NEW)
```

Each alignment score captures a different semantic relationship:

| Signal | High when | Low when |
|--------|-----------|----------|
| cos_sim_vt | Scene matches motion — human reaching toward a visible object | Novel scene, mid-reach switch |
| cos_sim_vl | The visible object matches the task description — e.g. sees a mug and text says "mug" | Scene has wrong object, occlusion |
| cos_sim_tl | Motion direction matches task — reaching toward the described object | Human is reaching toward wrong object |

The α gating uses all three, giving a richer confidence signal:

| Scenario | cos_sim_vt | cos_sim_vl | cos_sim_tl | α | Behavior |
|----------|-----------|-----------|-----------|----|----------|
| Normal: reaching for mug, text="mug" | ✅ High | ✅ High | ✅ High | High | Full assist |
| Wrong object: reaching for mug, text="bowl" | ✅ High | ❌ Low | ❌ Low | Low | No assist (task mismatch) |
| Novel scene: unfamiliar object, text="mug" | ❌ Low | ❌ Low | ❌ Low | Low | Safe fallback |
| Ambiguous: two mugs, text="left mug" | ✅ High | ✅ High | ✅ High | High | Text disambiguates |
| Mid-reach switch: starting toward mug, now toward bowl | Drops | Varied | Drops | Drops | Human regains control |

### 5. Decision Head

| Parameter | Value |
|-----------|-------|
| Architecture | MLP: 774 → 256 → 64 → 1 |
| Input | [z_v (256), z_t (256), z_text (256), cos_sim_vt (1), cos_sim_vl (1), cos_sim_tl (1), distance (3)] = 774 |
| Output | α ∈ [0,1] via sigmoid |
| Training target | α_target = need × capability |
| Loss | Binary cross-entropy |

The 7 input features provide complementary signals:
- **z_v, z_t, z_text**: Learned embeddings encoding scene, motion, and task semantics
- **cos_sim_vt**: How well vision explains the motion
- **cos_sim_vl**: How well the visual scene matches the task
- **cos_sim_tl**: How well the motion matches the task
- **distance**: Distances to nearest object (position + orientation) + distance to task-relevant object specifically *This is not practical nor accurate. 

### 6. Assistant Head (Chunk Output)

| Parameter | Value |
|-----------|-------|
| Architecture | MLP: 774 → 256 → 128 → (K × 6) |
| Input | [z_v (256), z_t (256), z_text (256), noisy_pose (6)] = 774 |
| Output | Δposes[1..K] — chunk of K future corrective poses |
| Default K | 5 (~165ms at 30Hz) |
| Training target | δ_t+i = smooth_pose[t+i] - noisy_pose[t] for i=1..K |
| Loss | MSE over all K outputs |
| Execution | Pose[0] is applied this timestep. Remaining K-1 cached as reference for next timestep. Re-predict every step. |

The text embedding allows the Assistant to produce **task-aware corrections**. For the same visual scene and motion, different task descriptions lead to different Δposes:

```
Same frame, same noisy_pose:
  Text="pick up the mug"   → Δpose moves hand toward mug
  Text="pick up the bowl"  → Δpose moves hand toward bowl
```

This is particularly important for **fine disambiguation** — two similar objects at similar distances, where the correction angle is the main differentiator.

### 7. Alignment Score → α Training Target (Updated)

With 3-way alignment, the capability signal becomes:

```python
capability = min(cos_sim_vt, cos_sim_vl, cos_sim_tl)
# OR:
capability = (cos_sim_vt + cos_sim_vl + cos_sim_tl) / 3
```

The **min** formulation is stricter — all three alignments must agree. This is usually safer (if any one signal is low, don't assist). The **mean** is more permissive.

**Recommended**: Use `min` during training, `mean` during deployment (slightly more conservative guard). Or simply let the Decision head learn the right combination from all 3 scores.

## Data Flow at Inference (30Hz)

```
Step Time (ms)  Component
────────────────────────────────────────
0      0       (One-time) Text → CLIP → z_text  [cached for episode]
────────────────────────────────────────
1      0       Camera frame capture
2      5       Frame → DINOv2 forward pass (GPU)
3     30       Vision projection head → z_v
4      0       Read last K noisy poses from buffer
5     32       Transformer forward → z_t
6     34       Compute cos_sims(z_v,z_t), cos_sim(z_v,z_text), cos_sim(z_t,z_text)
7     35       Decision head forward → α
8     36       Assistant head forward → chunk [Δpose_1..Δpose_K]
9     37       Apply α to chunk[0] → blend: final = noisy + α·chunk[0]
10    38       IK solver → joint targets
────────────────────────────────────────
Total ~40ms ≈ 25Hz (within 30Hz control rate)
```

On desktop GPU (RTX 3090), latency drops to ~16ms total.

### Chunk Execution Detail

```python
# Global cache persistent across timesteps
cache = None
z_text = text_encoder(task_string)  # computed ONCE per episode

def step(frame, noisy_poses_buffer):
    global cache
    
    # Encoders
    z_v = vision_encoder(frame)
    z_t = traj_encoder(noisy_poses_buffer)
    cos_vt = cos_sim(z_v, z_t)
    cos_vl = cos_sim(z_v, z_text)
    cos_tl = cos_sim(z_t, z_text)
    
    # Heads
    α = decision_head(z_v, z_t, z_text, cos_vt, cos_vl, cos_tl, distance)
    chunk = assistant_head(z_v, z_t, z_text, noisy_poses_buffer[-1])  # (K, 6)
    
    # Execute the FIRST correction
    commanded_pose = noisy_poses_buffer[-1] + α * chunk[0]
    
    # Optionally blend with cached future predictions for smoothness
    if cache is not None:
        commanded_pose = 0.7 * commanded_pose + 0.3 * (noisy_poses_buffer[-1] + α * cache[-1])
    
    cache = chunk
    return commanded_pose
```

## Safety Mechanisms

### 1. OOD Detection via Alignment

When any alignment score falls low, α is clamped:

```python
min_align = min(cos_vt, cos_vl, cos_tl)
α = α_model * min(1, max(0, (min_align - 0.2) / 0.3))
```

This is stronger than the vision-only version: even if vision↔trajectory agrees (human reaching toward *something*), if the scene doesn't match the task description, assistance is suppressed.

### 2. Text Mismatch Safety

If the operator says "pick up the mug" but the scene contains no mug:
- cos_sim_vl = low → α drops → human retains full control
- This prevents the system from "helping" toward the wrong object

### 3. Fast Exit on Target Switch

When human changes intention mid-reach:
- Trajectory shifts → cos_sim_vt drops
- The new motion doesn't match the task text → cos_sim_tl drops  
- α drops within ~200ms → human has full control

### 4. Physical Safety Limits

```python
Δpose = clip(Δpose, max_delta_position=5cm, max_delta_rotation=15°)
```

Plus joint velocity limits on the target robot.

## Text Input Interface

The text prompt is provided once per task and remains constant throughout the episode:

```
Interface options:
  1. Pre-recorded: Operator says task before starting teleop
     "pick up the red mug and place it in the bin"

  2. Keyboard/speech: During teleop, operator can speak or type new task
     Triggers re-computation of z_text (takes ~5ms)

  3. Default: If no text provided, system uses placeholder "pick and place"
     z_text becomes a neutral vector — effectively vision-only mode
```

For Phase 0 (Franka in sim), option 1 is sufficient. Option 2 is a natural upgrade.

## Comparison: 2-Modal vs. 3-Modal

| Dimension | 2-Modal (vision + trajectory) | 3-Modal (+ text) |
|-----------|-------------------------------|------------------|
| Object disambiguation | Relies on visual difference between objects | Text explicitly tells which object |
| Wrong-object detection | Not possible | Cos_sim_tl drops → system stays passive |
| Multi-step tasks | No task-phase awareness | Text can describe composite tasks |
| Data annotation | None needed | Text per episode (~5 seconds to annotate) |
| User friction | None | Millisecond-level (voice/keyboard) |
| Inference cost | ~38ms | ~40ms (one-time z_text, negligible per-step) |

The text modality adds object-level task reasoning with near-zero latency overhead (text embedding is computed once per episode and cached).

## Parameter Summary

| Component | Param Count | Trainable |
|-----------|-------------|-----------|
| DINOv2 ViT-B | ~86M | ❌ (frozen) |
| Vision projection head | ~197K | ✅ |
| Trajectory Transformer | ~830K | ✅ |
| Trajectory projection head | ~33K | ✅ |
| CLIP text tower | ~63M | ❌ (frozen) |
| Text projection head | ~131K | ✅ |
| Decision head | ~200K | ✅ |
| Assistant head | ~210K | ✅ |
| **Total trainable** | **~1.6M** | |

## Deployment Flow

```
Start
 │
 ├──► Operator provides task description (one-time)
 ├──► Compute z_text via CLIP (cached for episode)
 │
 ├──► Start camera stream (30Hz)
 │
 ├──► Loop:
 │     ├──► Read frame f_t
 │     ├──► Read noisy poses p[t-K:t] from teleop buffer
 │     ├──► z_v = vision_encoder(f_t)
 │     ├──► z_t = traj_encoder(p[t-K:t])
 │     ├──► cos_vt, cos_vl, cos_tl = compute_alignments(z_v, z_t, z_text)
 │     ├──► α = decision_head(z_v, z_t, z_text, cos_vt, cos_vl, cos_tl, dist)
 │     ├──► chunk = assistant_head(z_v, z_t, z_text, p_t)
 │     ├──► final = p_t + α * chunk[0]
 │     ├──► clip, IK, motor commands
 │     └──► Send to robot
 │
 └──► On disconnect: reset to pure teleop
```