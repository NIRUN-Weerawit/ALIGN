# ALIGN: Design Decisions

**Assistive Latent Intention-Guided Network (ALIGN)**

This document records key architectural and design decisions made during the development of ALIGN, the alternatives considered, and the rationale for each choice.

---

## Decision 1: Shared Encoder vs. Separate Models

**Question**: Should the Decision head and Assistant head use separate encoders or share one?

**Options**:
- **Option A (Separate)**: Two independent vision encoders, two trajectory encoders. Independent computation.
- **Option B (Hybrid Shared)**: One vision encoder + one trajectory encoder → shared embeddings → two lightweight heads.

**Chosen**: **Option B (Hybrid Shared)**

| Reason | Explanation |
|--------|-------------|
| **Latency** | One encoder forward pass vs. two ~ 26Hz vs. 16Hz on G1's Jetson Orin → 30Hz is barely within reach with one, impossible with two |
| **Multi-task learning** | Contrastive + decision + regression losses create richer features than any single objective |
| **Data efficiency** | Each training example contributes gradients to the encoder from both heads |
| **Architectural elegance** | Clean ablate: one component that serves two purposes |

**Downside accepted**: Encoder changes affect both heads. Mitigated by freezing the encoder after pretraining — heads can be independently retrained.

---

## Decision 2: Contrastive Alignment vs. Pre-Defined Goal Sets

**Question**: How should the system determine which object the human is reaching for?

**Options**:
- **Goal set (Javdani 2018)**: Object detector + pose estimator → discrete set of candidate goals → compute probability per goal
- **Contrastive alignment (Ours)**: Embed vision and trajectory into shared space → cos_sim as confidence

**Chosen**: **Contrastive alignment**

| Reason | Explanation |
|--------|-------------|
| **Novel objects** | Goal set approach needs retraining for new objects; contrastive works on unseen objects via semantic similarity |
| **No NMS/corner case handling** | No object detector → no missed detections, false positives, or non-max suppression tuning |
| **Continuous confidence** | cos_sim is inherently continuous — no artificial discretization or softmax over pre-defined categories |
| **Unified representation** | The same embeddings power both the gating and the correction |

**Downside accepted**: Contrastive embeddings are less interpretable than explicit object labels. The system learns "this looks like the objects I've seen paired with reaching motions" rather than "this is a mug."

---

## Decision 3: No Entropy-Based Gating

**Question**: Should we add an entropy signal (ensemble variance or goal-entropy) to the confidence calculation?

**Options**:
- **With entropy**: α = cos_sim × (1 - H/ H_max) where H is prediction uncertainty
- **Without entropy**: α = f(cos_sim, distance, embeddings)

**Chosen**: **Without entropy**

| Reason | Explanation |
|--------|-------------|
| **Redundancy** | cos_sim and ensemble entropy are highly correlated — when the system is uncertain about the visual scene, both cos_sim and assistant certainty drop together |
| **Compute cost** | Ensembles multiply inference cost 3-5× for minimal gain |
| **Clean paper story** | One signal (contrastive alignment) serves as both gating and OOD detection — simpler narrative |
| **Ablation evidence** | Can show that adding entropy doesn't improve results, making the system minimal-and-sufficient |

**Downside accepted**: No second opinion. If cos_sim is pathologically wrong (e.g., encoder failure), there's no redundant safety signal. Mitigated by a low-α fallback when the encoder produces NaN or frozen activations.

---

## Decision 4: Distance as Decision Head Input

**Question**: Should the Decision head receive distance-to-object or rely purely on learned features?

**Options**:
- **No distance**: Decision head only gets z_v, z_t, cos_sim
- **With distance**: Decision head also gets distance to nearest object

**Chosen**: **With distance**

| Reason | Explanation |
|--------|-------------|
| **Strong prior** | Distance provably correlates with task phase. The model should not have to learn this from scratch |
| **Small input cost** | 2 additional scalar dimensions (position distance + orientation distance) is negligible |
| **Better near-far differentiation** | cos_sim can be high even when far from an object (decisive reach across the table). Distance prevents premature assist |

**Downside accepted**: Introduces a minimal dependency on an object detector for distance calculation. Mitigated: "distance to nearest object" can be a simple raycast/point cloud query from the wrist camera depth, not a learned detector.

---

## Decision 5: Per-Step Correction vs. Chunk Output

**Question**: Should the Assistant head output a single Δpose or a chunk of K future poses?

**Options**:
- **Per-step Δpose**: Single 6D correction. Smoothness comes from α blending + trajectory encoder context.
- **Chunk (K poses)**: K consecutive 6D corrections. Inherently enforces temporal coherence.

**Chosen**: **Chunk output (default K=5)**

| Reason | Explanation |
|--------|-------------|
| **Built-in smoothness** | The model must output a coherent sequence — K poses that form a smooth arc toward the target. This is stronger than relying on α low-pass filtering. |
| **Closed-loop by design** | Execute chunk[0], re-predict at each timestep. The sliding window creates natural feedback: new observations update the plan. |
| **K=1 = per-step mode** | Chunk size is a parameter, not an architectural change. Can collapse to single-step Δpose instantly. |
| **Consistency with literature** | Diffusion Policy (Chi 2023) uses K=16 action chunks. Our K=5 is smaller (pick-and-place needs shorter horizons) but follows the same principle. |

**Downside accepted**: Slightly more compute (K×6 output dims instead of 6) — negligible for an MLP head.

---

## Decision 6: Frozen vs. Trainable Vision Encoder

**Question**: Should the vision backbone (DINOv2) be frozen or fine-tuned?

**Options**:
- **Frozen**: DINOv2 weights untouched, only projection head trains
- **Fine-tuned**: End-to-end backpropagation through DINOv2

**Chosen**: **Frozen**

| Reason | Explanation |
|--------|-------------|
| **Catastrophic forgetting** | Fine-tuning DINOv2 on our small dataset (~30K frames) would destroy its general-purpose features |
| **Compute budget** | DINOv2 backprop would need larger GPU memory and longer training |
| **Head independence** | Frozen encoder → heads can be iterated independently of the backbone |

**Downside accepted**: DINOv2 features might not be perfectly optimal for our specific object set. In practice, DINOv2 generalizes well enough that this is not a limitation.

---

## Decision 7: Phase 0 on Franka Panda (Not G1)

**Question**: Should development start on the full G1 humanoid or on a simpler arm platform?

**Options**:
- **G1 (arm only)**: Lock lower body, use just the 7-DOF arm
- **Franka Panda (sim)**: Fixed-base 7-DOF arm in Isaac Sim

**Chosen**: **Franka Panda in Isaac Sim**

| Reason | Explanation |
|--------|-------------|
| **Development speed** | No hardware boot/warmup cycles, no safety concerns, instant reset after failures |
| **Simpler IK** | Fixed base = no balancing, no leg coordination |
| **Isolate core contribution** | Gating + assist can be validated on an arm without confounding variables (base motion, stance switching) |
| **Transfer** | The architecture is arm-agnostic. Same head designs, same embedding dimensions, similar latency profile |

**Downside accepted**: Results on Franka need transfer validation on G1 for the paper to fully claim humanoid deployment. The transfer experiment is Phase 5.

---

## Decision 8: Ground Truth via Hybrid Smoothing (BaG + Motion Planner)

**Question**: How to produce ground-truth smooth trajectories from noisy teleop data?

**Options**:
- **SavGol only**: Fast, continuous, but may violate joint limits
- **Motion planner only**: Kinematically feasible, but ignores human path preferences (shortest path vs. intended path)
- **Expert re-demonstration**: Highest quality, but doubles collection time and introduces variability

**Chosen**: **Hybrid: SavGol (transit) + Motion planner (approach)**

| Reason | Explanation |
|--------|-------------|
| **Respects human intent** during transit (human chose that path, don't overrule it) | SavGol simply smooths, doesn't replan |
| **Ensures feasible approach** near the object (kinematic correctness matters for grasp) | Motion planner guarantees IK feasibility within 5cm of the target |
| **Simple to implement** | SavGol is a one-liner; motion planner is available in Isaac Sim (OMPL) |

---

## Decision 9: No Separate Pre-Defined α Thresholds

**Question**: Should α have hard thresholds (e.g., α < 0.3 → no assist)?

**Options**:
- **Continuous**: α = head output, used directly in blending
- **Thresholded**: α below 0.3 clamped to 0. α above 0.7 clamped to 1

**Chosen**: **Continuous (no hard thresholds)**

| Reason | Explanation |
|--------|-------------|
| **Smoother user experience** | Abrupt assist switches feel unnatural and cause aggressive correction |
| **Probability-like semantics** | α is the model's confidence — hard thresholds discard information |
| **Safety layer is separate** | We add a soft safety clamp (cos_sim < 0.2 → α aggressively reduced) but this is a gradual ramp, not a hard threshold |

---

## Decision 10: Staged Head Training

**Question**: Should both heads be trained simultaneously or sequentially?

**Options**:
- **Joint**: Both losses applied in every batch
- **Staged**: Train Decision first, then Assistant, then fine-tune jointly

**Chosen**: **Staged (Decision → Assistant → Joint)**

| Reason | Explanation |
|--------|-------------|
| **Prevents loss domination** | α is a single scalar (small loss); Δpose is K×6 (larger loss). Joint from start would ignore the Decision head |
| **Cleaner convergence** | Decision head converges quickly (~10 epochs), then Assistant has dedicated capacity |
| **Selective fine-tuning** | The final joint phase allows cross-head regularization |

---

## Decision 11: Adding a Third Modality (Text)

**Question**: Should we add task descriptions as a third contrastive modality alongside vision and trajectory?

**Options**:
- **2-modal (vision + trajectory)**: No text. α = f(cos_sim(z_v, z_t)). Simple, no user friction.
- **3-modal (+ text)**: CLIP text encoder, 3-way contrastive, 3 alignment scores.

**Chosen**: **3-modal (vision + trajectory + text)**

| Reason | Explanation |
|--------|-------------|
| **Object disambiguation** | Two identical-looking objects at similar distances — vision+trajectory can't tell them apart. Text explicitly specifies "the left mug" or "the red one" |
| **Wrong-object detection** | If text says "bowl" but human reaches for mug, cos_sim_tl drops → α drops → system doesn't fight the human. This is a genuine safety improvement. |
| **Task-conditional corrections** | Same visual scene + same noisy pose but different task → different Δposes. The assistant knows whether you're reaching to grasp vs. reaching to push. |
| **Negligible compute cost** | Text embedding is computed once per task (not per frame). ~5ms one-time, cached for the episode. |
| **3-way min capability is safer** | Using `min(cos_vt, cos_vl, cos_tl)` as the capability signal means all three modalities must agree. Stricter = fewer false-positive assists. |

**Downsides accepted**:
- User must provide text before or during teleop (voice or keyboard). Adds minimal friction.
- Text annotation overhead: ~5 seconds per training episode. Total ~20 minutes for 200 episodes. Negligible.
- CLIP text tower adds ~63M frozen params. No runtime cost (computed once, cached).

**Text variation during training**: Each episode is annotated with multiple text variants (specific → general). The model learns to handle both "pick up the red mug" and "pick and place," generalizing across specificity levels.

---

## Decision 12 (Future): Optional Autonomous Head

**Question**: Can ALIGN also operate as a fully autonomous system (α=1)?

**Current answer**: Theoretically yes, but α=1 at inference would feed the system's own smooth outputs back into the trajectory encoder, creating a distribution shift (the encoder was trained on noisy human trajectories, not smooth self-generated ones). This limits autonomous operation to short horizons (~2-3 timesteps).

**Future extension (not implemented yet)**: Add an optional autonomous head that operates alongside the existing Decision + Assistant heads, all sharing the same frozen 3-modal backbone:

```
Same shared encoder (z_v, z_text)
          │
          ├── Decision → α (for shared autonomy mode)
          │
          ├── Assistant → Δpose (for shared autonomy mode)  
          │
          └── Autonomous → absolute_pose (for α=1 mode)  ← NEW (optional)
```

| Property | Autonomous head |
|----------|----------------|
| Input | [z_v (256), z_text (256)] |
| Output | Absolute EEF pose (6D) |
| Training data | Demonstration trajectories (expert demos or smooth teleop) |
| Training target | Ground-truth smooth pose at timestep t |
| Role | Takes over fully when human releases control or α → 1 |

**Key benefit**: The shared backbone means the autonomous head benefits from the same rich 3-modal representation without additional encoder cost. The same vision encoder and text encoder serve all three heads.

**When to add**: After the core assistive system is validated and the paper is drafted — as a clear "spectrum of autonomy" contribution for a follow-up or journal extension.

---

## Decision 13: Synthetic Noise Injection for Training Data

**Question**: Should synthetic noise (Gaussian jitter, tremor, fatigue ramp) be added to recorded teleoperation data?

**Options**:
- **Yes**: Add Gaussian noise + tremor on top of real human teleop noise during collection
- **No**: Record raw human teleop as-is — the real human motion is already noisy enough

**Chosen**: **No (synthetic noise NOT used in default collection)**

| Reason | Explanation |
|--------|-------------|
| **Real noise is sufficient** | Physiological hand tremor (±1-5mm), VR tracking jitter (±1-2mm), arm instability (±5-15mm), and operator fatigue are all naturally present in real teleoperation |
| **Distribution mismatch** | Synthetic noise (Gaussian, sinusoidal) has a different statistical profile than real teleop noise. The model would learn to correct synthetic patterns that don't exist in deployment |
| **Wasted capacity** | The model has finite capacity — using it to learn "how to remove Gaussian jitter" consumes parameters that could be spent on understanding real motion |

**Retained as evaluation tool**: The `align_noise.py` module is kept for stress-testing and ablation experiments where controlled noise levels are useful for measuring system robustness.

---

## Decision 14 (Future): Latency Mitigation via Chunk Index Shift

**Question**: Can ALIGN's chunk output be used to compensate for teleoperation latency (MQTT + network + processing delay)?

**Current answer**: Yes, naturally — the Assistant head outputs a chunk of K=5 future corrective poses (~165ms horizon). If the system's round-trip latency is L ms, you simply shift the execution index:

```python
latency_steps = int(measured_latency_ms / dt)
final_pose = noisy_pose + alpha * chunk[latency_steps]
```

This lets the robot execute a "future" correction that compensates for the delay, making the response feel instantaneous.

| Latency | Chunk steps needed | Horizon |
|---------|-------------------|---------|
| 0-33ms | K=1 (chunk[0]) | Immediate |
| 33-100ms | K=3 | ~100ms |
| 100-165ms | K=5 | ~165ms (default) |
| 165-330ms | K=10 | ~330ms |

**Key benefit**: ALIGN does both latency compensation and motion smoothing in one forward pass. The chunk output serves double duty — no separate predictor needed.

**When to implement**: After the core system is validated. The chunk already exists; this is a deployment-time index shift, not an architecture change. Add as a section in the paper's discussion on practical deployment.

---

## Summary: Decisions at a Glance

| # | Decision | Chosen | Main Driver |
|---|----------|--------|-------------|
| 1 | Encoder sharing | **Hybrid** | 30Hz latency requirement |
| 2 | Goal representation | **Contrastive** | Novel object generalization |
| 3 | Entropy gating | **No** | Redundancy with cos_sim |
| 4 | Distance input | **Yes** | Strong prior for task phase |
| 5 | Output mode | **Chunk (K=5)** | Built-in trajectory smoothness |
| 6 | Vision backbone | **Frozen** | Prevent catastrophic forgetting |
| 7 | Starting platform | **Franka (sim)** | Development speed |
| 8 | Ground truth | **Hybrid (SavGol + planner)** | Respect intent + ensure feasibility |
| 9 | α thresholds | **Continuous** | Smooth user experience |
| 10 | Head training | **Staged** | Prevent loss domination |
| 11 | Third modality | **Text (CLIP)** | Object disambiguation + safety |
| 12 | Autonomous head | **Future** (optional) | Separate head for α=1; no distribution shift |
| 13 | Synthetic noise | **No** (evaluation only) | Real human noise is sufficient; synthetic distorts distribution |
| 14 | Latency mitigation | **Future** (index shift) | Chunk output inherently compensates; no architecture change |