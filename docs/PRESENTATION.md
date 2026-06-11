# ALIGN: Assistive Latent Intention-Guided Network

**Assistive Latent Intention-Guided Network (ALIGN)** — **Presenting an assistive shared autonomy framework for humanoid teleoperation**

W N — VRWIT Lab

---

## Slide 1: Problem

### Teleoperating a Humanoid Is Hard

**VR teleoperation of Unitree G1 / robot arms suffers from:**

- **Physiological tremor** — human hands oscillate at 8-12 Hz naturally
- **VR tracking jitter** — Quest 3 pose estimates have ±1-2mm noise
- **Arm instability** — holding a trigger while reaching creates oscillations
- **Fatigue** — noise and drift increase over long sessions
- **No haptic feedback** — no tactile cues to stabilize motion

**Result:** Noisy, jerky, inefficient trajectories. Failed grasps. Operator frustration.

The human **knows what they want to do** — the intention is meaningful — but the motor output is imperfect.

---

## Slide 2: Key Insight

### The Camera Already Knows What You Want

**The operator's egocentric camera sees the target object.**

When a human reaches toward a mug, the wrist camera sees the mug getting closer. The correlation between **what the camera sees** and **how the hand moves** is a natural signal for:

1. **What** the operator intends (which object)
2. **When** to assist (near the target = clear intention)
3. **What correction** is needed (snap the gripper into the right pose)

**Core idea:** Use contrastive learning to align vision and motion into a shared embedding space. The similarity between these embeddings becomes a learned confidence signal that gates assistance.

---

## Slide 3: System Overview

```
                     ┌──────────────────────┐
Task Description ───▶│  CLIP Text (frozen)   │──▶ z_text
("pick red mug")    └──────────┬───────────┘
                               │
                     ┌─────────┴──────────┐
Camera Frame ───────▶│  DINOv2 (frozen)    │──▶ z_v
(wrist RGB)          └──────────┬──────────┘
                                │
                     ┌──────────┴──────────┐
Noisy Poses ────────▶│  Trajectory          │──▶ z_t
(last 10 steps)      │  Transformer (trained)│
                     └──────────┬──────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐    ┌─────────────────┐
          │ DECISION HEAD   │    │ ASSISTANT HEAD  │
          │ When to assist? │    │ What correction?│
          │ → α ∈ [0,1]     │    │ → chunk of K     │
          │                 │    │   future poses   │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
                   └──────┬───────────────┘
                          ▼
         final = raw_pose + α · chunk[0]
         ───▶ IK solver → motor commands
```

---

## Slide 4: 3-Modal Architecture

### Three Frozen Encoders → Two Small MLP Heads

**Vision** — DINOv2 ViT-B (86M params, frozen)
- Egocentric wrist camera → rich visual features → z_v embedding (256d)

**Language** — CLIP text tower (63M params, frozen)
- Task description → semantic grounding → z_text embedding (256d)
- Computed **once per episode** and cached (zero per-frame cost)

**Trajectory** — 3-layer Transformer (830K params, trained from scratch)
- Last 10 noisy poses → temporal motion pattern → z_t embedding (256d)

**Decision Head** — 3-layer MLP (~200K params)
- Input: [z_v, z_t, z_text, 3×cosine_similarities, distances]
- Output: α ∈ [0,1] — confidence to assist

**Assistant Head** — 3-layer MLP (~210K params)
- Input: [z_v, z_t, z_text, noisy_pose]
- Output: chunk of K=5 future corrective poses

**Total trainable: ~1.6M params** | **Total inference: ~40ms (25Hz on Jetson Orin)**

---

## Slide 5: How Assistance Works

### α = Learned Confidence Score

```
α = f(z_v, z_t, z_text, cos_sims, distances)

α ≈ 1.0 → Full correction applied → gripper snaps to ideal pose
α ≈ 0.5 → Partial blend → human still in control
α ≈ 0.0 → No correction → pure human teleoperation
```

**The three alignment scores determine α:**

| Alignment | Measures | High when | Low when |
|-----------|----------|-----------|----------|
| cos(z_v, z_t) | Vision ↔ Motion | Reaching toward visible object | Novel scene, mid-reach switch |
| cos(z_v, z_text) | Vision ↔ Text | Scene matches task description | Wrong object in scene |
| cos(z_t, z_text) | Motion ↔ Text | Motion direction matches task | Reaching for wrong object |

**α_target = need × min(cos_vt, cos_vl, cos_tl)**
— All three must agree for high confidence.

---

## Slide 6: Where It Helps

### Scenarios Before and After ALIGN

| Scenario | Pure Teleop | With ALIGN |
|----------|-------------|------------|
| **Normal** — reaching for a mug | Jerky approach, miss-align gripper | Smooth approach, auto-align gripper |
| **Two identical mugs** | Ambiguous, may grab wrong one | Text "left mug" → disambiguates |
| **Wrong object** — reaching for mug, text says "bowl" | Assist fights human | α drops → human stays in control |
| **Mid-reach switch** — change target mid-way | System fights human | α drops in ~200ms → human regains control |
| **Novel object** — never seen before | No assist | Low alignment → α≈0 → safe fallback |

**The system degrades gracefully** — it doesn't confidently output wrong corrections when uncertain.

---

## Slide 7: Why Contrastive Learning?

### Building Representations That Suppress Noise

**Without contrastive pretraining:**
- Trajectory encoder sees: `[pose-pos, pose-pos, ..., noisy_pose]`
- Learns features that encode **both tremor and intention** — can't separate them

**With contrastive pretraining:**
- Positive pairs: `(frame_of_mug, trajectory_toward_mug)`
- Negative pairs: `(frame_of_mug, trajectory_toward_bowl)`
- The encoder learns: *"what parts of this motion correlate with what I see?"*
- Tremor doesn't correlate with visual features → **encoder suppresses it**
- The directional motion toward the target matches the visual scene → **encoder amplifies it**

**Result:** The trajectory embedding z_t represents clean intention, not noisy execution.

---

## Slide 8: Training Pipeline

### Single Dataset, Three Stages

**Stage 1: Data Collection (~2 days)**
- 200 teleop episodes on Franka Panda (Isaac Sim)
- Each episode: wrist camera frames + noisy poses + task description
- Operator provides text: "pick up the red mug"

**Stage 2: Contrastive Pretraining (~1 day)**
- 3-way InfoNCE: align z_v, z_t, z_text
- Positive pairs: frame+trajectory+text from same episode
- Negative pairs: frame+trajectory+text from different episodes
- Result: frozen 3-modal backbone

**Stage 3: Head Training (~1 day)**
- Train Decision head (BCE loss on α_target)
- Train Assistant head (MSE loss on K future corrections)
- Fine-tune jointly
- Frozen backbone → fast iteration, heads can be retrained independently

**Total dataset: ~200 episodes, ~30,000 frames, ~20 min text annotation**

---

## Slide 9: Ground Truth

### How We Generate the "Correct" Smooth Trajectory

The biggest challenge: what is the "correct" trajectory?

**Our approach — Hybrid Generation:**

```
Phase 1 (far from object, d > 8cm):
  Savitzky-Golay filter on raw trajectory
  → Removes noise while respecting human's chosen path

Phase 2 (near object, d < 8cm):
  Motion planner (RRT-Connect) from current hand → grasp pose
  → Ensures kinematic feasibility near the target

Phase 3 (blend):
  Smoothstep transition between transit and approach
```

**Decision head target:** α_target = need × capability
- need = ||noisy - smooth|| / threshold (how bad is the current pose?)
- capability = min(cos_vt, cos_vl, cos_tl) (can the model understand the scene?)

**Assistant head target:** Δpose[t+i] = smooth[t+i] - noisy[t] for i=1..K

---

## Slide 10: Novelty & Contributions

### What Makes This Publishable?

**Three contributions:**

| # | Contribution | Why Novel |
|---|-------------|-----------|
| 1 | **3-way contrastive alignment as gating signal** | No existing shared autonomy uses contrastive vision-trajectory-language alignment to derive α. Previous work uses hand-tuned thresholds, discrete goal sets, or always-on blending. |
| 2 | **Unified 3-modal shared encoder + dual heads** | Single backbone for both "when" and "what." Multi-task learning (contrastive + BCE + MSE) produces richer features. |
| 3 | **Real hardware deployment on humanoid** | Most shared autonomy work is simulation-only or 2D navigation. Deploying on Unitree G1 at 25Hz with real operators is rare. |

**Closest work:** Yoneda et al. 2023 "Diffusion for Shared Autonomy" — they use diffusion for blending (always-on), we use contrastive for gating (learned when-to-assist). Ours is faster, safer, and OOD-aware.

**Target venue:** ICRA 2026 / RA-L

---

## Slide 11: Why This Is Better Than Alternatives

| Approach | Object Detection Needed? | Hand-tuned α? | OOD Safe? | Novel Objects? | Latency |
|----------|------------------------|---------------|-----------|----------------|---------|
| **Pure teleop** | — | — | ✅ | ✅ | 0ms |
| **Distance-gated** | ✅ YOLO/DINO | ✅ threshold | ❌ | ❌ | ~10ms |
| **Goal-set (Javdani 2018)** | ✅ | ✅ priors | ❌ | ❌ | ~50ms |
| **Diffusion blending (Yoneda 2023)** | ✅ | ❌ | ❌ | ❌ | ~50-100ms |
| **VLM-based (RT-2)** | ✅ (implicit) | ❌ | ❌ | ✅ | >500ms |
| **ALIGN (Ours)** | **❌** (embeddings) | **❌** (learned) | **✅** | **✅** | **~40ms** |

**ALIGN is the only system** that combines: no object detector, no hand-tuned thresholds, OOD safety via alignment, novel object generalization, and <50ms latency for real-time control.

---

## Slide 12: Development Plan

### 7 Phases, ~14 Weeks Total

| Phase | What | Duration | Status |
|-------|------|----------|--------|
| **0** | Isaac Sim Franka + VR teleop + data recorder | 2 weeks | ✅ **Complete** |
| **1** | Data collection (200 episodes) + ground truth | 2 weeks | 🔜 Next |
| **2** | 3-way contrastive pretraining | 1 week | |
| **3** | Decision + Assistant head training | 1 week | |
| **4** | Simulation evaluation + 7 ablations | 2 weeks | |
| **5** | Real Franka deployment + user study (N=5-8) | 3 weeks | |
| **6** | G1 humanoid port + cross-platform validation | 3 weeks | |
| — | Paper writing (parallel from Phase 4) | 6 weeks | |

**Critical path:** ~11 weeks to submission-ready results. Starts after Phase 4.

---

## Slide 13: What's Already Built (Phase 0)

### Scripts in ~/VRWIT/ALIGN/scripts/

| Script | Purpose | Lines |
|--------|---------|-------|
| `align_data_recorder.py` | Records episodes: frames + poses + text + metadata | 345 |
| `align_noise.py` | Noise injection module (for evaluation only) | 219 |
| `collect_episodes.py` | Full Isaac Sim Franka + VR teleop + data collection | 655 |

**Also completed:** Full design documentation (9 documents, 2,200+ lines):

- `ARCHITECTURE.md` — Complete 3-modal system architecture
- `TRAINING_PIPELINE.md` — 4-stage training with 3-way contrastive
- `LITERATURE_REVIEW.md` — 15+ papers analyzed, gap analysis
- `CONTRIBUTION.md` — Novelty claims, paper structure, venue targets
- `DESIGN_DECISIONS.md` — 13 key tradeoffs with rationale
- `TASK_PLAN.md` — Full phased checklists with 28 completed items

---

## Slide 14: Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VR teleop latency >30ms | Medium | High | Keyboard/joint GUI fallback for data collection |
| Contrastive pretraining doesn't converge | Low | High | Debug with 10 episodes first; InfoNCE is robust |
| Sim-to-real gap | Medium | Medium | Test on real Franka early (Week 5) |
| Participant recruitment | Medium | Medium | Lab members as fallback |
| G1 SDK unavailable | Low | High | G1 is Phase 6; paper works with Franka |
| DINOv2 too slow on Jetson Orin | Medium | Low | Switch to DINOv2 ViT-S (smaller, faster) |

---

## Slide 15: Questions?

### ALIGN — Assistive Latent Intention-Guided Network

**Summary:**
1. 3-way contrastive alignment (vision + motion + language) produces a natural gating signal
2. Shared encoder + dual heads learn both *when* and *what* to assist
3. No object detectors, no hand-tuned thresholds, no OOD blind spots
4. Deployable on real hardware at 25Hz
5. 7-phase plan: Phase 0 done, Phase 1-4 next

**Contact:** W N — VRWIT Lab