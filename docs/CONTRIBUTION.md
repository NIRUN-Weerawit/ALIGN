# ALIGN: Contributions and Paper Plan

**Assistive Latent Intention-Guided Network (ALIGN)**

## Core Contribution

> **A 3-way contrastive vision-trajectory-language alignment framework for learned assist gating in shared autonomy, where the same embedding space determines *when* to assist and *what* correction to apply.**

## Three Novel Elements

### 1. 3-Way Contrastive Alignment as a Gating Signal (New)

Previous shared autonomy work gates assistance using:
- Pre-defined goal sets (Javdani 2018)
- Distance to objects (nontrivial but hand-tuned)
- Task-specific confidence metrics (Newman 2025)
- Always-on blending (Yoneda 2023)

ALIGN gates using the **minimum of three pairwise cosine similarities** (vision↔trajectory, vision↔language, trajectory↔language), learned through 3-way contrastive pretraining. This is:
- **Continuous** (not discretized to known goals)
- **Visual + Language** (not just distance-based)
- **Learned** (not hand-tuned)
- **Generalizable** (no object detector needed, works on novel objects)
- **OOD-aware** (drops on unseen scenes → safe fallback)
- **Semantically grounded** (text mismatch suppresses assist — "pick up the mug" while reaching for bowl = low α)

### 2. Unified 3-Modal Shared-Encoder Architecture (New in Shared Autonomy)

While shared encoders exist in multi-task learning, applying a 3-modal (vision, motion, language) shared encoder to the specific problem of "when to assist + what to do" is novel. The decision head and assistant head:
- Share vision, motion, and language features
- Receive complementary supervision (classification + regression + contrastive)
- Are jointly trained on a frozen 3-modal backbone
- Run at 25Hz on G1's Jetson Orin (~40ms total)

### 3. Hardware Feasibility on a Humanoid

Most shared autonomy works are:
- Simulation-only
- Wheelchair-mounted arms
- 2D navigation

ALIGN is designed and validated on the Unitree G1 humanoid — demonstrating that learned, text-conditioned assistive teleoperation is practical on real humanoid hardware at 25Hz.

---

## Paper Title Options

1. **"ALIGN: 3-Way Contrastive Vision-Trajectory-Language Alignment for Learned Assist Gating in Shared Autonomy"**
2. **"ALIGN: A Unified 3-Modal Shared Autonomy Framework for Learned Assist Gating"**
3. **"ALIGN: Learning When and How to Assist via Vision-Motion-Language Contrastive Alignment"**

---

## Proposed Structure

```
1. INTRODUCTION
   Problem: Noisy teleoperation, human tremor, unstable grasps
   Existing approaches and their limitations
   Our approach: ALIGN at a glance
   Contributions (3 bullet points)

2. RELATED WORK
   Shared autonomy / assistive teleoperation
   Diffusion-based shared autonomy (Yoneda)
   Visual intention prediction
   Contrastive learning for robotics
   Assist gating / confidence arbitration
   → Clear gap statement: "no existing work uses contrastive 
     vision-trajectory alignment for learned assist gating"

3. SYSTEM OVERVIEW
   Architecture diagram
   Components: vision encoder, trajectory encoder, 
               decision head, assistant head
   Final command blending

4. CONTRASTIVE PRETRAINING
   Positive/negative pair construction
   InfoNCE loss
   Encoder architectures
   Convergence checks

5. HEAD TRAINING
   Decision head: α_target = need × capability
   Assistant head: Δpose = smooth - noisy
   Ground truth generation (SavGol + motion planner)
   Training schedule

6. EXPERIMENTS
   6.1 Setup: Franka Panda in Isaac Sim → real Franka → G1
   6.2 Metrics: task success rate, task completion time,
       jerk, user effort (NASA TLX), α trajectory analysis
   6.3 Ablations (MANDATORY):
       - Pure teleop (α=0)
       - Distance-gated α
       - Always-on assistant (α=1)
       - ALIGN (Ours, full)
   6.4 Novel object generalization
   6.5 Mid-reach switch test
   6.6 OOD degradation analysis

7. RESULTS

8. DISCUSSION
   Failure cases
   Limitations
   Future work

9. CONCLUSION
```

---

## Ablation Experiment Table

| Method | α Signal | Success Rate | Task Time | Jerk (↓) | TLX | Notes |
|--------|----------|-------------|-----------|----------|-----|-------|
| Pure teleop | α = 0 always | baseline | baseline | high | high | No assistance |
| Distance-gated | α = f(d) | ? | ? | ? | ? | Non-visual baseline |
| Always-on | α = 1 | ? | ? | best | ? | No gating, may fight human |
| **ALIGN (Ours)** | α = cos_sim | **?** | **?** | **?** | **?** | Full system |
| ALIGN - contrastive | α = MLP only | ? | ? | ? | ? | Ablate alignment |
| ALIGN - no cos_sim | α = MLP w/o sim | ? | ? | ? | ? | Ablate gating signal |

The key comparison is **ALIGN vs. Distance-gated**. If ALIGN beats distance-gated on novel objects and on mid-reach switch scenarios, the contribution is validated.

---

## Venue Targets

| Venue | Deadline (approx) | Acceptance Rate | Fit | Strategy |
|-------|-------------------|-----------------|-----|----------|
| **ICRA 2026** | Sep 2025 | ~45% | High | Primary target. Need real Franka results + user study. 8 pages. |
| **RA-L (± ICRA option)** | Rolling | ~30% | High | Faster review (6-8 weeks). Can submit shorter version. |
| **IROS 2026** | Feb 2026 | ~45% | Medium | Backup. Slightly lower prestige than ICRA. |
| **CoRL 2025** | Jun-Jul 2025 | ~25% | Medium | Very competitive, but strong fit. Top conference for robot learning. |

**Recommendation**: Target **ICRA 2026** with RA-L as backup. Submit around Sep-Oct 2025.

---

## Key Arguments to Make to Reviewers

### Why contrastive alignment and not just distance?

> "Distance to nearest object does not resolve ambiguity between two equidistant objects, does not incorporate visual object identity, and cannot detect OOD scenes. Our contrastive alignment signal subsumes distance (it naturally correlates with proximity) while adding semantic understanding."

### Why not just use diffusion (Yoneda 2023)?

> "Yoneda et al. use diffusion for blending, but blending assumes assistance is always beneficial. Our learned gating explicitly handles the case where assistance would be worse than nothing — novel objects, mid-reach switches, uncertain intention. Additionally, diffusion requires 10-50 iterative denoising steps; our single-pass heads run at 30Hz on edge hardware."

### Why not two separate models?

> "A single shared encoder forces both heads to learn from each other's supervision signals. The contrastive loss trains the encoder to produce features that align vision and motion; the decision loss trains it to separate assist-worthy from not; the regression loss trains it for precise spatial corrections. This multi-task learning produces richer features than any single task could."

### How do you handle mid-reach intention changes?

> "When the human switches targets, the trajectory embedding shifts to the new goal while the vision embedding still shows the original object — and the new motion no longer matches the task description. All three alignment scores (cos_vt, cos_vl, cos_tl) drop within ~200ms, reducing α and returning control to the human. The 3-way signal is faster and more robust than vision alone because text provides a stable reference: the motion diverging from the described task is detected even before the visual scene changes."

---

## Potential Weaknesses (Address in Paper)

1. **"This is just A+B"** — Strong ablations showing WHY contrastive gating specifically beats alternatives will be crucial. The multi-factor learned α ablation is key.

2. **Small scale** — Pick-and-place on one robot arm. Mitigate by testing on novel objects and mid-reach switches. Show generalization not just performance.

3. **No user study** — Essential. Even N=5 operators doing 10 trials each gives statistical power for TLX and success rate.

4. **cos_sim is not entropy** — Don't call it entropy in the paper. Frame it as "alignment score" or "visual-motion coherence."

5. **"Training α_target uses need × capability where capability = cos_sim"** — Some may argue this is circular. Counter: cos_sim from pretrained encoders is independent of the Decision head, so there's no information leak.

6. **Autonomous mode requires distribution shift handling** — Setting α=1 feeds smooth self-generated trajectories back into the trajectory encoder, which was trained on noisy human inputs. Full autonomy requires a separate autonomous head (trained on demonstration data) or retraining the encoder on corrected trajectories. Both are left as future extensions — the core paper focuses on shared autonomy, not full autonomy.