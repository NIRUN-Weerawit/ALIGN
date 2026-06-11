# ALIGN: Literature Review

**Assistive Latent Intention-Guided Network (ALIGN)**

## Areas Covered

1. Shared Autonomy / Assistive Teleoperation
2. Diffusion-Based Shared Autonomy
3. Visual Intention Prediction
4. Contrastive Learning for Robot Manipulation
5. Assist Gating and Confidence Arbitration

---

## 1. Shared Autonomy / Assistive Teleoperation

### Foundational Works

**Dragan et al. (2013)** — "A Policy Blending Formalism for Shared Control"
- Introduces policy blending: final action = (1-α) · human + α · autonomous
- Uses confidence-based weights, but confidence is hand-tuned, not learned
- No visual context

**Javdani et al. (2018)** — "Shared Autonomy via Hindsight Optimization"
- Uses Bayesian inference over discrete goal sets to infer human intention
- Assistance is computed as the optimal policy toward the most likely goal
- **Limitation**: Requires a pre-defined set of possible goals
- **Relevance**: Our work removes this requirement — no pre-defined goals, just continuous embeddings

**Losey et al. (2020)** — "Shared Autonomy with Learned Latent Actions" (arXiv:2005.03210)
- Learns a low-dimensional latent action space from human teleoperation
- Maps noisy human input to learned latent actions that represent meaningful behaviors
- Foundational for learning-based shared autonomy
- **Limitation**: No visual context, latent actions are task-specific, no learned assist gating
- **Relevance**: Shares our motivation (learn a better representation of human intent) but differs in approach (we use contrastive vision-trajectory alignment)

**Losey et al. (2022)** — "ARRO: Controlling Robots Instinctively"
- Online adaptation to user's control model
- Learns mapping from human input to intended action via iterative correction
- **Limitation**: Slow online adaptation, no visual features
- **Relevance**: Complementary approach — ARRO learns online; ALIGN learns offline and provides real-time assistance

### Recent Shared Autonomy

**"Gaze to Grasp: Shared Autonomy in VR Robot Teleoperation" (2025)**
- Uses gaze direction + hand motion as multimodal signal for intent
- **Limitation**: Requires eye-tracking hardware, gaze is not always aligned with reach target in manipulation
- **Relevance**: Different signal modality — our approach uses vision + trajectory only

**"End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy" (arXiv:2511.00139, 2025)**
- VR teleoperation + autonomous VLA policy for data collection
- Uses vision-language models as autonomous agent
- **Limitation**: Computationally expensive (VLMs), not designed for real-time assistance
- **Relevance**: Our approach is lightweight enough for real-time control on-orbit

---

## 2. Diffusion-Based Shared Autonomy

**Yoneda et al. (2023)** — "To the Noise and Back: Diffusion for Shared Autonomy" (arXiv:2302.12244)
- **This is the closest existing work to ALIGN** (citations: ~30+)
- Uses conditional diffusion to blend human input with autonomous control
- The diffusion denoising process naturally produces smooth trajectories
- **Our key differences**:
  - Yoneda uses diffusion for *blending*; ALIGN uses contrastive alignment for *gating*
  - Yoneda assumes assistance is always active; ALIGN learns *when* to assist
  - Yoneda has no explicit OOD detection; ALIGN's cos_sim provides natural OOD safety
  - Yoneda needs multiple diffusion steps (~10-50); ALIGN uses single forward pass through heads (~2ms)
  - ALIGN's dual-head architecture is simpler and faster for real-time deployment on G1

**Chi et al. (2023)** — "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (arXiv:2303.04137, 500+ citations)
- Diffuses actions conditioned on visual observations for robot manipulation
- Foundational work for diffusion-based visuomotor policies
- **Limitation**: Designed for full autonomy, not shared autonomy / human-in-the-loop
- **Relevance**: If we were to replace our Assistant head with a diffusion process, this would be the architecture — but we choose single-pass MLP for speed

---

## 3. Visual Intention Prediction for Teleoperation

**Reddy et al. (2022)** — "Implicit Behavioral Cloning"
- Predicts actions directly from observation + goal images
- Uses visual goals (images of target state) to condition action prediction
- **Limitation**: Requires goal images at inference time, not designed for continuous trajectory correction

**Shi et al. (2023)** — "RoboTap"
- Visual affordance prediction from egocentric video for manipulation
- Predicts where to grasp from RGB observation
- **Relevance**: Could serve as a component for grip snap prediction in ALIGN's Assistant head

**"EgoIntent"** — Egocentric vision + trajectory for intention prediction
- Uses first-person video to predict manipulation intent
- **Limitation**: Predicts discrete goal labels (which object), not continuous trajectory corrections

**Gopinath et al. (2017)** — "Fast Intent Prediction for Shared Control"
- Bayesian inference over possible goals from joystick input
- **Limitation**: Pre-defined goal locations, 2D navigation only

---

## 4. Contrastive Learning for Robot Manipulation

**Contrastive Learning in General**
- SimCLR, MoCo, CLIP: align different views/modalities in embedding space
- InfoNCE loss: pull positive pairs together, push negatives apart

**Applications to Robotics**
- R3M (Residual Reinforcement Learning from Manipulation): contrastive pretraining from human video for robot manipulation
- VIP (Value-Implicit Pretraining): contrastive learning for goal-conditioned value functions
- Voltron: language-conditioned contrastive pretraining

**Our unique use**: Not pretraining for downstream tasks in general, but specifically using the **alignment score itself** as a gating mechanism for shared autonomy. Nobody has done this.

---

## 5. Assist Gating and Confidence Arbitration

**Newman et al. (2025)** — "Enhancing Shared Autonomy... Confidence-Aware Arbitration"
- Confidence-aware arbitration for shared autonomy under network delay
- Very recent (2025, 0 citations — independent concurrent work)
- Uses task-specific confidence metrics
- **Difference**: Their confidence is computed from network delay and task context; ours comes from learned visual-trajectory alignment

**"Robot Health Indicator" (arXiv:2303.06776, 2023)**
- Visual cue for level-of-autonomy switching
- Human decides when to switch, not the system
- Inverse of our approach — ALIGN decides autonomously when to assist

**Distance-Gated Assistance (common baseline)**
- α = f(distance to nearest object)
- Simple, effective, but has key weaknesses:
  - Doesn't handle ambiguous scenes (two objects at same distance)
  - No OOD detection
  - Fixed threshold, no per-scene adaptation
- We use this as a **baseline** in our ablation experiments

---

## 6. Text-Conditioned Shared Autonomy (New Section)

**CLIP (Radford et al., 2021)** — "Learning Transferable Visual Models from Natural Language Supervision"
- Vision-language contrastive pretraining on 400M image-text pairs
- Natural zero-shot classifier across arbitrary object categories
- **Relevance**: We use CLIP's frozen text tower as our third modality. Its text encoder produces embeddings that align with visual features, making it a natural fit for our 3-way contrastive pretraining.

**RT-2 (Brohan et al., 2023)** — "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
- Uses web-pretrained VLM to condition robot actions on natural language
- Demonstrates that language conditioning improves generalization in manipulation
- **Relevance**: Shows the value of text conditioning in robot learning. ALIGN brings this same principle to the shared autonomy setting.

**"End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy" (arXiv:2511.00139, 2025)**
- Shared autonomy with vision-language models
- **Limitation**: VLM inference is too slow for real-time control (~500ms+)
- **Relevance**: ALIGN uses CLIP text encoder (one-time ~5ms, cached) instead of an online VLM — fast enough for 30Hz.

**VLA-CLIP / Mobile ALOHA with Language** (various, 2024-2025)
- Emerging trend: using vision-language pretraining for robot conditioning
- Usually full autonomy (not shared), or too slow for online gating
- **Relevance**: ALIGN is the first to use text as a *gating modality* (via 3-way contrastive alignment), not just as a task conditioner.

### Our Position Relative to Text-Conditioned Works

| Property | VLM-based (RT-2, etc.) | LLM-based (CoT, etc.) | ALIGN (Ours) |
|----------|------------------------|----------------------|--------------|
| **Inference latency** | 500ms-2s | 1-10s | **~40ms** |
| **Runs on Jetson Orin** | ❌ | ❌ | **✅ 25Hz** |
| **Text conditioning** | ✅ Yes | ✅ Yes | **✅ Yes** |
| **OOD detection** | Not built-in | Not built-in | **✅ via cos_sim** |
| **Online adaptation time** | None | None | **<200ms (target switch)** |
| **Gating mechanism** | None | None | **✅ Learned α** |

ALIGN is the only system that combines text conditioning with fast, learned assist gating — the text doesn't just *describe* the task, it directly *gates* the assistance via 3-way contrastive alignment.

---

## 7. Gap Analysis: Where ALIGN Fits

| What's Missing in Existing Work | How ALIGN Addresses It |
|--------------------------------|------------------------|
| **Learned assist gating** — most work uses hand-tuned α thresholds | α is learned from contrastive alignment, not hand-tuned |
| **No pre-defined goals** — existing intent recognition requires discrete goal sets | Continuous embedding space, no pre-defined goals |
| **Visual + motion alignment for confidence** — gating is rarely vision-informed | cos_sim(z_v, z_t) encodes how well vision explains motion |
| **No language conditioning in shared autonomy** — existing text-robotics works are either too slow (VLM) or full-autonomy | Frozen CLIP text tower (one-time ~5ms), cached for episode — 3-way contrastive alignment with vision + trajectory |
| **Wrong-object assist prevention** — no existing gating checks whether the target matches the task description | cos_sim_tl drops if motion ≠ text description → α drops → safe fallback |
| **Object disambiguation beyond vision** — two identical objects at same distance are indistinguishable | Text explicitly specifies target ("left mug", "red one") |
| **OOD detection** — no graceful degradation for novel scenarios | min(cos_vt, cos_vl, cos_tl) drops on unseen objects → α → 0 → safe fallback |
| **Single unified model for when + what** — decision and assistance are separate pipelines | Shared encoder + dual heads |
| **Online hardware deployment** — most work is simulation-only | Designed for 30Hz control on G1's Jetson Orin |

### The Core Claim

**ALIGN is the first system to use 3-way contrastive vision-trajectory-language alignment as a learned gating signal for shared autonomy, in a unified architecture that simultaneously learns when to assist and what correction to apply — all from a single dataset with per-episode text annotations.**

---

## Relevant Papers Summary

| Paper | Year | Citations | Relevance | Key Limitation |
|-------|------|-----------|-----------|----------------|
| Yoneda — "Diffusion for Shared Autonomy" | 2023 | ~30 | **Highest** — same space | No gating; always-on assist; no OOD detection |
| Losey — "Learned Latent Actions" | 2020 | 89 | High — learning-based shared autonomy | No visual context |
| Chi — "Diffusion Policy" | 2023 | 500+ | Medium — visuomotor diffusion | Full autonomy, not shared |
| Dragan — "Policy Blending" | 2013 | ~300 | Medium — foundational blending | Hand-tuned α |
| Javdani — "Hindsight Optimization" | 2018 | ~150 | Medium — intent inference | Pre-defined goal sets |
| Newman — "Confidence-Aware Arbitration" | 2025 | 0 | Medium — concurrent work | Task-specific, not visual |
| Gopinath — "Fast Intent Prediction" | 2017 | ~100 | Low | 2D navigation only |
| Shi — "RoboTap" | 2023 | ~20 | Low | Discrete affordances only |