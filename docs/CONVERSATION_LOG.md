# ALIGN: Conversation Log

This document traces the evolution of ALIGN through the design discussion, capturing key questions, decisions, and reasoning at each stage.

---

## Session Context

- **Date**: June 2026
- **Model**: DeepSeek V4 Flash (via ollama-helios)
- **Platform**: Telegram DM

---

## Thread 1: Problem Framing

**Q**: "Develop a system to help teleoperating on a Unitree G1 humanoid for pick-and-place tasks smoothly. Idea: intention-prediction model as a corrective system."

**Response**: Proposed reactive/corrective system with vision-conditioned diffusion policy. Suggested:
- Egocentric camera → scene tokenizer
- Trajectory denoising via conditional diffusion
- Alpha blending with confidence

**Key insight**: Visual context disambiguates intent from noisy human input.

---

## Thread 2: Related Works

**Q**: "Are there existing baselines? Or similar works?"

**Key findings**:
- Dragan 2013 — Policy blending formalism (hand-tuned α)
- Javdani 2018 — Hindsight optimization (discrete goal sets)
- Losey 2020 — Learned latent actions (no visual context)
- Reddy 2022 — Implicit behavioral cloning (goal images needed)
- Chi 2023 — Diffusion Policy (foundational, full autonomy)
- **Yoneda 2023** — Diffusion for shared autonomy (closest paper, ~30 citations)

**Gap**: Nobody uses contrastive vision-trajectory alignment for learned assist gating.

---

## Thread 3: First Concern — System Downsides

**Q**: "Reasoning with all possible downsides of our system"

**Identified failure modes** (ranked by severity):
1. Human changes behavior → distribution shift (critical)
2. Mid-reach target switch → fighting the human (critical)
3. Latency on G1 hardware (high)
4. No graceful degradation (high)
5. Oversmoothing fine tasks (medium)
6. Ground truth ambiguity (medium)
7. Robot dynamics limits (medium)

---

## Thread 4: Proximity-Gated Assistance

**Q**: "What if assistance strength is influenced by closeness to an object?"

**Proposed**: α = f(distance) with smoothstep transition. Far = no assist, near = full snap.

**Assessment**: Works for most failure modes (mid-reach switch only matters near objects, where intention is unambiguous). But distance alone is too thin for a paper.

---

## Thread 5: Elevating the Approach

**Q**: "Is distance alone strong enough for a paper?"

**Proposed levels**:
- Level 1: Multi-factor learned α (better but still thin)
- Level 2: Goal-entropy gated (information-theoretic)
- Level 3: Contrastive latent intention (learned embeddings)
- Level 4: Learned assist value function (most principled)

**Chosen**: Level 3 (contrastive latent intention) — good balance of novelty and feasibility.

---

## Thread 6: Two-Model Architecture Clarification

**Q**: "Correct me: we need two models — decision and motion control?"

**Confirmed**: Two models, one dataset, two training stages (contrastive pretraining → head training).

**Key insight**: Decision head learns α_target = need × capability, where capability comes from contrastive alignment score.

**Also confirmed**: No pre-defined object detection — vision embeddings instead.

---

## Thread 7: One Model vs. Two Models

**Q**: "What about one model that outputs correction with magnitude determined by confidence?"

**Comparison**:
- **Two models (explicit α)**: Safe, debuggable, clear ablation story
- **One model (implicit magnitude)**: Simpler but MSE training forces non-zero outputs even when uncertain — no inherent confidence scaling

**Reality check**: One-model approach fails without explicit uncertainty modeling (ensembles/evidential). Two models is safer and more principled.

---

## Thread 8: Separate Models vs. Hybrid Shared Encoder

**Comparison**:
- **Option A (Separate)**: ~60ms, 16Hz, independent, fault-isolated
- **Hybrid (Shared)**: ~38ms, 26Hz, complementary representations, one encoder

**Winner**: Hybrid. The latency advantage (30Hz target on G1) and multi-task learning benefits decisively outweigh the coupling cost.

---

## Thread 9: Ground Truth and Starting Platform

**Q**: "What's the ground truth for the smooth trajectories and decision head?"

**Ground truth plan**:
- Smooth trajectories: SavGol filter (transit) + motion planner (approach)
- α_target: need × capability = clip(deviation) × cos_sim(z_v, z_t)
- One dataset, ground truth derived post-collection

**Starting platform**: Franka Panda in Isaac Sim → real Franka → G1 (phased)

---

## Thread 10: Contrastive Pretraining Details

**Q**: "How to train contrastive pretraining?"

**Design**:
- Positive pairs: (frame, traj) within W_pos window
- Negative pairs: all other pairs in batch
- InfoNCE loss with temperature 0.07
- DINOv2 frozen + projection head trainable
- Transformer trajectory encoder fully trainable
- Batch size 64-128

**Key insight**: Contrastive pretraining's main value is building good representations where trajectory encoder learns to suppress tremor (tremor doesn't correlate with visual features).

---

## Thread 11: Novelty Check

**Q**: "Check existing works. Is ours novel enough to publish?"

**Finding**: Yoneda 2023 (Diffusion for Shared Autonomy) is the closest paper. Three key differentiators:
1. Contrastive alignment for gating (new in shared autonomy)
2. Unified shared-encoder architecture (new for this problem)
3. Real hardware validation on humanoid (rare)

**Verdict**: Novel enough for ICRA/RA-L with strong ablations and user study.

---

## Thread 12: What Does Contrastive Actually Buy?

**Q**: "What good does contrastive do if in deployment trajectory and scene are always aligned?"

**Honest answer**: During normal operation, cos_sim is mostly high. The real value is:
1. **Representation quality**: Encoder learns to extract intention-relevant features and suppress noise
2. **OOD detection**: Low cos_sim on novel objects/scenes
3. **Safety**: Graceful degradation when uncertain

---

## Thread 13: To Entropy or Not?

**Q**: "Should we add entropy to the system?"

**Decision**: No. cos_sim already serves the same role (confidence + OOD detection). Adding entropy would:
- Require 3-5× compute (ensembles)
- Not improve results (cos_sim and entropy are correlated)
- Dilute the clean paper story

**Better approach**: Show in ablations that entropy adds nothing, proving the system is minimal and sufficient.

---

## Thread 14: System Name

**Proposed**: **ALIGN** — **A**ssistive **L**atent **I**ntention-**G**uided **N**etwork

**Alternative**: ALIGN, GUARD, FUSE, LATCH, SharedAlign, VisTraGate, ContrastAssist, CoAlign

**Chosen**: ALIGN — reads naturally, acronym fits, "alignment" captures both contrastive pretraining and system purpose.

---

## Thread 15: Chunk Output for Smooth Trajectories

**Q**: "How does the model actually generate smooth paths?"

**Realization**: The original design (single Δpose per step) relied on α blending for smoothness — not a planned trajectory. That's indirect and fragile.

**Decision**: Change Assistant head to output a **chunk of K future corrections** (default K=5). Ground truth becomes smooth[t+1..t+K] - noisy[t]. Execute chunk[0], re-predict every step, cache the rest for optional blending.

**Why it's better**:
1. Inherent temporal coherence — K poses must form a smooth sequence (MSE over all K)
2. Closed-loop via sliding window — re-predicting with each new observation creates natural feedback
3. K=1 collapses to original single-step mode — tunable parameter, not architectural change
4. Consistent with Diffusion Policy (Chi 2023) paradigm

## Thread 16: Adding Text as a Third Modality

**Q**: "Should we also add text prompt as another modality to guide/give hints?"

**Initial instinct**: Keep it as future work, avoid complexity.

**User decision**: Add it now. The benefits are genuine and the cost is low.

**Final architecture**:
- Frozen CLIP text ViT-B/32 tower → z_text (256d)
- Computed once per task (not per frame) → ~5ms one-time, cached
- 3-way contrastive loss: InfoNCE(v,t) + InfoNCE(v,l) + InfoNCE(t,l) / 3
- Decision head input grows from 515 to 774 (3 embeddings + 3 cos_sims + 3 distances)
- Assistant head input grows from 518 to 774 (+ z_text)
- Capability signal: min(cos_vt, cos_vl, cos_tl) — all three must agree

**Key benefits**:
1. Object disambiguation — "the left mug" vs "the right mug"
2. Wrong-object detection — text says "bowl", human reaches for mug → low α
3. Task-conditional corrections — same scene, different text → different Δpose
4. Negligible compute cost (one-time, cached)
5. 3-way min is stricter = safer than 2-way

**Downsides accepted**:
- ~20 minutes of text annotation for 200 episodes
- Minimal user friction (voice/keyboard input)
- ~63M frozen params from CLIP (no runtime cost)

## Thread 17: Write Documentation

**Action**: Create ~/VRWIT/ALIGN/ directory with structured documentation:
- README.md
- INITIAL_PLAN.md
- ARCHITECTURE.md
- TRAINING_PIPELINE.md
- LITERATURE_REVIEW.md
- CONTRIBUTION.md
- DESIGN_DECISIONS.md
- CONVERSATION_LOG.md (this file)

---

## Key Principles Established

1. **Safety first**: The system must degrade gracefully to pure teleoperation when uncertain.
2. **One dataset**: Both heads are trained from the same teleoperation episodes.
3. **No hand-tuned thresholds**: α is learned from data through contrastive alignment.
4. **No object detector**: Vision embeddings generalize to novel objects.
5. **30Hz target**: System must run on G1's Jetson Orin NX at control frequency.
6. **Arm first, full body later**: Start simple, add complexity after validation.
7. **Strong ablations**: The paper needs to prove WHY contrastive alignment beats simpler alternatives.