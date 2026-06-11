# ALIGN: Anticipated Professor Questions & Answers

**Assistive Latent Intention-Guided Network (ALIGN)**

## Theory & Architecture

### Q1: "Why contrastive learning? Why not just use a standard supervised objective?"

**Answer:**

Two reasons — representation quality and a natural gating signal.

**For representation quality:**
Standard supervised training on `(frame, trajectory) → correction` forces the encoders to learn features *only* useful for predicting that specific correction. Contrastive pretraining forces the encoders to learn a structured embedding space where the *relationship* between vision and motion is meaningful — similar motions produce similar embeddings regardless of the exact correction value.

Concretely: the trajectory encoder trained with MSE alone sees two noisy trajectories (both reaching toward objects, different objects) and optimizes to predict different Δposes. The contrastive encoder instead learns "these two trajectories are both 'reaching toward something' motions" — a higher-level feature. When you then train the head on top, it has richer features to work with.

**For the gating signal:**
The cosine similarity between z_v and z_t falls out of contrastive pretraining *for free*. We didn't design it — it's a natural property of the InfoNCE loss. If we used only supervised regression, we'd have no natural gating signal and would need to hand-design α from distance, velocity, or some other proxy. The contrastive approach gives us both better features AND the gating signal from one training stage.

---

### Q2: "Why three modalities? Is text really necessary if DINOv2 can distinguish objects visually?"

**Answer:**

DINOv2 can distinguish objects visually, but it cannot **disambiguate two identical-looking objects at the same distance.** Consider: two red mugs side by side. The vision encoder sees "two red mugs." The trajectory shows the hand moving between them. Which one does the operator want? The system can't tell.

Text resolves this: "pick up the left mug" or "grab the one near the bowl." The embedding z_text anchors the task in semantic space, and cos_sim(z_t, z_text) tells us whether the motion matches the instructed target.

The second reason is **safety.** If the operator says "pick up the bowl" but reaches for the mug (distracted, misheard, changed mind), cos_sim(z_t, z_text) drops immediately, α drops, and the system doesn't fight them. Without text, the system sees "human reaching toward mug" and assists toward mug — which is wrong.

The cost is negligible: CLIP is frozen (no training cost), and z_text is computed once per episode (~5ms) and cached. It's essentially free at runtime.

---

### Q3: "Why an MLP head instead of a diffusion model or transformer decoder?"

**Answer:**

Three reasons: data regime, latency, and task complexity.

**Data regime:** We have ~200 episodes (~30K frames). A diffusion policy or transformer decoder has 5-50M parameters. That ratio (30K samples / 5M params) guarantees severe overfitting — the model memorizes training episodes and fails on novel objects. Our MLP has ~200K params (150:1 sample-to-param ratio), which is well within the safe regime. If we had 10,000+ episodes, we'd revisit this.

**Latency:** Diffusion requires 10-50 iterative denoising steps per timestep. On Jetson Orin, that's 50-100ms for the assistant alone, pushing total inference past 100ms (<10Hz). Our MLP takes ~1ms. At 25Hz, the operator feels the robot responds instantly. At 10Hz, it feels sluggish and the VR experience degrades.

**Task complexity:** The correction space near a known object is roughly unimodal — there's one correct grasp approach direction. Diffusion's main advantage is handling multi-modal distributions. A transformer decoder's main advantage is long-horizon autoregressive generation. Neither applies here: we're predicting K=5 local corrections (~165ms) re-predicted every step from fresh observations. An MLP is sufficient.

---

### Q4: "How do you handle the distribution shift when the assistant's own outputs are fed back into the trajectory encoder?"

**Answer:**

We don't feed corrected outputs back. The trajectory encoder always receives **raw noisy human poses**, not the blended final command.

```
Trajectory encoder input:  [noisy_{t-9}, noisy_{t-8}, ..., noisy_t]
                            ↑ always raw teleop
                            
Final command:  corrected_t = noisy_t + α · Δpose_t
                 ↑ sent to robot, NOT fed back to encoder
```

This means the encoder always operates within its training distribution (noisy human inputs). The correction is applied *after* the encoder — the encoder never sees its own corrections.

The only risk is if α=1 for extended periods (full autonomous mode). In that case the robot moves smoothly, the human sees it moving and might stop moving themselves (relaxing their hand), so the noisy poses become static → trajectory encoder sees "no motion" → z_t changes → cos_sim drops → model degrades. We handle this with the optional autonomous head design (Decision 12), which operates on z_v + z_text only, bypassing the trajectory encoder entirely.

---

### Q5: "What happens when the operator's view is occluded? What if the wrist camera doesn't see the target?"

**Answer:**

Valid concern. In real teleoperation, there are two cases:

**Partial occlusion (hand blocks object partially):** DINOv2's features are surprisingly robust to occlusion — it attends to visible parts and maintains a strong object representation. The trajectory encoder's temporal context (last 10 poses) also disambiguates: if the hand was moving toward the mug before occlusion, z_t still encodes "reaching for mug." cos_sim stays high.

**Full occlusion (object completely hidden):** cos_sim(z_v, z_t) drops because the vision encoder doesn't see the expected object. α drops. The system becomes transparent — pure teleoperation. This is correct behavior: if the operator can't see the target, the model can't confidently assist either.

**The text modality helps here:** Even if the object is occluded, cos_sim(z_t, z_text) may remain high if the motion still matches the instructed task ("I can't see the mug but I know I'm reaching for it"). This partial signal keeps some assistance active during brief occlusions.

If occlusion is a major problem in practice, we could add a second camera (agentview / overhead) to the vision encoder — the architecture supports multiple vision inputs trivially since they'd all project to the same 256d embedding space.

---

## Ground Truth

### Q6: "How do you know the 'correct' smooth trajectory? This seems like a circular problem — you want to correct noise, but you train on 'ground truth' that you generate from the noisy data."

**Answer:**

This is the hardest question in assistive teleoperation, and we have a pragmatic answer, not a perfect one.

**We use a hybrid approach:**

For the **transit phase** (far from objects, d > 8cm): We apply a Savitzky-Golay filter. This preserves the human's chosen path shape (which is meaningful — they chose that trajectory intentionally) while removing high-frequency noise. We're not replacing their path, just cleaning the signal.

For the **approach phase** (near objects, d < 8cm): We replace the noisy trajectory entirely with a motion-planned trajectory from current hand position to the detected grasp pose. This is justified because, near the object, the human's intent is unambiguous (grasp that object) and kinematic precision matters more than path preference.

**Why this is valid for training:**
The Assistant head learns to output the *difference* between noisy and smooth. A noisy trajectory that is 3cm left of the smooth trajectory gets a correction of +3cm right. A noisy trajectory that is already on the smooth path gets zero correction. The model learns the mapping from `(noisy, scene, task) → correction`, not from `(noisy) → smooth`.

**For evaluation (not training):** We also collect expert slow demonstrations where the operator moves deliberately and slowly. This gives us an independent "ground truth" for testing without any post-processing.

---

### Q7: "How do you measure success? What metrics convince you the system works?"

**Answer:**

Four metrics, ordered by importance:

1. **Task success rate** — Did the robot grasp the target object? (Binary)
2. **Jerk** — Mean jerk over trajectory (cm/s³). Lower = smoother.
3. **Completion time** — Seconds from start to grasp.
4. **NASA TLX** — Subjective operator effort (for user studies).

**The critical comparison is ALIGN vs. Distance-gated assistance.** If we beat a simple distance-based α on all four metrics, the contribution of learned contrastive gating is validated. If we also beat it specifically on:
- Novel objects (where distance works but contrastive generalizes better)
- Wrong-text scenarios (where distance has no defense)
- Mid-reach switches (where distance is too slow)

...then the paper is strong.

---

## Practical Concerns

### Q8: "How long until this runs on real hardware? What's the risk of it not working on the real robot?"

**Answer:**

**Timeline:** We anticipate 3 weeks from sim evaluation to real Franka deployment (Phase 5). The codebase is already designed to be platform-agnostic — the encoders and heads receive standardized inputs (RGB frames, 6D poses, text).

**Risk of failure:** Moderate, but manageable. The main risks:

1. **Camera domain gap** — DINOv2 was trained on internet images, not Isaac Sim renders. Moving to a real camera might shift feature distributions. Mitigation: DINOv2 is surprisingly robust to this (it's used extensively in real-robot settings), and we can fine-tune the small projection head (~197K params) on ~10 real frames if needed.

2. **Latency on Jetson Orin** — DINOv2 ViT-B takes ~25ms on Orin, which dominates our budget. If total latency exceeds 50ms (target: 40ms), we swap to DINOv2 ViT-S (smaller, ~15ms, still good features). This is a configuration change, not an architecture change.

3. **IK differences** — Franka and G1 have different kinematics. The ALIGN model outputs EEF pose corrections, which are kinematic-agnostic. The IK solver handles the robot-specific conversion. No model retraining needed between platforms.

---

### Q9: "200 episodes seems like a lot of data collection. What if the operator gets fatigued?"

**Answer:**

200 episodes at ~10 seconds each = ~33 minutes of active teleoperation. Spread across 2-3 operators and multiple sessions, this is ~1-2 hours of total work — well within what's practical for a thesis project.

We've designed the collection process to minimize fatigue:
- **Auto-reset:** Button A resets the sim instantly
- **Auto-finalize:** Timeout-based episode ending if operator forgets to release trigger
- **Break-friendly:** Can stop mid-session and resume later (episodes are saved individually)
- **Noise is natural:** No synthetic noise injection — the operator's real fatigue adds ecological noise to the data

If collecting 200 episodes proves too time-consuming, we have a quick-start path: collect 100 episodes (still statistically meaningful), skip the motion planner for ground truth (use SavGol only), and you can have a workshop-ready result in 5 weeks.

---

### Q10: "How is this different from just using a low-pass filter or Kalman filter on the teleoperation input?"

**Answer:**

A low-pass filter or Kalman filter applies **frequency-based smoothing** — it removes high-frequency content regardless of what that content means. This is the key weakness:

| Scenario | Low-pass filter | ALIGN |
|----------|----------------|-------|
| Tremor (8-12 Hz) | ✅ Removes it | ✅ Removes it |
| Intentional fast motion | ❌ Blurs it | ✅ Preserves it (recognizes it as intentional) |
| Near-object snap (corrective) | ❌ Can't generate it | ✅ Snaps gripper into grasp pose |
| Reaching toward wrong object | ❌ Can't detect it | ✅ Detects via cos_sim mismatch |
| Mid-reach switch | ❌ Smooths through it | ✅ Detects and releases control |
| Novel object | ❌ Smooths same as always | ✅ Detects OOD and releases control |

A low-pass filter is a **one-size-fits-all frequency knife**. ALIGN is a **semantic scalpel** — it knows what the operator is trying to do and helps only when it's confident about the intention.

Concretely: a Kalman filter with 5cm/s velocity limits would smooth tremor but also smooth an intentional quick correction when the operator realizes they're off-target. ALIGN sees the camera frame showing the target, sees the motion direction, understands the intent, and applies the correction decisively.

---

### Q11: "What happens if the operator provides a wrong or ambiguous text description?"

**Answer:**

**Wrong description:** If text says "pick up the bowl" but operator reaches for mug, cos_sim(z_t, z_text) drops immediately. α drops. The system becomes transparent — the operator has full control. The text doesn't override the operator's actual motion; it only helps when the motion matches the instruction.

**Ambiguous description:** "pick up the object" is less specific than "pick up the red mug." Both work — the difference is in disambiguation power. With "the object," the model falls back to vision+trajectory disambiguation (essentially 2-modal mode). With "the red mug," it has explicit target grounding. We train with multiple specificity levels per episode so the model handles both gracefully.

**No description at all:** We use a default "pick and place" placeholder. The text embedding becomes a neutral vector, and the system operates in 2-modal mode. Still functional, just without text-level disambiguation.

---

## Paper & Novelty

### Q12: "What's the actual contribution? Is this a method paper or a system paper?"

**Answer:**

It's a **method paper with system validation.** The core contribution is methodological:

> **Using 3-way contrastive alignment as a learned gating signal for shared autonomy — where the alignment score naturally determines when to assist, without hand-tuned thresholds.**

The three components of novelty:
1. Contrastive alignment → gating (never done in shared autonomy)
2. Shared encoder + dual heads for "when + what" (new architecture for this problem)
3. Real humanoid deployment at 25Hz (rare)

The closest paper, Yoneda et al. 2023 "Diffusion for Shared Autonomy," uses diffusion for *blending* — it assumes assistance is always on. Ours learns *when* to assist through alignment, which is a fundamentally different capability. We explicitly handle the failure modes (mid-reach switch, novel objects, wrong targets) that always-on blending can't.

Target venue: ICRA 2026 or RA-L. Both accept papers with real hardware validation and clear methodological contribution.

---

### Q13: "Where could this fail in interesting ways? What's the negative result you're worried about?"

**Answer:**

Two failure modes I'm genuinely worried about:

**1. The "always-low-α" failure.** The model might learn to output α ≈ 0 always — it's the safe answer, BCE loss is low, the assistant head never gets exercised. We prevent this with staged training (train Decision first with α_target that includes need > 0 cases), but it's a risk.

**2. The "chunk oscillation" failure.** The MLP outputs K=5 future poses as one vector. If the ground truth chunks have high temporal variance (e.g., the SavGol filter didn't fully smooth the transit phase), the model learns to output oscillating chunks. At inference, this creates a 5-step oscillation that feels worse than the original teleop noise. Mitigation: we visualize chunk predictions during training validation and clamp Δpose velocity limits at inference.

The most interesting negative result would be: **ALIGN doesn't beat distance-gated assistance on normal objects, but beats it significantly on novel objects.** This would validate the contrastive approach (generalization) while acknowledging that simple baselines work well in-distribution. It's still a publishable result — the paper's contribution shifts from "better overall" to "generalizes better."

---
