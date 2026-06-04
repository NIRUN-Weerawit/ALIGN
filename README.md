# ALIGN: Assistive Latent Intention-Guided Network

**Assistive Latent Intention-Guided Network (ALIGN)** is a shared autonomy framework for assistive teleoperation of robotic manipulators. It learns *when* to assist (decision) and *what* to do (assistant) from a single shared **3-modal** visual-motion-language representation — using contrastive vision-trajectory-language alignment as a natural gating signal.

### Core Idea

A 3-way contrastive pretraining step aligns vision embeddings (from egocentric camera frames), trajectory embeddings (from noisy teleoperation poses), and language embeddings (from task descriptions) into a shared space. The pairwise cosine similarities between these embeddings serve as a confidence score that:

1. **Gates assistance** — when all three modalities agree (vision, motion, and task description), assistance activates. When any disagrees (wrong object, uncertain intention, novel scene), it defers to the human.
2. **Disambiguates objects** — text explicitly specifies the target ("the left mug", "the red one") when vision alone can't distinguish.
3. **Detects out-of-distribution inputs** — low alignment across any pair → safe fallback to pure teleoperation.
4. **Filters tremor** — the trajectory encoder learns to suppress noise that doesn't correlate with visual features or task semantics.

### Key Design Decisions

- **Hybrid shared encoder** — single vision + trajectory + text encoder with two lightweight heads (Decision, Assistant). Text is computed once per task (~5ms) and cached. Runs at 25Hz on G1's Jetson Orin.
- **3-way contrastive alignment** — InfoNCE on all three pairs (vision↔trajectory, vision↔language, trajectory↔language). Gating uses min of all three for maximum safety.
- **No object detector** — frozen DINOv2 + CLIP encoders work on novel objects without retraining.
- **No hand-tuned α** — the contrastive alignment provides a principled, data-driven gating signal from three modalities.
- **Text-conditioned corrections** — same visual scene + same motion but different task description → different Δpose. The assistant is task-aware.
- **Graceful degradation** — any alignment score drops on OOD inputs → min drops → α drops → human stays in control.

### Text Specificity Spectrum

The system does NOT require strict directions. It works across the full spectrum:
- **Specific** ("pick up the red mug") → highest α, object disambiguation via CLIP
- **Descriptive** ("pick up the mug") → high α if only one mug is visible
- **Neutral** ("pick and place") → moderate α, general smoothing, no disambiguation
- **None** → vision-only fallback, cos_sim_vt still gates

Training uses multiple text variants per episode so the model learns to calibrate confidence to specificity. Neutral text helps less because it *should* — the system is appropriately uncertain.

### Future Work

- **Enriched trajectory encoder input** — expand from (K,6) EEF pose to (K,13) by adding
  EEF velocity, angular velocity, and gripper state. Already recorded, trivially computed.
  Gives the Transformer explicit motion dynamics without the joint-level noise and 60% zero-padding
  of RDT-1B's full 128-dim unified action space.

### Development Plan

| Phase | Platform | Scope |
|-------|----------|-------|
| 0 | Isaac Sim + Franka Panda | Simulated pick-and-place, data collection, offline training |
| 1 | Franka Panda (real) | Real hardware validation, user studies |
| 2 | Unitree G1 arm-only | Full humanoid, fixed-base pick-and-place |
| 3 | Unitree G1 full-body | Add locomotion coordination |

### Venue Target

ICRA 2026 / RA-L. Core contribution: using contrastive vision-trajectory alignment as a learned assist gating mechanism — the first unified framework where the same embedding space drives both the decision to assist and the correction itself.

### Directory Structure

```
ALIGN/
├── README.md              ← This file
├── INITIAL_PLAN.md        ← Problem statement, motivation, overview
├── ARCHITECTURE.md        ← System architecture, components, data flow
├── TRAINING_PIPELINE.md   ← Data collection, ground truth, training stages
├── LITERATURE_REVIEW.md   ← Related works, baselines, gaps, position
├── CONTRIBUTION.md        ← Novelty, paper framing, ablation design, venues
├── DESIGN_DECISIONS.md    ← Trade-offs, alternatives considered, rationale
├── TASK_PLAN.md           ← Detailed phased checklists, timeline, risks
├── PRESENTATION.md        ← Professor presentation deck (15 slides)
├── CONVERSATION_LOG.md    ← Discussion history and decision log
└── scripts/
    ├── align_data_recorder.py
    ├── align_noise.py
    └── collect_episodes.py
```