# Request for Senior Engineering Review

## Context

I'm a robotics researcher working on **ALIGN**, a shared-autonomy
system for Franka/G1 robots in Isaac Sim targeting ICRA 2026. The
system learns shared embeddings (vision, trajectory, text) and
attempts to give the human operator corrective assistance.

## Current architecture

The current pipeline has three training stages:

| Stage | What | Output |
|-------|------|--------|
| **Phase 1a/1b** | Contrastive pretraining (InfoNCE) | Trained encoders (vision, traj, text) + mixer |
| **Phase 2 Stage A** | Future prediction head (cosine loss) | Predicted K future embeddings from K past |
| **Phase 2 Stage B** | **Corrective** assistant head (MSE) | K corrective deltas to add to human's action |

The α gating signal comes from the Stage A head's prediction error:
`α = 1 - cos_error / 2` (bounded [0, 1]).

At inference, the executed action is:

```python
executed = human_action + alpha * correction
```

## The problem I want to discuss

The current **Assistant head predicts corrective deltas** (small
adjustments to add to the human's action). I'm having doubts about
this design and want senior feedback on whether to redesign it as
**future trajectory prediction** instead.

## Why I think corrective vectors are wrong

### 1. Coupling to noisy human input
The corrective is only meaningful if the human's action is sensible.
If the human sends garbage, no corrective can rescue it — we just add
noise on top of noise.

### 2. α semantics are awkward
α currently gates "how much of a small fix to apply". This is two
layers of gating (the corrective itself is small, then we scale it
by α). The simpler formulation is "do we trust the model's plan?".

### 3. Mode collapse in training
Stage B train loss is ~0.0001 — suspiciously small. My hypothesis is
the model is collapsing to near-zero corrective output, because the
optimal constant prediction is the mean corrective (≈ 0). I'd like
to verify this is real, not a measurement error.

### 4. Decision head overlap
The Decision head's job is to predict the next K trajectory
embeddings. The Assistant head is then doing essentially the same
task in a different embedding space. Two heads with overlapping
responsibilities.

## My proposed alternative

Replace the Assistant head's job: instead of "predict corrective
deltas to add to the human's action", have it "predict the next K
absolute EEF poses" (in the same units as the dataset).

```python
# Proposed
future_poses = assistant_head(z_v, z_t, z_text)  # (B, K, 6)
# No current_action needed — the model plans from the current state
```

At inference, project the first future pose to an action (finite
difference or IK), and use α to gate whether to execute the model's
plan or the human's plan:

```python
if alpha > threshold:
    executed = model_poses  # autonomous control
else:
    executed = human_action  # follow the human
# Or blend:
executed = alpha * model_poses + (1 - alpha) * human_projected
```

This is essentially **gated autonomous control** — the model always
has a plan, α just decides whether to use it.

## Comparison table

| Dimension | Corrective (current) | Future trajectory (proposed) |
|-----------|---------------------|------------------------------|
| What the model predicts | K corrective deltas | K absolute EEF poses |
| α semantics | "How much of a small fix to apply" | "How much to trust the model's plan" |
| Dependency on human | Inherently coupled to noisy input | Independent of human input |
| Task alignment | Indirect | Direct (the actual goal) |
| Decision head overlap | Significant | Clean separation |

## My specific questions for you

### Question 1: Is the conceptual redesign correct?
**Does "gated autonomous control" (model has its own plan, α gates
execution) match your intuition for how shared autonomy should work?
Or is there a better formulation I'm missing?**

I'm worried I might be biased by the mode collapse I'm seeing and
overreacting. The corrective approach *could* be fine if I just fix
the training (e.g., better loss, harder regularization).

### Question 2: Decision/Assistant head consolidation?
The Decision head predicts K future *embeddings* of the trajectory
(using cosine loss). The Assistant head would predict K future
*poses* in the original space (using MSE). Should I:
- (a) Keep them separate (current plan): Decision = confidence
  signal, Assistant = plan
- (b) Collapse into one head: predict poses, derive α from
  prediction-vs-actual consistency
- (c) Predict poses from Assistant, but have Decision do something
  else (e.g., classification: "is the human on track?")

### Question 3: Loss function for future-pose prediction?
For training the Assistant head to predict K future poses, should I:
- (a) Plain MSE on all K poses (equal weight per timestep)
- (b) Weighted MSE: weight the first pose higher than the 5th (the
  near-term is what gets executed)
- (c) Some form of trajectory loss (e.g., directional error,
  Chamfer distance, dynamic time warping)

I have no intuition for which is right for robot pose prediction.

### Question 4: How to project the predicted future pose to an action?
At inference, I have `model_poses[0]` (the immediate next pose) and
need to convert it to an action the simulator can execute. Options:
- (a) Finite difference: `action = model_poses[0] - current_eef_pose`
- (b) Model-predictive control (MPC): track the predicted K-step
  trajectory with horizon=1, solving IK each step
- (c) Just send `model_poses[0]` as a target pose (depends on env
  interface)

### Question 5: What about the noisy human action at inference?
If I switch to "model has its own plan, α gates execution", what
happens to the human's action during the loop?
- (a) Ignore it entirely (model plan vs not, no human in the loop)
- (b) Use it as a hint to seed the model's prediction (e.g., concat
  it as input to the Assistant head)
- (c) Use it as the low-α fallback (when α < threshold, do what
  the human said)

I'm leaning toward (c) for safety (when the model is uncertain, let
the human drive) but want a second opinion.

### Question 6: Is this a known pattern in shared autonomy literature?
I haven't done a thorough literature review. Is "gated autonomous
control with learned α" a known thing? Are there better-established
alternatives (e.g., HMM-based intent inference, RL-based policies,
shared latent spaces)?

## What I need from you

- **Validation** of the conceptual redesign (or pushback if I'm
  wrong)
- **Suggestions** for the open questions above
- **References** to relevant papers (especially ICRA / RSS / T-RO
  work on shared autonomy or assistive teleop)
- **Pitfalls** to watch for in implementing this

## Things I am NOT asking

- Whether the rest of the architecture is right (encoders, mixer,
  contrastive loss) — that's a separate conversation
- Implementation details (PyTorch, HDF5, etc.) — those are fine
- Help with the ICRA 2026 deadline — I have my own plan for that

## Background for context (read if you have time)

The full project is at `github.com/NIRUN-Weerawit/ALIGN`. The
relevant files are:

- `models/align_model.py` — the model architecture (AssistantHead
  class around line 325)
- `training/train_heads.py` — Stage A and Stage B training
- `eval/eval_libero_trajectory.py` — the eval loop (where α is
  applied to the corrective)
- `docs/FUTURE_TRAJECTORY_ASSISTANT.md` — the design doc I just
  wrote, with more details
- `docs/DESIGN_DECISIONS.md` — the design history, in case you want
  to see why we got to the current architecture

Specific metrics I have:

| Metric | Value | Concern |
|--------|-------|---------|
| Stage B train loss (corrective) | 0.00006 | Too small, mode collapse? |
| Stage B eval RMSE | 0.0117 | Small corrective residuals |
| α (Decision head) | 0.49 ± 0.05 | Barely discriminative |
| with-align vs no-align | Mixed, often negative | No real improvement |

The α staying near 0.5 is the biggest red flag — it means the
Decision head isn't actually distinguishing "human is on track" from
"human is off track". This might be solvable separately (need
harder negatives in the contrastive loss, or a better α aggregation
strategy) but the corrective architecture limits the upside.

## TL;DR

I'm thinking of replacing the corrective-vector Assistant head with
a future-trajectory Assistant head that predicts K absolute EEF
poses. This makes the Assistant head a true forward-prediction
head, conceptually similar to the Decision head but in the original
pose space instead of embedding space. The α from the Decision head
then gates "model plan vs human plan" instead of "how much of a
small fix to apply".

Want your honest opinion on whether this is the right move before I
spend the implementation time.
