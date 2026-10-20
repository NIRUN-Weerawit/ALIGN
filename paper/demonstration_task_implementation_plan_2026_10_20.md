# Concrete plan: task context learned from expert demonstrations

Date: 2026-10-20. Original status: proposed implementation and experiment contract. Model sizes, schedules, and loss weights below are initial engineering settings, not measured optima. The initial plan preceded implementation; subsequent milestone status is recorded below.

First-milestone follow-up: the target builder and completed-demonstration teacher
are now implemented and pass the development/confirmation information gates.
See [implementation and measured results](task_context_teacher_results_2026_10_20.md)
for the selected temporal readout/decoder, frozen artifacts, and limitations.
The causal prefix student milestone is also now implemented, trained, and
confirmed. See [student results](task_context_student_results_2026_10_20.md):
useful causal information passed; longer-history advantage over an independently
trained 16-frame control did not. Student-conditioned policy training and actual
handover evaluation below remain proposals.

## 1. Deliverable and scope

Build an instruction-free ALIGN variant that observes a human-controlled prefix and produces:

1. A distribution over compact goal/procedure representations.
2. A separate execution-state representation that changes as the task proceeds.
3. Action chunks conditioned on those representations and current visual/state features.

Complete expert episodes supervise learning. At inference, no future frame, completed demonstration suffix, task ID, or text enters the context encoder. Optional task-group IDs are offline supervision and evaluation metadata only. A complete prior demonstration is a different conditioning mode and is excluded from the first experiment.

Keep DINOv2 frozen and retain the existing diffusion action generator. Hold the existing episodic-bank configuration fixed within each comparison. Use Mamba as the initial temporal backend; compare TTT only after establishing useful task-context learning. This isolates the representation/objective question from the memory-update question.

## 2. Exact representation and interfaces

| Tensor | Proposed shape, excluding time | Meaning and use |
|---|---|---|
| `task_tokens` | `(B, 4, 128)` | Joint compact representation of desired outcome and relevant procedure. Token roles are learned; no claim that token 1 necessarily means the target object. |
| `belief_logits` | `(B, 4)` | Weights of four continuous latent hypotheses, not four enumerated task classes. |
| `belief_mean`, `belief_log_std` | `(B, 4, 4, 128)` each | Each hypothesis predicts a distribution over the four task tokens. |
| `execution_token` | `(B, 1, 128)` | Current execution state: progress, recent events, and readiness for the next action. It is not required to be a scalar percentage complete. |
| `memory_state` | Backend-specific | Episode-local Mamba state and, when enabled, the existing spatial bank. It is distinct from the task representation. |

The first `4` in the belief mean is the number of hypotheses; the second is the number of tokens. Four modes need not cover all possible ambiguity; held-out likelihood and behavior determine whether to increase capacity.

The initial action interface is a separately normalized residual into the existing global condition:

```text
c_t = c_current_visual_state
      + alpha_z * Project_z(LayerNorm(flatten(task_tokens)))
      + alpha_p * Project_p(LayerNorm(execution_token))
```

Both projections return the existing global condition width. Initialize the trainable residual scales to 0.01, so the existing policy is a useful warm start. Preserve the live patch-detail pathway. The new context replaces the old intent conditioning in this variant; do not concatenate a second uncontrolled intent branch. A matched baseline uses the same normalized projections.

Freezing one selected hypothesis during takeover is the first deployment policy. Do not average incompatible task-token means before sending them to the controller. Hypothesis weights are not calibrated success probabilities, and high task confidence does not establish manipulation competence.

## 3. Observation encoder

Produce one 256-dimensional summary per real observation. Inputs are:

- Per-camera frozen DINO CLS features, projected to a combined width of 256.
- A 2×2 grid average of raw patch features in each camera, projected to 64 per cell; with two cameras this supplies 512 spatial-summary dimensions.
- Measured robot pose/gripper, encoded to width 64, with an explicit validity mask and a documented orientation convention.
- The previous human command and previous executed controller command, each projected to width 32, when those signals actually exist.
- Camera/command validity, control-source indicator, and elapsed time since the previous observation.

Concatenate these and project to width 256. Preserve camera and grid-slot identity. Train normalization statistics on training episodes only. Do not provide normalized episode progress or total future episode length to the causal encoder.

When a legacy dataset lacks separately recorded commands, mask those inputs unavailable and call the run observation-only. An action array is a human-command source only after its provenance and timing are verified. Do not derive raw intent from suppressed executed motion, or pass the current target action into its own predictor.

Implement two encoders over this shared summary:

- **Training teacher:** two bidirectional Transformer blocks, width 256, four heads, with four learned pooling queries projected to four 128-D task tokens. It consumes the full real observation sequence in order, with padding masks and actual timestamps. Start with every cached frame; if memory forces downsampling, record a fixed cadence and retain annotated event/terminal frames. Do not silently erase short contact events through uniform sampling.
- **Causal student:** a width-256 Mamba1 sequence encoder. Read its state into the mixture parameters and a 128-D execution token. Reuse the existing recurrence/streaming implementation pattern, but the changed input width requires a newly initialized context branch. Existing Mamba weights cannot simply be reshaped. Warm-start only compatible common modules.

The teacher's bidirectionality is a training privilege. Student batch and streaming outputs must agree on every tested prefix. The compact Mamba control uses the identical summary input and readout sizes, so compression and patch access cannot masquerade as a training-objective gain.

## 4. Demonstration record and GT contract

Use a sidecar manifest initially instead of rewriting existing HDF5 recordings. New recordings should support these fields explicitly:

```text
episode_id, collection_session, operator_id, scene_instance_id
objective_group_id                 # optional; never a policy input
success, quality_flags
observation_timestamp[T]
camera_valid[T,V], state_valid[T]
human_command[T,7], human_command_valid[T]
executed_command[T,7], executed_command_valid[T]
control_source[T]                  # human / robot / mixed
terminal_interval = [start,end]
events = [(timestamp,event_type,source,confidence), ...]
handover_eligible_intervals = [...]
simulator_initial_state            # when reproducible simulation is available
```

Keep observation time, command time, coordinate frame, action units, and execution delay explicit. The online summary at time `t` uses only commands already available under the declared timing convention. The supervised action chunk is `a[t:t+8]` after observation `o_t`.

Minimal human annotation per clean demonstration is success/failure, a stable completion interval, and optional same-objective group membership. Add sparse event and takeover-eligibility markers for the controlled benchmark. Generic event types such as grasp/release/contact are physical labels, not task sentences. Automatic markers retain their provenance and are masked when unreliable.

Failures/retries remain useful context, but their terminal state is not automatically a desired outcome. Only quality-approved expert continuation segments contribute imitation targets. A successful recovery can supervise continuation after a failed attempt.

For simulation, record object states and completion/event predicates as privileged targets. For hardware, use available tracking and contact sensing; unavailable object pose or force is masked rather than invented. Claims about gentle contact require force/contact observations or a valid task-specific measurement.

## 5. Deterministic training targets

Build a versioned target cache once from frozen features and verified metadata. The representation has no hand-authored embedding GT; these observed quantities supply supervision:

| Target | Exact initial construction | What it tests |
|---|---|---|
| Terminal visual outcome | Average normalized per-camera CLS and 2×2 spatial summaries over the approved stable terminal interval. | Which scene outcome was achieved. |
| Terminal geometric outcome | Object-relative target poses/relations when available, with field masks. | Object binding and precision beyond global appearance. |
| Ordered procedure evidence | Eight chronological visual/state-change anchors spanning the approved procedure, plus ordered event labels when present. | Whether operations/order beyond the endpoint are retained. |
| Local future changes | Existing +16/+64-frame feature/state changes, masked at true episode ends. | Whether the execution token supports predictive state tracking. |
| Expert action chunk | Eight consecutive actions at an approved anchor, with no repeated terminal padding. | Whether causal context supports control. |

Use temporal intervals/indices only to construct teacher targets. They are not student inputs. Ordered visual/state anchors describe the observed realization; they are weaker procedure supervision than event predicates and do not establish unobserved constraints. Global visual targets may be insensitive to small object changes, which is why the controlled object-relative benchmark is necessary.

Use the current orientation/state conventions consistently in encoder inputs. Compute geometric targets in an appropriate rotation representation, not by subtracting equivalent axis-angle branches. Record physical frame rate, so +16/+64 means the same horizon within a run. A later resampled dataset needs an explicitly versioned horizon policy.

Keep the existing development split for engineering comparisons, but reserve a fresh episode/session split for final claims. The existing validation episodes have already supported repeated development and are not an untouched test set.

## 6. Training stages and losses

### A. Train an informative whole-episode teacher

Freeze DINO. Train the shared observation summary, teacher, target decoders, and a disposable teacher-conditioned action head on training episodes only. The teacher action head conditions on live features at an anchor and the teacher tokens; it is an oracle diagnostic, not a deployable policy.

Initial normalized objective:

```text
L_teacher = L_action_oracle
            + 0.5 L_terminal
            + 0.5 L_ordered_procedure
            + 0.1 L_group
            + 0.01 L_variance
```

Average target losses separately over valid modality groups before applying these weights. Terminal/procedure losses combine normalized-feature Smooth L1 with masked geometric/event losses where available. `L_group` is supervised contrastive learning on a separate 32-D projection for verified same-objective groups. It does not force all four task tokens equal across layouts, because instance-specific targets must remain distinguishable. Do not treat every different episode as a negative. `L_variance` discourages a constant representation but cannot by itself establish useful semantics.

Teacher gate: on validation episodes, real tokens must improve relevant terminal/procedure prediction over constant/current-state controls, and incompatible donor tokens must impair the corresponding predictions or continuation. A teacher that merely encodes scenery or episode identity does not pass. Save a fixed teacher/summary checkpoint and hash; freeze it before training the student. There is no moving teacher target in version 1.

### B. Learn a causal distribution over teacher task tokens

For each episode, create the frozen target `z* = flatten(teacher(episode))`, width 512. The student produces a four-component diagonal Gaussian mixture over that space. Train:

`L_belief = -log sum_k pi_k * Normal(z*; mu_k, diag(sigma_k^2)) / 512`.

Constrain log standard deviations initially to [-3,1] in the normalized teacher-token coordinates. Monitor mode utilization, latent distances, and held-out likelihood; the number of components is capacity rather than an uncertainty guarantee. The teacher itself can misrepresent a task, so this likelihood is only one diagnostic.

Sample 16 anchors per episode. Initial allocation: eight uniformly over valid action anchors, four near approved takeover/event intervals, four near observed goal-disambiguating behavior. When annotations are absent, fill from uniform sampling and log that distinction. Stream all intervening observations once; sparse action loss must not mean resetting or skipping the context prefix. Reset between unrelated episodes.

For balanced, visually identical early prefixes with competing outcomes, likelihood training receives both possible teacher targets. A point-estimate MSE target would encourage an intermediate representation; a mixture can retain alternatives. Verify this explicitly. Train on early ambiguity even though version 1 does not autonomously take control there.

Train the execution-token predictor with the existing masked multi-horizon future-change objective. It sees the prefix state, not teacher tokens. Do not call elapsed-time prediction physical task progress.

### C. Fit the deployed policy to student-derived context

Freeze the teacher and observation-summary target space. Train the student readouts/execution branch and existing action head through expert continuation. Begin with matched common-module warm starts; later joint fine-tuning is a separate comparison.

For latent-mode training, assign each expert example to a compatible mixture component using its teacher target. Feed that component's **student-predicted mean**, reshaped to four tokens, into the action head. The teacher supplies a training responsibility/label, not a future-informed token vector. Stop gradients through the discrete assignment. Keep `L_belief` active so component weights learn which outcomes are plausible from the prefix.

This assignment is privileged training supervision, as in a latent-variable model. At evaluation, no teacher responsibility is available: the policy must select using the student weights. Report deployable top-weight-mode results separately from any oracle best-mode score. The latter is only a diagnostic. If mode coverage is good but top-mode selection is poor, more imitation training is not evidence that the inference problem is solved.

Initial combined objective:

```text
L_student = L_action
            + 0.2 L_belief
            + 0.2 L_future_execution
            + 0.1 L_outcome_from_student
```

Decode student outcomes with the frozen teacher outcome decoder, initially under teacher-based component responsibility. All inference metrics use the live component-selection rule. Compare learning curves and gradient scales before expanding a run; these numerical weights are a starting point.

Optimizer starting settings: AdamW, learning rate 1e-4 for new context modules and 1e-5 for a warm-started action head, weight decay 1e-4, gradient clipping 1. Use cached frozen vision, effective episode batch 8 through accumulation, BF16 trainable modules where supported, and FP32 loss/distribution calculations. First profile a 100-update bounded run to set feasible sequence length and batch size; do not promise a full-training GPU-hour budget before that measurement.

### D. Train and evaluate actual handover continuation

Training windows in stage C learn to predict near each anchor. To test takeover, also train sampled suffixes that hold the task latent selected at handover fixed while the execution state continues to evolve. Use the expert suffix to update execution state during offline training; then evaluate the policy's own suffix in simulation. This exposes the gap between a changing per-prefix inference target and a fixed inferred task at execution time.

First use explicit human handover requests. Do not fit an automatic takeover threshold from latent variance alone. Subsequent autonomy calibration needs held-out task correctness and execution-success outcomes as separate signals.

## 7. Runtime state machine

**Human control:** every real observation updates the causal context and execution state. Only actual human commands are marked as human evidence. Maintain the hypothesis distribution and normal spatial memory.

**Requested handover:** select the top-weight task hypothesis using the student alone and save that task-token snapshot. Preserve the observed prefix and all episode-local state; do not reset on the authority change. The initial pilot logs ambiguity and evaluates handover at predeclared intervals.

**Robot control:** hold task tokens fixed. Continue updating execution state and spatial memory from real observations and actual executed actions. Do not turn the model's commands into additional demonstrations of human intent. Predict eight actions with the existing head; fix and record the action-execution horizon across compared arms. Start with horizon 1 for diagnostic responsiveness and compare a larger horizon separately.

**Human correction/resumption:** return to the human-evidence inference branch using the new human interaction and an explicit source/time-gap indication. Test this transition; it must not silently treat an unobserved gap as consecutive human demonstrations. Goal-changing and goal-preserving corrections can be evaluated separately later.

**Episode end:** clear both temporal states and the spatial bank. Save learned weights/config separately from ephemeral episode state. No permanent deployment parameter updates are part of version 1.

## 8. Concrete experiments and data budget

### Existing-data engineering pilot

Use the existing LIBERO Goal data/cache configuration for target construction, causal parity, and a bounded teacher/student run. Verify the actual availability of commands, success indicators, terminal observations, timestamps, and reset states rather than assuming fields exist. Task grouping can use verified existing metadata without encoding descriptions or supplying IDs to the policy. Existing scenes may reveal task identity; good LIBERO task decoding alone does not pass the memory/context test.

### Controlled collection pilot

Choose three contrasting task families compatible with the available simulator/hardware:

| Family | Two objectives to contrast | Required evidence |
|---|---|---|
| Target choice | Put the same object into either of two destinations. | A human movement reveals the target; later current observations can be matched or occluded. |
| Procedure order | Perform the same two operations in opposite orders, with matched final configurations. | Event ordering, not only terminal image. |
| Fine alignment | Complete one of two target-specific alignment/insertion variants. | Relative geometry, handover state, and a measurable completion condition. |

An initial collection budget is **30 complete expert episodes per objective: 180 total**. Pair objectives within matched scene configurations; split 20/5/5 scene/session groups per objective into train/development/test, keeping paired objectives in the same split. This is a feasibility budget, not a sample-complexity guarantee. Hardware tasks that lack the required target or contact measurements need a different observable success definition.

For an initial simulator screen, evaluate the 30 held-out episode prefixes with two matched policy-noise seeds across three trained seeds: 180 rollouts per arm, but only 30 distinct held-out episode configurations. Report uncertainty clustered by episode/task rather than pretending all calls are independent. Expand scene/operator coverage before strong generalization claims.

Use exact saved simulator initial states and identical recorded prefixes across arms. Record whether a task was already completed by the prefix. Count assistance success only for the intended remaining stage. For hardware, report how prefix equivalence was achieved; do not label randomized resets as exact demonstration replay.

### Arms, in order

1. Current-only/short-history baseline.
2. Existing Mamba intention baseline with the same action-conditioning normalization.
3. Compact Mamba with the same new summary and token/conditioning budget, trained by action loss only.
4. Full proposed teacher/student training on that compact Mamba.
5. Teacher-token oracle for diagnosis only.
6. TTT backend under the same learned representation contract, only after arm 4 shows useful historical conditioning.

Keep the spatial bank either disabled in all initial isolated arms or identically configured in every arm; test its addition afterwards. Test human-command availability as a matched ablation rather than mixing datasets across architecture comparisons.

## 9. Acceptance criteria and failure diagnosis

The first substantive gate is a paired closed-loop gain on target/procedure-sensitive handovers over arm 3, with no unexplained gripper/contact regression. Predeclare the minimum useful gain for the actual application before final evaluation; a 10-percentage-point absolute success gain is an initial planning target, not a claim that the proposed evaluation is powered to establish it.

Also require:

- Correct history beats conflicting history at fixed current observations and matched diffusion noise. Same-objective donor histories transfer better than incompatible histories after controlling instance geometry.
- Before a distinguishing cue, competing outcomes remain represented; after it, the correct outcome is selected more reliably. Mode entropy alone is not calibration.
- Order-sensitive tasks remain distinguishable when endpoints match.
- Teacher-oracle improvements are substantially retained by the deployed student selector; oracle success alone does not count.
- Whole-episode and streaming prefixes agree; no future length, labels, commands, or target frames enter causal features.
- Handover does not reset context; robot-generated actions do not update the frozen task belief; corrections and episode resets obey the state contract.
- End-to-end p50/p95 latency and peak memory fit the measured deployment budget. Report teacher training costs separately from online student cost.

If the teacher is weak, revise target supervision/data. If the teacher is strong but the student fails after observable disambiguation, investigate causal inference/memory. If both predict outcomes but the policy ignores swaps, revise conditioning and history-dependent action supervision. If task choice is correct but insertion fails, improve local control/sensing rather than task clustering. This separates causes that a blanket Mamba-to-TTT replacement would mix.

## 10. Implementation work packages

| Order | Proposed new/changed files | Reviewable output |
|---|---|---|
| 1 | New target/manifest builder under `tools/`; loader extension under `data/`; recorder fields in `scripts/align_data_recorder.py` when new collection starts. | Versioned split/target manifest, field provenance, masks, and example episode/prefix inspection. |
| 2 | New `models/task_context_teacher.py`, target decoders/losses, and a cached-feature training entry point. | Teacher checkpoint, immutable target-cache hash, held-out teacher diagnostics. |
| 3 | New `models/task_context_encoder.py` and mixture/progress losses; stream-state support. | Student likelihood/selection diagnostics and batch/streaming causal parity. |
| 4 | `models/align_intention.py`, `training/train_intention.py`, cached ablation runner, `IntentionStream`, checkpoint reconstruction, both rollout evaluators. | Opt-in context conditioning and complete human-prefix-to-model-suffix execution. |
| 5 | New context-intervention and handover assessment tool under `tools/` or `eval/`. | Paired correct/conflicting/empty history, oracle/student, and control-source reports. |
| 6 | TTT backend behind the same interface. | A matched recurrence comparison, without changing targets and policy conditioning simultaneously. |

Save a new context configuration/version, teacher hash, target-schema version, command provenance, split keys, and every state-machine option in checkpoints. Original checkpoints continue to reconstruct their original architecture. Warm starts copy only explicit shape-compatible common modules; no silent coercion of one intent vector into four tokens.

Meaningful implementation checks cover leakage by future-input mutation, variable-length padding and unavailable targets, event timing, missing-command masks, mixture/component shapes and finite gradients, teacher immutability, train-only target normalization, task/phase gradient paths, streaming parity, state isolation, fixed latent after handover, and exact checkpoint roundtrips. End-to-end validation includes a small real GPU optimizer run and paired simulator prefixes, followed by the controlled experiment above.

The first implementation milestone is the manifest/target builder plus whole-episode teacher diagnostics. It makes the proposed representation inspectable before committing to student-policy training or a new memory backend. This plan extends the [representation assessment](demonstration_task_representation_2026_10_20.md) and uses the existing [future-intention supervision](intention_future_implementation_2026_10_18.md) as one auxiliary objective.
