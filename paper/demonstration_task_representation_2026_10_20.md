# Demonstration-derived task and context representation for ALIGN

Date: 2026-10-20. Design/research note; no policy or recorder implementation changed.

## Intended use

ALIGN observes a human teleoperating the robot and assists or takes over a difficult stage after the human's behavior has supplied enough evidence of the intended task. Task text is not required at deployment. Complete recorded episodes are available during training; only the current episode's causal prefix is available at handover. A complete prior demonstration, if supplied in a different deployment mode, is additional context and must be evaluated separately.

The central design recommendation is **a learned, probabilistic representation of intended outcomes and procedure, combined with a changing representation of progress and remembered evidence**. A short instruction, one class ID, or a single unconstrained pooled vector is unlikely to cover all these roles. The correct representation is the one that supports the right continuation under changed scenes, operators, and trajectories.

## What should be represented?

There is a useful distinction between three kinds of information:

| Information | Example | Desired behavior |
|---|---|---|
| Task objective and constraints | Insert this connector into that socket; preserve its orientation; keep contact gentle. | Persist through approach, contact, and insertion, while allowing revision if the human changes intent. |
| Episode context and historical evidence | Which socket was selected earlier, the connector's observed orientation, a previous failed attempt. | Preserve evidence even when it is occluded, and update facts when the world changes. |
| Progress and immediate control state | Connector grasped, approaching contact, partial insertion, current force and relative pose. | Change throughout execution; support precise local control. |

Uncertainty belongs alongside these representations. Before a distinguishing action, more than one objective can be consistent with the same history. The system should maintain those possibilities rather than average incompatible goals into one target.

One conceptual interface is

`C_t = {belief_over_goal_and_procedure, progress, historical_evidence}`.

The belief is conditioned on observations, measured states, and past human commands. The policy receives both this context and the live spatial observations. High-level context should not replace wrist-camera detail or measured contact feedback.

For a first experiment, use a few learned continuous tokens, for example four to eight tokens of width 128 or 256. Keep goal/procedure conditioning separate from the recurrent progress state and spatial memory. These sizes are pilot choices, not literature-established optima. Token names alone do not guarantee factorization: outcome, sequence, and control objectives must make the intended roles useful. If goal ambiguity matters, add a multimodal latent distribution or multiple hypotheses; a single diagonal Gaussian may not represent distinct task branches well.

An interpretable object/relation graph can complement the tokens when reliable tracking is available. It might store a target identity, object-relative destination, or an observed contact event. This is useful supervision and debugging scaffolding, rather than a requirement to manually enumerate every possible task. Appearance-only embeddings may miss small geometric or contact differences.

## Alternatives and their limits

| Representation | Useful information | Main limitation for this application |
|---|---|---|
| Task ID | Groups repeated instances of known tasks. | Omits instance-specific geometry, preferences, and progress; closed-set classification alone does not support new objectives. |
| Goal image or short terminal clip | Supplies a concrete desired visual outcome. | Does not distinguish two procedures that end in the same state, or identify invisible force constraints. |
| Full demonstration video | Shows ordering, object interactions, and execution style. | Expensive to retain verbatim; contains scene/operator details that may not define the task. A future demo suffix is unavailable during an ongoing handover. |
| End-effector trajectory | Captures motion and timing. | Can encode an accidental path instead of the intended outcome; absolute replay fails when geometry changes. |
| Object-relative state changes and event sequence | Captures which entities interact and how the procedure unfolds. | Depends on tracking, event observability, and annotation or simulation state. |
| Learned goal/procedure tokens with progress and memory | Can combine multiple sensory signals and supervise useful continuations. | Can collapse or exploit shortcuts without appropriate data and behavioral tests. |

Language can still describe an abstract objective or an otherwise unobservable constraint. It need not be the bottleneck, latent training target, or policy input. Conversely, video cannot reveal an unexpressed preference: if the operator has not demonstrated or otherwise communicated a constraint, observational inference has no evidence for it.

## What is the ground truth?

**The latent coordinates have no unique ground truth.** Two encoders can use different coordinates and support the same behavior. Collect evidence about intended and achieved behavior, then train the embedding through predictions and control. A successful trajectory supplies observed targets; it does not prove the operator's unobserved internal objective.

| Collected signal | Training target it supplies | Annotation burden |
|---|---|---|
| Synchronized external/wrist RGB, measured pose/joints/gripper, timestamps | Future features, object-relative changes, progress evidence. | Automatic logging; preserve observation validity. |
| Raw human input, command sent to the controller, and measured executed motion | Distinguishes operator intent, controller transformations, and actual effects. | Automatic logging with explicit action frames/units. |
| Expert continuation after a sampled prefix | Next action chunk and longer continuation. | Existing high-quality teleoperation recording. |
| Successful terminal interval | Achieved goal features and final object relations. | One success/failure decision, with a stable final observation interval. |
| Optional phase/contact/handover/correction markers | Event ordering, assistance eligibility, and recovery examples. | Footswitch/button markers or sparse post-hoc annotation. |
| Object identities/poses and contact/force | Target geometry and delicate-control feedback. | Easy in simulation; needs sensing or tracking on hardware. |
| Optional same-objective grouping across episodes | Which demonstrations should share task content despite different trajectories. | A group identifier or pairwise judgment; no sentence is required. |

The inspected [DataRecorder](../scripts/align_data_recorder.py) already writes per-camera frames, poses, gripper values, timestamps, optional reference/absolute poses, and episode metadata. Its `step` interface does not separately log raw human commands, executed controller commands, per-step control source, or force/torque. Other datasets/loaders have action fields; their actual semantics must be audited before treating them as human-intent labels. Derived pose differences cannot recover a command that was suppressed by contact or a controller limit.

## Collection procedure

1. Record complete successful human-controlled demonstrations with synchronized cameras, measured state, and commands. Retain failures and recoveries with explicit outcome/quality markers; do not train every failed action as an expert target or label every failed terminal state as the intended goal.
2. Preserve a brief stable interval after completion so the outcome can be observed. Record a minimal success flag and optional objective-group identifier. For tasks with procedural constraints, record relevant events as well as the endpoint.
3. Vary object arrangement, operator, approach path, and speed while preserving the same intended outcome. This provides evidence that the task can remain the same despite different trajectories.
4. Collect competing goals from the same or closely matched initial scenes, including shared early prefixes followed by different choices. This prevents the model from recognizing the background or robot pose instead of reading the human's behavior.
5. Sample training cut points throughout each episode, especially before/after the first disambiguating behavior and around the intended takeover stage. The encoder sees only the prefix; future actions and observations are targets.
6. Add supervised takeover/recovery trials after a basic policy is competent. Start from realistic human-led prefixes and include small deviations from expert states. Save requested/actual handover, human corrections, and outcomes. These examples address the difference between intent recognition and autonomous execution competence.

Split data by full episode and collection session, with held-out scenes/operators as appropriate. Prefixes and continuations from one episode must stay in the same split. A large number of overlapping windows from a handful of demonstrations does not create independent task evidence.

## A proposed training design

Use complete demonstrations as privileged training evidence, while training an online encoder to work from prefixes. This is an adaptation of latent-plan learning ideas to ALIGN's particular handover setting; it has not been implemented here.

```text
TRAINING
complete successful episode -> goal/procedure teacher -> latent target distribution
human-controlled prefix    -> causal context encoder -> inferred distribution
current observation + context -> existing action head -> expert continuation
context -> outcome/event heads -> recorded outcomes and procedure evidence

DEPLOYMENT
observations + measured state + past human commands
    -> causal context encoder -> goal belief + progress + memory
    -> existing action head with live visual/state features
```

The full-episode teacher should be trained to represent desired outcomes and relevant procedure, not merely reproduce every camera pixel or memorize episode identity. For example, supervise terminal object-relative relations, future task-relevant features, and ordered event evidence. If no reliable object tracker exists, start with frozen visual targets and measured state changes, while retaining current spatial detail at the policy. Check whether those targets actually distinguish the candidate goals.

Sample prefixes and train a distributional predictor to cover the teacher's episode latent. Early ambiguous prefixes must be allowed to remain ambiguous. An exact vector-matching loss at every prefix can reward averaging or impossible certainty. A mixture prior with latent-plan likelihood/KL matching is one possible implementation; the precise distribution and weighting need comparison. Teacher outputs are learned targets, not infallible GT. Use fixed/slowly updated targets and useful decoding losses to avoid jointly collapsing teacher and student.

Train the actual action head using the prefix-derived representation, including action supervision at intended handover stages. If the head only sees a teacher embedding containing the future, good offline action reconstruction can coexist with a failed deployed student. A teacher-conditioned head can be an upper-bound diagnostic, but final claims must use the causal encoder.

Use several complementary losses:

- **Expert continuation:** diffusion/flow action loss from causal context, emphasizing valid, high-quality expert segments.
- **Outcome prediction:** predict distributions over future/terminal visual features or object-relative relations; include a persistence baseline.
- **Procedure/progress:** predict events, future changes, or phase boundaries when annotations are available. Endpoint-only supervision cannot distinguish open-then-close from never-opened.
- **Cross-demonstration consistency:** encourage compatible goal content across different demonstrations of the same objective, with pose/path/operator variation. Keep progress content free to differ.
- **Distribution matching:** transfer full-episode evidence to the prefix predictor without forcing early certainty. Evaluate predictive calibration and unseen-goal rejection rather than treating latent variance as a calibrated confidence score.

The action head may ignore context if current vision/state already predicts the demonstration actions. Train and evaluate ambiguous-scene examples where earlier history changes the correct continuation. A shuffled-context intervention is therefore as important as a low reconstruction loss. Avoid using every other episode as a contrastive negative: two episodes can accomplish the same task. Group labels are useful offline even though task labels are absent at deployment.

The existing [future intention predictor](../models/intention_future.py) is a useful starting point for multi-horizon supervision, but currently predicts horizons 16/64 from an intent embedding and does not implement a full-episode teacher, terminal/task-relation targets, human-command input, or calibrated task belief. Its [implementation report](intention_future_implementation_2026_10_18.md) documents only smoke-scale training, rather than demonstrated task-inference improvement.

## How to judge whether the representation is useful

Use repeated task variants with controlled historical differences:

- Same current scene, different earlier human choices: correct context should select the appropriate continuation, and conflicting context should change it in the corresponding way.
- Same intended task, different layouts/operators/paths: the goal representation and successful continuation should transfer.
- Same endpoint, different required event ordering: the procedure representation should preserve the distinction.
- Before versus after an informative human action: ambiguity should decrease when evidence arrives, rather than merely with elapsed time.
- Intended task recognized, but contact geometry changed: evaluate control competence and correction need separately from goal recognition.

Measure human-to-model handover completion, wrong-object choices, intervention rate, gripper/contact behavior, force violations where observable, and calibration of assistance decisions. Include current-only, short-history, correct-history, conflicting-history, and no-human-command controls. Task clustering or task-ID classification alone cannot establish correct assistance.

For a small first pilot, use a simple two-target manipulation family with matched scenes and varied trajectories. Then add a task where the same endpoint can hide different procedures, followed by the delicate stage of interest. Pick the number of demonstrations and evaluation trials based on held-out performance and uncertainty, rather than assuming a fixed universal sample count.

## Relation to TTT and relevant precedents

Mamba, an attention encoder, and TTT are competing mechanisms for accumulating context. The semantic objectives and data determine what task information that context contains. TTT fast weights are internal memory state; they do not need GT weight values or automatically become an interpretable task representation. Establish the causal training/evaluation contract before attributing a gain to a memory backend.

[Learning Latent Plans from Play](https://proceedings.mlr.press/v100/lynch20a.html) learns a latent plan space from teleoperated play without task labels, supporting behavior-derived supervision. Its goal-conditioning setting differs from inferring an unspecified goal from an ongoing prefix.

[Learning to Share Autonomy Across Repeated Interaction](https://arxiv.org/abs/2107.09650) is particularly close to ALIGN's intended setting: it embeds observed human behavior into a distribution over latent tasks, learns assistance from repeated interactions, and handles uncertainty about unfamiliar behavior. This is a relevant baseline, rather than evidence that the proposed larger visual system will automatically generalize.

[MOSAIC](https://arxiv.org/abs/2110.13423) studies demonstration-conditioned multi-task one-shot visual imitation with attention and contrastive learning. [MimicPlay](https://arxiv.org/abs/2302.12422) learns latent plans from human play to guide robot control. Both support demonstration-derived conditioning; observing a completed external demonstration is distinct from identifying intent before the current episode finishes.

[Time-Contrastive Networks](https://arxiv.org/abs/1704.06888) learns multi-view video representations through self-supervision. [XIRL](https://proceedings.mlr.press/v164/zakka22a.html) learns progress-aware embeddings across embodiments. These motivate visual/progress supervision, while leaving task ambiguity and assistance competence to be evaluated explicitly.

The recommended ALIGN target is therefore a **causal, demonstration-derived task belief that predicts useful continuations and outcomes**, supported by progress state and sensory memory. Whole episodes supply training evidence; handover behavior supplies the decisive validation.
