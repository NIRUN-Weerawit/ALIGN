# Named task estimates on failed expert-replay episodes

Date: 2026-10-21. Policy: `context_policy_1/context_no_memory/intention_best.pt`.

**The task estimator is correct at 40% handover in 13 of the 15 episodes that
failed the expert replay check.** Two bowl-on-plate episodes are confused with
other bowl destinations. A correct handover estimate does not ensure successful
control: nine of the eleven unsuccessful prefix-policy rollouts have a correct
estimate at handover. Estimates can also drift as the observed trajectory changes.

The evaluator now supports an optional fixed task-name decoder, comparing its
predictions with the episode's recorded task name at every observation. It logs
the continuing student estimate separately from the task tokens used by the
actor, including when the latter are snapshotted at handover. Task names and
decoder predictions are diagnostic outputs, not policy inputs.

## Which episodes and which estimator?

The 15 cases are all protected development episodes. Thirteen expert replays
never complete the task; two (`ep_000177`, `ep_000210`) complete it but fail the
pose-alignment check. No case in this requested subset is omitted. No
confirmation episodes were opened for this analysis.

The pretrained neural encoder and policy remain unchanged. A single
`StandardScaler` plus balanced multinomial logistic regression (`C=0.1`) was
fitted on all nine previously declared query points from each of the **340
training episodes**, giving 3,060 training token bundles. No development examples
or outcomes were used to fit or select it. This pooled-prefix decoder can run
at any observation without knowing episode length, normalized progress, phase
or task name. Its inputs are the four native student task tokens, normalized by
the encoder's training-only latent buffers.

This differs from the earlier separability experiment, which fitted a separate
classifier at each query. That 40%-specific classifier scored 14/15 on failed
expert-prefix cases. The new fixed online decoder scores 13/15; it additionally
confuses `ep_000170`. Neither result reflects a change in neural weights.

The decoder maps tokens to ten known training-task labels. Its probabilities
are uncalibrated and cannot establish competence or serve as validated takeover
confidence. A wrong decoded label may reflect a decoder limitation, an ambiguous
prefix, or a changed latent representation; it is not a literal declaration of
the student's internal goal.

## Accuracy through the observed trajectory

All entries below are episode counts, using the same fixed decoder:

| Observation stage | Expert replay | Autonomous start, no expert prefix | 40% expert prefix then policy |
|---|---:|---:|---:|
| Initial observation | 2/15 | 1/15 | 1/15 |
| At 40% of demonstration duration | **13/15** | 1/15 | **13/15** |
| Last demonstration observation | 12/15 | 0/15 | 11/15 |
| Final evaluation observation, step 299 | 12/15 | **0/15** | **4/15** |

For autonomous start, the 40% row is only a matched time index; it does not
contain an expert prefix. Of the thirteen cases where expert replay actually
fails task completion, eleven are classified correctly at the expert 40% prefix
and ten at the end of expert control.

Nine estimates that are correct at prefix-policy handover are wrong at step
299. For six of those nine, the first post-handover error occurs after the
demonstration's recorded duration; for three it occurs within it, with some later
recovery. The late metric therefore includes extended trajectories and possible
post-completion behavior. Two of these nine episodes are successful policy
completions, so late drift must not automatically be attributed as a cause of
failure. Of the seven unsuccessful cases that lose a correct handover estimate
by the end, causality is also not established: changing actions, scene state,
elapsed time and errors can all affect the representation and decoder.

At final autonomous-start observation, fourteen of the fifteen token bundles
are decoded as "turn on the stove" and one as "open the middle drawer of the
cabinet". None of the fifteen assigned tasks is stove activation. This describes
the decoder's behavior on non-expert policy trajectories; it does not prove that
the latent vectors have numerically collapsed or that a task label caused the
robot's actions. The initial task choice is weakly observed without human action.

![Task-label correctness throughout reconstructed episodes](../reports/task_context_task_estimation_20261021/analysis/task_estimate_correctness.png)

Green means the inferred label matches the recorded task; orange means it does
not. White dots mark 40% of recorded demonstration duration, and black ticks its
last observation. Correct task recognition and task completion are separate.

## What each failed-replay episode is classified as

The table uses the original human-prefix policy trajectories. Names are shortened
for readability; JSON sidecars contain the complete recorded and predicted names.

| Episode | Recorded task | Estimate at 40% handover | Estimate at step 299 | Policy completes task |
|---|---|---|---|---|
| ep_000002 | Wine → rack | Wine → rack ✓ | Wine → rack ✓ | No |
| ep_000008 | Top drawer + bowl | Top drawer + bowl ✓ | Top drawer + bowl ✓ | Yes |
| ep_000032 | Push plate | Push plate ✓ | Push plate ✓ | Yes |
| ep_000057 | Wine → cabinet | Wine → cabinet ✓ | Top drawer + bowl ✗ | No |
| ep_000078 | Cheese → bowl | Cheese → bowl ✓ | Push plate ✗ | No |
| ep_000130 | Wine → rack | Wine → rack ✓ | Wine → rack ✓ | No |
| ep_000160 | Wine → rack | Wine → rack ✓ | Push plate ✗ | No |
| ep_000170 | Bowl → plate | Bowl → cabinet ✗ | Wine → rack ✗ | No |
| ep_000174 | Push plate | Push plate ✓ | Turn on stove ✗ | No |
| ep_000177 | Wine → cabinet | Wine → cabinet ✓ | Wine → rack ✗ | Yes |
| ep_000184 | Bowl → plate | Bowl → stove ✗ | Turn on stove ✗ | No |
| ep_000210 | Push plate | Push plate ✓ | Turn on stove ✗ | No |
| ep_000215 | Top drawer + bowl | Top drawer + bowl ✓ | Push plate ✗ | No |
| ep_000271 | Cheese → bowl | Cheese → bowl ✓ | Wine → rack ✗ | No |
| ep_000320 | Bowl → cabinet | Bowl → cabinet ✓ | Middle drawer ✗ | Yes |

Of four successful prefix-policy episodes, all four are correctly classified at
handover; two remain correct at step 299. Of eleven unsuccessful ones, nine are
correct at handover and two at step 299. The numbers are descriptive for this
small, selected failure cohort and one policy seed, not an overall benchmark.

## Online evaluator behavior and validation

Pass `--task-estimator PATH.npz` to the existing trajectory evaluator. It adds:

- Recorded task, predicted task, uncalibrated top-one probability and correctness
  at each observation, with no task label supplied to the encoder or decoder.
- Initial, handover, 40%-time, last-demonstration and final checkpoint summaries.
- Separate predictions from the current student tokens and actor-used tokens.
- Per-episode `*_task_estimates.json` and `.npz` files containing named estimates,
  native task tokens and class probabilities, even with `--no-video`.
- Aggregate accuracy for all episodes, replay pass/fail, actual expert-task
  failure, and successful/unsuccessful policy groups in `result.json`.

Unknown recorded task names are flagged as outside the decoder vocabulary and
excluded from accuracy denominators; the classifier remains a known-class
decoder. Every selected episode still runs regardless of replay outcome.

The exported NumPy classifier matches sklearn probabilities on real training
vectors with maximum absolute error zero. **82 targeted tests pass**, including
unchanged executed actions and Torch RNG with diagnostics enabled, independence
of predictions from GT labels, encoder-fingerprint binding, unknown labels,
snapshot/current-token separation and existing context/evaluator regressions.

Two real 300-step policy evaluations further exercise online logging with a 40%
expert prefix. Both selected episodes fail manipulation:

- `ep_000002`: wine-rack estimate is correct at handover and remains correct at
  the final observation. This provides a concrete execution failure despite a
  correct decoded task.
- `ep_000184`: bowl-on-plate is decoded as bowl-on-stove at handover and remains
  so at the final observation. The handover score is 0.855 despite the wrong
  label, illustrating why scores are not calibrated takeover confidence.

These are integration cases, not a replacement full-set performance evaluation.

## Provenance of reconstructed observations

Old rollout traces did not contain tokens or images. The diagnostic therefore
restores their simulator setup, executes their saved actions, renders observations
and runs only frozen perception/context plus the label decoder. It reconstructs
45 trajectories × 300 observations: expert replay, autonomous start and human-
prefix policy for the fifteen failed-check cases. **Maximum EEF position error
against every corresponding saved trajectory is zero.** No action policy is
resampled for these reconstructions; the two online integration cases above
are separate new policy runs.

The current `context_policy_1_corrected_dynamic_04/result.json` uses another
comparison episode set, as documented in the earlier token report. However, all
fifteen relevant original prefix rollout files still match the SHA256 values
recorded in `context_policy_1_results_20261021/analysis.json`. Only these verified
traces and archived task/seed/handover/outcome records are used. A clearly marked
derived diagnostic reference is stored separately; the replaced full result
JSON has not been recovered, and no current evaluation input is overwritten.

Reset order matches the respective saved expert and policy phases. LIBERO
fixture poses are outside the flat simulator-state vector; the expert and policy
reset sequences differ, so comparisons do not claim identical fixture scenes
between those phases. Reconstruction checks saved EEF trajectories, not historical
pixel identity. Scene, control and duration confounds remain when interpreting
differences across trajectories.

See [machine-readable analysis and full names](../reports/task_context_task_estimation_20261021/analysis/analysis.json),
[expert/autonomous reconstructed estimates](../reports/task_context_task_estimation_20261021/failed_expert_reconstruction/result.json),
[original prefix-policy reconstructed estimates](../reports/task_context_task_estimation_20261021/failed_expert_handover_reconstruction/result.json),
and [live evaluator checks](../reports/task_context_task_estimation_20261021/live_policy_04/result.json).

## Running named task estimation

The local fitted artifact is
`reports/task_context_task_estimation_20261021/task_estimator.npz`. Use the ALIGN
environment with LIBERO configured and a new output directory:

```bash
python eval/eval_libero_v4_trajectory.py \
  --data data/libero_goal.h5 \
  --checkpoint checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --task-estimator reports/task_context_task_estimation_20261021/task_estimator.npz \
  --n-episodes 44 --switch-at 0.4 --task-context-mode dynamic \
  --max-steps 300 --seed 42 --no-video --no-plot \
  --out-dir reports/context_policy_1_named_tasks_04
```

The checkpoint's saved development split is used by default. The decoder is
bound to the context encoder, observation summary, normalization and configuration
and rejects an incompatible context branch. The action policy is unchanged.

To export the same type of decoder from completed cached token features:

```bash
python tools/export_task_label_estimator.py \
  --policy checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --probe reports/task_context_token_probe_20261021 \
  --output reports/task_estimator_new.npz
```

If the cached feature directory is absent, first run
`tools/probe_task_context_tokens.py` as documented in the previous token report.

## Implication for the next experiment

Three issues are now measurable separately: incorrect destination at handover,
loss of an initially correct task estimate during later observations, and failed
execution despite a correct task estimate. The current evidence supports testing
goal retention and controlled task-snapshot versus dynamic-context ablations,
alongside action-policy ablations. A snapshot can also preserve an incorrect
estimate, as in the two bowl-on-plate cases. No change to the deployment behavior
or claim of a causal fix follows from task-label decoding alone.
