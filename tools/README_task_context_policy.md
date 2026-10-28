# Task context in ALIGN policy training

The pretrained observer now connects to `ALIGNIntentionModel`, the production
sequential trainer, the intention/memory ablation runner, and checkpoint/rollout
evaluation. Legacy configurations retain their original intention encoder.

The context configuration creates one causal Mamba context encoder and omits the
old intention encoder. It supplies four task tokens and one execution token.
The diffusion head adds separately normalized task/execution projections to its
existing current visual/state condition, with scales initialized to 0.01. The
base conditioning width and live patch pathway are retained. Optional memory
retrieves visual/state evidence; task context stays outside cognitive retrieval.

Optional `--task-context-fusion` choices are `additive` (the compatible default),
`gated_residual` (live-observation task gating), `film` (bounded task-conditioned
modulation of the live condition), and `cross_attention` (live observations query
the four task tokens). `--task-context-fusion-dim` defaults to 128; attention
uses four heads. Fusion configuration is saved and restored by the production
training, ablation, and evaluation paths. The execution residual remains
separate and live. These options change how tokens are read, independently of
the rollout snapshot-selection mode.

## Controlled token utilization experiments

`tools/context_fusion_cache.py` packs frozen actor features and expert action
chunks at 40%, 50%, 70%, and 90% prefixes. It verifies the policy/observer/cache
bindings and includes only the existing 340 training and 44 development episodes.
The confirmation partition stays closed. Build the compact cache once:

```bash
python tools/context_fusion_cache.py \
  --checkpoint checkpoints/context_policy_2/context_no_memory/intention_best.pt \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2.external \
  --token-probe reports/task_context_token_probe_20261021 \
  --prefix-cache reports/task_context_student_20261020/prefix_cache \
  --output reports/context_fusion_new/features.pt --gpu-memory-fraction 0.20
```

Run an equal-budget adapter screen:

```bash
python tools/experiment_context_fusion.py \
  --checkpoint checkpoints/context_policy_2/context_no_memory/intention_best.pt \
  --features reports/context_fusion_new/features.pt \
  --output reports/context_fusion_new/screen \
  --arms additive_dynamic additive_mixed gated_mixed film_mixed attention_mixed \
  --steps 200 --batch-size 32 --seeds 123 124 125 --gpu-memory-fraction 0.20
```

Only context adapters and optional fusion layers train; the action decoder,
visual/state encoders, and causal observer remain frozen. Existing trained
context adapters are copied, and optional interactions start neutral, giving
the same initial policy. `additive_dynamic` trains with current tokens;
`*_mixed` randomly uses current task tokens or the same episode's 40% snapshot,
while keeping execution tokens current. `execution_only` and `task_only_mixed`
are additional controls for which adapters receive updates. Training uses a
fixed update budget with matched batches/noise; development never supplies
optimizer updates or early stopping.

Every run saves a complete loadable policy checkpoint, exact protocol,
per-anchor development errors, no-task controls, and different-task donor
controls. These measure cached expert-state imitation, **not simulator task
completion**. Offline `freeze40` corresponds to evaluator
`--task-context-mode handover --switch-at 0.4`; evaluator `freeze` uses the
distinct highest-score selection rule. Do not equate the two.

Use `--steps 1000` with a new output directory to compare longer equal budgets.
Run `--arms execution_only task_only_mixed` separately for adapter-update
controls. Combine completed directories with multiple matched diffusion draws:

```bash
python tools/assess_context_fusion_experiment.py \
  --runs reports/context_fusion_new/screen reports/context_fusion_new/extended \
  --features reports/context_fusion_new/features.pt \
  --output reports/context_fusion_new/assessment \
  --noise-seeds 20261022 20261023 20261024 --gpu-memory-fraction 0.20
```

The assessment keeps optimizer budgets separate and reports paired episode
bootstrap intervals against both the original checkpoint and equally trained
additive conditioning. It emphasizes first-action error for action horizon 1.
Existing draw results are reused; additional draws never update policy weights.
See [the October 22 experiments](../reports/context_fusion_experiment_20261022/summary.md).

The normal `--policy-warm-start` intentionally excludes context adapters;
use the dedicated experiment to reproduce identical adapter-continuation
initialization. Fresh full-policy training can select a fusion with the normal
`--task-context-fusion` option. The main trainer's context sequence remains
dynamic; the mixed snapshot schedule above belongs to this bounded experiment.

By default, the pretrained observation summary and context encoder stay frozen.
The current visual/state modules, context adapters, action policy, and optional
memory train from expert continuation. Context inference receives no text, task
IDs, whole-episode teacher tokens, or current target actions.

## Protected split and inputs

Every arm in a context experiment uses the target manifest from pretraining.
Hash, episode disjointness/coverage, lengths, camera order, and observer binding
are checked. All arms inherit 340 training / 44 development / 44 confirmation
episodes. Confirmation is excluded from policy training and checkpoint selection.
Do not generate a fresh random split for these pretrained weights.

Training streams complete episodes at the declared 20 Hz cadence, with 16 action
anchors by default. Every intervening observation updates recurrence/memory.
State conversion matches pretraining: translation, two rotation-matrix columns
derived from the rotation vector, and the causal gripper fallback. Time is frame
index / fps, never normalized progress. Context experiments share FP32 matmul
across all arms and save this precision setting.

## Ablation training

Use the existing ALIGN CUDA environment and fetch pretrained weights with
`git lfs pull`. The checked local Python executable is
`/home/whinnoy/.venvs/align-ablation/bin/python`.

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2.external \
  --output checkpoints/task_context_policy_full \
  --task-context-checkpoint checkpoints/task_context_student_20261020/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_teacher_20261020/targets/manifest.json \
  --variants no_intent_no_memory no_intent_memory intent_no_memory \
             context_no_memory context_memory context_scratch_no_memory \
  --epochs 80 --batch-size 2 --micro-batch-size 1 --num-workers 0 \
  --head-window-batch-size 8 --diffusion-loss-repeats 1 \
  --gpu-memory-fraction 0.24
```

This is a full-training template, not a completed or optimized 80-epoch run.
Profile the actual hardware first. For a bounded integration pilot use
`--epochs 3 --max-steps 12 --max-val-batches 4` and a new output directory.
Those limits are recorded; these are not full-dataset epochs.

| Variant | Temporal context | Memory |
|---|---|---|
| `no_intent_no_memory` | Current observation | Off |
| `no_intent_memory` | Current observation + retrieval | On |
| `intent_no_memory` | Original intention encoder | Off |
| `intent_memory` | Original intention encoder | On |
| `context_no_memory` | Pretrained context | Off |
| `context_history` | Pretrained context + ordered visual/state window at the action head | Off |
| `context_memory` | Pretrained context | On |
| `context_scratch_no_memory` | New context, action-only training from scratch | Off |
| `context_scratch_memory` | New context, action-only training from scratch | On |

`context_history` reuses the `plain_history` action-head attention over an ordered
window of visual features and robot states. Set its length with
`--plain-history-size` (default 4), keeping `--history-size 1` for the other
context variants. The current task/execution tokens condition every history row;
the decoder attends to the rows with learned temporal positions. At episode start,
the first observation pads the head window. The context encoder still processes
each real observation once across the complete causal episode, including frames
between supervised anchors. This variant always uses the dense episode loader,
even when `--plain-history-loader sparse` is selected for other arms. It adds no
separate memory bank and keeps the observer frozen by default.

To compare against the current-observation context policy:

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2.external \
  --output checkpoints/context_history_comparison \
  --task-context-checkpoint checkpoints/task_context_student_20261020/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_teacher_20261020/targets/manifest.json \
  --variants context_no_memory context_history --plain-history-size 4 \
  --epochs 80 --batch-size 2 --micro-batch-size 1 --num-workers 0 \
  --head-window-batch-size 8 --diffusion-loss-repeats 1 \
  --gpu-memory-fraction 0.24
```

Saved checkpoints record the actual history length and support the existing
trajectory evaluator without extra history flags. This is an available training
variant; its effect on simulator success requires training and paired rollouts.

Scratch controls share the frozen observation summary/normalization but initialize
the causal backbone/readouts afresh. During trainable-context training, a
straight-through selector supplies weight gradients while its forward value
remains one top-weight hypothesis. Evaluation uses the actual top-weight mode.

`--task-context-trainable` enables optional action-only encoder fine-tuning at
0.1× the policy LR; the summary stays frozen. It does not add teacher-token or
future-prediction retention losses. The validated first policy stage freezes
the encoder; staged joint training with representation retention remains separate.

`--policy-warm-start path/to/intention_best.pt` copies compatible common policy
weights to every arm, retaining the observer and fresh context adapters. It
checks action dimension, horizon, camera count, compressed width, and head type.
It does not restore optimizer state or learned context adapters. Use `--resume`
with unchanged arguments for continuation. Observer/split hashes, trainability,
precision, and warm-start identity are checked on resume.

## Main training entry point

```bash
python training/train_intention.py \
  --data data/libero_goal.h5 --output-dir checkpoints/context_policy_main \
  --use-precomputed-dinov2 --dinov2-path data/libero_goal.dinov2.external \
  --task-context-checkpoint checkpoints/task_context_student_20261020/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_teacher_20261020/targets/manifest.json \
  --history-size 1 --chunk-size 8 --state-dim 64 --compressed-dim 4 \
  --head-d-model 64 --batch-size 1 --num-workers 0 --no-sample-during-train
```

Keep legacy intent tokens and text disabled. This entry point uses the same
complete-episode loader and sequential training/validation functions, bypassing
the old window-level random split. Distributed loaders retain the episode split.

## Checkpoints and assessment

Policy checkpoints contain context architecture/weights, normalization, observer
identity, episode split, and adapters. `eval.eval_intention.load_intention_model`
reconstructs them without the original observer file. Missing trained context
parameters fail explicitly. Legacy checkpoint loading remains supported.

```bash
python tools/assess_task_context_policy.py \
  --data data/libero_goal.h5 \
  --checkpoints checkpoints/task_context_policy_full/context_no_memory/intention_best.pt \
                checkpoints/task_context_policy_full/context_memory/intention_best.pt \
  --output reports/task_context_policy_full/development.json
```

This evaluates all development episodes by default, matches noise seeds, keeps
selected checkpoints fixed, and leaves confirmation closed. `--max-batches`
bounds evaluation. The generic `eval/eval_intention.py` also dispatches saved
split-bound policies to complete-episode evaluation. Its saved DINO cache path
must exist for cached evaluation; live-image rollout uses the vision encoder.

## Expert-prefix handover

The trajectory evaluator's `--switch-at` protocol replays expert commands while
streaming actual simulator observations. Starting with evaluation protocol v4,
`--task-context-mode freeze` retains the native task-token bundle with the
highest decoder top-task score observed so far. Selection starts at observation
zero, continues before and after handover, and replaces the snapshot only on a
strictly higher score. Equal or lower scores keep the earlier bundle. It requires
`--task-estimator`. No recorded task label enters this selection.

Execution state, observation recurrence, and visual/state memory keep updating.
A new score record clears queued predictions and immediately replans with the
selected task tokens. `--task-context-mode handover` preserves the legacy fixed
handover snapshot. `dynamic` uses every current task estimate. `auto` selects
highest-score freezing with a human prefix and dynamic mode at `--switch-at 0`.
Explicit `freeze` also supports autonomous starts. All snapshot state resets per
episode. The fraction refers to the original demonstration length, with `1`
meaning expert control throughout the simulation budget.

The evaluator automatically uses the checkpoint's exact development keys and
camera order. In default `--episode-set development` mode, an explicit
`--val-episodes` file may subset those keys; training and confirmation keys are
rejected. `--episode-set comparison --val-episodes FILE` uses the named cases
regardless of training exposure, and records train/development/confirmation counts
and held-out status. Unknown and duplicate keys are rejected in either mode.
The split-export script honors saved keys instead of reconstructing a random
window split.

The context checkpoint's 44 development episodes differ from
`checkpoints/piper_v4_ablation_val_episodes.txt`: only six overlap. To match the
earlier Piper scenarios, add:

```bash
--episode-set comparison \
--val-episodes checkpoints/piper_v4_ablation_val_episodes.txt
```

For `context_policy_1`, that list contains 33 training, six development and five
confirmation episodes. It supports a comparison on identical cases, not a shared
held-out benchmark. That requires a common split for representation pretraining
and action-policy training across the compared architectures.

```bash
python eval/eval_libero_v4_trajectory.py \
  --data data/libero_goal.h5 \
  --checkpoint checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --n-episodes 44 --switch-at 0.4 --task-context-mode handover \
  --out-dir reports/context_policy_1_corrected/handover_0.4 \
  --preflight-only
```

Run the same command with `--replay-only --no-video --no-plot` instead of
`--preflight-only` to validate expert replay without loading the policy/DINO.
Then remove `--replay-only` for actual policy rollouts; omit `--no-video` to save
videos. Keep distinct output directories for replay, handover and dynamic runs.
For highest-score selection, use `freeze` with `--task-estimator` as shown below;
the estimator is omitted in replay-only runs.

Expert replay always runs, even without video. Each run resets controllers and
restores the same captured simulator state using LIBERO's `set_init_state`.
An optional H5 `initial_sim_state` dataset or `--initial-states states.npz`
supplies original states. The NPZ must map each converted episode key to its
exact flat simulator state; do not guess a mapping from task names or the public
benchmark's initialization index. Missing states use an explicitly labelled
seeded reset, not a claimed restoration of the original demonstration.

Replay passes only if the explicit LIBERO task predicate succeeds and both mean
and initial pre-action EEF error are within `--replay-position-tolerance`
(default 0.02 m). This checks replay usability, not exact object-state identity.
Replay checks are informational: **every selected episode gets a policy rollout,
including failed expert replays**. No override flag is required. The old
`--allow-unverified-replay` flag remains accepted for compatibility. Horizon
`done` is not treated as task success. Success before handover is attributed to
the expert and excluded from the policy-eligible denominator, with counts saved.
Both expert replay and policy rollout end immediately on the first task success,
including success before handover. Episodes already successful at reset execute
zero actions. A failed episode continues until `--max-steps`. Traces end at the
actual termination step, and `reference_steps` retains the original demonstration
length. Later diagnostic checkpoints are unavailable if the episode ended first.

`result.json` includes `evaluation_complete`, `n_evaluated_episodes`,
`expert_replay_all_passed`, per-episode replay checks, handover mode and
phase-specific success. Replay failure does not suppress `success_rate`.
`success_rate` counts policy completions among episodes still incomplete at
handover; `episode_success_rate` counts any task completion across every evaluated
episode, including expert-prefix completions. The latter matches the old Piper
reports' completion scope, but episode/split/protocol differences still matter.
`qualified` is a deprecated alias of `evaluation_complete`, with no replay gate.
Replay-only has no policy success rate. `.npz` traces preserve
initial simulator state, executed actions, predicted actions and EEF errors.
Missing cameras/poses fail explicitly. Results stay inside `--out-dir` so old
checkpoint sidecars are preserved.

The explicit API is `IntentionStream.freeze_task()` / `resume_human()`. This
does not implement calibrated automatic takeover or a source-aware hardware
command recorder. Fixed-cadence simulator rollout is the supported target.
Actual manipulation success, contact competence, and history-dependent benefit
still require closed-loop trials and appropriate measurements.

See [integration results](../paper/task_context_policy_integration_2026_10_20.md).

## Checking whether task tokens distinguish tasks

`tools/probe_task_context_tokens.py` loads the exact frozen context branch from
a trained policy, checks its observer/cache/split identity, and fits diagnostic
task classifiers on the 340 training episodes only. It evaluates all 44
development episodes at multiple causal prefixes and fixed elapsed times, with
individual-token, execution-token, current-observation and time controls.
No neural model is trained and confirmation is excluded.

```bash
python tools/probe_task_context_tokens.py \
  --policy checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --cache reports/task_context_student_20261020/prefix_cache \
  --data data/libero_goal.h5 \
  --observer checkpoints/task_context_student_20261020/observer_frozen.pt \
  --output reports/task_context_token_probe_new
```

Use a fresh output directory. Outputs include the pre-scoring protocol,
feature arrays, confusion matrices, within/between-task similarities and figures.
The [completed analysis](../paper/task_context_token_probe_2026_10_21.md) finds
84% balanced known-task accuracy at 40% prefix, versus 10% chance. Current-summary
features score 87.5%, so decodability does not establish a history or policy benefit.

`tools/probe_task_context_handover.py` additionally extracts live simulator tokens
after an expert prefix using captured development replay states/actions. It
requires LIBERO/EGL and an exact development-only reference report. It validates
saved EEF trajectories, includes failed replays, and applies the same train-only
probes without adapting the encoder or classifiers to simulator development data.

## Predicted task name versus recorded task during evaluation

The `--task-estimator PATH.npz` flag adds named task predictions to
context-policy rollouts. A fixed label decoder is trained separately on training
tokens only. In highest-score `freeze` mode its top-task score selects the native
task-token snapshot used by the actor. In `dynamic` or legacy `handover` mode the
decoder remains diagnostic. Recorded GT task names never condition the actor or
the snapshot selector. The neural encoder remains frozen. To export a decoder from completed token-probe
features:

```bash
python tools/export_task_label_estimator.py \
  --policy checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --probe reports/task_context_token_probe_20261021 \
  --output reports/task_estimator_new.npz
```

This pools all predeclared training prefixes into one online classifier. It does
not need episode progress or the task name to make predictions. It validates the
context fingerprint on loading and runs without sklearn in the evaluator.

```bash
python eval/eval_libero_v4_trajectory.py \
  --data data/libero_goal.h5 \
  --checkpoint checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --task-estimator reports/task_context_task_estimation_20261021/task_estimator.npz \
  --n-episodes 44 --switch-at 0.4 --task-context-mode freeze \
  --max-steps 300 --seed 42 --no-video --no-plot \
  --out-dir reports/context_policy_1_peak_task_04
```

The local fitted artifact in this command already exists. When exporting a new
artifact, substitute its path. The checkpoint's development split remains the
default. Replay-only does not load a context policy and cannot use this flag.

Each episode saves `*_task_estimates.json` with predicted/recorded names,
correctness and uncalibrated scores, plus `.npz` with native tokens and all class
probabilities. Current inferred tokens and actor-used task snapshots are logged
separately. `result.json` summarizes checkpoint accuracy and replay/policy-outcome
subgroups. Unknown recorded names are flagged and excluded from accuracy.
Highest-score runs also save `snapshot_steps`, `snapshot_scores`,
`snapshot_updated`, and `candidate_scores` in the NPZ. The JSON records every
selection and per-episode score-record events. Protocol metadata explicitly marks
that decoder scores control task selection. Original v3 `freeze` reports remain
historical handover-snapshot evaluations; the new behavior is recorded as v4.

`tools/evaluate_task_estimates_from_replays.py` reconstructs expert and policy
task estimates under saved commands and checks the saved trajectories, avoiding
new action-policy sampling. `tools/assess_task_estimation_failures.py` builds the
named failure table and correctness plot. The
[completed failed-episode analysis](../paper/task_context_failed_episode_estimation_2026_10_21.md)
reports 13/15 correct at human-prefix handover, 11/15 at the demonstration's
last frame and 4/15 at step 299, with the long-tail and completion caveats stated.

## Quick evaluation summaries

The evaluator automatically prints a compact summary and writes `summary.md`
and `summary.json` beside `result.json`. These include policy completion,
split membership, per-task outcomes, expert replay failures, and task recognition
when a decoder was enabled. Episodes completed before policy control are kept
separate from policy-eligible outcomes. Failed expert replays remain included.

To summarize existing results and compare two runs without rerunning simulation
or loading a model:

```bash
python3 tools/summarize_libero_evaluation.py \
  reports/context_policy_1_corrected_dynamic_04_decode_2/result.json \
  --compare reports/context_policy_1_corrected_dynamic_04_decode/result.json
```

Add `--full` to print per-task results and individual handover mistakes too.
The saved Markdown always contains the detailed report. `--output-dir DIR`
places the derived summaries elsewhere; raw evaluation results remain unchanged.

Comparisons identify different episode sets and report paired outcomes only for
shared policy-eligible episodes. Recognition at first success uses matching
per-step `*_task_estimates.json` logs, with usable-log coverage reported explicitly.
Missing logs or unknown task labels produce unavailable metrics rather than
incorrect predictions. Final-frame recognition can include observations after
demonstration duration or task completion; inspect handover and first-success
recognition alongside it. Diagnostic class scores are not calibrated confidence.

## Does the action policy use task context?

`tools/probe_context_policy_dependence.py` runs matched-noise interventions on
all 44 saved development episodes, at 40%, 50%, 70%, and 90% expert prefixes.
It holds visual/state observations fixed and compares normal context, frozen
handover tokens, training-mean tokens, different-task and same-task training
donors, and removal of task/execution residuals. An identical-input repeat
control must produce exactly identical actions. No model weights are changed.

```bash
python tools/probe_context_policy_dependence.py \
  --checkpoint checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2.external \
  --prefix-cache reports/task_context_student_20261020/prefix_cache \
  --token-probe reports/task_context_token_probe_20261021 \
  --output reports/context_dependence_new
```

Outputs include `protocol.json`, `result.json`, and `summary.md`, with per-task,
per-prefix, and per-episode action errors. Bootstrap intervals resample episodes.
Action sensitivity and cached imitation errors do not establish completion-rate
benefit. The visual-zero arm is an out-of-distribution positive control.

`tools/evaluate_context_interventions.py` additionally runs a matched simulator
pilot. It defaults to the first development episode of each task, selected
without outcome filtering, with six arms and a full repeated-baseline control
on the first episode. Use a LIBERO-configured environment and a fresh output:

```bash
MUJOCO_GL=egl python tools/evaluate_context_interventions.py \
  --checkpoint checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --data data/libero_goal.h5 \
  --reference reports/context_policy_1_corrected_dynamic_04_decode_2/result.json \
  --token-probe reports/task_context_token_probe_20261021 \
  --output reports/context_interventions_new
```

If needed, add the installed LIBERO checkout to `PYTHONPATH`. The runner verifies
matching initial model geometry, simulator state, and both camera observations,
as well as identical executed expert prefixes and prefix trajectories. Each
observation step uses matched diffusion noise. All arms use 40% handover,
300 steps, and action horizon 1. The expert prefix and continuing encoder are
unchanged; interventions act only on action-head conditioning after handover.
Historical results supply episode/state provenance, not the paired baseline.

Set `--episodes-per-task 0` for all saved development episodes. The default
10-episode pilot is a mechanistic check, not a reliable architecture ranking.
Task-mean and swapped tokens come from the training partition only. Same-task
donors are available with `--arms`; the actor never receives task names. The
diagnostic uses recorded task identity only to select donor groups.

The completed [action-dependence study](../paper/task_context_policy_dependence_2026_10_21.md)
contains valid cached-observation results. The local simulator pilot failed its
identical-baseline repeat control, so it provides no intervention completion
comparison. The runner aborts on that failure and marks its report invalid.
It uses a shared environment/render context with soft resets and explicitly
refreshes cameras after each simulation step; this diagnostic observation
schedule is recorded separately from the main evaluator. Renderer repeatability
must be resolved before using the pilot to attribute success changes to tokens.

## When successful episodes develop task evidence

`tools/plot_successful_task_tokens.py` analyzes the saved traces of every
successful, policy-eligible episode in a completed development evaluation.
It plots the continuing task estimate, actor-used task snapshot, and all four
individual task tokens. The bundle uses the exact evaluator decoder; individual
tokens use separate diagnostic classifiers fitted on the 340 training episodes
and all nine predeclared training queries. No neural weights are updated.

```bash
python tools/plot_successful_task_tokens.py \
  --result reports/context_policy_2_corrected_freeze_05_1/result.json \
  --token-probe reports/task_context_token_probe_20261021 \
  --task-estimator reports/task_context_task_estimation_20261021/task_estimator.npz \
  --output reports/successful_token_evidence_new
```

Outputs include overview heatmaps, a four-panel figure per successful episode,
PNG/SVG exports, `timings.csv`, and `summary.md`/`result.json`. All plots mark
handover, the observation before the first successful action, and demonstration
end. The default strong-score display threshold is 0.8; stability requires one
second of consecutive observations. Timing records distinguish the retrospective
start of that interval from the later time at which it can be confirmed.
These criteria are uncalibrated diagnostics, not takeover confidence.

A positive centroid-gap curve means the recorded task's centroid is closer
than every competing task centroid, using cosine similarity after training-only
standardization. Strong predictions of the wrong task are shown explicitly.
Successful-episode selection does not establish performance on failures.

Older rollout traces contain only task tokens. The tool labels its cached
expert-demonstration comparison separately when showing execution, observation,
camera, state, and time features. It never substitutes these for missing rollout
execution tokens. Future evaluations with `--task-estimator` also save the native
`execution_tokens` array in the task-estimate NPZ, without changing policy actions.

## Correct-task oracle diagnostic

The oracle test asks whether correcting task identity helps the existing frozen
policy. It supplies native student vectors selected using the known task label.
This guarantees correct identity under the saved diagnostic decoder, rather
than certifying every detail of the context representation or setting a score
to 1.0. Future demonstration observations may be used, so these runs are
privileged diagnostics, not deployment performance.

Build an episode-specific bank using the earliest correctly decoded predefined
student query at or after handover. The builder checks the checkpoint, cache,
split, and decoder bindings and uses no simulator outcomes:

```bash
python tools/build_task_context_oracle.py \
  --checkpoint checkpoints/context_policy_2/context_no_memory/intention_best.pt \
  --data data/libero_goal.h5 \
  --prefix-cache reports/task_context_student_20261020/prefix_cache \
  --token-probe reports/task_context_token_probe_20261021 \
  --task-estimator reports/task_context_task_estimation_20261021/task_estimator.npz \
  --handover-fraction 0.4 \
  --output reports/task_context_oracle_new/oracle_bank.npz
```

Run three evaluations with the same checkpoint, 44 development episodes, seed,
40% handover, 300-step limit, and action horizon. Use `baseline`, `repair`, and
`episode` as the output subdirectories. Baseline omits both oracle arguments;
the other two use the command below with the corresponding mode and output:

```bash
MUJOCO_GL=egl python eval/eval_libero_v4_trajectory.py \
  --data data/libero_goal.h5 \
  --checkpoint checkpoints/context_policy_2/context_no_memory/intention_best.pt \
  --n-episodes 44 --switch-at 0.4 --task-context-mode handover \
  --max-steps 300 --seed 42 --device cuda --no-video --no-plot \
  --task-estimator reports/task_context_task_estimation_20261021/task_estimator.npz \
  --oracle-task-bank reports/task_context_oracle_new/oracle_bank.npz \
  --oracle-task-mode repair \
  --out-dir reports/task_context_oracle_new/repair
```

`repair` replaces only incorrectly decoded live handover snapshots. `episode`
replaces every snapshot with the correctly decoded same-demonstration vector.
Both freeze the selected task context after handover and retain live execution
context. Failed expert replays remain included. Ordinary evaluation defaults
are unchanged; oracle outputs explicitly record GT usage and privileged status.

After all three evaluations finish:

```bash
python tools/assess_oracle_task_context.py \
  --checkpoint checkpoints/context_policy_2/context_no_memory/intention_best.pt \
  --features reports/context_fusion_experiment_20261022/features.pt \
  --oracle-bank reports/task_context_oracle_new/oracle_bank.npz \
  --task-estimator reports/task_context_task_estimation_20261021/task_estimator.npz \
  --output reports/task_context_oracle_new/offline --device cuda
python tools/summarize_task_context_oracle.py --root reports/task_context_oracle_new
```

The comparison checks episode pairing and every actor-used oracle identity. It
also reports outcome changes in untouched repair episodes as a variability
control. A negative result cannot establish that the whole representation is
adequate: correctly decoded category identity can hide missing goals, objects,
or procedural context. The completed experiment is in
[the oracle report](../reports/task_context_oracle_20261023/summary.md).

## Full Piper checkpoint evaluation matrix

`tools/run_piper_context_evaluations.py` inventories every best checkpoint under
`checkpoints/piper_context_*/*/intention_best.pt`, including separate memory,
no-memory, history, compression, and head-width variants. It copies the weights,
embedded configs, and Python source into a fresh batch directory before running.
Context models run dynamic, highest-score freeze, and handover with a 40% expert
prefix. Text models run once from 0%, with no context-mode sweep. Each run uses
44 saved validation episodes, seed 42, 300 steps, action horizon 1, and no video
or plots. Existing output directories are never overwritten.

```bash
/home/whinnoy/.venvs/align-ablation/bin/python -u \
  tools/run_piper_context_evaluations.py \
  --output reports/piper_context_evaluations_new --max-parallel 4
```

The runner fits a decoder for the Piper frozen context coordinate system using
only the 384 training episodes and nine predeclared prefix queries. It validates
the exported NumPy probabilities and binds every context checkpoint to that
decoder. Freeze uses its uncalibrated top-class score for snapshot selection;
dynamic and handover use it only for reporting. No policy weights are trained.
The current runner expects the existing Piper-384 prefix cache and installed
ALIGN/LIBERO environment; their paths are recorded in the batch source.

GPU admission checks free memory before starting each subprocess. A batch stores
`batch_manifest.json`, `checkpoints.json`, individual commands and logs,
`batch_status.json`, and a live `summary.md`. When every run is complete:

```bash
python tools/summarize_piper_context_batch.py \
  --batch reports/piper_context_evaluations_new
```

This writes `comparison.md` and `comparison.json`, separating autonomous text
scores from assisted context scores and including per-task expert completion,
task recognition, selected epochs, configurations, and paired mode outcomes.

## Paired same-task / wrong-task token evaluation

```bash
/home/whinnoy/.venvs/align-ablation/bin/python -u \
  tools/run_task_token_counterfactuals.py \
  --reference-batch reports/piper_context_evaluations_20261026 \
  --output reports/piper_task_token_counterfactual_new \
  --seeds 42 43 44 --max-parallel 4
```

This pins the cp4/no-memory checkpoint from the reference batch and runs normal,
repeated normal, same-task training donor, wrong-task training donor, and removed
task contribution conditions. All conditions use a 40% expert prefix, fixed task
context after handover, live execution context, 300 steps, and action horizon 1.
The normal rollout captures explicit reset states for the other arms of its seed.
Diffusion noise is keyed by episode and observation step and audited without
advancing its generator. Environment auditing leaves camera refresh and physics
schedules unchanged. A simulator smoke test precedes the full matrix.

Training donors use native student tokens at 40%, filtered for correct decoder
identity. Related wrong goals are declared before rollout outcomes. Same/wrong
donor selection uses the recorded task label for this diagnostic; no evaluation
future observations or teacher tokens select donors. No policy weights change.
The no-task arm rebuilds conditioning from current observations and execution
context, removing the entire task contribution, including its projection bias.

Where objects and region definitions exist in the original scene, the runner
also checks the supplied wrong goal using LIBERO predicates. It never replaces
the original task-success criterion, and alternative-goal success does not stop
the rollout. Per-step payloads, removal masks, sampler noise hashes, and passive
environment audits are saved under each run's `counterfactual/` directory.

After all arms finish:

```bash
python tools/summarize_task_token_counterfactuals.py \
  --root reports/piper_task_token_counterfactual_new
```

The summary includes repeated-baseline variability, episode-cluster confidence
intervals across evaluation seeds, same-versus-wrong donor effects, a cohort
selected by correct normal task identity before intervention, and results by task.
This measures dependence and benefit within a frozen policy. A separately trained
no-task baseline is needed to establish a benefit of the architecture itself.
# Correct-task donor comparison across non-text Piper policies

Run the same privileged, GT-selected native student donor intervention across
all eight non-text checkpoints in the immutable Piper evaluation batch:

```bash
/home/whinnoy/.venvs/align-ablation/bin/python -u tools/run_correct_task_donor_batch.py \
  --output reports/piper_correct_task_donor_all_20261027 --max-parallel 4
```

This reuses the six completed cp4/no-memory normal/same-task runs explicitly and
adds 42 paired runs for the other seven policies. Each condition uses the same
44 validation episodes, seeds 42/43/44, predeclared training donors, per-seed
initial simulator states, 40% handover, live execution context, 300-step cap,
and action horizon 1. Memory/history features remain enabled as saved in each
checkpoint. No policy or encoder weights change. GT task labels select donors;
this is a diagnostic and is not deployable inference performance.

The runner snapshots sources, checkpoints, commands, donor data and configs,
runs one-episode checks for both arms of each new policy, and writes
`status.json`. After all full evaluations finish, it writes `summary.md` and
`result.json`, with paired episode-cluster intervals, per-task expert and policy
completion rates, and audit checks for scene state, expert prefix and noise.
Use the same command with `--resume` after an interrupted run; incomplete
directories are archived before retrying. Completed validated runs are retained.

## Alternating context/observation attention head

`--task-context-fusion aci` selects the new action head for diffusion or flow
matching. State and denoising time condition FiLM; action features alternately
attend to task/execution tokens and current camera patch tokens. See the
[ACI head guide](../scripts/README_context_aci_head.md) for architecture, training,
compatibility constraints and verification.

For ACI, `--policy-temporal-mamba` adds an independent Mamba trained jointly with
the action head while the pretrained context observer remains frozen. Its extra
temporal token joins the original task/execution tokens in context attention.
Width/layers are configurable with `--policy-temporal-width` and
`--policy-temporal-layers`; see the ACI guide above.
