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

The trajectory evaluator's existing `--switch-at` protocol replays expert
commands while streaming actual simulator observations. At handover it now
snapshots the selected task, clears queued pre-handover predictions, and plans
fresh actions. Task tokens remain fixed while execution state and visual/state
memory update. Recurrence is preserved across handover and reset per episode.

The explicit API is `IntentionStream.freeze_task()` / `resume_human()`. This
does not implement calibrated automatic takeover or a source-aware hardware
command recorder. Fixed-cadence simulator rollout is the supported target.
Actual manipulation success, contact competence, and history-dependent benefit
still require closed-loop trials and appropriate measurements.

See [integration results](../paper/task_context_policy_integration_2026_10_20.md).
