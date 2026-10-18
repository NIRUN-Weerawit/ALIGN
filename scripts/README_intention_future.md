# Predictive supervision for the intention projection

This implements the first proposed experiment: action loss plus multi-horizon
future-feature prediction. It is opt-in and does not add task text, text masking,
terminal-outcome labels, or anti-collapse losses. Existing commands default to
`--intent-future-weight 0` and retain their original objective/checkpoint layout.

The final contextual intent embedding, before memory fusion, is the **only
input** to a small predictor. Each configured horizon has an output layer that
predicts future per-camera DINO CLS features, their change from the current CLS,
and change in raw robot state. Text and live observations cannot bypass the
intent embedding into this predictor. Future observations are training targets
only; inference never calls the predictor.

Targets use detached, per-camera layer-normalized raw DINO features, not the
trainable visual compressor/state encoder. Raw state differences use stored
translation units, shortest Euler-angle differences divided by pi, and raw
gripper differences. The three groups use equal-weight Smooth L1 losses.
Unaugmented targets are captured before camera/state dropout, so targets do not
inherit synthetic missing observations. Cache generation freezes DINO; raw
image training retains the existing frozen backbone and detaches target values.

For each supervised action anchor, only future positions strictly inside that
episode/segment are eligible. No terminal-frame repetition, cross-episode
targets, or padded targets. For crop sampling, choose a segment long enough
for the largest horizon; episode sampling is recommended. Horizons are **frame
offsets**, not seconds.

The objective is:

```
action_loss + intent_future_weight * future_loss
future_loss = sum(valid pair losses) / (valid action anchors * number of horizons)
```

This makes microbatch accumulation match the logical batch objective even when
future-target coverage differs. Anchors without any valid future have no
auxiliary contribution. FP32 target/loss computation still backpropagates through
the final intention projection and, unless frozen, through Mamba/input encoders.
An invalid nonfinite auxiliary loss aborts before updating parameters.

## Warm-start pilot: no memory, no task text

Run from the repository root, using a new output directory:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/whinnoy/.venvs/align-ablation/bin/python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2.external \
  --output checkpoints/intent_future_no_text_h16_h64 \
  --variants intent_no_memory \
  --warm-start checkpoints/piper_v4_ablation_no_text_4_cp_dim_64_bastet \
  --freeze-intent-backbone \
  --intent-future-weight 0.1 --intent-future-horizons 16 64 \
  --temporal-sampling episode --history-size 1 --compressed-dim 64 \
  --epochs 20 --batch-size 4 --micro-batch-size 1 --num-workers 1 \
  --lr 0.00002 --gpu-memory-fraction 1.0
```

`--warm-start` takes the parent run directory containing `intent_no_memory/`,
not the checkpoint file. The runner preserves matching trained parameters,
initializes the new predictor, and uses a fresh optimizer. A lower fine-tuning
LR than the original training LR is a starting choice, not a validated optimum.

`--freeze-intent-backbone` freezes vision, state, and the Mamba encoder, then
re-enables `intention_encoder.intent_proj`. The action head, future predictor,
and any existing text-conditioning modules remain trainable. This mode requires
a warm start or resume so it cannot silently freeze a random encoder.
Frozen encoders remain in evaluation mode during training, including disabled
state-encoder dropout.

For the text-conditioned pilot, use a new output directory, replace the source
with `checkpoints/piper_v4_ablation_use_text_4_cp_dim_64_bastet`, and add
`--use-task-text`. Text stays supplied normally to the action head. No
text-masked examples are introduced.

Remove `--freeze-intent-backbone` to train the full existing trainable model
with the auxiliary objective. `training/train_intention.py` supports the same
future-weight/horizon/hidden-dimension flags for joint training; its entry point
does not support the frozen pilot because it has no checkpoint warm start.
For its cropped segments, increase `--segment-max-mult` enough to reach the
longest horizon (the default five-frame setting cannot supervise 64 frames).

Auxiliary training supports `intent_no_memory` and `intent_memory`, and both
diffusion and flow matching. Disabled-intent variants reject the auxiliary flag.
`--intent-future-hidden-dim` defaults to 256. Resume with the same arguments and
`--resume`, increasing `--epochs` if desired. Changing weight, horizons, hidden
dimension, or freezing mode during resume is rejected; use a new warm-start run
for a changed experiment. Future heads are not copied across changed horizons.

## Metrics and checkpoints

`train/loss` and `val/loss` retain the action-loss meanings used by old runs.
Auxiliary runs additionally log action/total losses and:

- `intent_future_loss`: objective-normalized future loss.
- `intent_future_pairs_h16`, etc.: valid target coverage.
- `intent_future_loss_h16`: mean loss per valid horizon pair.
- `intent_future_visual_h16`, `intent_future_visual_delta_h16`,
  `intent_future_state_delta_h16`: component losses.
- `intent_future_persistence_h16`: the same loss for copying current vision
  and predicting zero changes. This is a reference, not another trained model.

Training versions have a `train/` prefix; validation metrics retain the runner's
existing unprefixed diagnostic convention. Total/action metrics use valid-anchor
weighting; legacy `train/loss` and `val/loss` preserve their existing averaging.
Checkpoint selection remains the configured action metric, rather than silently
switching to auxiliary loss. New checkpoints save predictor weights/settings;
standard evaluation and intervention loaders reconstruct them for strict loading
but do not execute the future predictor during action inference.

Before claiming improvement, require better held-out future prediction relative
to persistence, repeat intention probes and same/different-task interventions,
and evaluate simulator success. Task decoding alone still does not establish
inference of a future objective.

## Implementation validation, 2026-10-18

Bounded GPU smoke runs are under `checkpoints/intent_future_pilot_20261018`
(two training microbatches per epoch, four epochs including resume checks) and
`checkpoints/intent_future_text_pilot_20261018` (one microbatch per epoch, two epochs).
These contain only four and two optimizer updates respectively; they are not
quality-trained models. Logs/protocol checks are under
`reports/intent_future_implementation_2026_10_18/`.
