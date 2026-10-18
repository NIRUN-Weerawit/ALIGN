# Frozen intention usage and history assessment

Run after `probe_intent_clustering.py` has produced its 21-position reference
probe. This assessment currently supports intention-enabled, memory-disabled
diffusion checkpoints with direct observation history size 1 and causal gripper
inputs.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/align-probe-mpl \
/home/whinnoy/.venvs/align-ablation/bin/python tools/assess_intention_usage.py \
  --probe reports/intent_task_probe_2026_10_18/no_text_intent \
  --output reports/intent_usage_assessment_2026_10_18/no_text_intent \
  --cache data/libero_goal.dinov2.external \
  --windows 8 32 128 --noise-draws 3 --head-batch-size 32
```

Repeat with `text_intent` for the instruction-conditioned checkpoint. Frozen
backbone construction is omitted because the cache already supplies its exact
features. All remaining weights load strictly, including the complete head.
Encoder hashes must agree with the reference probe. Extraction uses FP32 with
TF32 disabled. No policy parameters are trained or changed.

## History diagnosis

Use the same 384 training / 44 held-out episodes as the reference. At 0, 25, 50,
75 and 100% observation progress, compare:

- Full causal history and an independent single-frame reset.
- Last 8, 32, and 128 frames, clipped to the available prefix without padding.
- Reversed past observations, retaining the real current observation last.
- Repeated current observation/state for the original prefix length. This is a
  counterfactual control for sequence length and state accumulation; it adds no
  new past information and is not a valid deployment history.
- The last Mamba observation output, the learned-query output before its final
  projection, and the projected intention embedding. These help locate where
  task information becomes less linearly accessible. Their dimensions differ.

Classifiers and scalers fit training episodes only. Native appended-query
readouts must match the saved causal reference within a stated FP32 tolerance.
PCA measures unscaled training-feature variance concentration, not exact rank.

## Action-head interventions

Replay held-out expert episodes, selecting five positions through the last
complete action chunk. No future target padding. Current visual/state features
are fixed across intent interventions. Three diffusion noise draws are matched
across conditions. Evaluate complete DDIM action sampling and epsilon error at
three saved-schedule noise levels.

Cases: correct intent, identical-condition repeat, zero intent, another episode
of the same task, another task, the episode's initial intent, reset/recent-window
intents, reversed-past and repeated-current intents. Donors are different
episodes at matching normalized valid-anchor progress; three donor maps are
recorded. Same/different-task swaps can also change episode-specific information.

Direct visual/state zeroing are positive controls. For text-trained models,
the correct instruction is supplied to the action head and held fixed across
intent interventions; a different-task instruction is an additional control.
No task instruction enters the Mamba encoder.

Report action MSE, action changes relative to baseline, position/rotation/gripper
metrics, gripper flips and balanced accuracy, and epsilon MSE. Paired bootstrap
intervals resample whole episodes after averaging their progress/noise draws.
Identical-condition controls must produce exactly identical actions/epsilon
errors. Explicitly compare different-task vs same-task token replacement.

These are offline dependence and imitation-error tests. Sensitivity alone does
not establish useful task reasoning, and offline benefit does not establish
simulator success. Zeroing/counterfactual histories can be out of distribution.

## Artifacts

- `history_features.npz`, `head_features.npz`: reusable frozen features.
- `history_results.json`: per-progress task decoding and final PCA statistics.
- `head_results.json`: intervention summaries, per-episode/per-phase errors,
  donor maps, saved diffusion schedule and matched-noise controls.
- `protocol.json`: checkpoint/config/hash, anchor definitions and limitations.

Use `--reuse-features` to repeat analysis/sampling without extraction. Keep the
reference checkpoint, windows, data, and output directory consistent.
