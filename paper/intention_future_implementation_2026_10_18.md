# Predictive intention supervision implementation, 2026-10-18

Implemented the first proposed experiment: existing action loss plus an opt-in
multi-horizon future-feature objective on the final contextual intention
embedding. No task instruction enters the auxiliary predictor. No text masking,
terminal-outcome supervision, or variance/covariance regularizer was added.

## Behavior

- `--intent-future-weight` defaults to zero. Existing state-dict keys, policy
  initialization and objective remain unchanged when disabled. Auxiliary
  initialization preserves the policy's RNG stream for fair comparisons.
- Horizons default to frame offsets 16 and 64; hidden width defaults to 256.
- Predictor input is only flattened contextual intention tokens, before memory
  fusion. Each horizon predicts normalized future DINO CLS, CLS change, and
  normalized raw-state change. These are fixed detached observation targets,
  not trainable state/visual embeddings. Text and current observations have no
  independent predictor path.
- Only valid within-segment targets at supervised action anchors contribute.
  Missing horizons are excluded, with no padding or terminal-frame repetition.
  Raw states/CLS targets are captured before synthetic observation dropout.
- Smooth L1 components have equal group weights; normalization by action-anchor
  count and horizon count preserves the logical objective under microbatch
  accumulation. Logs include valid counts, components, combined/action losses,
  and a copy-current/zero-change persistence baseline.
- Frozen pilot mode requires a warm start or resume. Vision/state/Mamba remain
  fixed and in evaluation mode, including state dropout. The intent projection,
  action head and auxiliary predictor train; existing text conditioning remains
  supplied normally when enabled. Joint training is also available.
- Checkpoint configs include the objective options; resume rejects changes to
  those options. Warm starts do not copy auxiliary heads across changed
  horizons. Standard evaluation and intervention loaders support the new
  weights; action inference never invokes the auxiliary predictor.

## Verification

65 regression tests passed for future supervision, padding/gripper/memory
contracts, ablation configuration, two-process synchronization and evaluation
dispatch. New tests cover detached future targets, masked NaN padding, angle
wrapping, unavailable horizons, unequal microbatch weighting, original-policy
initialization, strict state-dict roundtrips, frozen dropout and projection
updates in both diffusion and flow-matching production loops.

Bounded actual-GPU checks used the existing compression-64 intent-only models:

| Check | No text | Text-conditioned |
|---|---:|---:|
| Optimizer updates | 4 | 2 |
| Validation episodes | 44 | 44 |
| Valid h16 targets in validation | 161 | 161 |
| Valid h64 targets in validation | 95 | 95 |
| Peak allocated GPU memory | 2.26 GiB | 2.32 GiB |
| Frozen trained encoder tensors unchanged | 52 | 52 |
| Intention projection updated | Yes | Yes |
| Resume completed | Yes | Yes |

The standard GPU evaluation loader loaded the new no-text checkpoint strictly
(321 tensors) and returned finite `(1,8,7)` actions while a hook would raise if
the future predictor were invoked.

These are implementation smoke tests, **not quality-trained models**. After
four/two optimizer updates, future prediction remains worse than persistence:

| Final horizon loss | No text | Text-conditioned | Persistence |
|---|---:|---:|---:|
| h16 | 0.22345 | 0.22929 | 0.14560 |
| h64 | 0.26756 | 0.26990 | 0.22288 |

No task-inference or simulator-success improvement is claimed. A full training
comparison, future-vs-persistence validation, intention probes and token
interventions remain necessary before accepting this as an improvement.

## Files and runs

- Implementation: `models/intention_future.py`, `models/align_intention.py`,
  shared V4 trainer, cached ablation runner, checkpoint evaluation/loaders.
- [Training commands and details](../scripts/README_intention_future.md).
- Smoke checkpoints: `checkpoints/intent_future_pilot_20261018/` and
  `checkpoints/intent_future_text_pilot_20261018/`.
- Logs, encoder equality checks and inference check:
  `reports/intent_future_implementation_2026_10_18/`.

The frozen-pilot dropout guard was added during verification; the final resumed
epoch of each smoke run exercises the corrected deterministic mode. The runs
are intentionally too short for policy-quality conclusions.
