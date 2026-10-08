# Diffusion, gripper, and trained-head dependence audit

Implemented in the main working tree on 2026-10-08. Existing 80-epoch checkpoint weights were preserved.

## Diffusion correction

The old `num_steps` loop began at `num_steps - 1`. Validation passed action chunk size 8 and therefore started at noise index 7, while deployment started at index 9. Solver steps now select a descending grid over the entire usable schedule, independently of action chunk length. Validation and training diagnostics use the head default, matching deployment. DDIM uses the next selected index for alpha, and the final transition uses clean alpha=1. The epsilon denoiser, state and inversion run in FP32 even when conditioning was encoded under BF16 autocast; low-signal inversion otherwise amplifies BF16 prediction rounding.

Fresh schedules normalize alpha[0] to 1 and clamp terminal alpha to 1e-4 instead of the mathematically singular zero-signal endpoint. Training draws only noisy indices 1..10. Tests with an analytic epsilon oracle recover clean actions using 1, 4, 8, and 10 solver steps.

Legacy checkpoint buffers retain their exact saved values and load strictly. Their unusable zero-signal index 10 is skipped; the original usable 9..0 default schedule remains available. With 8 solver steps the new sampler still begins at index 9, fixing the horizon coupling. Resumed legacy training excludes both clean index 0 and the singular endpoint. Old checkpoints do not acquire the fresh schedule or better gripper learning merely by changing the inference code.

## Gripper correction

LIBERO predictions are dataset commands: 0=open and 1=close. A continuous score is thresholded at 0.5; the next observation carries this executed binary command, and the simulator receives the matching +1=open/-1=close polarity. Both synchronous and asynchronous V4 evaluators share this conversion. Pose scaling affects only the first six action dimensions. Causal dataset fallback remains the previous executed command, never the current target.

All supported training paths and validation use configurable gripper dimension weights, with default 1.0 instead of 0.01. Equal weight matters for diffusion because the target noise has the same variance in every dimension; the old setting suppressed gripper gradients by 100x. The standard trainer now defaults to validation threshold 0.5 like LIBERO deployment; signed-command datasets can explicitly request threshold 0.0. Existing ablation manifests retain their original weight; resuming requires matching it explicitly.

## Existing trained heads

GPU experiments used all 44 held-out episodes, their original deterministic 20-frame crops, history size 1, and anchors 0/6/12. Each intervention used identical initial action noise and identical epsilon-probe noise at indices 1/5/9. Intention interventions modify only the final head tokens; memory interventions bypass fusion or swap all bank streams across distinct tasks with current queries fixed. Real bank histories are restored after each intervention. Each variant contributes 132 predicted chunks / 1,056 action predictions, which are correlated samples rather than independent trials. Identical-input controls produced exactly zero action and epsilon differences for all four checkpoints.

| Checkpoint | Intervention | Position MSE | Change from baseline | Position prediction delta RMS |
|---|---|---:|---:|---:|
| no_intent_no_memory | baseline | 0.099378 | +0.00% | 0.000000 |
| intent_no_memory | baseline | 0.102030 | +0.00% | 0.000000 |
| intent_no_memory | intent_zero | 0.102092 | +0.06% | 0.009744 |
| intent_no_memory | intent_shuffle | 0.102227 | +0.19% | 0.010874 |
| no_intent_memory | baseline | 0.091671 | +0.00% | 0.000000 |
| no_intent_memory | memory_bypass | 0.102334 | +11.63% | 0.135313 |
| no_intent_memory | memory_shuffle | 0.092660 | +1.08% | 0.025310 |
| intent_memory | baseline | 0.089231 | +0.00% | 0.000000 |
| intent_memory | intent_zero | 0.100489 | +12.62% | 0.053367 |
| intent_memory | intent_shuffle | 0.091552 | +2.60% | 0.042427 |
| intent_memory | memory_bypass | 0.189956 | +112.88% | 0.359554 |
| intent_memory | memory_shuffle | 0.089785 | +0.62% | 0.031502 |

The intention-only head is weakly dependent on its token: zeroing or shuffling changes position error by less than 0.2%. The combined head shows stronger dependence: zeroing intention worsens position error by 12.6%, and cross-task token shuffling worsens it by 2.6%.

Memory-only and combined heads depend on the memory module. Bypassing it worsens position error by 11.6% and 112.9%, respectively. However, cross-task bank shuffling worsens error by only 1.1% and 0.6%. This distinction matters: sensitivity to removing a learned fusion transformation is not sufficient evidence of strong use of historical information. There is measurable bank-content sensitivity, but its prediction benefit is small in this diagnostic.

Gripper accuracy does not track position improvements. In the combined head, baseline accuracy is 71.2%, intention zeroing gives 69.8%, and memory bypass gives 73.3%. Thus memory is not consistently improving gripper prediction in these old weights. Zeroing the visual condition produces large motion errors and near-chance gripper accuracy across variants; those out-of-distribution controls only establish direct visual dependence.

These are action-prediction diagnostics, not new simulator success rates or statistically established policy improvements. One training seed, correlated chunks, one matched initial-noise draw per anchor, and out-of-distribution zero/bypass interventions limit causal quality conclusions. Cross-task shuffles are stronger evidence of information use than zeroing alone. Intention sensitivity does not establish semantic meaning.

Full results: `checkpoints/ablation_libero_goal_h1_e80_20261008/condition_dependence_cross_task_fp32_20261008/{summary.json,comparison.md}`. A preliminary within-batch episode shuffle is preserved under `condition_dependence_20261008`.

Reproduce with `scripts/probe_condition_dependence.py --run checkpoints/ablation_libero_goal_h1_e80_20261008 --output <new_directory>`. Run fresh controlled training with the corrected defaults before comparing training-quality gains.

## Verification

60 regression tests passed, covering analytic DDIM inversion, full-grid subsampling, strict legacy checkpoint loading, autocast precision, invalid budgets, actual executed gripper feedback, pose scaling, all supported training loss paths, padding, memory contracts, Mamba recurrence, and distributed reductions. `git diff --check` passed.

All four configurations completed a production GPU smoke run with one optimizer update each and full 44-episode validation using equal gripper weight and fresh diffusion schedules. Subsequent fresh-checkpoint GPU checks verified FP32 sampling, finite outputs, and indices 10..1 for every configuration. Artifacts are in `checkpoints/diffusion_gripper_fix_smoke_20261008`, including `fp32_sampling_checks.json`. These untrained policies have large action errors and establish execution stability only. No 80-epoch retraining or new simulator success evaluation was run for this fix.
