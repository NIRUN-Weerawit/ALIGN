# Frozen intention usage assessment

Both checkpoints use compression 64, one 128-D intention embedding, no memory, causal gripper input, and diffusion action heads. Best checkpoint epochs: no-text 164, text-trained 200. No policy weights were changed.

## Action-head dependence

44 held-out expert episodes, five valid action anchors each, three matched diffusion noise draws. Every 8-step target chunk is complete. Current visual/state features and normal task instruction are held fixed across intent interventions. Donors are distinct held-out episodes at matched normalized progress. Repeat-condition controls produced exactly zero differences.

| Intervention | No-text action MSE | Text-trained action MSE |
|---|---:|---:|
| baseline | 0.024085 | 0.015938 |
| intent_zero | 0.025867 | 0.016026 |
| intent_same_task | 0.024071 | 0.015962 |
| intent_other_task | 0.024113 | 0.015942 |
| intent_reset | 0.023617 | 0.015991 |
| intent_recent_8 | 0.024085 | 0.015938 |
| visual_zero | 0.981626 | 1.013926 |
| state_zero | 0.026203 | 0.018280 |
| text_other_task | unavailable | 0.050128 |

Different-task vs same-task intent swap action-MSE differences (paired whole-episode 95% bootstrap):
- no_text_intent: 0.00004159, interval [-2.358868205192266e-05, 0.00012391742784529925].
- text_intent: -0.00002002, interval [-6.166047660371987e-05, 1.274879080028769e-05].

No clear offline task-specific benefit of the correct intent over natural same/different-task donors is established. Zeroing intent increases no-text MSE by 7.40% (absolute delta 0.001782, interval [0.001393, 0.002167]); zero is out of distribution, so this can reflect generic conditioning/calibration. Text-trained zeroing changes MSE by only 0.55%, with its interval including zero. Wrong text increases text-trained MSE from 0.015938 to 0.050128 (about 3.15 times baseline), demonstrating strong instruction dependence.

The no-text zeroing penalty is concentrated at the first anchor: MSE changes
from 0.031767 to 0.040519. Later anchors change much less. Exploratory phase-wise
same/different-task comparisons show tiny, inconsistent differences, rather
than a growing task-specific advantage through the episode. These phase-wise
intervals are not adjusted for multiple comparisons; the full phase results
are in each model's `phase_head_results.json`.

## Locate task information

Each row uses a train-only scaler and linear task classifier on the same whole-episode split. Final-position balanced accuracy, chance 10%. The PCA alternative fits only training features without task labels; its task classifier then uses training labels.

| Representation | No text | Text-trained |
|---|---:|---:|
| observation_hidden | 97.5% | 97.5% |
| readout_hidden | 78.5% | 97.5% |
| full | 35.3% | 54.8% |
| reset | 70.7% | 77.0% |
| recent_8 | 35.8% | 54.8% |
| recent_32 | 33.8% | 54.8% |
| recent_128 | 35.3% | 54.8% |
| reverse_past | 53.7% | 53.2% |
| repeat_current | 62.7% | 79.5% |
| Query hidden → training-only PCA 128-D | 97.5% | 97.5% |

The native query readouts match the previous causal-prefix reference within FP32 tolerance. The task signal is available before the learned final projection; 128 output dimensions alone do not explain poor decoding. Three random 128-D projections of the same query output yield 49.7–53.7% without text and 91.0–97.5% with text, also exceeding the learned projection. PCA is a diagnostic, not a drop-in policy fix: the action head was trained in the learned embedding coordinates. Different feature dimensions/scaling affect classifier performance, so this locates reduced linear accessibility rather than proving total information loss.

The learned projection weight matrices themselves are not nearly rank-one:
their singular-energy participation dimensions are 56.4 and 79.7. Low-rank
output variation therefore should not be equated with a rank-one weight matrix.

## History and scale

Truncating to eight frames changes sampled actions by RMS 0.000069 without text and 0.000017 with text, with zero gripper-class flips. The full-history task-decoding scores are also almost reproduced by eight frames. These checkpoints exhibit little practical action dependence on observations older than that window. Repeated-current histories decode better (62.7%, 79.5%) but are counterfactual and do not establish a valid deployment improvement.

Average head input L2 norms: visual 191.76 / 182.77, state 7.99 / 7.96, intent 0.85 / 0.39 (no-text / text-trained). This flags a conditioning-scale imbalance to inspect alongside the measured weak intent dependence; norms alone do not determine influence because learned weights can compensate.

## Assessment and next changes

1. Do not conclude Mamba cannot represent the task: its observation output and a dimension-matched PCA of its query output support 97.5% decoding on these expert episodes.
2. Focus on the learned query/projection and its training objective. Current action training does not yield a stable, task-discriminative final intention embedding. Investigate a training-only future-outcome/subgoal auxiliary objective, while keeping instruction out of encoder inputs.
3. Give intention a separately normalized/projected conditioning path and verify that different-task replacement hurts more than same-task replacement after retraining. Do not assume wider intention vectors or a longer observation window will resolve the issue.
4. Preserve an explicit current-only/short-history baseline. Adding more history to these fixed weights produces almost no action change beyond eight frames.
5. Evaluate any resulting model in simulator rollouts. These offline expert-episode errors and probes do not establish task success or prediction of an unobserved future goal.

No architecture/training changes were made. The best epochs and microbatch settings differ, and only one training seed and 44 held-out episodes were assessed. Zeroing, reversed histories, and repeated current inputs can be out of distribution. Donor swaps also change episode-specific information.

## Artifacts

Per-model protocol, frozen features, head summaries/episode rows, projection
controls, plots and history probes are saved under
`reports/intent_usage_assessment_2026_10_18/` (regeneratable, ignored by Git).
See [run instructions](../tools/README_intention_usage.md).

Validation: 15 tests passed, including causal prefixes, strict trained-head
loading, distinct task donors, paired resampling, and real Mamba streaming
parity. Identical-condition action/denoising controls were exactly equal in
both real-checkpoint assessments. No policy parameters were changed.
