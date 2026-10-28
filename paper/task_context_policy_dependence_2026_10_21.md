# Does the context policy use its task tokens?

The fixed `context_no_memory` checkpoint responds to task-token interventions,
but cached expert-state measurements show substantially stronger dependence on
its execution token. These results do not establish that task tokens are ignored,
or that removing them improves autonomous completion.

## Completed cached-observation study

[Detailed results](../reports/context_policy_dependence_20261021/summary.md)
and [raw measurements](../reports/context_policy_dependence_20261021/result.json).

The study uses all 44 protected development episodes, four causal observation
prefixes (40%, 50%, 70%, 90%), and three matched diffusion noise draws per anchor:
528 matched predictions per intervention. Ten interventions produce 5,280 action
chunks. No training or confirmation evaluation occurred. The checkpoint is
`f345a86ca7449808f8b18579af8ab2940d6a89182549885092809e4ded0d9a3e`.

All repeated-condition action predictions are exactly equal. Fresh observation
summaries agree with the bound prefix cache to a maximum absolute error of
2.8610e-6. Replacement tokens are converted from the probe's normalized space
back into the native encoder space. Means and donors use 340 training episodes
only. Same-task donors control for incidental episode differences; different-task
donors are deliberately selected using labels for this diagnostic only.

| Intervention | Action MSE | Change vs baseline | First-action MSE |
|---|---:|---:|---:|
| Normal context | 0.035927 | — | 0.033426 |
| Freeze task at 40% | 0.036257 | +0.9% | 0.033567 |
| Training-mean task tokens | 0.036998 | +3.0% | 0.035867 |
| Different-task donor | 0.036639 | +2.0% | 0.034320 |
| Same-task donor | 0.035158 | −2.1% | 0.033177 |
| Remove task contribution | 0.034806 | −3.1% | 0.032705 |
| Remove execution contribution | 0.044198 | +23.0% | 0.040902 |
| Remove both contributions | 0.046365 | +29.1% | 0.042993 |

MSE averages all seven action dimensions in dataset units and all eight predicted
actions. First-action error is included because the evaluated controller executes
one action before replanning. Neither metric is a success rate.

Different-task swaps change position actions by RMS 0.0696 and flip 1.7% of
predicted gripper decisions, versus RMS 0.0244 and 0.2% for same-task swaps.
This establishes sensitivity to task-token content, but not correct semantic use.

The episode-bootstrap 95% interval for removing task context is
[−0.002546, +0.000254] in absolute MSE change; the interval includes zero.
The corresponding execution-removal interval is [+0.004583, +0.012066].
The task-removal benefit therefore remains uncertain, while execution removal
consistently worsens this offline metric. These intervals describe this selected
checkpoint and development set, not independent test-set generalization.

The task contribution has mean RMS 0.0380, execution contribution 0.0792, and
full condition 1.0207. Contribution norms are supporting diagnostics, not causal
importance scores. Direct interventions supply the more useful evidence.

## Interpretation and next training experiment

The evidence supports weak or inconsistent task-context benefit in the current
conditioning pathway. It does not support a blanket claim that task tokens are
unused. Effects vary by task: removing task tokens worsens wine-on-rack imitation
MSE by 0.00687, while improving several other tasks. A global accuracy number
would conceal this difference.

The next controlled training experiment should compare the existing residual
adapter against stronger task-token conditioning, keeping the encoder, data,
action head capacity budget, and split fixed as far as possible. Preserve the
execution pathway, which contributes useful information in these measurements.
Only after isolating the adapter effect should joint encoder fine-tuning with
representation-retention objectives be introduced. More task-name classifier
training alone cannot change the deployed actor, which never consumes its output.

A simulator pilot is handled separately by
`tools/evaluate_context_interventions.py`. Its results must pass the initial
scene, expert-prefix, and repeated-baseline controls before completion changes
are interpreted. Cached imitation measurements above remain valid independently
of simulator repeatability.

## Simulator pilot: repeat control failed

The [pilot report](../reports/context_policy_interventions_pilot_20261021/summary.md)
is explicitly invalid for intervention completion comparisons. The predeclared
selection was one development episode per task (lexicographic first key), with
no filtering on expert replay or policy outcome. The first repeated baseline,
`ep_000002`, diverged, so the remaining interventions were aborted.

The latest diagnostic matched the initial model geometry, flat simulator state,
and camera observations. It used the same environment/render context, soft
resets, deterministic Torch settings, matched per-step diffusion seeds, and
forced camera refresh at the current state. Nonetheless, camera hashes first
differed after step 53, visual/task conditions differed at step 54, and executed
actions first differed at handover step 60. Saved flat simulator states matched
until that first autonomous action. Thus this repeat cannot isolate token effects
in closed-loop outcomes. The precise root cause of the image mismatch remains
unresolved; these observations do not establish that the neural policy or the
original demonstrations are defective.

A Mesa software-renderer fallback could not initialize successfully in this
environment and exited with SIGSEGV before scoring any rollouts. Failed attempts
are preserved separately. The main evaluator and trained checkpoint were not
modified by these simulator experiments. Reproducible rendered observations are
required before interpreting a matched closed-loop intervention comparison.
