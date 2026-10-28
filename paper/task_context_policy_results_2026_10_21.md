# Trained context policy: complete dynamic-context evaluation

**Provenance update, 2026-10-21:** this is an archived analysis. Its prefix input
had SHA256 `a0d05e89...`, recorded in the linked analysis JSON. The referenced
`context_policy_1_corrected_dynamic_04/result.json` has since been replaced
(observed SHA256 `73f2da98...`) with a comparison-set run containing 33 training,
six development and five confirmation episodes. The current start/prefix files
overlap on only six episode IDs. The paired statistics below describe the
previous inputs and must not be applied to the current files. See the
[independent task-token diagnostic](task_context_token_probe_2026_10_21.md).

Date: 2026-10-21. Model: `context_policy_1/context_no_memory`, selected epoch 199
from the 200-epoch action-policy run. Dynamic context remains enabled throughout
both evaluations. No text or separate episodic memory is used.

**On the same 44 development episodes, success is 3/44 (6.8%) from autonomous
start and 17/44 (38.6%) after a 40% expert prefix: +14 successes, +31.8 percentage
points.** All three original successes are retained; 27 episodes fail in both
conditions. No episode is excluded because expert replay failed.

This demonstrates useful continuation after expert control in this trained
system. It does not isolate the contribution of task inference: the prefix also
moves the robot and objects closer to completion. The trained policy remains
weak, especially from autonomous start and on several placement tasks.

## Inputs and integrity checks

Input reports:

- [Autonomous start](../reports/context_policy_1_corrected_dynamic_00/result.json)
- [40% expert prefix](../reports/context_policy_1_corrected_dynamic_04/result.json)

Both use the checkpoint's saved **44 development episodes**, with zero training
or confirmation episodes. They are the same between these two runs, but differ
from the shared Piper evaluation list: only six IDs overlap with that list.
No new checkpoint selection or training was performed for this analysis.

The two reports identify the exact same checkpoint, evaluator, protocol helper,
camera order, seed 42, 300-step budget, action scale 1, action horizon 1, no
temporal ensembling, dynamic task context and LIBERO rotation convention.
Only `switch_at` and output directory differ. The checkpoint SHA256 is
`f345a86ca7449808f8b18579af8ab2940d6a89182549885092809e4ded0d9a3e`;
it matches the current local file. The recorded evaluator/helper hashes also
match the current source.

All **88 saved rollout traces** were checked:

- Every requested episode received a policy rollout, lasting 300 steps.
- All executed/predicted actions, positions and reference errors are finite.
- Initial simulator states are exactly identical for all 44 pairs; their expert
  replay error sequences are also exactly identical.
- Trace lengths, handover indices and mean reference errors match the JSON
  records. Episode task names and random seeds match across runs.
- Both report zero expert-completed episodes before handover. Thus all 44 remain
  policy-eligible, and policy success equals overall episode success here.
  Every credited success occurs after the policy takes control.

Both protocols set `replay_filters_policy_episodes=false` and
`evaluation_complete=true`. `qualified` is the deprecated completion alias in
this evaluator version; it does not mean that every expert replay passed.
Original demonstration simulator states were unavailable; paired runs restore
captured seeded resets instead. These facts remain part of the reported setup.

## Complete results

| Measure | Autonomous start | 40% expert prefix |
|---|---:|---:|
| Requested / evaluated episodes | 44 / 44 | 44 / 44 |
| Successful policy completions | **3/44 (6.8%)** | **17/44 (38.6%)** |
| Already complete before policy control | 0 | 0 |
| Mean EEF reference error over all recorded reference frames | 0.33189 m | 0.07225 m |
| Mean EEF reference error over the same post-boundary reference frames | 0.41080 m | 0.11171 m |

The EEF means weight episodes equally, matching the input summaries. The common
window begins at each episode's 40% handover index and ends when its recorded
demonstration poses end. It compares the same timestep ranges across the two
runs; it does not extend through all 300 simulation steps where reference poses
are unavailable. The full-reference prefix mean also includes expert-controlled
steps, so it should not be interpreted as autonomous policy tracking alone.
EEF agreement with a particular demonstration is secondary to actual task
completion and is not a measurement of task recognition accuracy.

Paired completion outcomes:

| Outcome | Episodes |
|---|---:|
| Both succeed | 3 |
| Only prefix-assisted run succeeds | 14 |
| Only autonomous-start run succeeds | 0 |
| Both fail | 27 |

These are descriptive results from one sampling seed and 44 episodes in ten
known task groups, not an unseen-task or multi-seed generalization result.

### Task breakdown

| Task | Autonomous start | 40% expert prefix |
|---|---:|---:|
| Open middle drawer | 0/4 | 1/4 |
| Open top drawer and put bowl inside | 0/4 | 1/4 |
| Push plate to front of stove | 0/3 | 1/3 |
| Put bowl on plate | 0/5 | 0/5 |
| Put bowl on stove | 0/5 | 3/5 |
| Put bowl on top of cabinet | 0/5 | 3/5 |
| Put cream cheese in bowl | 0/4 | 0/4 |
| Put wine bottle on rack | 0/4 | 0/4 |
| Put wine bottle on top of cabinet | 0/5 | 3/5 |
| Turn on stove | 3/5 | 5/5 |

![Success by task](../reports/context_policy_1_results_20261021/task_success.png)

Autonomous start succeeds only on stove activation. The expert prefix extends
success to seven task groups in total, but bowl-on-plate, cream-cheese-in-bowl,
and wine-rack placement remain at zero. The records identify task failures;
they do not establish whether each failure arose from goal inference, grasping,
contact, gripper timing, geometry or recovery. Videos/contact annotations would
be needed for that attribution.

### Replay diagnostics, with every episode included

The full expert action sequence completes its task in 31/44 episodes. The
stricter replay check passes 29/44: 13 expert runs fail the task and two more
exceed the fixed initial/mean EEF tolerance. These outcomes are identical across
the two evaluations and do not filter the policy denominator.

| Diagnostic subgroup | Episodes | Autonomous start | 40% expert prefix |
|---|---:|---:|---:|
| Replay check passes | 29 | 3/29 | 13/29 |
| Replay check fails | 15 | 0/15 | **4/15** |
| Expert never completes the task, irrespective of pose check | 13 | 0/13 | **3/13** |

The final row overlaps the failed-check row; it is not an additional group.
Four prefix-run successes would have been missed by evaluating only replay-pass
episodes. Three occur even where full expert replay never succeeds. This is
consistent with useful autonomous continuation under imperfect replay, but does
not prove a learned recovery mechanism. It supports the requested inclusive
evaluation protocol. Replay imperfections remain documented conditions of the
benchmark, rather than grounds for removing difficult cases.

## Reflection and next experiment

1. **The intended human-prefix workflow is more promising than cold start.**
   All 14 paired gains occur with a human-controlled prefix. However, task
   evidence and physical progress change together. An ambiguous initial scene
   may limit task identification, but these reports alone do not measure that
   ambiguity or identify the cause of cold-start failure.
2. **Context pretraining has not yet demonstrated a policy advantage.** There
   is no fully trained, matched current-only, execution-only or scratch-context
   policy in these two reports. The earlier offline intervention found useful
   execution information but no clear extra benefit from the four task tokens.
   These closed-loop results do not resolve that attribution. Both new runs use
   dynamic context, so they also do not compare frozen against dynamic inference.
3. **Continuation quality is still limited.** A 38.6% success rate after expert
   progress leaves 27 failures. The three zero-success placement groups are
   priority targets for failure inspection and corrective demonstrations.
   Simply adding memory, changing Mamba to TTT, or extending training is not
   established as the remedy by these measurements.
4. **Use matched controls next.** Train comparison policies on the existing
   protected context split and evaluate the same 44 episodes with identical
   prefix lengths, physical states, seeds and action budgets. Existing
   `no_intent_no_memory` and memory baselines can use that manifest; an
   execution-only training control would require an additional variant. To
   isolate dependence on task history, hold the current state/prefix outcome
   fixed and compare correct, shortened, absent and incompatible task context.
   Such interventions test context usage; matched trained controls test whether
   that representation improves policy performance.

The earlier 19/44 result used the older Piper episode list with frozen task
conditioning and an earlier evaluator. This new 17/44 result uses the context
development list and dynamic inference. The older list contains 33 context
training episodes, six development episodes and five confirmation episodes.
The numerical difference is not a measured regression from dynamic inference.
Matching only episode count or instruction names is not a controlled comparison.

## Reproduction and evidence

The [paired assessment](../reports/context_policy_1_results_20261021/analysis.json)
contains input/source hashes, per-task counts, per-episode transitions,
replay-subgroup counts, common-window errors and SHA256 values for all 88 traces.
The figure is also available as
[SVG](../reports/context_policy_1_results_20261021/task_success.svg).

```bash
python tools/assess_context_policy_pair.py \
  --start reports/context_policy_1_corrected_dynamic_00/result.json \
  --prefix reports/context_policy_1_corrected_dynamic_04/result.json \
  --output reports/context_policy_1_results_20261021
```

This reads the supplied results/traces and regenerates the assessment/figure.
It performs no training, simulation or checkpoint selection. Input evaluation
artifacts remain unchanged.
