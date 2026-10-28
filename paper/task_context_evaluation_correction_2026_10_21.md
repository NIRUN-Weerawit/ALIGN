# Context policy evaluation correction and verification

Date: 2026-10-21. Checkpoint:
`checkpoints/context_policy_1/context_no_memory/intention_best.pt`.
The evaluator is corrected and has been exercised in real LIBERO simulation.
**Subsequent user-directed protocol update:** every selected episode now receives
a policy rollout, regardless of expert replay outcome. The 15 failed replay
checks below remain diagnostic and do not suppress policy scores. The default
context development set differs from the shared Piper set (only six of 44
overlap); explicit `--episode-set comparison --val-episodes FILE` matches cases
and records training exposure. See the current
[evaluation instructions](../tools/README_task_context_policy.md).

An archived complete policy analysis recorded **3/44 (6.8%)** from autonomous start and
**17/44 (38.6%)** after a 40% expert prefix, both with dynamic context and all
44 development episodes included. The [full results and reflection](task_context_policy_results_2026_10_21.md)
audit all 88 traces, document 14 paired gains and no losses, and separate
continuation performance from unproven context-encoder benefits. The single-case
checks below describe earlier evaluator verification, not these full results.
The prefix report has since been replaced with a different comparison episode
set; see the provenance notice in that analysis before using the current two
result files as a paired comparison.

**Additional simulator limitation found during the
[task-token diagnostic](task_context_token_probe_2026_10_21.md):** LIBERO fixture
model poses are sampled during reset and are not part of the flat simulator-state
vector saved by this evaluator. Matching that vector proves agreement of dynamic
state, not of all fixture placement after independently sampled resets. The
token diagnostic exactly reproduces the expert-reset sequence; historical
expert/policy fixture equality has not been verified. Complete scene/fixture
preservation remains necessary for a fully paired scene audit.

## Corrections

- The checkpoint's saved 44 development episodes and camera order are defaults.
  Explicit lists may subset development; training, confirmation, unknown and
  duplicate keys are rejected before model loading. The exact split-export
  script now uses checkpoint keys rather than a new random window split.
- `--task-context-mode auto` updates task inference from the start and freezes
  only after a nonempty human prefix. `dynamic` supports a paired handover
  comparison. Explicit freezing without a prefix is rejected. The switch uses
  the original demonstration length, and expert-only means expert-only for the
  full simulation budget.
- Expert replay always runs, including with `--no-video`. Replay and policy
  restore the same simulator state after resetting controllers. Original states
  can be supplied as H5 `initial_sim_state` or episode-keyed `--initial-states`
  NPZ. Without them, reports explicitly say "captured seeded reset".
- Replay validation requires actual task success and mean/initial pre-action EEF
  errors no larger than the predeclared 0.02 m tolerance. These checks are now
  informational: failed replay never filters policy episodes or suppresses the
  score. The initial verification recorded below used the earlier gating rule;
  its archived reports preserve that historical behavior.
- The explicit task-success predicate replaces `done`. Completion during the
  expert prefix is recorded separately and excluded from the policy-eligible
  denominator. EEF comparisons now align pre-action simulator and recorded poses.
- Missing cameras/poses fail explicitly. Replay tails hold the last gripper
  command with zero motion. Both replay and rollout use the same binary gripper
  convention. Traces save executed commands, predictions, errors, initial state
  and handover boundary even without video.
- Motion-magnitude diagnostics compare only the six pose-action channels on
  matching recorded timesteps; gripper values and padded tails do not distort
  the ratio. The evaluator no longer suggests a gain change from that ratio.
- Corrected results stay inside their output directories, preserving historical
  sidecars. `--preflight-only` validates inputs without DINO/simulation;
  `--replay-only` validates simulation without loading the policy or DINO.

Code: [evaluator](../eval/eval_libero_v4_trajectory.py),
[protocol helpers](../eval/libero_protocol.py),
[split exporter](../scripts/save_val_episodes.py), and
[evaluation instructions](../tools/README_task_context_policy.md).

## Verification performed

**91 tests passed** across new protocol regressions and existing policy,
state/gripper, generative-head, memory, checkpoint-loading and XVLA tests.
Preflight passed for all 44 development episodes at both 0 and 0.4 switch
fractions. The actual checkpoint rejects the previously used contaminated
validation list before model/DINO construction.

The actual trained checkpoint also completed observation-replay integration
checks with real cached DINO features, validating dynamic task inference at zero
prefix, frozen task inference after a prefix, dynamic continuation, advancing
execution state, finite actions and preserved recurrence. Those checks do not
simulate physical action consequences; the following tests do.

An official LIBERO checkout and configuration were created under `/tmp`, using
existing installed dependencies; no packages were installed or environments
modified. Real physics used robosuite 1.4.1, MuJoCo 3.5.0, EGL rendering and the
official LIBERO source commit recorded in the verification manifest. Training
used a different server/runtime, so these results describe this documented
verification environment.

### Full development expert-replay audit

All **44 development episodes**, 300 simulation steps each, were audited without
opening confirmation or loading the policy:

| Check | Episodes |
|---|---:|
| Expert completes the task | 31/44 |
| Expert completion and pose-alignment validation pass | **29/44** |
| Expert does not complete the task | 13/44 |
| Expert completes it but exceeds pose tolerance | 2/44 |

The two alignment failures are `ep_000177` (mean 0.03167 m; initial 0.02796 m)
and `ep_000210` (mean 0.02192 m; initial 0.01529 m). The tolerance was not changed
after examining these results. All runs used captured seeded resets because the
converted dataset contains no simulator states. Expert replay success does not
prove identity of all original object states. Failures may reflect initial-state
or simulator/data compatibility differences; this audit does not isolate them.

Result: [full replay audit](../reports/context_policy_1_evaluation_corrected_20261021/replay_development/result.json).
This historical replay-only report sets `qualified=false` and
`success_rate=null` under the initial protocol. Current reports separate
`evaluation_complete` and `expert_replay_all_passed`; replay-only still has no
policy success rate.

### Real trained-policy rollouts

The replay-valid development episode `ep_000024` (wine bottle on rack) was used
for three bounded, 300-step integration runs with the real policy, live RGB/DINO,
diffusion sampling and simulator action execution. Policy GPU allocator use was
capped at 24%. These are single-episode checks, not a representative benchmark.

| Case | Resolved task behavior | Policy success |
|---|---|---:|
| Switch 0, auto | Dynamic from first frame | 0/1 |
| Switch 0.4, auto | Freeze at step 70 | 0/1 |
| Switch 0.4, dynamic | Continue task updates | 0/1 |

All three have exactly equal initial simulator states. The two 0.4 runs have
exactly equal executed human-prefix commands. The expert replay passes in all
three, and the policy still fails, demonstrating that real policy limitations
remain after fixing the evaluation. No ranking between context modes follows
from one episode.

[Failure video](../reports/context_policy_1_evaluation_corrected_20261021/policy_smoke_0.4/ep_000024_libero_goal.mp4)
contains dataset, expert-replay and policy panels. Each case also has a JSON
report and compressed action/state trace.

### Camera convention

Initial-frame comparisons for both cameras in three development demonstrations
all select **both vertical and horizontal flips** as the closest orientation
among the four possibilities. The existing default is correct for these data;
it was retained. Placement differences remain visible, especially in the wrist
view. This check is not a full camera calibration or state-identity proof.

[Camera comparison](../reports/context_policy_1_evaluation_corrected_20261021/ep_000002_camera_orientation.png)
and [numeric results](../reports/context_policy_1_evaluation_corrected_20261021/camera_orientation.json).

## Running corrected evaluation

Use the Python environment containing LIBERO on the target machine:

```bash
python eval/eval_libero_v4_trajectory.py \
  --data data/libero_goal.h5 \
  --checkpoint checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --n-episodes 44 --switch-at 0.4 --task-context-mode auto \
  --max-steps 300 --no-video --no-plot \
  --out-dir reports/context_policy_1_corrected/replay \
  --replay-only
```

Remove `--replay-only` and choose a fresh output directory for policy evaluation
on all selected episodes. Use `dynamic` in another directory for the paired 0.4
comparison; use auto at switch 0 for progressive task inference. Omit
`--no-video` to save videos. No external validation list is needed.

If available, add `--initial-states path/to/states.npz`, where each key is the
converted episode key and each value is its exact original flattened simulator
state. Do not substitute an arbitrary public initialization index for an episode
without a verified mapping. Retain replay failures in reported diagnostics and
evaluate every requested episode. Evaluating only the 29 passing episodes would
define a different benchmark. Matching the old Piper list evaluates 33 context
training episodes, six development episodes and five confirmation episodes;
it matches scenarios but does not create a shared held-out benchmark.

The [verification manifest](../reports/context_policy_1_evaluation_corrected_20261021/verification.json)
records source/checkpoint hashes, versions, simulator source commit, paired-run
checks, tests and limitations. The evidence directory includes exact source
snapshots, logs, the correct development list and per-episode traces. No policy
weights were changed by this evaluation work.
