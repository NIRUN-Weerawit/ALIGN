# X-VLA LIBERO reproduction audit — 2026-10-23

## Conclusion

The existing ALIGN XVLA runs do not reproduce the released X-VLA evaluation
protocol. They establish performance under ALIGN's custom rollout protocol,
not a failed reproduction of the paper's 98.1% average. Several differences can
affect control substantially; their individual success-rate effects remain
unmeasured. No new policy rollout was run during this audit.

## Which reported result?

- The [paper, Table 13](https://arxiv.org/html/2510.10274v1) reports Spatial
  98.2%, Object 98.6%, Goal 97.8%, Long 97.6%, average 98.1%.
- The [author repository](https://github.com/2toinf/X-VLA) associates 98.1%
  with `2toINF/X-VLA-Libero`.
- [LeRobot's documentation](https://huggingface.co/docs/lerobot/xvla) advertises
  `lerobot/xvla-libero` at 93%. This is the model downloaded locally, revision
  `12e8783e996944f5c97e490d37d4c145484ed70a`. These advertised numbers should
  not be conflated; this audit does not establish whether the two releases have
  identical weights.

## Confirmed protocol differences

Reference: [author client](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/libero_client.py),
inspected on the audit date. Local implementation: `eval/xvla_libero_adapter.py`
and the evaluation arguments embedded in `checkpoints/xvla/**/*.json`.

| Contract | Released author client | Saved ALIGN XVLA runs |
| --- | --- | --- |
| Controller | Native absolute OSC; `use_delta=False` | Convert absolute goals to scaled delta commands and clip to input limits |
| Chunk execution | Queue and execute the returned chunk before querying again | Horizon 1; discard remaining predictions and replan each step; local model chunk length is 30 |
| Initialization | Suite-provided fixed initial states, then 10 settling actions | Seeded random resets for selected HDF5 episode/task labels |
| Goal rollout limit | 800 policy steps | 300, 500, or 1000 |
| Trial selection | Default 50 trials per task | 44 total held-out episode labels across Goal tasks |
| Later-call proprioception | Previous chunk's final predicted XYZ/rotation | Current measured XYZ/rotation |

The 50-trial count is a released-client default, not independent proof of the
paper's exact trial count. LeRobot's own processor also uses measured state;
the proprioception difference is specifically against the author client.
New evaluation reset options in the working tree do not retroactively change
the saved runs and do not automatically select official benchmark states.

The absolute-to-delta conversion can alter the requested target. A CPU check
of the actual adapter with standard OSC scaling found that a target 0.15 m
ahead becomes delta command 1.0, producing a controller goal only 0.05 m ahead.
Small in-range deltas can be equivalent; clipping makes general equivalence
false. This check demonstrates a contract difference, not its rollout impact.

The [author's controller explanation](https://github.com/2toinf/X-VLA/issues/15)
specifies training on controller goal positions/orientations and execution
through native absolute OSC. Controller goals and observed achieved poses
must not be treated interchangeably when preparing absolute action labels.

## Existing local evidence

All full 44-case results below use horizon 1 and the LeRobot checkpoint.

| Result file under `checkpoints/xvla/` | Success | Limit | Expert prefix |
| --- | --- | --- | --- |
| `xvla.mujoco_eval_0.0_val.json` | 20/44 | 300 | None |
| `500/xvla.mujoco_eval_0.0_val.json` | 23/44 | 500 | None |
| `1000/xvla.mujoco_eval_0.0_val.json` | 23/44 | 1000 | None |
| `1000/xvla.mujoco_eval_0.4_val.json` | 24/44 | 1000 | 40% |

Longer time limits alone did not resolve the gap. These are different initial
conditions and trial weighting from the public benchmark. Static inspection
found agreement with the LeRobot policy contracts for camera flips, ImageNet
normalization, consecutive rotation columns, 20-D state, domain ID 3, task
instruction input, and 10 flow-sampling steps. This is not an end-to-end
numerical equivalence test.

## How the paper trains the model

The paper describes 0.9B parameters, heterogeneous pretraining on roughly 290K
episodes, and LIBERO adaptation with absolute EEF actions, batch 128, and 60K
updates. Adaptation initially updates prompts/action heads for 1000 iterations,
then warms up for 1000 iterations toward joint training. See Sections 4.2 and
Appendices G/H. This is substantial pretrained capacity and benchmark-specific
adaptation, rather than a model trained only on ALIGN's held-out-split dataset.

## Reproduction order

1. Evaluate the existing LeRobot checkpoint through its native absolute-control
   pipeline with all 30 queued actions, fixed benchmark initial states, settling,
   correct task text, 10 flow steps, and an 800-step Goal horizon. Verify actual
   controller goals and input tensors on a short rollout first.
2. Compare against the current adapter on paired initial states. Separate the
   controller change and chunk-execution change to measure their contributions.
3. Run equal trial counts per task, reporting per-task and aggregate success.
   Keep ALIGN's 44-case comparison separately labeled.
4. If native LeRobot results remain substantially below its documented 93%,
   investigate runtime/version and numerical differences. Reproduce the author's
   `2toINF/X-VLA-Libero` checkpoint with their deployment/client pipeline before
   comparing directly with 98.1%.

Simulator runs on this host must include
`PYTHONPATH=/home/whinnoy/.local/share/align/LIBERO` and the appropriate headless
MuJoCo rendering settings. No adapter behavior was changed by this audit.
