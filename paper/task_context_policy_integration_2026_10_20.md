# Task context connected to ALIGN policy training

Date: 2026-10-20. Status: opt-in integration implemented and verified on real
cached training data and live RGB feature extraction. The bounded policy pilot
is complete. Full action-policy training and closed-loop qualification remain.

Update, 2026-10-21: the user has completed the 200-epoch `context_no_memory`
policy run. An archived paired evaluation recorded 3/44 from autonomous start
and 17/44 after a 40% expert prefix, including failed expert replays. The prefix
result file has since been replaced with a different comparison episode set;
see the provenance notice in
[trained-policy results and reflection](task_context_policy_results_2026_10_21.md)
before interpreting the current files together. The independent
[task-token diagnostic](task_context_token_probe_2026_10_21.md) measures task
information on the protected 44 development episodes.
The pilot results below remain historical integration evidence, not the final
trained-policy measurements or a matched demonstration of context benefit.

## Architecture and training path

`ALIGNIntentionModel` can now instantiate the pretrained context branch in place
of the old intention encoder. Four task tokens and one execution token condition
the existing diffusion policy through separately normalized residual projections
initialized with scale 0.01. The current visual/state condition retains its
original width and live visual-detail pathway. Optional episodic memory retrieves
visual/state features without modifying the selected task tokens.

The default first policy stage freezes the pretrained summary and encoder.
Adapters, current visual/state modules, diffusion policy, and optional memory
train from expert actions. Scratch context controls share the frozen summary
and normalization, but train a newly initialized causal backbone/readout through
action loss. Optional pretrained-encoder action-only fine-tuning is exposed at
0.1× the policy learning rate; representation-retention losses and a staged
unfreezing schedule are not included in this milestone.

Both [main training](../training/train_intention.py) and the
[ablation runner](../scripts/run_intention_ablation.py) use complete episodes
and the frozen pretraining split: 340 train / 44 development / 44 excluded
confirmation episodes. Manifest integrity, episode coverage/lengths, camera
order, and observer identity are enforced. All comparison arms inherit the same
split. Future length, text, task labels, and teacher tokens are not context inputs.

The [integration guide](../tools/README_task_context_policy.md) documents commands,
warm starts, scratch controls, self-contained checkpoints, and rollout behavior.

## What ran

1. A two-epoch, two-update-per-epoch GPU smoke test exercised pretrained context,
   context plus memory, and scratch context.
2. The main `training/train_intention.py` entry point completed one real optimizer
   update, validated all 44 development episodes, and saved its checkpoint.
3. A six-arm pilot completed three bounded epochs × 12 episode-batch updates:
   **36 optimizer updates per arm**, batch size one, 16 anchors per episode,
   AdamW 1e-4, seed 42. Four development episodes selected checkpoints during this
   deliberately small integration run. These were not full-dataset epochs.
4. Those six fixed checkpoints were then evaluated on **all 44 development
   episodes**, with matching noise seeds and complete observed prefixes.
   This did not reselect checkpoints or open confirmation.

The current-only baseline uses the existing sparse loader. Other arms observe
all intervening frames for recurrence/retrieval. Common shape-compatible initial
parameters and action-supervision budgets are shared. The old-intention reference
has its original conditioning width; the new pretrained/scratch context pair
shares the new architecture. This is not an equal-total-compute benchmark.

## Bounded pilot results

Offline sampled-action errors after only 36 updates remain large. The following
numbers demonstrate a functioning training/evaluation path, **not useful robot
control or a benefit from context pretraining**.

| Fixed policy | Development diffusion loss | Position-channel action MSE | Rotation-channel action MSE | Gripper accuracy |
|---|---:|---:|---:|---:|
| Current only | 0.863851 | 1.194894 | 0.994434 | 0.4963 |
| Current + memory | 0.815375 | 1.187483 | 0.988385 | 0.4980 |
| Original intention | 0.860150 | 1.194580 | 0.994007 | 0.4961 |
| Pretrained context | 0.862429 | 1.194864 | 0.994271 | 0.4961 |
| Pretrained context + memory | 0.815151 | 1.187487 | 0.988366 | 0.4980 |
| Scratch context | 0.861486 | 1.194782 | 0.994199 | 0.4959 |

Action errors are in native dataset action units, not Cartesian metres or
degrees. The very short training budget and near-identical context/control
results do not establish a performance gain. There is one seed, no controlled
history-disambiguation benchmark, and no completed real simulator success trial.

Peak allocated GPU memory in the six-arm pilot ranged from 0.185 GiB for the
sparse current baseline to 2.434 GiB for context plus memory. The arms' three
training/selection epochs took about 9–44 seconds each under a shared GPU load.
Those bounded times do not estimate an 80-epoch full-training budget.

## Checkpoint and handover verification

- **155 tests passed**, including existing training, memory, diffusion, checkpoint
  I/O, and cached loading regressions. New tests cover exact state/visual
  conversion, future isolation, sequence/stream parity, context gradients,
  frozen weights, scratch-selector gradients, protected splits, warm starts,
  main-loader dispatch, and checkpoint loading without the original observer.
- Pretrained context weights and summary tensors remained bitwise identical
  during policy training. Scratch encoder weights changed while its summary
  remained frozen. Context arms contain no old `intention_encoder` parameters.
- A policy checkpoint reconstructed through the production evaluator processed
  real RGB observations with DINO and produced finite `(1,8,7)` action chunks.
  Task freezing preserved the selected tokens without resetting recurrence.
- Mock simulator tests verify fresh action planning exactly at handover, removal
  of queued pre-handover predictions, preserved recurrent state, fixed task
  tokens, and continuing execution-token updates. This is a state-contract test,
  not a physical manipulation-success measurement.

The existing trajectory rollout's expert-prefix `--switch-at` protocol now
activates context streaming and snapshots the task at the control boundary.
There is no automatic takeover-confidence threshold. The fixed 20 Hz cadence
and legacy previous-gripper fallback remain part of the input contract.

Evidence:

- [Full-development assessment](../reports/task_context_policy_20261020/development_pilot.json)
- [Integration/source/frozen-weight verification](../reports/task_context_policy_20261020/integration_verification.json)
- [Pilot manifest](../reports/task_context_policy_20261020/pilot_manifest.json)
  and [training curves](../reports/task_context_policy_20261020/pilot_metrics.json)
- [Main-entry-point metrics](../reports/task_context_policy_20261020/main_smoke_metrics.jsonl)
- [Real-RGB checkpoint inference](../reports/task_context_policy_20261020/raw_image_inference.json)

Bounded policy checkpoints remain local under
`checkpoints/task_context_policy_20261020/pilot_v3`; they are not published as
qualified pretrained action policies. The original pretrained context artifacts
remain unchanged.

## Next training decision

The integration is ready for sustained policy training with the same frozen
context/control arms. Keep confirmation closed until development selection and
handover criteria are fixed. A useful policy result still requires paired
closed-loop handovers, especially matched current scenes with different earlier
goal cues and same-endpoint/different-order procedures. No TTT replacement is
supported by this integration pilot.
