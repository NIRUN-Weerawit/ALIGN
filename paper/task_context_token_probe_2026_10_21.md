# Does the task-context encoder distinguish tasks?

Follow-up: [named task estimates on failed expert-replay episodes](task_context_failed_episode_estimation_2026_10_21.md)
now includes per-episode predicted versus recorded names throughout expert and
policy trajectories and an optional online evaluator decoder. It uses one
training-only classifier pooled across prefixes; the per-query separability
probes reported below remain unchanged.

Date: 2026-10-21. Policy: `checkpoints/context_policy_1/context_no_memory/intention_best.pt`.

**Yes: the frozen task-token representation contains decodable known-task
information after observing an expert prefix.** On 44 development episodes from
ten tasks, a linear classifier fitted on 340 training episodes achieves 84%
balanced accuracy at the 40% prefix. Same-task tokens are substantially closer
than different-task tokens. This rules out a collapsed, task-independent
representation on this dataset. It does not establish complete task/context
understanding, useful long-history dependence, or effective use by the actor.
An additional live simulator expert-prefix diagnostic scores **86% balanced
task accuracy** on the same 44 development episodes, without classifier
adaptation to simulator observations.

## Protocol and boundaries

The diagnostic uses the exact context branch stored in the trained policy,
SHA256 `f345a86ca7449808f8b18579af8ab2940d6a89182549885092809e4ded0d9a3e`.
Its encoder, observation summary and state normalization are bitwise identical
to `observer_frozen.pt`. No policy or encoder parameters were updated.

The existing protected partition supplies 340 training and 44 development
episodes. The 44 confirmation episodes were not opened for this diagnostic.
Development episodes were already used in earlier model selection, so these
are post-hoc development measurements, not an untouched final test.

Only cached, frame-local observation summaries enter the causal encoder. Text,
task IDs, privileged whole-episode tokens, future targets and normalized episode
progress are not encoder inputs. Task-group IDs serve only as classifier targets.
The selected highest-weight hypothesis supplies four 128-dimensional task tokens;
they are normalized using the encoder's train-only teacher-latent buffers.

At each query, a `StandardScaler` and balanced logistic regression (`C=0.1`,
maximum 3,000 iterations) fit training episodes only. The regularization matches
the existing teacher probe and was not searched on development results. The
primary 40% query matches the intended handover protocol. Both the scaler and
classifier are refitted separately for each query and feature control, always
on training episodes. Class-balanced accuracy averages recall across ten tasks;
its chance reference is 10%.

Fractional queries use frame `floor(episode_length * fraction)`, including that
observation. Fractions are analysis selectors, not network features. Fixed-time
queries at 1 and 2 seconds are also included. The first development episode's
40% readout agrees with running only its observed prefix to maximum absolute
native-token error `9.54e-7`, checking that later observations do not affect it.

## Task information over the observed prefix

| Observations available | Correct episodes | Balanced task accuracy |
|---|---:|---:|
| Initial frame | 9/44 | 19.5% |
| First 10% | 33/44 | 78.0% |
| First 25% | 40/44 | 92.0% |
| First 40% | 36/44 | 84.0% |
| First 50% | 36/44 | 84.0% |
| First 70% | 44/44 | 100.0% |
| First 90% | 44/44 | 100.0% |
| First 1 second | 40/44 | 92.0% |
| First 2 seconds | 37/44 | 86.0% |

These estimates are not monotonic: the representation evolves with execution,
and each row has its own fixed training-only classifier. There are only three
to five development episodes per task. Perfect late-prefix decoding on these
44 known-task episodes does not imply general recognition of unseen tasks.

Fixed-time results remain far above the 10% constant-time control, so the result
does not rely solely on selecting a task-correlated relative-duration frame.
Scene configuration and robot motion can still reveal task identity.

![Prefix probes, confusion matrix and pair similarities](../reports/task_context_token_probe_20261021/task_token_separability.png)

## Are different tasks separated, and is the representation stable within a task?

At the same 40% query, across the 44 development episodes:

| Measure | Result |
|---|---:|
| Same-task episode pairs | 77 |
| Different-task episode pairs | 869 |
| Mean same-task cosine similarity | 0.775 |
| Mean different-task cosine similarity | -0.085 |
| Mean same-task Euclidean distance | 8.31 |
| Mean different-task Euclidean distance | 29.95 |
| Same/different task discrimination using pair cosine, ROC AUC | 0.966 |
| Nearest training-episode task, balanced accuracy | 86.0% |
| Nearest training-task centroid, balanced accuracy | 84.0% |

The pair counts are not independent statistical samples: many share episodes.
All 44 development vectors differ numerically, including episodes of the same
task. That is expected for a continuous context representation and would not
alone demonstrate task information; the class decoding and within/between-task
comparisons are the meaningful checks.

Shuffling development labels 1,000 times while holding the fitted 40% classifier's
predictions fixed yields mean balanced accuracy 10.17%, with a central 95% null
range of 2.0–19.5%. This is a label-association control, not a confidence interval
for generalization accuracy or an independent retraining experiment.

![Train-fitted PCA of development task tokens](../reports/task_context_token_probe_20261021/task_token_projection.png)

PCA is fitted on training vectors only. The two-dimensional projection is an
illustration; the classifiers and pair metrics use the complete representation.

## What the four tokens encode, and remaining ambiguity

At 40%, all four individual 128-dimensional token probes score 84%, matching
their concatenated 512-dimensional bundle. This indicates overlapping task-ID
information. It does not assign distinct semantic roles such as object, goal,
constraint and phase to individual tokens. Four tokens are not four task classes.
The four-way mixture mode index alone scores only 40%; its index is also not
task identity.

All eight bundle classification errors occur among three related tasks:

| True task | Correct / total | Main confusions |
|---|---:|---|
| Bowl on plate | 2/5 | Three predicted bowl on stove |
| Bowl on cabinet | 3/5 | One plate, one stove |
| Bowl on stove | 2/5 | Three predicted bowl on plate |
| Other seven tasks combined | 29/29 | None |

Distinguishing the destination of the same manipulated object remains a useful
target for improving task inference and for constructing a harder evaluation.

## Does the encoder add information beyond the current observation?

The same training-only probes at 40% give:

| Input representation | Balanced task accuracy |
|---|---:|
| Four task tokens | 84.0% |
| Execution token | 82.0% |
| Current frozen observation summary | 87.5% |
| Current camera CLS features | 80.0% |
| Current geometric state | 82.7% |
| Elapsed time only | 34.2% |
| Mixture mode index only | 40.0% |

The current summary already distinguishes these known tasks at least as well as
the task tokens in this small diagnostic. The execution token also exposes task
identity. These observations prevent a claim that the task branch supplies
exclusive information, that its designated tokens have cleanly separated task
and execution semantics, or that full-history Mamba is necessary here. Earlier
student tests likewise did not establish an advantage over a short history.

The initial representation is weak, but an initial scene can admit multiple
legitimate goals. Inferring an unobserved choice before any distinguishing human
action is not generally possible. Demonstration coverage should include shared
initial scenes with different intended destinations and action prefixes.

## Live simulator expert-prefix diagnostic

The second diagnostic replays the first 40% of each development demonstration
in LIBERO, renders both cameras, runs the same FP32 DINO path as the policy, and
updates the exact frozen context encoder one observation at a time. Captured
initial states and expert commands come from the intact development-only
autonomous-start evaluation's independent expert replay files. No action policy
is run. These are expert-controlled simulator prefixes, not recorded autonomous
policy latents or a rerun of the noisy handover evaluation.

The classifiers still fit the 340 cached training episodes only, using the
same settings as above. No simulator-specific adaptation or threshold tuning
is performed. All 44 development episodes are included, comprising 29 passing
and 15 failing expert replay checks. Confirmation is excluded.

| Input representation at simulated 40% prefix | Correct episodes | Balanced task accuracy |
|---|---:|---:|
| Four task tokens | 37/44 | 86.0% |
| Each individual task token | 37/44 | 86.0% |
| Execution token | 35/44 | 82.0% |
| Current observation summary | 39/44 | 89.5% |
| Current geometric state | 36/44 | 82.7% |

All seven task-token errors remain among bowl-on-plate, bowl-on-cabinet and
bowl-on-stove. The same-task/different-task cosine means are 0.775/-0.084,
and pair-similarity ROC AUC is 0.958. Simulator inputs therefore retain strong
known-task information under this expert-prefix setup, while current observations
remain a competitive source of task identity.

The saved expert trajectories are reproduced with **zero maximum EEF position
error** over every observed prefix frame in all 44 episodes. The first episode's
CUDA streaming and causal sequence task readouts differ by at most `7.15e-7`
in native token coordinates. Runtime was approximately 126 seconds on a shared
RTX 5060 Ti with the Torch allocator capped at 24% of GPU memory. Runtime
versions and per-episode hashes are stored with the
[simulator results](../reports/task_context_token_probe_20261021/simulator_handover_v2/results.json).

The first simulator attempt intentionally stopped when its strict `1e-4 m`
trajectory-reproduction check detected a mismatch: it omitted the evaluator's
initial setup reset. LIBERO samples fixture `model.body_pos/body_quat` during
reset; these are outside its flat simulator-state vector. Matching the original
setup-reset followed by expert-replay-reset sequence resolves the mismatch.
The incomplete first attempt is retained separately and contributes no scores.

This also limits what a saved flat-state comparison proves: matching those
vectors alone does not establish identical fixture scenes after independently
randomized resets. The diagnostic matches the expert-replay reset sequence;
it does not validate fixture equality between historical expert and policy runs.
Future paired simulator evaluations should preserve fixture model poses or the
complete scene as well as dynamic simulator state.

With LIBERO available on `PYTHONPATH` and its config already set:

```bash
MUJOCO_GL=egl python tools/probe_task_context_handover.py \
  --policy checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --reference reports/context_policy_1_corrected_dynamic_00/result.json \
  --probe reports/task_context_token_probe_20261021 \
  --data data/libero_goal.h5 \
  --output reports/task_context_simulator_probe_new \
  --gpu-memory-fraction 0.24
```

## Reproducing the cached-observation diagnostic

Run from the repository root in the ALIGN environment. This trains small
diagnostic classifiers; it does not retrain the neural models. Use a fresh output
directory on each run.

```bash
python tools/probe_task_context_tokens.py \
  --policy checkpoints/context_policy_1/context_no_memory/intention_best.pt \
  --cache reports/task_context_student_20261020/prefix_cache \
  --data data/libero_goal.h5 \
  --observer checkpoints/task_context_student_20261020/observer_frozen.pt \
  --output reports/task_context_token_probe_new
```

The [machine-readable results](../reports/task_context_token_probe_20261021/results.json)
include all feature-control confusion matrices, pair geometry, provenance hashes,
the causal check and null control. Extracted training/development vectors and
the protocol written before scoring are in the same output directory.

## Implication for the policy

Task tokens are not collapsed, and known task identity is often recoverable
after a short expert prefix. Poor manipulation success therefore does not by
itself show that task inference learned nothing. Conversely, task classification
does not show that the diffusion policy uses the tokens or encodes all the
context needed for a delicate action.

The next useful behavioral comparison holds current observations, physical
progress, training budget and episode set fixed while comparing correct task
context with removed or appropriately mismatched task context. Shared scenes
with different destinations are especially informative. Measure task completion
on every selected episode, including failed expert replays, and log task tokens
before and after handover to distinguish inference drift from action execution.
Use matched policy ablations to establish training-time benefit; a frozen-policy
intervention alone can introduce distribution shift.

## Rollout-file provenance update

During this diagnostic, `reports/context_policy_1_corrected_dynamic_04/result.json`
was found to differ from the prefix input recorded by the earlier paired-results
analysis. The archived analysis records prefix hash `a0d05e89...`, whereas the
current file was observed at hash `73f2da98...`. The current 40% report uses the
Piper comparison list: 33 training, six development and five confirmation
episodes. The autonomous-start report still uses all 44 protected development
episodes; only six episode IDs overlap between the current two files.

Consequently, the archived paired gains must not be applied to the current two
files. This token diagnostic independently uses the saved development partition
and does not pool comparison episodes or open confirmation data.
