# Intention-token task discrimination and convergence, 2026-10-18

The contextual intention embeddings contain task information, but these
checkpoints do not demonstrate ten stable, compact task clusters. Their task
information is weaker than information already available in observations/state.
Recurrence does not consistently improve task decoding over a reset encoder.

Follow-up [usage/history assessment](intention_usage_assessment_2026_10_18.md)
locates strong task information upstream of the final projection: a training-only
128-D PCA of the query output reaches 97.5% decoding in both compression-64
models. The action head shows little pooled task-specific dependence on natural
intent-token swaps. These results refine the initial probe; they do not support
concluding that Mamba is incapable of representing the task.

## Protocol

- Frozen trained encoders, one 128-dimensional contextual intention token.
  Probe the Mamba/readout output before memory fusion, not the learned query
  parameter (which is shared across episodes).
- LIBERO Goal: the same checkpoint-recorded 384 training / 44 held-out whole
  episodes, ten tasks. Each held-out task has 3–5 episodes. No random-frame
  train/test split. Verify episode lists agree across checkpoints.
- Twenty-one causal observation prefixes per episode, using real per-frame
  states and cached DINO CLS features. No task text, action targets, or future
  observations enter the encoder. Progress is an analysis coordinate only.
- FP32 extraction, exact encoder weight loading with recorded SHA256 hash.
  Old ablations reproduce their training-time zero-gripper convention;
  the newer checkpoint uses causal gripper state. Supplementary BF16 runs
  preserve the broad conclusion, but some individual classifications change.
- Train-only scaling and logistic regression, C=0.1, balanced classes, no
  held-out hyperparameter tuning. Separate classifiers at each progress point;
  an additional final-position classifier measures transfer across time.
- Train-fit clustering/PCA and original-space held-out geometry. Joint t-SNE
  is for visualization only. No policy weights change.

## Task decoding

All values below are held-out balanced accuracy. Chance is 10%.

| Checkpoint | Epoch | Start | 40% observed | End | End 95% episode-bootstrap interval | End with reset recurrence |
|---|---:|---:|---:|---:|---:|---:|
| `piper_v4_ablation_no_text_2/intent_memory` | 183 | 17.5% | 67.5% | 69.3% | 58.0–80.8% | 66.2% |
| `piper_v4_ablation_use_text_2/intent_memory` | 189 | 11.5% | 82.5% | 74.2% | 61.7–86.2% | 95.5% |
| `piper_v4_ablation_use_text_4_cp_dim_64_bastet/intent_no_memory` | 200 | 19.0% | 68.5% | 54.8% | 42.0–67.7% | 77.0% |
| `piper_v4_ablation_no_text_4_cp_dim_64_bastet/intent_no_memory` | 164 | 19.8% | 41.7% | 35.3% | 23.0–48.3% | 70.7% |

The first two checkpoints use compression dimension 8; the latter two use 64 and
newer data handling. This is a checkpoint diagnostic, not a matched estimate of
the causal effect of memory or compression. Text-trained means text was used
by the action head during training; no instruction is supplied in this probe.

Each final token decoder exceeds all 99 shuffled-training-label trials
(permutation p=0.01, the test's minimum resolution). This supports readable
task information, not a claim of inferred future goals.

Controls at the final position:

| Feature | Balanced task accuracy |
|---|---:|
| Current raw DINO CLS + raw robot state | 97.5% |
| Raw robot state alone | 93.5% |
| Causal history-average CLS/state + current state | 100.0% |
| Random Mamba/readout + the same trained state encoder | 97.5%, 97.5%, 100.0%, 100.0% respectively |

The random control has the same token dimension as its trained counterpart.
Raw feature controls have larger dimensions. These are probe scores, not robot
success rates or results from retraining policies with these features.

## Clustering and temporal consistency

| Checkpoint | Final cosine silhouette | Final K-means ARI | Final-position decoder tested at 40% |
|---|---:|---:|---:|
| No-text intent+memory | -0.020 | 0.193 | 13.0% |
| Text-trained intent+memory | 0.055 | 0.215 | 31.2% |
| Newer text-trained intent-only | -0.340 | 0.037 | 27.8% |
| Newer no-text intent-only | -0.508 | -0.022 | 15.7% |

The low/negative silhouettes and low ARIs show overlap rather than ten clean
clusters. For the no-text model, refitting a classifier at 40% yields 67.5%,
while the final-position classifier reaches only 13.0% there. Task information
changes direction across phases; successful per-phase decoding does not imply
a stable goal representation.

The no-text model's within/between task-distance ratio decreases from 8.10 at
the start to 0.49 at the end, so relative task structure does emerge. This is
not monotonic collapse to one point per task: different tasks remain mixed,
and per-task trajectories vary. Per-task plots compare held-out tokens with
fixed centroids from the final 20% of training episodes, including the nearest
other-task centroid.

## What this supports and what remains untested

1. **Tokens contain task information.** They are neither constant nor wholly
   uninformative. Their information is less linearly accessible than in the
   current inputs or random control.
2. **Useful temporal task abstraction is not established.** Reset recurrence
   is often stronger, particularly in the text-trained checkpoints. This
   suggests investigating information loss/phase dependence in the learned
   recurrence/readout; the probe alone cannot identify its training cause.
3. **Early ambiguity is expected.** Identical initial scenes can serve
   different objectives. An instruction-free encoder cannot identify an
   unobserved choice until observations or operator behavior reveal it.
4. **These are expert-episode probes.** Late observations contain consequences
   of task execution. Readability there does not establish prediction of a
   future objective, closed-loop success, or use by memory/action prediction.
5. **One point per task is not a necessary goal.** A useful token can retain
   task phase and uncertainty. A stable objective component plus a changing
   phase component may be more appropriate than collapsing the whole token.

The next useful experiment is a matched-phase token intervention: correct,
other-task, and reset-history tokens through the frozen action head, measured
on held-out episodes and simulator rollouts. This tests whether task
information is actually used. Future-state/subgoal prediction probes and
training-only auxiliary targets can then test/improve goal inference without
introducing task instructions as encoder inputs. No such architecture or loss
change was made in this probe.

The newer no-text intent-only checkpoint in
`piper_v4_ablation_no_text_4_cp_dim_64_bastet` was initially truncated. Both
files were subsequently replaced with readable copies, and the best checkpoint
(epoch 164) was added to this report. It performs worse on task decoding than
the other evaluated checkpoints. Its full-history token peaks at 44.5% task
accuracy across sampled positions and finishes at 35.3%, whereas reset-history
tokens finish at 70.7%. Final K-means ARI is approximately zero and silhouette
is strongly negative. These results do not support task-specific clustering
or improving task abstraction through recurrence for this checkpoint.

Additional training-only PCA of its **unscaled** 128-dimensional embeddings
finds 97.04% of all-progress variance in PC1 and 99.98% in the first two PCs.
Even at the final position alone, PC1 explains 88.98% and the first two PCs
99.88%; the participation-ratio dimension is 1.24. This is strong low-rank
concentration rather than total constancy, with different task labels mixed
along the dominant directions. It is consistent with a weak task-discriminative
readout, but does not identify the cause. These unscaled PCA numbers differ
from the figure's PCA, which scales features using training statistics.
See `no_text_intent/representation_rank.json` for this additional diagnostic.

### Direct comparison of compression-64 intent-only checkpoints

Both use one 128-D intention embedding, no memory bank, causal gripper inputs,
and exactly matching train/held-out episode lists. No task instruction is
provided to either encoder during probing. The text-trained checkpoint has
higher observed decoding: 54.8% vs 35.3% at the final position, a 19.5 percentage
point difference. A stratified paired bootstrap of the same 44 held-out
episodes gives a 95% interval of **−1.0 to +39.3 percentage points** for this
difference, conditional on the fitted decoders. The interval includes zero;
the sample does not establish a reliable population improvement.

Clustering is also better numerically with text training (silhouette −0.340
vs −0.508; ARI 0.037 vs −0.022), but both remain poor. Resetting recurrence
improves final decoding in both models: 77.0% vs 54.8% with text training, and
70.7% vs 35.3% without. Thus text training does not resolve the lack of a
consistent task-decoding benefit from history.

Unscaled final-token variance is even more concentrated in the text-trained
model: PC1 accounts for 99.25% vs 88.98%, with participation-ratio dimensions
1.02 vs 1.24. These describe variance concentration, not exact mathematical
rank or proof that small residual directions contain no information.

The best checkpoints occur at different epochs (200 vs 164), and training
microbatch settings differ (16 vs unset). This single-seed comparison should
not be interpreted as an isolated causal effect of task text. Text only
conditions the action head in these models; differences in raw intention
embeddings arise indirectly through training gradients.

The paired comparison and plot are saved as
`reports/intent_task_probe_2026_10_18/intent_only_text_comparison.{json,png}`.

## Reproduction and artifacts

See [probe instructions](../tools/README_intent_task_probe.md). Results, exact
checkpoint configs/hashes, raw token trajectories, confusion matrices, task
labels, PCA/t-SNE, progress curves, and per-task convergence are in
`reports/intent_task_probe_2026_10_18/{no_text_memory,text_memory,text_intent,no_text_intent}/`.
Reports are local regeneratable artifacts and are ignored by Git.

Validation: 10 tests passed for exact loading/split isolation and real Mamba
causal-prefix/streaming parity. The analysis produced no captured warnings.
