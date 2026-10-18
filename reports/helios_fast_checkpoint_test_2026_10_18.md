# Helios fresh-model local checkpoint test — 2026-10-18

The old 30.6 GiB flat-memory model was excluded. Its recovery/controller stayed
stopped and the abandoned local partial copy was removed. Original NFS saves
were retained. A fresh current-architecture model completed two bounded epochs,
then resumed from epoch 2 and completed epoch 3 with optimizer state intact.

Remote experiment: `/data/wee_ucl/align-fast-checkpoint-20261018-v2`.
Raw measurements and exact command: [JSON](helios_fast_checkpoint_test_2026_10_18.json).
An earlier attempt in `...-v1` exposed the encoder cuDNN SDPA execution-plan
failure and is preserved as a failed attempt; all results below are from v2.

## Configuration and scope

- GPU 0, H100 NVL; PyTorch 2.10.0+cu128. The unrelated GPU 1 process was left alone.
- Current source snapshot of HEAD d961811 plus uncommitted changes; the snapshot
  SHA-256 is recorded in the JSON artifact and checkpoint configuration.
- `intent_memory`, diffusion, no task text, compression 32, history 1, chunk 8,
  20-frame crops, batch/microbatch 2, workers 0, one training batch and one
  validation batch per epoch. Original 384/44 episode split, real Goal cached
  features and correctly aligned causal gripper states/actions.
- Explicit multiscale visual context, patch memory, inferred-intention visual
  queries, two retrieval blocks; 53,916,200 trainable parameters.
- Source, logs, checkpoint staging, and mirror destinations were on local SSD.
  Input HDF5/cached features were read from NFS. This test did **not** benchmark
  NFS checkpoint mirroring or establish a full-epoch speedup.

## Measurements

| Epoch | Train wall s | Data wait s (included in train) | Validation s | Foreground checkpoint s | Total epoch s |
|---|---:|---:|---:|---:|---:|
| 1 | 17.08 | 15.96 | 4.53 | 0.87 | 22.48 |
| 2 | 7.66 | 7.17 | 0.33 | 1.81 | 9.80 |
| 3, resumed | 12.89 | 11.77 | 0.23 | 1.10 | 14.22 |

Foreground checkpoint timing includes the immutable CPU snapshot and local
serialization, and excludes asynchronous mirror drain. The fresh two-epoch
process took 72.09 s; the resumed process took 55.29 s, including startup and
final mirror drain but excluding the controller's subsequent verification.
The first two epochs peaked at 3.13 GiB allocated GPU memory. These small-batch,
short-segment figures do not predict memory usage at larger training settings.

Final recovery `training_last.pt`: 648,821,938 bytes (619 MiB), containing weights
and AdamW state. Best weights `intention_best.pt`: 217,299,099 bytes (207 MiB).
The old 30.6 GiB recovery file belonged to a different, much larger architecture;
the size reduction is not an apples-to-apples storage speed benchmark.

## Verification and fixes

- Both fresh and resumed processes exited successfully. Recovery epoch and
  optimizer step counts were 2/2, then 3/3. SHA-256 comparisons verified local
  canonical and mirrored best/recovery files matched after both processes.
- A cuDNN plan error initially occurred in `StateConditionalCrossAttn` at
  compression 32. Moved the existing action-head backend selection into
  `models/attention_backend.py` and applied it to state/visual attention too.
  Flash, efficient and math SDPA remain available; the context restores the
  caller's settings. No model parameter or target definition was changed by
  this backend fix.
- Added `--max-val-batches` for bounded infrastructure tests (default 0 means
  full validation). The limit is recorded and fixed on resume. Added source
  snapshot provenance support for isolated deployments without `.git`.
- CUDA regression tests: 77 passed in 12.38 s across visual-memory conditioning,
  diffusion sampling, ablation setup, and checkpoint I/O. Compilation and
  `git diff --check` passed.
- Before committing, a clean export of the exact staged source passed 217
  regression tests in 36.09 s, covering cached loaders, gripper contracts,
  intention/future supervision, Mamba recurrence, episodic memory, sparse
  history, checkpointing, distributed training, and evaluation dispatch. Earlier
  uncommitted XVLA/diagnostic files were excluded from that source export.
- GPU 0 was released after the test. No ongoing training/controller was left
  scheduled by this experiment.

Local checkpoint saving works. Shared-storage input waits still dominated
these tiny training batches (91–94% of train wall time). Staging training data
locally is the next throughput experiment. Three optimizer updates provide no
evidence about policy quality or memory usefulness; that requires full training
and the planned paired simulator evaluations.
