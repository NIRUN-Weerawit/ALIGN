# Plain history diffusion ablation

The `plain_history` variant has no intention encoder and no memory bank. It
keeps the last `--plain-history-size` encoded observations in temporal order.
Set this flag to any positive frame count; `1` is a single-observation plain
model, while larger values retain that many ordered observations. Use a
separate `--output` directory for each count because checkpoint weights and
the saved configuration depend on it. Evaluation reads the count from the
checkpoint automatically.
Diffusion cross-attention excludes the cuDNN SDPA backend because some
GPU/PyTorch/cuDNN combinations cannot build an execution plan for its short
action queries and history keys. Flash and efficient attention remain enabled,
with PyTorch math attention available as a fallback. The backend selection is
local to these attention calls and does not affect model/checkpoint structure.
Each observation contributes its 4,096-dimensional compressed visual feature
and 64-dimensional encoded robot state in the current LIBERO Goal setup.
The diffusion U-Net conditions its FiLM blocks on the current observation and
uses cross-attention at the bottleneck to read the full ordered history.
Task text, when enabled, is projected into the condition width and added to
each history step.

Training defaults to the sparse loader: it selects the same supervised
anchors as the reference trainer, loads the union of their required history
frames directly into final batch storage, and encodes each unique frame once.
There is no intermediate per-episode feature gather allocation. Packed observations avoid padding
whole episodes to the longest episode in a batch. The first
observation is repeated on the left until a complete history is available;
this preserves the same action supervision anchors as a one-frame model.
MuJoCo evaluation encodes each new camera frame once and retains encoded
visual and state features in `IntentionStream`. No past raw frame is encoded
again at later action calls.

The first comparison is text-conditioned `plain_history` with four frames
against `checkpoints/piper_v4_ablation_use_text_2/intent_memory`. It uses the
same LIBERO Goal data, whole-episode split, seed 42, 200 epochs, batch size
10, learning rate 0.0002, and diffusion settings. The history head adds a
projection and cross-attention parameters; this comparison tests the complete
history-head architecture rather than isolating memory capacity alone.

Run directory: `checkpoints/piper_v4_ablation_use_text_plain_history_h4`.
Training progress and metrics are written to `run.log` and
`plain_history/metrics.jsonl`. The run resumes from the last complete epoch
and runs under the `align-plain-history-h4` user service. The
`align-plain-history-eval` service waits for training to complete, then runs
the same 44 held-out MuJoCo episodes used by the baseline. It writes a paired
task-success comparison to `heldout_comparison.md` in the run directory.

The reproducible training command is:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2 \
  --output checkpoints/piper_v4_ablation_use_text_plain_history_h4 \
  --variants plain_history --use-task-text \
  --history-size 1 --plain-history-size 4 \
  --temporal-sampling episode --epochs 200 --batch-size 10 \
  --micro-batch-size 2 \
  --plain-history-loader sparse --plain-head-batch-size 32 --no-pin-memory \
  --num-workers 1 --prefetch-factor 1 --compressed-dim 8 --lr 0.0002 \
  --gpu-memory-fraction 1.0 --seed 42
```

The 10-episode logical batch is loaded in five microbatches of two episodes,
with gradients accumulated before each optimizer step. This bounds host
shared-memory use for long episodes while retaining the optimizer batch size
and episode groups. One worker with prefetch factor one bounds the queued
batch count on this 32 GB machine; explicitly use `--num-workers 0` to disable
prefetch. Expandable CUDA segments reduces fragmentation
across variable-length episode batches.
The DINOv2 sidecar is accessed through `data/libero_goal.dinov2`, which now
points to a verified local copy on NVMe. The original external-drive target is
preserved at `data/libero_goal.dinov2.external`.

Sparse loading is automatically enabled for `plain_history` only. Other
variants retain the sequential episode path to build intent and memory.
`--plain-head-batch-size` bounds supervised points per loss/sampling call;
it does not change the optimizer batch. Pinning defaults off for the sparse
loader to avoid a second large locked host allocation. `--pin-memory` enables
it explicitly; `--plain-history-loader dense` retains the reference path.
These runtime options can change when resuming an existing checkpoint.

A GPU benchmark on ten identical epoch-60 microbatches (20 episodes, two
optimizer updates, four-frame histories, warm cache, pinning off in both
cases) measured 6.11 seconds dense versus 1.70 seconds sparse, about 3.6x
faster. Cached feature payload fell from 3,701 MiB to 1,544 MiB; peak process
RSS fell from 5.89 GiB to 2.29 GiB; peak GPU allocation fell from 1.61 GiB to
0.56 GiB. This short benchmark is not a full-epoch throughput guarantee.
Reproduce each mode with `scripts/benchmark_plain_history.py --run-directory
checkpoints/piper_v4_ablation_use_text_plain_history_h4/plain_history --mode
sparse --microbatches 10 --epoch 60 --output /tmp/benchmark.json`, then repeat
with `--mode dense`.
Condition/target and deterministic-gradient parity tests cover missing
observations, initial padding, uneven episode lengths, and microbatch
accumulation. Dropout/noise regrouping preserves their distributions but does
not produce bit-identical stochastic training trajectories.

The interrupted local run resumed from epoch 59 with sparse loading. Its
first complete optimized epoch (60) took 62.83 seconds, compared with the
293.32-second median of the preceding ten dense epochs (about 4.7x faster).
Data wait fell from a 202.99-second median to 42.12 seconds; validation took
6.55 seconds. Epoch 60 recorded validation loss 0.04922, position MSE 0.05406,
gripper accuracy 0.93874, and peak GPU allocation 0.590 GiB. These measurements
describe the first resumed epoch; they do not establish closed-loop success.

Detailed loading profiles are enabled with `--profile-data-stages` on the
runner or benchmark. The runner writes the additional stages to epoch
`metrics.jsonl`; without the flag, per-item CPU timers and detailed CUDA
events are omitted. CUDA events do not synchronize each individual stage.
For workers > 0, CPU stage totals sum worker work and can overlap, so they
should not be added to reconstruct main-process wall time.

Two measured passes over the same 40 microbatches (80 episodes, seed/anchor
epoch 153, four-frame history, zero workers, unpinned CPU batches) averaged:

| Stage | Seconds | Clock |
|---|---:|---|
| Episode metadata and mmap setup | 0.15 | CPU wall |
| Feature materialization | 4.60 | CPU wall |
| Batch packing | 2.03 | CPU wall |
| Host-to-GPU transfer | 0.84 | CUDA stream |
| Unpacking, encoders, history/head condition | 0.76 | CUDA stream |
| Diffusion loss | 0.88 | CUDA stream |
| Backward and optimizer | 1.74 | CUDA stream |
| Overall benchmark | 11.20 | Wall |

Materialization includes disk/page-fault stalls and the initial NumPy gather
copy. Packing is a second copy into contiguous batch storage. Each pass
materialized 5.74 GiB of feature payload and incurred about 4.29 GiB of
physical disk reads; repeating the sample did not produce a fully resident
cache. Two existing jobs shared the GPU during profiling, so GPU timings
include contention. The largest remaining opportunities are eliminating the
extra packing copy and overlapping bounded loading with GPU work. CPU
metadata/HDF5 setup and H2D transfer are smaller contributors in this sample.
Raw measurements are saved in the run directory as `data_stage_profile.json`.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/benchmark_plain_history.py \
  --run-directory checkpoints/piper_v4_ablation_use_text_plain_history_h4/plain_history \
  --mode sparse --microbatches 40 --epoch 153 --profile-data-stages \
  --output /tmp/plain_data_stages.json
```

The next loading optimization was checked against a frozen model/optimizer
checkpoint and identical epoch-153 episode groups. Direct loading copies
contiguous runs from each episode mmap straight into the final tensor. Worker
output uses shared storage to avoid another copy across the IPC queue.
`--plain-history-copy-mode materialized` retains the older extra-copy path
for comparisons. Plain-history-only runs now default to one worker; explicit
worker arguments and the defaults of other variants remain respected.

| Loading configuration | Full training-pass wall time | Reduction |
|---|---:|---:|
| Materialized features, zero workers | 49.37 s | Reference |
| Direct copies, zero workers | 36.64 s | 25.8% |
| Direct copies, one worker/prefetch one | 27.81 s | 43.7% |

Each pass contains 190 microbatches / 380 episodes, with five warmup
microbatches before timing. Validation and initial worker startup are excluded.
Zero-worker direct/reference runs performed similar physical disk reads
(17,417 versus 17,253 MiB). In shorter warm-cache comparisons, direct/one
worker took 7.44–7.58 seconds, versus 9.68 seconds materialized/zero workers;
two workers took 7.48–7.51 seconds and offered no additional benefit. An
initial cold-cache materialized sample took 24.74 seconds and is excluded
from warm-cache comparisons. GPU workload contention remains a limitation.
Parent RSS measurements exclude worker RSS, and main-process disk counters
exclude worker reads. CPU profiling totals include overlapping worker work.
Raw runs, logs and the comparison are in `loading_copy_benchmark/` under the
run directory. Use the benchmark's `--copy-mode`, `--num-workers`,
`--prefetch-factor`, and `--warmup-microbatches` flags to reproduce this.

The local run completed 200 epochs using the preceding sparse loader. The
automatic post-training evaluator then failed to import LIBERO. Its generated
comparison reused an older evaluation and is marked invalid for the final
run; the evaluation script now requires a fresh result before comparing.

After training, evaluate the same 44 held-out episodes with `--switch-at 0.0`,
`--val-episodes checkpoints/piper_v4_ablation_val_episodes.txt`, and
`--n-episodes 44`. Compare task success, per-task failures, gripper behavior,
and inference time with the text-conditioned intent-plus-memory model.
