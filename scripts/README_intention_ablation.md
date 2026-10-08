# Sequential intention/memory ablations

`run_intention_ablation.py` trains these four models **in sequence**, finishing
all epochs of each before starting the next:

1. No intention tokens, no memory.
2. Intention tokens, no memory.
3. No intention tokens, memory enabled.
4. Intention tokens and memory enabled.

All models receive the same CLI arguments. Only the intention/memory flags
change. Shared parameters start from identical initial weights; the episode
split, training crop schedule, batch order and validation RNG seeds also match.
Disabled-intention models omit the Mamba encoder entirely. Memory-only models
still retrieve perceptual/state information.

Run from the ALIGN repository using a Python environment with CUDA PyTorch,
`mamba_ssm`, h5py and the project dependencies installed:

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 \
  --cache data/libero_goal.dinov2 \
  --output checkpoints/ablation_goal_sequential \
  --epochs 80 \
  --batch-size 2 \
  --history-size 1 \
  --segment-length 20 \
  --chunk-size 8 \
  --head-type diffusion \
  --lr 1e-4
```

Use the spatial `.h5`/`.dinov2` paths to evaluate LIBERO Spatial instead.
The default cameras are `image wrist_image`; `--cameras` must match the cache's
camera order. The cache must contain 256 patches and one CLS token per camera,
each with width 768. Frozen raw-image vision stays on CPU during cached training.

Run `python scripts/run_intention_ablation.py --help` for every argument.

| Shared setting | Argument | Default |
|---|---|---:|
| Epochs per model | `--epochs` | 80 |
| Batch size | `--batch-size` | 2 |
| Optimizer updates per epoch | `--max-steps` | 0: full loader |
| Observation window / initial warmup | `--history-size` | 1 |
| Training segment length | `--segment-length` | 20 |
| Action chunk length | `--chunk-size` | 8 |
| Action head | `--head-type` | diffusion |
| Learning rate | `--lr` | 0.0001 |
| Weight decay | `--weight-decay` | 0.0001 |
| Gradient norm clipping; 0 disables | `--grad-clip` | 1 |
| State embedding width | `--state-dim` | 64 |
| Compressed patch width | `--compressed-dim` | 4 |
| Action head width | `--head-d-model` | 64 |
| Legacy Mamba hidden projection width | `--mamba-output-dim` | 128 |
| Mamba state / convolution / expansion | `--mamba-d-state`, `--mamba-d-conv`, `--mamba-expand` | 16 / 4 / 2 |
| Tokens in intention-enabled variants | `--num-intent-tokens` | 1 |
| Intent embedding width | `--intent-dim` | 128 |
| Memory capacity | `--memory-bank-len` | 8 |
| Gripper loss weight | `--gripper-loss-weight` | 0.01 |
| Predicted gripper accuracy cutoff | `--gripper-threshold` | 0.5 for binary 0/1 |
| Validation fraction per task | `--validation-fraction` | 0.1 |
| Random seed | `--seed` | 42 |
| Per-process CUDA allocator fraction | `--gpu-memory-fraction` | 0.40 |
| CPU threads | `--cpu-threads` | 2 |

History size 1 gives the action head the current observation. Mamba retains
causal context across the training segment, resetting between random segments.
Each training epoch samples one reproducible crop from every training episode;
the default runs all batches. Validation uses fixed held-out whole episodes and
fixed crops. Select `flow_matching` to run the same four comparisons with that
head. Training and validation both apply the configured gripper loss weight.

Each variant gets its own configuration, epoch metrics, best checkpoint and
latest model/optimizer recovery checkpoint. The output root stores the shared
manifest and combined `comparison.md`/`summary.json`. Existing outputs are
protected against accidental overwrite. Resume with the **same arguments** and
add `--resume`; completed variants are skipped. Attempts lacking an optimizer
checkpoint are archived and restarted rather than pretending to resume exactly.

For a smoke run, add `--epochs 1 --max-steps 1` and use a new output directory.
For the four-model sequential workflow, leave `--variants` and
`--worker-variant` unset. Those options are retained for selected-model and
parallel-worker workflows.
