# Sequential intention/memory ablations

`run_intention_ablation.py` trains these four models **in sequence**, finishing
all epochs of each before starting the next:

1. No intention tokens, no memory.
2. Intention tokens, no memory.
3. No intention tokens, memory enabled.
4. Intention tokens and memory enabled.

All models receive the same CLI arguments. Only the intention/memory flags
change. Shared parameters start from identical initial weights; the episode
split, training supervision schedule, batch order and validation RNG seeds also match.
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
| Memory capacity | `--memory-bank-len` | 16 |
| Gripper loss weight | `--gripper-loss-weight` | 1.0 |
| Predicted gripper accuracy cutoff | `--gripper-threshold` | 0.5 for binary 0/1 |
| Validation fraction per task | `--validation-fraction` | 0.1 |
| Random seed | `--seed` | 42 |
| Per-process CUDA allocator fraction | `--gpu-memory-fraction` | 0.40 |
| CPU threads | `--cpu-threads` | 2 |

History size 1 gives the action head the current observation. Mamba retains
causal context across every observation in the episode. The default
`--temporal-sampling episode` resets memory and Mamba between episodes, retains
all intervening observations, and samples `--supervision-points 16` deterministic
action-loss anchors per episode. Targets are contiguous future action chunks.
`--temporal-sampling crop` restores the bounded segment protocol;
`--segment-length` controls that mode. Validation uses the same sampling mode
with fixed anchors. The default runs all batches. Select `flow_matching` to run the same four comparisons with that
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

### Sampling, gripper, and saved-head diagnostics

Action chunk length is independent of denoising steps. Diffusion and flow
validation use the head's default 10 solver steps, matching deployment. Fresh
diffusion heads use a normalized cosine schedule with a positive terminal
signal and train on noisy indices 1..100 by default (`--diffusion-train-steps 100`).
`--diffusion-loss-repeats 4` draws independent noise/timestep samples per
condition. `--diffusion-clip-sample` clips predicted clean actions to normalized
motion bounds and the binary gripper range. The denoiser and DDIM inversion use FP32 even under outer BF16 autocast.
Legacy checkpoints keep their original
buffers; sampling skips their singular zero-signal endpoint.

New training gives gripper the same loss weight as each motion dimension,
rather than 0.01. To resume the old 80-epoch experiment, explicitly use
`--gripper-loss-weight 0.01` so its saved configuration remains consistent.
For binary LIBERO actions, both validation and simulator execution use cutoff
0.5. Simulator feedback carries the last **executed binary command** in dataset
units, while pose action scaling applies to the first six dimensions.

Measure dependence of existing diffusion checkpoints without changing weights:

```bash
python scripts/probe_condition_dependence.py \
  --run checkpoints/ablation_libero_goal_h1_e80_20261008 \
  --output checkpoints/condition_dependence
```

This uses all held-out episodes, fixed validation crops, matched noise draws,
and cross-task shuffles. It compares final-head intention zeroing/shuffling,
whole-memory bypass, and bank-content shuffling, restoring each episode's real
memory after interventions. Zeroed visual/state controls and an identical-input
control help interpret the result. Dependence does not establish policy benefit.

### Episodic memory and policy selection

New runs use `--memory-mode episodic`, detached raw writes, and timestamp age
encoding on retrieval keys. `--memory-write-fused` and
`--no-memory-detach-writes` expose controlled write-policy comparisons.
`--memory-patch-retrieval` preserves spatial tokens;
`--visual-token-attention` adds a residual visual cross-attention path in the
diffusion/flow U-Net. These architecture options default off and require
matched ablations. Legacy checkpoints load with their original contracts.

Use `--warm-start EXISTING_RUN` for exploratory adaptation; matching weights
are copied but the diffusion schedule is fresh. Different variant warm starts
do not give a controlled cross-model comparison. Within-model memory
interventions remain paired. Add `--episode-anchors` to the condition probe
for beginning/middle/end diagnostics on full-episode runs.

Training checkpoint selection by position error is provisional. Add
`--keep-epoch-checkpoints`, then use `scripts/select_policy_checkpoint.py`
with immutable `--candidates`, a shared `--episodes` file, and
`--interventions normal bypass empty`. It selects by simulator success. Use
independent episodes/seeds for final reporting, and keep task and action
protocols identical between candidates.

`--observation-dropout-prob` is an optional training-only missing-camera
experiment (default 0). Visibility masks are deterministic per episode/epoch
and shared across variants; targets stay unchanged and validation stays fully
observed. Hidden cameras contribute neither current CLS tokens nor encoded
patches. Use `--visual-occlusion` in the condition probe for paired current-view
latent dropout with correct, shuffled and bypassed history. Report these
partial-observation diagnostics separately from unmodified simulator success.

`--drop-state-with-all-views` extends that experiment to complete observation
packet loss. Current state features are then unavailable along with the cameras;
completely missing packets are excluded from bank writes. Streaming
`encode_step`/`IntentionStream.observe` accept the corresponding camera/state
availability masks and preserve physical observation indices across outages.

`--memory-context-only` is an opt-in experiment: the retrieved branch uses historical attention output in its FFN and residual, with current features preserved separately by the fusion gate. Existing checkpoints default to the original query-residual branch. Its LIBERO policy benefit is not established; compare correct and cross-task shuffled histories before adopting it.
