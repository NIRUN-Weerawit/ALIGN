# Alternating context/observation action head

Use `--task-context-fusion aci` with `--head-type diffusion` or
`--head-type flow_matching`. This selects a separate U-Net denoiser architecture;
existing additive, gated, FiLM and global-cross-attention heads are unchanged.
The pretrained observer is frozen unless `--task-context-trainable` is explicit.

The U-Net retains temporal convolutions and the original diffusion/flow loss and
sampler. Its FiLM layers receive ONLY the encoded current robot state and the
64D denoising-time embedding. Four residual attention stages run in this order:

| Stage | Queries | Keys/values |
| --- | --- | --- |
| Encoder after `enc2` | Half-resolution action features | Task tokens + execution token |
| Bottleneck | Quarter-resolution action features | Current camera patch tokens |
| Decoder after `dec1` | Half-resolution action features | Task tokens + execution token |
| Output after `dec0` | Full-resolution action features | Current camera patch tokens |

Attention has four heads; `--task-context-fusion-dim` sets their total width
(default 128). Action queries have sinusoidal positions. Visual keys/values have
camera and spatial position embeddings. Context tokens have learned slot
embeddings, including a distinct execution slot. Each modality has its own
attention softmax. Residual attention outputs and the final action projection
start at zero, following the existing denoiser initialization convention.

No action self-attention or context-token self-attention is added. This is an
ACI-style temporal U-Net, not a reproduction of RDT-1B's transformer. The output
is the same chunk of step-wise robot commands (including the gripper).

## Training

Run from the repository root in the ALIGN environment, with a new output path:

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2.external \
  --output checkpoints/context_aci_diffusion \
  --variants context_no_memory \
  --task-context-checkpoint checkpoints/task_context_student_20261020/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_teacher_20261020/targets/manifest.json \
  --task-context-fusion aci --task-context-fusion-dim 128 \
  --head-type diffusion --history-size 1 --temporal-sampling episode \
  --chunk-size 8 --compressed-dim 8 \
  --epochs 80 --batch-size 2 --micro-batch-size 1 --num-workers 2 \
  --head-window-batch-size 8 --diffusion-loss-repeats 1
```

To use flow matching, select `--head-type flow_matching`, choose a new output
path, and use the existing `--flow-*` options. The same attention/FiLM architecture
is used. Change `--compressed-dim` to test visual bottlenecks; wider attention
cannot recover detail already lost by patch compression.

Start a bounded integration pilot with `--epochs 3 --max-steps 1
--max-val-batches 1` before a long run. These limits do not represent full epochs.
No full training or simulator-success improvement is claimed by this change.

## Compatibility and efficiency

- Start a fresh experiment. An old additive-head checkpoint cannot resume as
  ACI. Shape-compatible warm starts are partial transfers, not identical initial
  behavior: FiLM inputs and parameter shapes changed.
- The initial ACI path requires `history-size=1`, mean head history, and no
  `--visual-token-attention` or `--visual-context-mode multiscale`. Context memory
  variants can still supply their current retrieved/fused visual/state features;
  a full visual-history attention sequence is not implemented here.
- At the production interface, conditions use a packed tensor containing state,
  individual context/visual token values, and context-presence flags. Packing
  preserves existing temporal batching, microbatching and checkpoint interfaces;
  it does not fuse these modalities globally. The denoiser unpacks the tokens
  before attention. Only state enters FiLM.
- Both modalities' keys/values are projected once per sampling call and reused
  through all denoising/ODE steps. Loss repeats reuse the same projected keys and
  values without detaching gradients.
- Existing padded-sample masks are applied before key/value projection.
  `no_task`, `no_execution`, and `no_context` evaluation interventions use true
  context-attention masks, including suppression of output biases when every
  context token is removed. They do not merely replace token values with zeros.
- Architecture reconstruction uses saved `task_context_fusion`,
  `task_context_fusion_dim` and existing model configuration. Evaluation does not
  require the original observer artifact when the policy contains its weights.

## Checks

`tests/test_context_aci_head.py` covers attention order; state/time-only FiLM;
visual, task and execution gradients; key/value reuse; NaN padded samples;
context-removal masks; strict production checkpoint loading; and production
training/validation on CPU and CUDA when available. Existing-head regression
coverage is retained.

Initial ACI verification: **177 targeted/regression tests passed**, including
the original 14 ACI tests. Production-loop checks ran three optimizer updates followed by
validation on both CPU and CUDA for both objectives, confirmed nonzero context
projection gradients, and verified bitwise unchanged observer weights. They use
small synthetic cached-feature episodes, not a dataset-quality benchmark.
