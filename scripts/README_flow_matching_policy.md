# Flow matching with the current policy pipeline

Use `--head-type flow_matching` in either `scripts/run_intention_ablation.py`
or `training/train_intention.py`. The flow head now uses the same U-Net,
visual/state conditioning, configurable task-token reader, live execution token,
fusion choices, ordered history attention, memory retrieval, and optional visual
cross-attention as the diffusion head. DistilBERT task-text training uses the
same text-conditioning path. Context and text remain separate ablations.

Both heads use the existing causal episode split, action normalization,
supervision masks, gripper weights, batching, and optimizer configuration.
The flow loss independently resamples continuous time and noise **four times per
supervised action chunk** by default, matching diffusion's four noise draws.
Visual keys/values are projected once and reused across those draws without
detaching their gradients.

The objective remains velocity matching. With the default zero endpoint noise:

```
x(t) = (1 - t) * action + t * noise
velocity_target = noise - action
```

Sampling integrates from noise at `t=1` to actions at `t=0`, in FP32 even under
AMP. Only the final actions are clipped, using the same normalized bounds as
diffusion: `[-1, 1]`, with gripper channel 6 in `[0, 1]`.

| Flow option | New training default | Meaning |
|---|---|---|
| `--flow-inference-steps` | `10` | Number of ODE steps |
| `--flow-solver` | `euler` | `euler` or `heun`; Heun needs two network calls per step |
| `--flow-loss-repeats` | `4` | Independent time/noise draws per supervised chunk |
| `--flow-clip-sample` | enabled | Terminal action clipping; disable with `--no-flow-clip-sample` |
| `--flow-sigma-min` | `0.0` | Residual Gaussian noise at the data endpoint; keep zero for the standard comparison |

`--diffusion-*` options affect the diffusion head only. Flow settings are saved
in manifests/checkpoints, checked on flow training resume, and restored by the
production evaluation loader and ablation reevaluation path.

## Context-policy example

After generating the ViT-L feature cache and pretraining the corresponding
observer described in [the backbone guide](README_dinov2_vitl14.md):

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2_vitl14 \
  --vision-backbone dinov2_vitl14 --raw-dim 1024 \
  --task-context-checkpoint checkpoints/task_context_vitl14_384/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_vitl14_384/targets/manifest.json \
  --output checkpoints/piper_context_vitl14_flow_cp4 \
  --variants context_memory context_no_memory context_history \
  --head-type flow_matching --flow-solver euler --flow-inference-steps 10 \
  --flow-loss-repeats 4 --flow-clip-sample \
  --plain-history-size 4 \
  --epochs 400 --lr 0.0003 --batch-size 32 --micro-batch-size 8 \
  --compressed-dim 4 --head-d-model 64 --chunk-size 4 \
  --temporal-sampling episode --supervision-points 16 --num-workers 0 --seed 42
```

For an existing ViT-B observer/cache, explicitly select `dinov2_vitb14` and
`--raw-dim 768`, and supply the matching artifacts. Task-token count is inferred
from the observer, or validated using `--num-task-tokens K`.
For text, use the usual non-context variants plus `--use-task-text`; DistilBERT
is the training default. Use a separate output directory for each head type.

Existing flow checkpoints without the new fields retain Euler/10 steps, one
loss draw, zero endpoint noise, and disabled action clipping. To resume such a
flow run with unchanged settings, explicitly pass `--flow-loss-repeats 1
--no-flow-clip-sample`. Existing diffusion parameter names and computation are
preserved. Diffusion weights are not a trained flow policy: retrain the action
head with the flow objective. The pretrained context observer can be reused
when the backbone, cameras, token space, and data split match.

Compare simulator success under identical episode/handover conditions and seeds;
diffusion noise MSE and flow velocity MSE are different targets, so their raw
loss magnitudes do not rank the two policies. This update does not launch cache
generation, training, or simulator evaluation.
