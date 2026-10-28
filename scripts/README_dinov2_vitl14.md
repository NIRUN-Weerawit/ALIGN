# DINOv2 ViT-L/14 retraining

New action-policy training and feature generation default to `dinov2_vitl14`
(1024 channels). Select another supported DINOv2 using `--vision-backbone`;
`--raw-dim` is inferred and, when supplied, must match that backbone.
Images remain 224×224, with 256 spatial patches followed by CLS per camera.
The four 128-D task tokens and compressed patch dimension do not need to grow.

Existing checkpoints without vision metadata load as ViT-B/14 (768 channels).
Low-level model constructors preserve that historical default; new training
entry points explicitly select ViT-L. Legacy feature caches without backbone
metadata are interpreted as ViT-B and rejected by ViT-L training. New caches
record backbone, feature width, input size, patch size, camera order and source.
Existing ViT-B features, context observers and trained policy weights are not
interchangeable with ViT-L. Use separate paths for all new outputs.

## Inspect the cache job without generating anything

```bash
export PYTHON=/home/whinnoy/.venvs/align-ablation/bin/python
bash scripts/precompute_dinov2_vitl14.sh --dry-run
```

This reads dataset shapes and estimates FP32 disk usage. It does not download
weights, run the backbone, create an output directory or write features.

## Generate features when ready

```bash
bash scripts/precompute_dinov2_vitl14.sh --device cuda
```

Default output: `data/libero_goal.dinov2_vitl14/`. The wrapper uses four images
per forward call; adjust using `--batch-size`. Dataset, cameras and output may
also be supplied. The equivalent explicit command is:

```bash
$PYTHON scripts/precompute_dinov2.py \
  --data data/libero_goal.h5 --cameras image wrist_image \
  --output data/libero_goal.dinov2_vitl14 \
  --vision-backbone dinov2_vitl14 --raw-dim 1024 \
  --batch-size 4 --device cuda
```

Features are FP32 `(frames, cameras*257, 1024)` arrays. For two cameras that
is `(frames, 514, 1024)`, approximately 4/3 the storage of ViT-B. Default
behavior preserves existing output directories; only explicit `--overwrite`
permits replacement.

## Rebuild and pretrain context from the new visual space

After the cache exists:

```bash
$PYTHON tools/build_task_context_targets.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2_vitl14 \
  --output reports/task_context_vitl14_targets --fps 20

$PYTHON tools/refit_task_context.py \
  --source-targets reports/task_context_vitl14_targets \
  --report-dir reports/task_context_vitl14_384 \
  --checkpoint-dir checkpoints/task_context_vitl14_384 \
  --teacher-epochs 80 --student-epochs 100 --seed 42 --device cuda
```

Target construction records the backbone and visual width; the teacher derives
its input width from that manifest, and prefix/student dimensions follow the
new teacher. The refit retains the canonical Piper 384/44 episode assignments.
Use the newly exported observer, not the existing ViT-B observer.

## Train the downstream policies

After context pretraining finishes:

```bash
$PYTHON scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2_vitl14 \
  --vision-backbone dinov2_vitl14 --raw-dim 1024 \
  --task-context-checkpoint checkpoints/task_context_vitl14_384/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_vitl14_384/targets/manifest.json \
  --output checkpoints/piper_context_vitl14_cp4 \
  --variants context_memory context_no_memory \
  --epochs 400 --lr 0.0003 --batch-size 32 --micro-batch-size 8 \
  --compressed-dim 4 --head-d-model 64 --chunk-size 4 \
  --temporal-sampling episode --supervision-points 16 --num-workers 0 --seed 42
```

Saved configurations include the selected vision backbone and raw width;
evaluation restores these fields. Legacy checkpoints missing the fields use
ViT-B/14 and 768. To start another legacy-B training run, explicitly supply
`--vision-backbone dinov2_vitb14 --raw-dim 768` and the original B cache/context.
Training resume checks reject a change of backbone or feature width.
