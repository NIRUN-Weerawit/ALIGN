# Task-token count ablations

Use `--num-task-tokens K` (alias `--task-context-tokens K`) to configure the
whole-demonstration teacher/refit pipeline and to verify the expected count in
downstream policy ablations. Positive integer counts are supported; the
pretraining default remains **4**. The policy default is to infer the count from
the observer checkpoint. Token width remains 128, the execution token remains
separate, and the student still predicts four mixture hypotheses. Thus an
8-token experiment produces **four hypotheses of eight task tokens**, plus
one live execution token.

Token count is part of the learned latent space. A four-token observer cannot
be changed into an eight-token observer by a policy flag. Pretrain a new teacher
and student for each count and then train a new policy using that observer.
The policy rejects mismatches instead of slicing, padding or duplicating slots.

## Pretrain one representation for each count

The visual target cache is independent of task-token count and can be shared.
For example, after creating the ViT-L feature/target caches described in
[the ViT-L guide](README_dinov2_vitl14.md), run:

```bash
python tools/refit_task_context.py \
  --source-targets reports/task_context_vitl14_targets \
  --num-task-tokens 8 \
  --report-dir reports/task_context_vitl14_k8 \
  --checkpoint-dir checkpoints/task_context_vitl14_k8 \
  --teacher-epochs 80 --student-epochs 100 --seed 42 --device cuda
```

Repeat with counts **1, 2, 4, 8**, using distinct output directories for each.
The refit records the count, passes it to the teacher, exports matching latent
labels, and derives student dimensions from those labels. The teacher's
historically named `--pooling quarters` divides the demonstration into **K**
ordered intervals when K tokens are requested. These are global-context
readouts, not guaranteed semantic phases.

Direct teacher training also accepts `--num-task-tokens`. Student training
automatically reads the count from the prefix-cache manifest. Both teacher
export paths record the configured token shape rather than assuming 4×128.

## Train the matching policy ablation

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2_vitl14 \
  --vision-backbone dinov2_vitl14 --raw-dim 1024 \
  --task-context-checkpoint checkpoints/task_context_vitl14_k8/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_vitl14_k8/targets/manifest.json \
  --num-task-tokens 8 \
  --variants context_memory context_no_memory \
  --output checkpoints/piper_context_vitl14_k8 \
  --epochs 400 --lr 0.0003 --batch-size 32 --micro-batch-size 8 \
  --compressed-dim 4 --head-d-model 64 --chunk-size 4 \
  --temporal-sampling episode --supervision-points 16 --num-workers 0 --seed 42
```

The saved policy configuration records the actual observer count. Evaluation
reconstructs the corresponding head automatically; no evaluation override is
needed. Token-level probes enumerate every configured slot. Refit any task-label
estimator/donor bank for each representation; latent spaces differ between runs.

Keep the backbone, episode split, budgets, token width and policy settings fixed
when comparing counts. More slots increase both latent capacity and the task
projection parameter count, so this is a count/capacity ablation. It is unrelated
to `--num-intent-tokens`, which controls the older intention encoder.

No pretraining, feature-cache generation or policy training is launched by this
configuration change.
