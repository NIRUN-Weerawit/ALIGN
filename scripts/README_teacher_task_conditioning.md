# Fixed teacher task tokens with live student execution context

Both training entry points accept:

```bash
--task-context-source teacher \
--task-context-teacher-targets reports/task_context_refit_384_20261023/frozen_latents
```

The default remains `--task-context-source student`. Teacher mode requires the
observer checkpoint and its exact target split manifest. Point the targets flag
at the **exported latent directory**, whose manifest has schema
`align-privileged-task-tokens-v1`, not at the visual target or prefix cache.
It contains `manifest.json` and partition NPZs with `keys` and `task_tokens`.
No teacher inference or latent generation is performed during policy training.

For each episode, all task slots are replaced with that episode's saved teacher
tokens at every action anchor. The student still scans every observation and
provides its live execution token. A separate policy temporal token, if enabled,
is retained. The replacement supports memory/no-memory/history variants, ACI,
and diffusion/flow heads. Non-context arms retain their usual behavior.

The loader verifies teacher identity against observer metadata, native token
shape, finite values, file hashes, unique episode keys and exact episode
partitions. Tokens are looked up by episode key, never by shuffled batch position
or task-name nearest neighbor. A mismatched teacher space is rejected even when
the generic split override is enabled. Checkpoints record the conditioning source;
run manifests/configs record target provenance. Resume checks reject changes, and
ablation reevaluation validates the original targets again.

## Example using the existing ViT-B observer

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2 \
  --vision-backbone dinov2_vitb14 --raw-dim 768 \
  --task-context-checkpoint checkpoints/task_context_refit_384_20261023/observer_frozen.pt \
  --task-context-split-manifest reports/task_context_refit_384_20261023/targets/manifest.json \
  --task-context-source teacher \
  --task-context-teacher-targets reports/task_context_refit_384_20261023/frozen_latents \
  --variants context_memory context_no_memory context_history \
  --output checkpoints/piper_teacher_task_context \
  --epochs 400 --lr 0.0003 --batch-size 32 --micro-batch-size 8 \
  --compressed-dim 4 --head-d-model 64 --chunk-size 4 --plain-history-size 4 \
  --temporal-sampling episode --supervision-points 16 --num-workers 0 --seed 42
```

Use your actual feature-cache path. For ViT-L, supply its matching observer,
teacher-token export and split, plus `--vision-backbone dinov2_vitl14` (1024).
For flow matching, add `--head-type flow_matching`.

## Validation and deployment interpretation

Offline training validation uses the held-out episode's whole-demonstration
teacher tokens too. This is **privileged conditioning**, not a causal task-inference
score or evidence of execution from an independent demonstration. At deployment,
encode an available reference demonstration with the matching frozen teacher,
then hold those tokens fixed while the student updates execution context.

The observer branch supports externally supplied native tokens without needing
the training cache or original observer file after checkpoint loading:

```python
branch = model.task_context
cache = branch.new_stream(batch_size=1)
branch.set_reference_task_tokens(cache, reference_tokens)  # [1, K, token_dim]
payload, cache = branch.step(patches, cls, robot_state, cache)
# payload contains the fixed reference task tokens followed by live execution.
```

Set the reference again after every episode reset. The saved source flag records
the training condition; it does not automatically select a reference during
simulator deployment. Existing simulator context-intervention options remain
separate. A realistic autonomous evaluation should use a different reference
demonstration from the execution being evaluated. Retaining the evaluated
episode's own teacher tokens is an oracle experiment.

No policy training or simulation is launched by this feature change.
