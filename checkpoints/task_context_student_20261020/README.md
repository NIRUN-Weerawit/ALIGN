# Pretrained causal task-context encoder

These selected checkpoints are published with Git LFS. After checking out this
branch, install Git LFS and fetch the weight objects:

```bash
git lfs install
git lfs pull
```

| Artifact | Use | SHA256 of downloaded weights |
|---|---|---|
| `observer_frozen.pt` | Recommended inference bundle: frozen frame-local summary, causal encoder, train-only geometric normalization | `f97b16da0be2a74eebfc08bd6ea7eb7929f64757075d1620df123640a1859fb3` |
| `student_frozen.pt` | Student alone, taking frozen 256-D observation summaries | `a982f5c3ff1195e6fd1ed9f8bc4bf65051d6b7d46b6374448353b2cf306b6cbe` |
| `selector_full_seed42/student_best.pt` | Exact selected full-prefix experiment checkpoint; identical to `student_frozen.pt` | `a982f5c3ff1195e6fd1ed9f8bc4bf65051d6b7d46b6374448353b2cf306b6cbe` |
| `selector_current_seed42/student_best.pt` | Independently trained current-only control | `1d27c5f17258644401627ddfd7ec8c1a22017e71b7eb1f78652abbf4bd54fb10` |
| `selector_short_seed42/student_best.pt` | Independently trained 16-frame control | `45d40ce23cb5920b914030479f42cf88333b5a4f88e27dd482f7c49c403509aa` |

The [frozen teacher](../task_context_teacher_20261020/teacher_frozen.pt) is also
included for training-target reconstruction and offline assessment. Its SHA256
is `1c872b39f44d910dca6ecd57217dd28f577a3a864b056de7ee507e19546f8910`.
The teacher is not part of the deployed observer.

## Load the observer

Use an environment with PyTorch and Mamba1/causal-conv1d installed, as described
in the [encoder instructions](../../tools/README_task_context_encoder.md).

```python
from models.task_context_observer import load_observer

observer, metadata = load_observer(
    "checkpoints/task_context_student_20261020/observer_frozen.pt", device="cuda"
)
stream = observer.new_stream(batch_size=1)
# For each real observation, supply float32 tensors on the model's device:
# visual_slots: (1, 2, 5, 768), normalized CLS + 2x2 slots, camera order in metadata
# geometric_state: (1, 10), physical translation + two rotation-matrix columns
#                  + the documented previous-gripper-command fallback
# elapsed_seconds: (1,), strictly increasing episode-relative seconds
belief = observer.observe(visual_slots, geometric_state, elapsed_seconds, stream)
task_tokens = belief["tokens"]                 # (1, 4, 128), top-weight mode
execution_token = belief["execution_token"]    # (1, 1, 128)
```

Use `data.task_context_dataset.spatial_summary` to construct the visual slots
from camera-major 16x16 DINO patches followed by CLS. Allocate a new stream at
episode reset. No text, task ID, completed teacher tokens, or future frames are
observer inputs.

## Published evidence and data requirements

The branch includes selected weights, small target/prefix manifests and
normalization files, teacher latent labels, training configurations/curves/source
snapshots, development and confirmation reports, and per-prefix result arrays.
Large per-episode observation/target caches, raw demonstrations, and rejected
checkpoint runs are not included. Existing reports preserve their original
local provenance paths; those paths do not imply that the corresponding dataset
is distributed with the branch.

**Inference does not require the demonstration caches or teacher.** Rebuilding
training caches requires the audited LIBERO Goal HDF5 recording and matching
two-camera DINO cache described in the training instructions. Because the small
manifest directories are already published, choose new output directories when
running cache builders, which intentionally refuse to overwrite existing paths.

The [results](../../paper/task_context_student_results_2026_10_20.md) establish
useful causal outcome/procedure information on known tasks. Full history did not
clearly outperform the 16-frame control. This is a representation checkpoint,
not an integrated action policy or a validated automatic takeover system.
