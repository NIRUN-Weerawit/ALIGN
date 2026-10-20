# Causal task-context milestone

The [measured results](../paper/task_context_student_results_2026_10_20.md)
confirm useful prefix information. Longer history did not outperform the
independently trained 16-frame control enough to pass its separate gate.

This experiment learns a non-text task belief from the observation prefix. It
does not modify the production ALIGN controller. The whole-episode teacher,
observation summary, and teacher decoders stay frozen. A new two-layer,
width-256 Mamba1 outputs four Gaussian hypotheses over four 128-D teacher task
tokens, plus a separate 128-D execution token. Inference always selects the
top-weight hypothesis. Ground-truth teacher responsibility is training-only.

The input is one frozen 256-D summary per observation: both camera CLS features,
2×2 patch summaries, normalized geometric state, and elapsed time. A legacy
previous-gripper-command fallback is inherited from the audited target cache;
this is not a measured gripper or verified raw human-command recording.
Text, group IDs, actions being predicted, future frames, complete-episode
tokens, and normalized episode progress are not encoder inputs.

## Reproduce

Use the existing ALIGN environment with PyTorch, Mamba1, causal-conv1d, NumPy,
SciPy, h5py, scikit-learn, and pytest. The commands below use relative workspace
paths. Every builder/trainer/report requires a new output path rather than
overwriting an experiment. `PY` denotes the Python executable, for example
`/home/whinnoy/.venvs/align-ablation/bin/python`.

```bash
$PY tools/build_task_context_prefix_cache.py \
  --targets reports/task_context_teacher_20261020/targets \
  --latents reports/task_context_teacher_20261020/frozen_latents \
  --teacher checkpoints/task_context_teacher_20261020/teacher_frozen.pt \
  --output reports/task_context_student_20261020/prefix_cache

$PY training/train_task_context_encoder.py \
  --cache reports/task_context_student_20261020/prefix_cache \
  --teacher checkpoints/task_context_teacher_20261020/teacher_frozen.pt \
  --output checkpoints/task_context_student_20261020/profile_new \
  --max-steps 100 --epochs 3
```

Train three independent matched arms, each 100 epochs/4,300 updates, seed 42,
batch 8 episodes, 16 sampled anchors, AdamW 1e-4, BF16 training and FP32 losses
and validation. Invoke the trainer separately with `--history full`,
`--history current`, and `--history short --window 16`, using new output paths.
Each arm has the same architecture and loss/optimizer budget. Short history is
at most 16 observations (0.8 seconds at 20 Hz). Every intervening observation
is processed by the full student; sparse training anchors never reset it.

The loss is normalized mixture NLL, plus 0.2 execution-change loss, 0.2 frozen
outcome-decoder loss, and 0.2 mixture-selector responsibility cross entropy.
The last term was added after an initial development-only NLL run exposed
weak mode selection: normalizing the joint likelihood by 512 suppresses the
mixture-weight learning signal too. Its detached posterior targets rescale
selector gradients. They are not task-class labels. `--selector-weight 0`
reconstructs the original objective. Source snapshots and hashes distinguish
these runs. Inference has no posterior/teacher assignment.

Execution supervision predicts normalized CLS and geometric-state changes at
+16/+64 frames (0.8/3.2 seconds). It excludes unavailable future frames and
averages modality groups separately. It does not label contact, readiness,
success, or percentage complete. All 16 training anchors are uniform because
event, takeover, and disambiguation annotations are absent.

```bash
$PY tools/assess_task_context_encoder.py \
  --cache reports/task_context_student_20261020/prefix_cache \
  --teacher checkpoints/task_context_teacher_20261020/teacher_frozen.pt \
  --full checkpoints/task_context_student_20261020/selector_full_seed42/student_best.pt \
  --current checkpoints/task_context_student_20261020/selector_current_seed42/student_best.pt \
  --short checkpoints/task_context_student_20261020/selector_short_seed42/student_best.pt \
  --output reports/task_context_student_20261020/development_selector_seed42.json
```

Only after the development information gate passes, run the same tool with
`--split confirmation`, a new output, and
`--confirm-development-report reports/task_context_student_20261020/development_selector_seed42.json`.
Exact teacher, student, and target-cache hashes must match. Confirmation never
selects checkpoints. Those episodes were previously characterized for the
teacher, so a new scene/session-disjoint benchmark is still necessary.

The fixed protocols are `reports/task_context_student_20261020/protocol.json`
and `protocol_selector.json`. Useful causal information and an advantage from
longer history have separate gates. Analysis fractions select fixed evaluation
anchors only; neither duration nor fraction is a network feature. Bootstrap
confidence intervals resample whole episodes. Donor-history interventions keep
the recipient's current summary fixed but have unverified scene continuity.

## Streaming interface

```python
from training.train_task_context_encoder import load_checkpoint

encoder, metadata = load_checkpoint(student_path, device="cuda")
stream = encoder.new_stream(batch_size=1)
for summary_t in real_frame_local_frozen_summaries:
    belief = encoder.step(summary_t, stream)  # summary_t: (1, 256)
    task_tokens = belief["tokens"]          # (1, 4, 128), top-weight mode
    execution = belief["execution_token"]  # (1, 1, 128)
    weights = belief["logits"].softmax(-1)  # (1, 4)
```

Produce summaries with the exact frozen teacher summary and target-cache
normalization. `means`/`log_stds` describe train-normalized token coordinates;
use the saved `latent_mean`/`latent_std` to convert distributions. `tokens` are
already native teacher coordinates. Allocate a fresh stream on episode reset.
Stream state is ephemeral and never stored in model weights. Preserve it across
any later handover; take a detached task-token snapshot when the user requests
handover in the later policy integration milestone.

After confirmation passes the information gate, export a standalone observer:

```bash
$PY tools/export_task_context_observer.py \
  --teacher checkpoints/task_context_teacher_20261020/teacher_frozen.pt \
  --student checkpoints/task_context_student_20261020/selector_full_seed42/student_best.pt \
  --targets reports/task_context_teacher_20261020/targets \
  --cache reports/task_context_student_20261020/prefix_cache \
  --report reports/task_context_student_20261020/confirmation_selector_seed42.json \
  --output checkpoints/task_context_student_20261020/observer_frozen.pt
```

`models.task_context_observer.load_observer` reconstructs the frozen summary,
encoder, and train-only geometric normalization. Call `new_stream()` and
`observe(visual_slots, geometric_state, elapsed_seconds, stream)`. Inputs are
float32 tensors on the model's device. Visual slots use the existing
`data.task_context_dataset.spatial_summary` convention. Geometric state is
translation plus two rotation-matrix columns and the documented gripper fallback.
The observer validates finite inputs and increasing times. It accepts no teacher
tokens, text, or labels. The bundle records both information and history gates;
export does not establish a history advantage or authorize automatic handover.

The CPU reference implements the same Mamba equations and parameters as the
CUDA fused scan and online update. Tests compare all three, mutate future and
padding inputs, verify current/short controls, freeze the teacher while retaining
student gradients, test balanced ambiguous targets, and bind confirmation to
selected hashes. Run:

```bash
$PY -m pytest -q tests/test_task_context_encoder.py \
  tests/test_task_context_teacher.py tests/test_diffusion_sampling.py
```

This is a causal representation pilot on known LIBERO tasks. Decoder MSE is
not a rollout success rate; mixture weights are not calibrated takeover
confidence. A causal action-conditioned policy and actual human-prefix-to-robot
suffix evaluation are subsequent milestones. TTT can be compared behind this
same representation interface after useful historical conditioning is measured.
