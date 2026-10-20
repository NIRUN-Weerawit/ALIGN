# Whole-demonstration task teacher

This isolated first milestone builds a versioned target cache and trains a
completed-demonstration encoder producing four 128-D task tokens. It does not
change the deployed ALIGN policy, infer goals from an unfinished prefix, or
implement TTT. The teacher sees the complete episode, including its outcome.

## Targets and provenance

```bash
python tools/build_task_context_targets.py \
  --data data/libero_goal.h5 \
  --cache data/libero_goal.dinov2.external \
  --output reports/task_context_teacher_20261020/targets \
  --fps 20
```

Use a new output directory. The existing primary cache symlink can be broken
while the external cache still exists; select the actual cache explicitly.
Camera order must match the versioned DINO cache metadata.

The builder reads every cached observation in bounded blocks, stores normalized
per-camera CLS and 2x2 patch summaries, and derives rotation-matrix state
features. When measured gripper state is absent, the state uses the **previous**
recorded gripper command, with an explicit provenance label. No raw human-command
stream is fabricated from pose differences or the action targets.

The output contains per-episode `.npy` summaries and `.npz` targets, train-only
normalization, and a content-hashed `manifest.json`. It defines disjoint train,
development, and confirmation episodes, stratified by offline task group.
Hashed task descriptions provide those groups in existing LIBERO recordings;
descriptions/IDs do not enter the encoder. This is an episode split, not a
verified scene/session/operator-disjoint split.

The unannotated Goal recordings have no per-episode success flags or approved
terminal intervals. The default last-eight-observation interval is therefore
labeled an **observed terminal outcome**, not an approved goal. Procedure targets
are eight ordered samples of realized visual/state changes, not verified task
event or contact constraints. Declared 20 Hz timestamps are used when the
recording does not contain timestamps. Object poses, force, and independently
verified human intent are unavailable in this cache.

Optional `--annotations annotations.json` accepts per-episode records:

```json
{
  "ep_000000": {
    "success": true,
    "terminal_interval": [104, 112],
    "objective_group_id": "numeric-or-symbolic-offline-group",
    "events": []
  }
}
```

Explicitly failed episodes are excluded from outcome-supervised examples.
Raw recordings and the existing DINO cache are never rewritten. The builder
rejects stale camera/source metadata, nonfinite real values, short action
targets, and invalid terminal intervals.

## Training

Train three models with the same architecture, initialization seed, target
manifest, losses, observation summary, and update budget:

```bash
python training/train_task_context_teacher.py \
  --targets reports/task_context_teacher_20261020/targets \
  --output checkpoints/task_context_teacher_20261020/temporal_full_seed42 \
  --mode full --pooling quarters --decoder-style temporal \
  --epochs 80 --batch-size 8 --anchors 4
```

Repeat with `--mode current` and `--mode constant`, using separate output paths
`temporal_current_seed42` and `temporal_constant_seed42` and the same other flags.
The full teacher reads the episode bidirectionally; the current control reads
only the particular anchor observation; the constant control has learned
constant task tokens. Each action head retains the same live current input.
The current control cannot access initial/future frames through its encoder.

`--pooling quarters` gives each full-teacher readout query an ordered temporal
quarter of the completed episode. The bidirectional encoder still has global
access. This is a privileged teacher readout prior, not a causal progress input.
The option never runs in current/constant modes; those controls can therefore
be reused for this readout ablation. The assessment checks all other architecture
settings and training budgets. Tests verify exact control-output invariance.
`--decoder-style temporal` adds time-indexed access to the four tokens when
reconstructing outcome/procedure targets. It is active in all modes, so its
controls must be trained with the same decoder. Original `learned`/`flat` defaults
remain available to reproduce the earlier failed candidates.

The objective combines the existing diffusion U-Net loss, a standardized
deterministic continuation probe, terminal/procedure feature changes, geometric
state reconstruction, offline group contrast, and a variance penalty. The
deterministic probe is a diagnostic, rather than a newly supported production
action head. Targets and group labels are not model inputs.

Development selection averages normalized continuation, terminal, procedure,
and state prediction error, with one vote per episode. Confirmation episodes
never select checkpoints. Validation is FP32; CUDA training uses BF16 where
supported and FP32 losses. Training starts from scratch, with no DINO model
loading or modifications to existing policy checkpoints. New output paths are
required to preserve earlier runs.

`--max-steps 100` bounds a real optimizer/profile check. The default per-process
CUDA memory limit is 24% of the GPU; `--memory-fraction` changes it. Peak allocated
memory is logged. Set CPU thread count and data-loader process count carefully
when sharing a machine; this trainer uses two CPU threads and zero loader
workers.

## Information gate

```bash
python tools/assess_task_context_teacher.py \
  --targets reports/task_context_teacher_20261020/targets \
  --checkpoint checkpoints/task_context_teacher_20261020/temporal_full_seed42/teacher_best.pt \
  --current-checkpoint checkpoints/task_context_teacher_20261020/temporal_current_seed42/teacher_best.pt \
  --constant-checkpoint checkpoints/task_context_teacher_20261020/temporal_constant_seed42/teacher_best.pt \
  --split development --sample-actions \
  --output reports/task_context_teacher_20261020/development_temporal_seed42.json
```

The gate was specified before the first assessment:

- Terminal and ordered-trace MSE each improve at least 10% over both independently
  trained controls, with positive paired episode-bootstrap benefit intervals.
- Wrong-task tokens harm normalized continuation prediction relative to correct
  tokens and to distinct same-task donor tokens; both 95% paired intervals must
  exclude zero.
- A train-only-fitted linear task probe decodes teacher tokens above chance.
  Probe accuracy is supplementary; it cannot replace the intervention tests.

The evaluator holds the current input fixed while changing only task tokens.
Donors and action noise use fixed seeds. It also evaluates training-mean tokens,
current-image endpoint persistence, and population endpoint prediction. Optional
DDIM sampling reports actual sampled-action errors under paired initial noise,
separately from the deterministic continuation probe. Intervals use 10,000
resamples of whole episodes, retaining all three anchors within an episode;
they are pointwise diagnostic intervals, not simultaneous confidence bounds.

After development passes, rerun with `--split confirmation`, a new output path,
and `--confirm-development-report path/to/passing_development.json`. The evaluator
checks that the selected full/current/constant checkpoints and target manifest
are unchanged. Never optimize a new model using confirmation results while
continuing to call that split untouched.

Reports include per-episode metrics, donor identities, target/checkpoint hashes,
and held-out task tokens. A passing gate establishes useful information about
**completed demonstrated outcomes/traces and oracle continuation**. It does not
establish causal goal inference, procedure-constraint understanding, manipulation
success, or autonomous takeover readiness.

After confirmation passes, export frozen latent supervision:

```bash
python tools/export_task_context_teacher.py \
  --targets reports/task_context_teacher_20261020/targets \
  --checkpoint checkpoints/task_context_teacher_20261020/teacher_frozen.pt \
  --gate-report reports/task_context_teacher_20261020/confirmation_temporal_seed42.json \
  --output reports/task_context_teacher_20261020/frozen_latents
```

The exporter requires a passing confirmation assessment and verifies the selected
teacher/target hashes. It produces separate train/development/confirmation token
arrays and a provenance manifest. These tokens are privileged training targets,
and must never be substituted for a causal deployed encoder's predictions.

## Checks

```bash
python -m pytest -q tests/test_task_context_teacher.py tests/test_diffusion_sampling.py
```

Checks cover provenance, split and file integrity, train-only statistics, causal
gripper fallback, spatial slot order, physically equivalent rotations, masked
NaN padding, current-control input isolation, target/label input exclusion,
finite optimizer gradients, checkpoint roundtrips, paired donors, and unchanged
confirmation checkpoint enforcement.
