# Intention audit follow-up

Date: 2026-10-08. Branch: `main`.
Integration target: `main` in `/home/whinnoy/ALIGN`.

The main ALIGN checkout was on `feat/xvla-finetuning` with existing requirements
and dataset-decoder edits. Those edits are preserved separately while the
foundational fixes are integrated onto main.

## Requested decisions and changes

1. **Camera fusion disabling conflicts with the historical V4 docs.** On
   rechecking the original ALIGN checkout, its ignored local `docs/` directory
   contains `V4_SYSTEM_OVERVIEW.md` (lines 18, 56, 97), which specifies active
   cross-camera attention. `CHANGELOG_2026-07-22.md` (line 70) explicitly lists
   it as active; `NOTES_PATCH_LEVEL_VISION.md` (lines 14–21) also retains it.
   These files were missed because they are absent from the semantic worktree.
   No decision to disable it was found in those docs. The disabling change is
   excluded from main. The existing model fusion configuration and the versioned
   per-camera cache path are retained; reconciling train/cache/inference fusion
   remains separate work. The cross-camera module remains in the codebase.

2. **Gripper conditioning is causal.** Measured gripper state takes precedence.
   Otherwise the input at t carries the previous executed command a[t-1], never
   the target a[t]. Dataset reads obtain the previous command even when a window
   begins mid-episode. An `initial_gripper` HDF5 attribute supplies the initial
   state; unknown initialization uses neutral zero only once. Both simulator
   evaluators carry the last executed command in model/data units. Updating the
   next state does not rewrite gripper values in earlier observation history.
   Targets are unchanged.

   Conditioning on a current human command is valid if it is an explicitly
   available deployment input. In the audited pipeline it was used as a state
   substitute while deployment supplied zero, allowing the model to copy a
   training-only answer. Previous-command conditioning remains useful because
   it represents available history; correlation with future actions is not
   itself leakage. Existing simulator command-polarity conversion is a separate
   unresolved audit item.

3. **Generative padding masks are applied before loss reduction.** Both
   diffusion and flow matching filter targets and conditioning before generating
   training noise. Sequential training, batched training, and validation pass
   the actual sample mask. Segment objectives are normalized across valid
   sample/window pairs. Validation error metrics now aggregate every valid
   window. Batched no-history training and validation use the same observation
   width and existing dimension weights. The last eligible history window is
   included.

4. **Transformer/Mamba/hybrid action heads are future work.** Documentation
   marks their status, V4 training rejects them with an explicit explanation,
   and the trainer/model default to diffusion. Their legacy implementations
   remain available; this change does not attempt to finish them.

5. **Semantic hysteresis and supervision remain WIP.** README, paper, and CLI
   help explicitly distinguish existing sidecar loading from unconnected
   semantic objectives and inference hysteresis. They are not activated by this
   change.

6. **The intent-token flag controls intention encoder construction.** With
   `use_intent_tokens=False` (omit `--use-intent-tokens` at the CLI), no Mamba
   intention module or its parameters are constructed for that ablation. The
   encoder implementation remains in the codebase and is constructed when
   intent tokens and history are enabled. Observation
   history can still feed the policy; disabling intent tokens is distinct from
   disabling the observation window. Intent tokens require an enabled history
   encoder and invalid flag combinations fail clearly.

7. **Memory retrieves visual/state context without intent tokens.** One shared
   `condition_actions` path is used by sequential training, validation, and both
   simulator evaluators. With memory enabled it retrieves, gates, and stores
   visual/state context even when cognitive intent is absent. All fused streams
   reach the policy. Memory storage excludes unobserved padded rows and
   mixed-empty banks no longer enter an all-masked attention softmax. Async
   evaluation now samples flow-matching actions as well as diffusion actions.

## Validation and compatibility

Main integration checks: 26 targeted tests passed, including two-process CPU
training, versioned caches, and the Piper converter; Python compilation and
`git diff --check` passed. Semantic-label tests run on the feature branch.

Regression coverage checks causal dataset state across cropped windows,
measured-state precedence, episode boundaries, padded loss values/gradients,
actual trainer and validation masks, mixed-empty memory, gradient-bearing
retrieval without intent, cognitive memory with intent, simulator
gripper-history timing, and future-head rejection.

Tests use synthetic data and a stub DINOv2 backbone to avoid downloads. The
available Python environment has no Mamba kernels or LIBERO installation, so
real-model GPU parity, full training, and physical simulator rollouts remain
unverified. h5py was installed only into `/tmp/align-intention-test-deps` for the
tests; project requirements and the existing environment were not modified.

Old feature caches generated with random cross-camera fusion must be
regenerated. Older checkpoints can contain unused camera/disabled-intent
encoder weights; loading should report those mismatches. Gripper input timing
has changed, so new training or fine-tuning is required before assessing the
quality improvement. Correct masked/normalized losses are not numerically
comparable to the previous padded, summed objectives.

Other audit items, including diffusion schedule selection, episode-level
splits, legacy offline/live inference compatibility, intent readout cache
semantics, and memory cadence at longer action horizons, remain separate work.
