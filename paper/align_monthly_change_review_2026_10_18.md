# ALIGN change and defect review — 2026-10-18

## Comparison and limits

Current revision: `e6449dd` (2026-10-18). Baseline: `203c535` (2026-08-06),
the latest commit in current main's history with a committer date before
2026-09-19. There is no closer September snapshot in that history. October
branch migration/reorganization means this comparison does not reconstruct
every feature branch's exact checkout on September 18. A September-authored
commit (`f253a84`) was committed onto this history in October.

Evidence: baseline/current source diffs, the October system audit and follow-up,
committed implementation notes, saved experiment records, and two small CPU
optimizer reproductions during this review. No new policy training or simulator
rollouts were run. New architecture, correctness, offline prediction gains, and
closed-loop success are distinct claims throughout this report.

## 1. Confirmed correctness fixes

| Area | Earlier defect | Change and scope |
|---|---|---|
| Diffusion timestep selection | Solver count acted as a schedule bound; some callers passed action horizon. Different horizons started Gaussian noise at incompatible noise levels, and sufficiently large counts indexed outside the grid. | Solver steps now subsample the full usable noise grid independently of action horizon. Validation and deployment use the head's configured sampling budget. |
| DDIM inversion/transitions | Singular terminal signal, an inconsistent clean endpoint, incorrect transitions across skipped timesteps, and BF16 rounding amplified during epsilon inversion. | Fresh schedules have clean alpha[0]=1 and terminal alpha>=1e-4. Training samples noisy timesteps. DDIM uses the actual next selected alpha and final clean endpoint. Sampling/inversion use FP32. Clipped estimates reconstruct consistent epsilon. Legacy saved schedules remain legacy schedules. |
| Padded losses | A scalar batch loss multiplied by valid fraction still trained on padded examples; some paths did not even scale it. | Diffusion/flow filter invalid examples before noise generation and reduction. Segment and accumulated microbatch losses weight valid sample/window pairs. Padded rows do not become observations in memory. |
| Gripper input leakage | Missing measured state fell back to current action[t,6], which was also the target. Deployment instead supplied zero. | Prefer measured observations; otherwise use previous executed command a[t-1], including when crops start mid-episode. Initial state comes from an episode attribute, with zero only as unknown initialization. |
| Cached gripper loader | After the shared causal reader was fixed, cached ablation loading still supplied constant zero, so it bypassed the repair. | Dense and sparse cached loaders now share the causal reader. The real Goal check verified all 428 episodes/52,042 frames; manifests preserve the input convention and retain an explicit legacy mode. |
| Gripper objective | Gripper epsilon/velocity supervision was downweighted 100 times, inconsistently across training paths. | Configurable equal per-dimension weighting defaults to 1.0. Gripper remains part of the generative seven-dimensional action, not a newly introduced BCE classifier. |
| Gripper execution/metrics | Thresholds, continuous predicted feedback, and model/simulator polarity differed. | 0=closed/1=open, threshold 0.5, simulator +1=closed/-1=open, and the actually executed binary command feeds the next observation. Motion scaling excludes gripper. Class-aware diagnostics supplement overall accuracy. |
| Optional intention encoder | Mamba was allocated/computed with intent disabled even though its output had no action-loss path. | Do not construct the intention encoder in that ablation. Its implementation remains available and is constructed when intent/history are enabled. |
| Memory without intention | The code stored memory but gated retrieval behind the existence of intention tokens. | Visual/state retrieval is active without cognitive intention. One conditioning interface serves supported training/validation/evaluation paths. |
| Memory in evaluation | Fused visual/state outputs were computed and then ignored by the policy head. Flow head dispatch was also inconsistent. | Supported evaluators feed all fused streams to the configured diffusion/flow head; optional task text is forwarded too. |
| Empty/mixed-valid memory | A whole-batch empty test allowed all-masked softmax for empty rows in mixed batches. | Per-row safe masking and identity for empty banks. BF16 storage and snapshot/autograd regressions found during optimization were repaired. |
| Recurrent cache contamination | Intent readout tokens modified the persistent observation cache; cache allocation also treated a tuple as a tensor. | Clone both convolution/SSM caches for readout, preserve observation-only state, and move cache tensors individually. |
| Raw/step feature contract | Streaming could return raw patches rather than training's compressed/state-conditioned features. Camera layout differed across paths. | Streaming encoding now matches per-camera batch layout and patch processing. Batch/step and causal-prefix parity tests cover supported paths. |
| Validation aggregation | Metrics used only the final successful window and could include invalid rows. | Aggregate valid windows and report sampled position, rotation, gripper errors separately from noise/velocity loss. |
| Episode leakage/reproducibility | Generic training randomly split overlapping dataset windows; unseeded collators and changing validation crops/noise obscured comparisons. | The **ablation runner** now persists task-stratified whole-episode splits (384/44 Goal), deterministic per-epoch sampling and fixed validation RNG. This is not fixed globally: generic `build_datasets` still random-splits windows. |
| Cached vision | Precomputation could save randomly initialized cross-camera-fused features while training used per-camera DINO features; precision/layout metadata were inadequate. | Versioned raw per-camera cache, fixed camera order, dataset-path/layout validation, rejection of legacy caches, and FP32 frozen-backbone execution. Old incompatible caches need regeneration. |
| Simulator state coordinates | Shortest-axis-angle conversion differed from LIBERO training's positive-X branch. Physically similar poses could differ by about 2*pi in encoder coordinates. | Shared sync/async quaternion conversion preserves the training branch, including angles above pi. Earlier incompatible-input rollouts were explicitly superseded. |
| cuDNN SDPA plans | Certain action/history and state/visual attention shapes failed on real CUDA hardware. | Attention backend contexts retain Flash/efficient/math SDPA while excluding failing cuDNN plans, restoring caller settings afterward. |

Sources: [original audit](intention_system_audit.md),
[follow-up](intention_fixes.md),
[diffusion/gripper audit](diffusion_gripper_condition_audit.md),
[causal gripper/compression pilot](../reports/compression_gripper_2026_10_14/REPORT.md),
[LIBERO state conversion](../eval/libero_state.py),
[training/experiment record](memory_adjustment_work_log.md).

## 2. Model structure: deliberate changes, not all proven quality improvements

The two visual streams already existed in the baseline: per-camera CLS/state
for Mamba, and compressed patch features for the action head. Mamba was **not**
newly changed from consuming all patches to consuming CLS. The major recurrent
change is how state persists and how causal token readouts are produced.

The latest enhanced configuration is:

```text
images / cached DINOv2 features
  CLS per camera + encoded robot state
    -> causal Mamba observation state -> inferred intention token
  patch tokens + encoded state
    -> SE channel compression / state modulation
    -> patch-preserving episodic retrieval and gated fusion
       visual query optionally includes inferred intention
       two sequential retrieval blocks read the same historical bank
    -> explicit camera/grid visual context at three action U-Net scales
       + existing global FiLM state/visual/intention condition
    -> diffusion/flow action chunk, including gripper
```

- **Patch-preserving retrieval:** replaces the very wide flattened visual token
  with shared small-width token operations when enabled. Flattening did not
  delete feature values; it made retrieval/gates costly and gave the action
  head less direct spatial access. The old flat path remains for compatibility.
- **Stronger head access:** explicit `[B,512,32]` visual context for two
  256-patch cameras at compression 32; camera identity and 2D grid position;
  cross-attention at bottleneck/intermediate/final decoder scales. Existing
  global FiLM conditioning remains. K/V are shared across scales and reused
  across denoising steps; training gradients remain connected.
- **Inferred-intention visual queries:** a learned projection of the current
  causal intention guides visual selection. This does not require task text,
  does not change raw stored values, and switches off for no-intent ablations.
  It does not newly condition the state-bank query on intention.
- **Depth two:** sequential reads of the same bank, followed by final fusion
  and one observation write. Old checkpoint loaders retain depth one when the
  saved field is absent. New training CLIs default to two; explicit visual
  context and intention queries still require their flags.
- **Temporal memory:** real observation timestamps and relative-age encoding
  on keys replace reliance on slot order after consolidation. Default episodic
  writes are raw/detached. Retrieval and gates learn through action losses;
  detached writes are an intentional contract, not an uncorrected autograd bug.
- **Experimental controls:** context-only retrieval, per-grid temporal lookup,
  raw-value retrieval, pre-state visual storage, missing-field masks, recency
  penalties, and observation dropout are configurable experiments. Missing-field
  retention defects have reproduced fixes, but these optional settings are not
  all enabled by the latest training command or universally beneficial.
- **Plain history:** new configurable `plain_history` baseline passes ordered
  feature/state observations to diffusion attention. It removes temporal mean
  pooling for that variant. Standard legacy mean conditioning still exists;
  history=1 has no multi-observation average to remove. The full-history variant
  currently excludes intent/memory and supports diffusion.
- **Task text:** optional trainable word-average task encoder supplies the
  action head. This is neither general pretrained language understanding nor
  semantic supervision of the intention encoder. No text masking was added.
- **Future supervision:** optional intention-only multi-horizon future CLS,
  visual-change and state-change loss. Targets are detached and masked. Default
  weight is zero; the supplied full-run command does not turn it on. Execution
  checks passed; short pilots did not outperform persistence.
- **Unsupported heads:** Transformer/Mamba/hybrid action-head implementations
  were retained but marked future work for V4 and rejected by that training
  contract. They were not completed. The supported heads are diffusion/flow.
- **Semantic hysteresis:** remains WIP, not an activated or trained feature.

Sources: [visual implementation](visual_memory_implementation_2026_10_18.md),
[Mamba recurrence](mamba_recurrence.md),
[future objective](intention_future_implementation_2026_10_18.md).

## 3. Training and loss calculation

- Earlier action anchors replayed overlapping finite history windows from
  fresh Mamba states. Training now projects/causally convolves observations
  once and carries differentiable FP32 SSM state within each segment. Learned
  intent-token readouts fork that state; activation checkpointing preserves
  gradients across internal chunks. This is not inference-only in-place cache
  reuse during backward. State resets across episodes/unrelated crops.
- Episode sampling retains all intervening real observations for causal Mamba
  and memory and samples 16 loss anchors per episode by default. Future action
  chunks remain contiguous in original time; sparse loss anchors are not sparse
  observation updates for a recurrent/memory model. Crop sampling remains an
  explicit alternative. Action-head history defaults to 1.
- For diffusion, the basic objective remains weighted epsilon MSE; for flow,
  weighted velocity MSE. The substantive common correction is to exclude
  invalid examples before noise generation and to normalize across valid
  sample/window counts, including gradient accumulation. Flow's interpolant
  `x_t=(1-t)*data+t*noise`, target `noise-data`, and integration from t=1 to 0
  were internally consistent already; no reversed-flow fix is claimed.
- Fresh diffusion training defaults to a 100-step noise grid, independent of
  the usual 10 inference steps and action chunk size. Multiple noise draws per
  condition are configurable (model/new runner default four). This changes the
  stochastic training budget; it is not automatically the same experiment as
  old ten-step/one-draw training.
- Optional transition sampling keeps the anchor budget fixed while reserving
  some training anchors for gripper changes; uniform validation remains. Width-8
  pilot switch recall improved from 43.1% to 52.8%, but stable false switches
  worsened from 21.1% to 25.6%. It remains opt-in, default zero.
- Width-32 text-only plain pilots had lower action errors than width 8 under
  the tested short budget. They also had 3.7x as many trainable parameters.
  This neither isolates compression information loss nor proves a final policy
  improvement, and it does not justify wide flattened memory.
- Sequential four-way ablations, identical recorded splits/sampling budgets,
  CPU/mmap resume with optimizer state, extensible total epochs, and explicit
  compatibility checks make comparisons more reproducible. Training/evaluation
  noise grouping and recurrence semantics changed, so old metrics/weights are
  not directly interchangeable with new runs.
- Distributed training now synchronizes gradients from helper/submodule paths
  outside `model.forward`, including parameters unused on some ranks.

## 4. Data loading, throughput, and operational reliability

- HDF5 handles reopen per spawned worker; persistent workers observe epoch
  changes. Mmap views, direct copies into final batch allocations, bounded
  prefetch, microbatching, and length bucketing reduce duplicate host allocation.
- Sparse plain/current-only loading reads the union of required history and
  target-anchor frames, instead of materializing complete episodes. Recurrent
  and memory variants preserve chronological context. Sparse models batch many
  anchors through the action head; recurrent models group head losses after
  their ordered retrieval steps.
- Memory writes/consolidation are vectorized; unnecessary GPU-to-host empty
  checks/count/argmax transfers were removed. Validation samples valid anchors
  together and aggregates metrics before CPU transfer.
- Local SSD atomic checkpoint staging uses immutable CPU snapshots, bounded
  asynchronous mirrors, coalesced latest/best generations, and preserved
  optimizer state. Frozen DINO weights are omitted from best/recovery saves.
  Incomplete files are not treated as valid progress merely because summary
  metadata advanced. Errors retain completed recovery files and surface visibly.
- Stage profiling separates data wait, materialization, transfers, encoders,
  retrieval, head loss, backward, validation, and checkpointing. CUDA event
  elapsed time is not equivalent to exclusive hardware utilization.
- The abandoned old flat-memory run had a 30.6 GiB recovery checkpoint and
  spent 91.7–95.1% of observed epoch wall time saving to NFS. Fresh patch-memory
  smoke recovery is 619 MiB; local foreground save took 0.87–1.81 s and optimizer
  resume 2->3 passed. Architecture changed too, and mirror output was local SSD:
  this is not an apples-to-apples full-run NFS speedup benchmark. Data waits
  still consumed 91–94% of those tiny training batches.

Sources: [stage profile](../TRAINING_STAGE_PROFILE.md),
[training instructions](../scripts/README_intention_ablation.md),
[Helios test](../reports/helios_fast_checkpoint_test_2026_10_18.md).

## 5. Evaluation and scientific interpretation

- Config-aware supported evaluation constructs lazy modules before loading,
  preserves legacy options, rejects missing trained parameters, and consistently
  dispatches diffusion/flow. Omitted frozen vision is an explicit exception.
- Sync/async simulation updates causal observation context independently of
  action replanning, uses corrected gripper/rotation contracts, and forwards
  task text when required by the saved checkpoint. Action horizon, independent
  episode seeds, held-out episode selection, and protocol metadata are recorded.
- Matched-noise normal/bypass/empty/wrong-history comparisons distinguish a
  useful bank from the benefit of extra learned transformations. Removing a
  module and worsening predictions alone never established correct-history use.
- Strict frozen task probes use causal prefixes, raw intention before memory,
  train-only decoders/scalers, whole-episode splits, and reset/current/random
  controls. Usage probes add natural same/other-task token donors, recent/reset/
  counterfactual histories, projection diagnosis and episode-paired intervals.
- These diagnostics found strong task information upstream but weak task-
  specific use of final intention tokens in selected compression-64 policies.
  Final token decoding was 35.3%/54.8% for no-text/text-trained policies, whereas
  final Mamba observation outputs decoded at 97.5%. Recent eight-frame histories
  almost reproduced full-history actions. This identifies a readout/objective
  problem in those checkpoints, not a proof that Mamba cannot represent tasks.
- A developed memory policy achieved 9/20 seeded handover successes versus
  6/20 for a separate no-memory policy, but same-policy memory bypass achieved
  7/20. Gains are small, development-selected and seed-dependent. They do not
  establish broad policy improvement or validate the latest multiscale model.
- Added optional pretrained XVLA LIBERO inference in the same simulator loop,
  including native camera/state/absolute-action conversions and a separate
  runtime. It is a pretrained baseline with potentially overlapping data, not
  a matched-data ALIGN ablation. Model weights are not committed.
- Branch organization moved foundational fixes into main while keeping semantic
  and XVLA finetuning feature work separate. This was repository organization,
  not a robot-performance improvement.

## 6. Remaining defects and priorities found in current code

1. **Generic trainer still leaks overlapping windows across train/validation.**
   `training/train_intention.py:build_datasets` still uses `random_split` on the
   indexed dataset. `v4_segment_collate`/`head_collate` still construct unseeded
   `np.random.default_rng()`. The ablation runner fixes these contracts locally;
   migrate that split/sampling protocol before trusting generic-trainer scores.
2. **Legacy live inference remains unsafe for these V4 checkpoints.**
   `inference/align_inference.py` still partially reconstructs config and loads
   with `strict=False` before lazy head construction. The supported evaluator
   fixes do not repair this older entry point. Unify loading/conditioning or
   explicitly reject incompatible checkpoints there.
3. **Cross-camera transformer is not reconciled.** Historical V4 docs describe
   active fusion, but current cached/batch/step intention paths deliberately use
   per-camera DINO features. The old `VisionEncoder.cross_cam_attn` still lives
   inside `no_grad` when invoked. Cache consistency was repaired; active shared
   trainable cross-camera fusion was not. The new action-head token attention
   is a distinct mechanism, not restoration of that transformer.
4. **`--grad-clip 0` contradicts documented disable behavior.** A fresh tiny CPU
   reproduction through `sync_and_step` gave weight 1.0->1.0 and gradient
   2.0->0.0 at clip 0, versus 1.0->0.9 at clip 1. The unconditional
   `clip_grad_norm_(params, grad_clip)` zeros gradients at 0. This was not fixed
   during this read-only code review. Supplied full-training defaults use 1.
5. **Non-finite gradient protection is absent.** The same tiny helper test with
   an explicitly injected NaN gradient stepped the optimizer and made the
   parameter non-finite. Finite loss checks do not guard every backward failure.
   This is a reproduced missing guard, not evidence that saved runs hit it.
6. **Mamba optimizer tags are not respected.** Both runner/generic optimizer
   setup still use one weight-decay group; upstream special `_no_weight_decay`
   parameters are not separated. This is an optimization risk, not a proven
   source of poor success. Check train-only state/action normalization and
   conditioning scale as well; no universal fitted normalizer was added.
7. **State-conditioned patch attention still broadcasts one state query.**
   `StateConditionalCrossAttn` expands the same state-derived Q over every patch.
   It adds a global state-dependent visual summary plus local residuals. It is
   not unique per-patch query selection. Description/backend were corrected;
   changing that design remains a separate experiment.
8. **No robust final memory/intention quality result yet.** Multiscale/depth-two/
   inferred-intent retrieval passed implementation and checkpoint tests, not
   full-budget policy evaluation. Future supervision and semantic hysteresis
   likewise are not established quality solutions. The immediate experiment is
   matched training and paired normal/bypass/wrong-history rollouts with enough
   independent seeds; local data staging addresses the remaining I/O bottleneck.

Useful code entry points: [optimizer/split helper](../training/train_intention.py),
[ablation runner](../scripts/run_intention_ablation.py),
[legacy inference](../inference/align_inference.py),
[vision encoder](../models/align_model.py),
[patch/intention encoder](../models/intention_encoder.py).

This review adds documentation only. No new fix was silently applied, no new
checkpoint was trained, and no commit/push was performed for this report.
