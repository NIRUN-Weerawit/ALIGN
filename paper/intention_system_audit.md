# Intention system audit

Date: 2026-10-07. Repository: `/home/whinnoy/ALIGN`.
Branch: `feat/semantic-intent-hysteresis`; inspected commit: `2dcafa9`.

Scope: README, FIXES, paper method/training sections, experiment log and review notes; intention model/encoder, vision/state/text encoders, attention, memory bank, all four action heads, dataset/semantic sidecars, intention trainer, offline evaluator, trajectory/async evaluators, live inference, and relevant tests. The README's three `docs/V4_*.md` documents are not present in this checkout. The legacy cross-attention mixer is a separate path and is not invoked by `train_intention.py`.

No training implementation was changed. This report distinguishes reproduced failures, static implementation defects, and design risks. These findings do not establish the quality of any particular saved checkpoint because no checkpoint/dataset rollout was available in this audit.

## Priority and evidence

- P0: correctness blocker, target leakage, or incompatible feature contract; fix before trusting new runs or rollout conclusions.
- P1: materially weakens training, semantic learning, validation, or an advertised model configuration.
- P2: robustness, scaling, or design risk requiring measurements after correctness fixes.

## 1. P0 — Camera features differ between raw training, precompute, and deployment

References: `models/align_model.py:348`, `models/align_model.py:372`, `training/train_intention.py:298`, `models/align_intention.py:306`, `scripts/precompute_dinov2.py:59` and `:81`.

Raw training and batched model forward flatten cameras into the image batch before calling VisionEncoder. VisionEncoder consequently sees V=1 and skips `cross_cam_attn`. Precompute instead calls with `(1,V,H,W,3)`, enabling cross-camera attention. It initializes that transformer afresh, freezes it, and saves only features and shape metadata; its weights are not saved with the cache. The online raw model forward again skips this transformation, while `encode_step` enables a separately initialized transformer.

Additionally, the entire fusion call is inside `torch.no_grad()`, so the supposedly trainable cross-camera transformer receives no gradients even on a path that calls it. With two cameras, these are substantially different representations, not numerical precision differences.

Reproduction: a synthetic backbone and parameterized camera fusion produced different outputs for multi-camera versus flattened-camera calls, and the fused output had `requires_grad=False`.

Fix: cache only frozen per-camera DINOv2 outputs; apply one shared trainable camera-fusion module outside `no_grad` in training and deployment. Alternatively, disable fusion consistently. Store/check camera order, backbone revision, preprocessing configuration, dataset identity, and feature layout in cache metadata. Compare raw and cached outputs using the actual backbone/checkpoint before regenerating runs.

## 2. P0 — Gripper input leaks the target and changes at deployment

References: `data/align_dataset.py:193`, `:593`, `:1266`; `training/train_intention.py:362`; `scripts/decode_libero_to_hdf5.py:57`; `eval/eval_libero_v4_trajectory.py:735`.

When there is no dedicated gripper dataset, the loader fills state[t,6] from action[t,6]. The training target starts at action[t], so the current target gripper command is present directly in its conditioning state. The LIBERO conversion script truncates observed state to six dimensions and writes no measured gripper field, making this fallback relevant to that conversion path. Both simulation evaluators instead feed constant 0.0 for the gripper state.

Reproduction: calling the actual V4 collate function on a sample using the action-column fallback confirmed `states_segment[...,6] == actions_segment[...,6]`.

Fix: preserve measured gripper position/width from the source observation and use the matching simulator observation online. If the intended input is a command history, use the previous executed command, name it explicitly, and align its timing. Keep current human command as an explicitly specified input only if the shared-autonomy contract actually supplies it at deployment. Re-evaluate gripper quality after removing leakage.

## 3. P0 — Padding is not excluded from diffusion/flow training loss

References: `training/train_intention.py:433`, `:437`, `:672`, `:869`; `models/intention_head.py:550`, `:719`; `data/align_dataset.py:1345`.

Generative heads reduce the entire batch/time/dimension error to a scalar. Multiplying that scalar by the valid fraction does not remove invalid samples: their repeated padded action targets still generate gradients. The batched trainer does not apply even this scaling. Thus segment-length distribution changes the learned objective.

Reproduction of the exact reduction: per-sample losses [1,100], validity [true,false] yield 25.25 through the implemented formula instead of 1.0 for valid-sample mean.

Fix: select `target[valid_mask]` and `cond[valid_mask]` before calling the loss, or expose unreduced losses and apply a proper sample/time mask. Reduce by the number of valid elements across all windows. Exclude padding from metrics and avoid storing fabricated observations after each sample's observed segment ends.

## 4. P0 — Regression action-head configurations cannot execute as advertised

References: `models/align_intention.py:183`, `:172`; `models/intention_head.py:76`, `:114`, `:192`, `:216`; `training/train_intention.py:1045`, `:390`.

Three separate problems:

1. `ALIGNIntentionModel` never assigns `self.mamba_d_state`, `self.mamba_d_conv`, or `self.mamba_expand`, but lazy construction reads them for mamba/hybrid heads. The default head is mamba, so the default build raises AttributeError.
2. The model passes `intent_dim * num_intent_tokens` to transformer/Mamba heads, but those heads apply Linear separately to each `(B,N,intent_dim)` token. N=2 produces a matrix-width mismatch. Multiplying by N is appropriate for the flattened generative conditioning, not these per-token projections.
3. Transformer/Mamba heads require input-window length == output chunk_size. Default history is 20 and chunk size is 10. Active memory collapses inputs to one step, which also violates this assertion. These heads therefore cannot substitute for generative heads on the current V4 interface.

Reproduced missing Mamba field, two-token transformer projection error, and history20/chunk10 assertion using isolated CPU probes. Mamba's missing field fails before any Mamba kernel is required.

Fix: register constructor fields; distinguish per-token intent width from flattened intent width; separate observation history H from action horizon K. Prefer K learned future-action queries conditioned on H history tokens, or project a final temporal summary into K outputs. Validate all head/intent/memory combinations before training.

## 5. P0 — Offline evaluation and live checkpoint loading are incompatible with current models

References: `eval/eval_intention.py:300`; `inference/align_inference.py:69`, `:80`, `:166`; `models/align_intention.py:391`, `:424`, `:523`; `eval/eval_libero_v4_async.py:271`.

Offline evaluation checks obsolete head names (`flow`, `diffusion_policy`), passes `h_current` instead of current intent tokens, and supplies `z_sext=` to methods that do not accept it. This raises TypeError before useful metrics for current heads.

Live inference reconstructs only a small subset of checkpoint config, loads `strict=False` before lazy head/bank construction, and never checks missing/unexpected keys. Trained action-head weights can be discarded as unexpected keys, then a random head created later. The encoder step returns raw 3D patches, bypassing SE/state modulation and flattening, and builds head width from raw_dim rather than compressed_dim. Stacking these features produces a 4D tensor incompatible with current heads. It also does not implement V4 memory/intent conditioning.

Async evaluation recognizes diffusion but not flow_matching, so the latter returns a condition tensor through predict_actions rather than sampled actions. The synchronous evaluator's head dispatch was fixed, but that fix was not propagated.

Fix: use one config-aware loader that constructs all lazy modules before strict weight loading. Use one explicit head-dispatch and conditioning helper for all evaluators/inference. Fix encode_step to produce the same compressed feature contract as training. Fail loudly for unsupported or missing essential checkpoint weights.

## 6. P1 — Semantic supervision and inference hysteresis are not implemented end to end

References: `data/align_dataset.py:610`, `:1193`, `:1349`; `training/train_intention.py:1055`, `:449`; `models/align_intention.py:157`; `paper/main_paper.md:269`.

The branch loads validated event labels into `__getitem__`, but V4/head collators discard `semantic_event`. Training never reads semantic labels or texts, never calls the text encoder, and never computes anchoring loss. `anchor_weight` is declared and logged but has no effect. `use_text` allocates an unused encoder. The paper claims CLIP alignment with default 0.1, whereas the actual flag defaults to 0.0 and is unused at every value.

There is also no persistent inferred-intent switching/hysteresis mechanism. `segment_gripper_events` debounces offline gripper boundaries; that is a labeling helper, not policy intent hysteresis.

A single event is chosen at the end of the original dataset window. V4 then crops a random segment and uses multiple action anchors within it; forwarding that single event unchanged would supervise the wrong phase for many anchors.

Reproduction: the actual V4 collation output contains no semantic-event field.

Fix: carry dataset-qualified episode identity, absolute segment start, event intervals, semantic targets, grounding, confidence, and masks through collation. Resolve the event for each current prediction anchor. Add and separately log a meaningful semantic objective with valid-label coverage. Keep the text target space fixed or otherwise prevent both sides of an alignment loss from collapsing. Grounding labels must supervise the correct camera/time coordinates. Add inference intent state and switching policy only after defining its latency/transition behavior and validating it. Update docs to describe implemented behavior.

## 7. P1 — The history encoder is disconnected when intent tokens are disabled

References: `training/train_intention.py:369`, `:407`, `:501`, `:660`; `models/align_intention.py:261`; `models/intention_encoder.py:359`.

`h_current` is computed in training but never consumed by any current action head. Only intent tokens can carry the intention encoder's temporal output to the policy. Without intent tokens, Mamba intention-encoder parameters receive no action-loss gradients. The model may still have head-local temporal processing or averaged vision history, but `--use-history` does not connect the intention Mamba as its help text suggests.

Reproduction: actual IntentionEncoder forward with a stand-in sequence layer produced history requiring gradients, but the current head-conditioning/loss path left every encoder parameter gradient absent.

Fix: include a trained temporal summary in conditioning when intent tokens are disabled, or explicitly remove the encoder for that ablation. Verify gradient coverage for every intended trainable module. Remove unused duplicate patch encoder and legacy projections once checkpoint compatibility is addressed.

## 8. P1 — Memory has multiple train/deployment contract defects

References: `training/train_intention.py:383`, `:395`; `eval/eval_libero_v4_trajectory.py:770`, `:777`; `eval/eval_libero_v4_async.py:264`; `models/memory_bank.py:73`, `:348`, `:354`.

- Training retrieves/fuses only when `intent_emb` exists. `--use-memory-bank` without intent tokens only stores entries; its perceptual/state retrieval is never used, despite the module supporting optional cognitive memory.
- Simulation evaluators compute fused visual/state outputs and then ignore them, passing the original history window to the policy. Training uses fused current visual/state features. Only fused intent reaches the evaluator's action head.
- Training stores every prediction window; evaluation stores only model calls. Increasing action_horizon therefore changes memory's effective time scale and content.
- Empty-bank handling uses `bank_mask.any()` for the whole batch, not per sample. A mixed batch with one nonempty row and one empty row generates NaNs for the empty row.

Reproduced the mixed-empty case: finite flags [true,false]. Multi-step differentiable memory and token merge did successfully backward in a small CPU probe; no general in-place-autograd failure was established.

Fix: retrieve/fuse visual/state memory regardless of intent availability; share training/deployment conditioning; define memory updates at observation/control cadence independently of action replanning. Handle empty rows separately. Track real validity/timestamps. Choose explicitly between differentiable segment memory and detached/truncated memory based on the desired gradient horizon; do not detach blindly as an alleged bug fix.

## 9. P1 — Diffusion inference step count changes the noise distribution

References: `models/intention_head.py:451`, `:538`, `:562`; `training/train_intention.py:425`, `:867`.

Training samples indices 0..10 inclusive for the fixed 11-point schedule, including an effectively zero-signal terminal point. Default sampling starts at index 9, not 10. More seriously, sample(num_steps=N) interprets N as a schedule-array bound, not a resampling of the full noise-to-data interval. Many callers set N to chunk_size, tying denoising noise levels to action horizon.

Reproduction: N=10 starts at alpha_bar=0.0241; N=4 starts at alpha_bar=0.7868 despite both initializing pure Gaussian noise; N=20 raises IndexError. Thus short sampling is strongly mismatched with the input distribution used during training.

Fix: separate training noise-grid resolution, inference denoising steps, and action horizon. Choose inference timesteps spanning the supported noise schedule and use their actual predecessor alpha values. Avoid directly dividing an epsilon prediction by an effectively zero alpha at the terminal endpoint. Use a well-tested scheduler and check noisy-input distributions with oracle probes. Hugging Face's [DDIM scheduler implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) is a primary reference for explicit timestep selection and matching training/inference indices.

## 10. P1 — Gripper loss and decoding need a single documented contract

References: `training/train_intention.py:265`, `:672`, `:918`; `models/intention_head.py:543`, `:709`; `eval/eval_libero_v4_trajectory.py:518`, `:899`, `:901`.

Gripper noise/velocity error is downweighted 100x in sequential training and validation but not batched training. In diffusion, the target is standard-normal noise for every dimension; raw binary action scale does not establish that gripper epsilon loss dominates pose epsilon loss. The present rule severely reduces gripper supervision without a demonstrated justification. Direct regression uses different weighting again.

Validation classifies with >0, whereas deployment uses `1 if value <= 0.5 else -1`. For 0/1 targets, tiny positive predictions count as closed under >0; for native signed targets, the deployment mapping reverses signs. The dataset converter copies source actions unchanged, so the actual dataset convention must be checked before assuming either interpretation. Both expert and model actions receive this conversion, potentially corrupting the expert baseline too. Six-dimensional model actions also hit an out-of-bounds `final_action[6]` fallback.

Fix: store/check encoding and polarity in dataset/checkpoint metadata; share canonical conversion across training, metrics, replay and deployment. Use measured state as in finding 2. Consider a separate K-step BCE gripper head with class/transition balancing, or keep gripper in the generative head with justified weights and proper normalization. Report open/close precision/recall, transition timing, and task success. Add hysteresis to decoded decisions only with a measured delay budget. Pad to seven dimensions before indexing a missing gripper.

## 11. P1 — Train/validation split leaks overlapping trajectory windows

References: `data/align_dataset.py:271`, `:277`; `training/train_intention.py:144`.

The dataset index advances eight frames by default, but each item reads eight plus traj_window frames. The trainer random-splits these overlapping indexed windows, not episodes. The same trajectories, frames, and task phases can appear in both partitions. A low validation loss consequently does not demonstrate held-out-episode generalization.

Fix: split by dataset-qualified episode ID before constructing windows. Persist the split and use it for all offline tests/probes. Add task-held-out evaluation separately if task generalization is a claim. Measure actual overlap in any datasets used for reported results.

## 12. P1 — Validation metrics and baseline comparisons are unreliable

References: `training/train_intention.py:892`, `:909`, `:1405`; `data/align_dataset.py:1221`; `training/train_intention.py:658`, `:806`; `data/align_dataset.py:1138`.

Per-dimension validation metrics use only the last successful window and include padded/invalid batch rows. They do not summarize the full evaluated segments. Segments are randomly resampled on each validation pass, and generative validation samples fresh noise and random noise levels. Model selection uses that changing scalar loss.

The collators instantiate `np.random.default_rng()` without a seed. Calling `np.random.seed(args.seed)` does not control these generators. Same-seed ablations therefore do not receive equivalent sampled segments.

Batched no-history training uses H history features while sequential no-history validation uses one current feature; they also apply different gripper weights. V3 trains on past action windows, while V4 predicts from the current time forward. Disabling V4 segment mode therefore changes the target semantics, and current validate still expects V4 keys.

Fix: define a single target-time/conditioning/loss contract shared across all trainer paths. Evaluate fixed held-out segments; mask and accumulate all windows. Use seeded worker-aware generators and repeatable evaluation noise seeds, then report sample variability separately. Mark missing gripper predictions as unavailable instead of padding the target into the prediction. Compare BC denoising loss with sampled action errors and closed-loop success rather than selecting from a single inconsistent metric.

## 13. P1/P2 — State-conditioned attention is global broadcast, not per-patch query selection

Reference: `models/intention_encoder.py:137`.

The query is one projected state vector expanded identically across every patch. It contains no patch feature or positional term. All attention rows therefore retrieve exactly the same summary. The patch residual preserves local differences, so this is not total feature collapse, but the claimed unique state-conditioned per-position attention is absent.

Reproduction: both maximum query difference and maximum retrieved-attention difference across positions were exactly 0.0.

Fix: either describe/implement this efficiently as a global state-conditioned summary, or use queries containing each patch plus state and camera/spatial position. Keep spatial information and ablate the result. `attn_scale` starts at 1.0 despite the identity-initialization comment; select and document initialization intentionally. The cross-camera transformer also lacks explicit camera identity/position inputs beyond DINO features, worth measuring after fixing its path.

## 14. P1 for recurrent deployment — Intent readout modifies persistent observation state

References: `models/intention_encoder.py:393`, `:400`, `:410`, `:418`; `training/train_intention.py:369`.

Training reruns a finite history window from a fresh Mamba state, appending intent tokens only at the end. `forward_step(produce_intent=True)` instead processes intent tokens directly in the persistent observation cache and returns the modified cache. Subsequent observations therefore follow synthetic readout tokens that were absent inside training histories. Persistent full-episode inference also differs from finite-window training.

The upstream [Mamba implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) explicitly updates both cache tensors in place, so merely holding a tuple reference cannot isolate readout. `allocate_state()` additionally calls `.to(device)` on the cache tuple and raises AttributeError with that API.

Fix: clone both cache tensors for intent readout and preserve the post-observation cache; choose matching finite-window or persistent recurrence semantics for training/deployment. Add batched-versus-step parity checks against the installed Mamba version, including gradients/precision where relevant. No actual Mamba CUDA kernel was available for this audit.

## 15. P2 — Numerical, capacity, and label-integrity risks

- Mamba special A_log/D parameters are tagged `_no_weight_decay` upstream, but the trainer uses one AdamW group with nonzero weight decay for every trainable parameter. Respect those tags and consider norms/biases separately. This is a quality risk, not a reproduced training divergence.
- Loss finiteness is checked, gradient finiteness is not. A finite forward with nonfinite backward can still corrupt optimizer state. Check the gradient norm/gradients before stepping; log skip counts and fail after persistent instability.
- The paper's claim that the SSM always runs in FP32 is not enforced by an explicit wrapper here. Upstream Mamba preserves some internal parameters in FP32, so absence of a wrapper alone does not prove BF16 failure. Validate the exact installed kernels and cache dtype on target hardware before asserting the paper's stronger claim.
- Actions and robot states have no fitted normalization contract. Mixed scales and orientation representation can make dimensions unevenly learnable. Fit any normalization on training episodes only, save its statistics/rotation convention, and invert actions at deployment. Check source observation layout instead of assuming the first six values always mean position plus Euler angles.
- Memory attention and gate networks operate at full flattened patch width. Perceptual retrieval plus gates cost roughly 10D² weights, about 671M weights at D=8192, before the policy and cognitive/state streams. Project memory to a compact latent and measure whether the full-width cost earns its benefit. Per-sample Python loops and `.item()` also synchronize GPU execution during storage/merge.
- Memory PE represents current slot index, not observed timestamp; merges renumber entries and average all streams based solely on visual similarity. Visually similar frames can contain different operations or gripper transitions. Preserve timestamps and event-boundary/transition metadata, and measure a protected-merge policy.
- Sidecar loading validates syntax but not dataset identity, episode bounds, grounding camera availability, or grounding-offset upper bounds. Dataset samples lose absolute start and dataset identity. Strengthen these before using semantic targets. `segment_gripper_events` also uses `if not commands`, which fails on a multi-element numpy array; normalize inputs or check length explicitly. Its single threshold plus debounce is not a two-threshold hysteresis rule.
- Feature-only probing is advertised by `forward_with_probe`, but that method still calls image preprocessing on 768-dimensional cached tensors. Use the same feature dispatch as training. Probe tooling also partially reconstructs architecture from CLI rather than the trainer's saved config and can supply zero state; fix loading/state provenance before interpreting embeddings.

## Verified and not verified

Executed on CPU with PyTorch 2.9.1:

- 16 existing semantic-schema/VLM parsing tests passed.
- Synthetic probes reproduced findings listed above, including label loss during collate, fallback gripper target leakage, dead history-encoder gradients, mixed-empty memory NaNs, regression-head failures, identical attention queries, fusion path mismatch, unseeded generator behavior, and diffusion schedule/index failures.
- Flow-matching noise-to-data integration with an oracle constant velocity recovered the target within approximately 1.2e-7. Its velocity sign and integration direction are internally consistent despite conflicting introductory comments.
- Five memory steps through consolidation with bank_len=2 successfully completed backward. Do not report a generic autograd mutation bug without a failing case.
- The diffusion noise-prediction and flow velocity losses themselves have the expected squared-error structure. Their main confirmed issues are masking, weighting/contracts, and diffusion schedule selection.

Synthetic probe source: `/tmp/align_intention_audit_probes.py`. No dependencies were installed. The available environment lacked h5py, mamba_ssm, and CLIP, so full dataset integration tests, actual DINOv2 parity, Mamba CUDA parity, checkpoint roundtrip, end-to-end training, and simulator rollout were not executed. The dataset collate function was exercised directly from its AST with real numpy inputs; vision and sequence dependencies were replaced only in the specifically described isolated checks.

## Recommended fix order

1. Fix feature/camera preprocessing, strict checkpoint loading, action dispatch, and all head-shape contracts.
2. Establish measured state/action timing and one gripper encoding/decoder; remove fallback target leakage.
3. Correct loss masks/reduction, validation accumulation, episode splits, seeded sampling, and trainer parity.
4. Connect the intention encoder and memory in every supported configuration; match recurrent/memory cadence and readout behavior online.
5. Correct diffusion schedule selection; recheck all action horizons and inference step counts. Keep flow matching as a useful comparison once common pipeline issues are fixed.
6. Implement per-anchor semantic supervision with confidence/grounding masks and measurable intent-switch behavior. Update paper/README to match the actual system.
7. Run a tiny overfit check, head/configuration smoke tests, gradient coverage, raw/cache and batch/step parity, gripper transition checks, and held-out closed-loop rollouts. Only then spend compute on large ablations and architecture tuning.
