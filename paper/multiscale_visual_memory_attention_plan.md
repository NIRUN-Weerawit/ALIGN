# Plan: visual memory access across the diffusion U-Net

Recorded: 2026-10-18 (Asia/Tokyo).
Status: first prototype implemented; matched policy experiments remain pending.
Implementation: [visual memory implementation](visual_memory_implementation_2026_10_18.md).
Related review: [MemoryVLA success review](memoryvla_success_review_2026_10_18.md).

## Objective and hypothesis

Give memory-enhanced visual tokens direct access to several action-generation
layers. Test whether this improves closed-loop manipulation and useful reliance
on historical content, beyond improving processing of the current image.

The intended deployed policy receives no task instruction. The intention encoder
infers a latent objective from causal observation/state history using knowledge
learned during training. Text-conditioned models are diagnostic scaffolds, not a
required input to the final architecture. When the observed history cannot
distinguish objectives, the policy cannot uniquely infer one from that history.

Current intention tokens query the cognitive bank and condition the action head.
Visual and state retrieval currently use their own queries without directly
conditioning those queries on intention. Intention-conditioned visual/state
queries are a separate optional experiment: use the encoder's current causal
intent token, keep historical values intact, and preserve existing retrieval when
intent tokens are disabled. Visual intention-guided queries are now implemented
behind a separate flag. This experiment must be isolated from the attention
location comparison below.

The existing optional visual attention in `models/intention_head.py` acts only
at the diffusion U-Net bottleneck. For an eight-action chunk, this location has
approximately two temporal positions. Subsequent decoder layers reconstruct
individual actions without another visual-attention lookup. Flattening itself
retains feature values; the proposed change concerns where and how the head
accesses those values.

## First prototype

Keep the existing diffusion U-Net, sampler, action representation, and global
FiLM conditioning. Add a separate visual-context pathway:

```text
State + task text + intent + diffusion timestep
    -> existing global FiLM conditioning

Current visual patch features
    -> patch-preserving memory retrieval and fusion, when memory is enabled
    -> explicit visual context [B, cameras * patches, compressed_dim]
    -> cross-attention at three U-Net locations
```

With two cameras, 256 patches per camera, and compression width 32, visual
context has shape `[B, 512, 32]`. Without memory, provide current visual patch
tokens through the same interface and attention architecture.

### Explicit context interface

- Pass visual tokens separately from the combined global condition instead of
  reconstructing them from its first dimensions. Task conditioning currently
  adds a projection to that combined condition.
- Initially pass the existing fused current-plus-memory visual tokens. Preserve
  the patch axis and use patch-preserving memory retrieval for this experiment.
- Do not average the visual tokens or pass every raw bank entry to every layer.
- Carry this context consistently through head loss, noise prediction, sampling,
  training, validation, checkpoint reconstruction, and simulator inference.
- Preserve the existing head path and legacy checkpoint behavior when the new
  option is disabled. Save the new architecture settings in checkpoint config.

### Three attention locations

| Location | Temporal positions for an eight-action chunk | Intended role |
|---|---:|---|
| After bottleneck | Approximately 2 | Broad movement context |
| After intermediate decoder (`dec1`) | Approximately 4 | Refine movement segments |
| After final decoder (`dec0`), before output projection | 8 | Refine individual actions |

At each location, action features query visual context:

```text
Q = project_query(LayerNorm(action_features))
K = project_key(visual_context)
V = project_value(visual_context)
action_features += project_output(attention(Q, K, V))
```

Initial settings:

- Attention width: 128; four heads.
- Camera identity and 2D patch-position embeddings.
- Stage-specific query/output projections to accommodate U-Net channel widths.
- Zero-initialized attention output projections so new branches initially
  preserve the existing policy function. Avoid simultaneously zeroing another
  multiplicative gate that would prevent the branch from learning.
- Global FiLM conditioning continues to expose state, instruction, intention,
  and diffusion timestep to action features.
- Project/cache static visual keys and values across denoising steps during
  inference. Training reuse must preserve gradients; do not detach the new
  context pathway as an optimization.

The intended benefit is more direct access to visual detail for pose refinement,
contact, and gripper timing. These are hypotheses, not established outcomes.

## Controlled experiment matrix

The primary experiment targets instruction-free operation. Hold task-text usage
fixed across all compared variants; the no-text setting is the target condition.
A separate text-enabled matrix may be used as a diagnostic reference. Compare:

| Memory | No visual attention | Existing bottleneck attention | Proposed three-level attention |
|---|---|---|---|
| Disabled | Control | Current-image attention control | Current-image multiscale control |
| Enabled | Memory control | Existing visual-attention control | Proposed memory pathway |

Use the same episode splits, observation cadence, supervision/anchor schedule,
compression width, memory settings, optimizer budget, effective batch size,
sampling settings, and training seeds. Hold patch-preserving retrieval fixed
across memory-enabled attention variants. Record parameter counts and compare
multiple seeds before a final performance claim. Start with a fixed compression
width, rather than changing compression and attention locations together.

The no-memory models must receive the same new attention blocks: otherwise,
better current-image processing could be misattributed to historical retrieval.

## Verification and acceptance criteria

1. Verify shapes, camera/grid ordering, memory masks, and causal observation
   inputs. Future observations and current action targets must not enter context.
2. Verify gradients reach the new pathway once its zero-initialized output
   projection begins learning, and that disabled/initial branches preserve the
   legacy function. Check training/inference context parity and checkpoint reload.
3. Run matched-noise held-out probes with correct, shuffled, and empty historical
   banks. Keep current observations and task instructions identical. Whole-module
   bypass is an additional diagnostic; it also removes learned transformations.
4. Measure position and rotation error, actual gripper-switch recall, false
   switches, and switch timing. Average gripper accuracy alone is insufficient.
5. Confirm results with paired closed-loop rollouts using identical tasks, reset
   seeds, action horizons, and step limits. Select checkpoints on a development
   set and report final results on independent episodes/seeds.
6. Measure inference latency, training wall time, peak VRAM, and parameter count.

Accept a useful memory improvement only when correct historical content gives a
repeatable advantage over incorrect/empty history and improves closed-loop
control. Sensitivity to changed memory, lower denoising loss, or improvement over
module bypass alone does not establish that advantage. If multiscale attention
only improves the no-memory baseline, report it as a visual-conditioning gain.

## Follow-up experiment, after the first prototype

Expose current and retrieved visual tokens separately, with source embeddings,
so the head can distinguish present evidence from historical evidence. Add
timestamp embeddings only when individual temporal entries are exposed; fused
tokens summarizing several ages do not have one exact historical timestamp.
Evaluate this separately rather than combining it with the initial change.

## Scope

This note records the design. The explicit three-scale context pathway and the
separate intention-guided query option are implemented; policy training remains
pending. See the implementation note for flags, compatibility and verification.
Keep results in the existing memory development log. Transformer-head
replacement, memory-capacity expansion, and simultaneous semantic-encoder changes
are separate experiments.
