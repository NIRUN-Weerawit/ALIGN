# Visual memory and intention-guided retrieval

Implemented 2026-10-18. Architecture/gradient verification only; matched policy
training and closed-loop success comparisons remain pending.

## Changes

1. `--visual-context-mode multiscale` supplies explicit camera-major patch tokens
   to the diffusion/flow U-Net, independently of the global FiLM/text condition.
   With two cameras and 256 patches each this is `[B,512,compressed_dim]`.
   Memory-enabled models automatically use patch-preserving episodic retrieval.
   No patch averaging is used in this side channel. The existing global FiLM
   condition remains available; flattening its visual input is a reshape and
   does not itself discard values.
2. Action features attend to this context at the bottleneck, intermediate
   decoder (`dec1`), and final decoder (`dec0`). For an eight-action chunk these
   have 2, 4, and 8 action positions. Attention width is 128 with four heads,
   learned camera embeddings and fixed 2D grid encodings. Residual output
   projections start at zero to preserve the initial U-Net function.
3. `--memory-intent-query` adds a projected, normalized **current causal intent**
   to the visual query before retrieval. The projection starts at zero and learns
   from the action objective. It does not require task text. Raw historical
   values, writes, relative-age encodings, and the current token used by the
   fusion gate retain their existing meaning. Without intention tokens the
   option has no effect and visual/state memory remains operational.
4. `--memory-retrieval-depth 2` applies two independent attention/FFN retrieval
   blocks to the same past bank, refining the query between reads. It applies
   to visual, state and (when enabled) cognitive retrieval. Fusion and the
   current-observation write happen once after the complete retrieval stack.
   Existing context-only, raw-value and field-mask modes are respected.

The explicit context follows the same fused patch features in ordered training,
grouped head losses, validation, sparse plain/history training, simulator
inference, and memory/dependence probes. It is passed as `visual_context` to head
loss/prediction/sampling rather than recovered by slicing a text-modified vector.
Projected visual K/V are shared across the three stages, projected once across
diffusion noise repeats (with gradients retained), and cached across sampling
steps. Plain controls receive the current patch grid through the same head;
plain-history also retains its existing ordered history attention.

## Training

Enable all three changes without text:

```bash
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2 \
  --output checkpoints/visual_memory_v2 \
  --variants intent_memory \
  --visual-context-mode multiscale --memory-intent-query \
  --memory-retrieval-depth 2 \
  --history-size 1 --temporal-sampling episode \
  --epochs 100 --batch-size 2 --micro-batch-size 1
```

For a matched four-condition comparison replace `--variants intent_memory` with
`--variants no_intent_no_memory intent_no_memory no_intent_memory intent_memory`.
For separate attribution compare depth 1/2, guidance off/on, and visual context
none/multiscale while keeping compression width, initialization protocol, split,
noise schedule, supervision cadence, text usage and update budget fixed.
Use correct/shuffled/empty histories and paired rollouts to establish useful
memory, rather than relying only on sensitivity or denoising loss.

Multiscale context replaces the older `--visual-token-attention` option. Do not
combine them. With raw-value retrieval also enable `--memory-patch-temporal`.

## Compatibility

- New training CLIs default to depth **2**. Visual multiscale attention and
  intention-guided queries require their explicit flags above.
- Model constructor and old checkpoint loaders default to depth **1**, no
  explicit visual context, and no intention-guided query when fields are absent.
  The first block keeps its original parameter names. Old checkpoint architecture
  is preserved rather than silently adding untrained blocks at inference.
- Resume checks include all three architecture choices. To resume an old
  one-block run, explicitly pass `--memory-retrieval-depth 1` and its original
  context/query settings. Architecture changes require a new run; `--warm-start`
  can transfer matching weights with a fresh optimizer.
- Checkpoint configs preserve the new choices; evaluation reconstructs them.
  Use a new output directory for the enhanced architecture.

## Verification

`tests/test_visual_memory_conditioning.py` covers three-scale visual access,
zero-initialized functional preservation, explicit-context gradients, padded
sample exclusion, K/V reuse, intention-dependent selection with untouched raw
writes, depth-two gradients/empty-bank identity, no-intent memory, CPU/CUDA BF16
training and validation, sparse plain/history controls with task text, and
checkpoint reconstruction through the production evaluator.

These are implementation contracts using synthetic inputs, not a measurement of
LIBERO policy quality. No full training or simulator success claim is made.

Final regression run: **156 passed** in 21.47 seconds (three existing robosuite
deprecation warnings). This includes CPU and CUDA BF16 production-loop smoke
tests using two-camera cached-feature shapes, diffusion and flow matching,
gripper/data contracts, sparse histories, future-supervision compatibility,
and checkpoint loading. Compilation and `git diff --check` also passed.
