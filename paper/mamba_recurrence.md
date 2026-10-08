# Causal Mamba recurrence and efficiency

Implemented on main, 2026-10-08. Existing Mamba1 parameters and learned appended
intent tokens remain compatible with existing state dictionaries.

## Training and validation

Previously every action window replayed its H observations through a fresh
Mamba state. With overlapping windows, the encoder repeatedly projected the
same observations and could not retain context earlier than H frames.

`IntentionEncoder.forward_sequence` now projects and causally convolves the
observation segment once. A differentiable FP32 SSM recurrence carries state
through the segment. Readout at time t processes learned tokens on a fork of
the observation state after t; the next observation receives the observation
state, never the token-readout state. Readouts for bounded chunks are batched.
Activation checkpointing bounds saved recurrent activations while preserving
gradients across chunk boundaries. The default internal chunk size is 16.

`train_v4_epoch` and `validate` perform this scan once per batch segment, through
the last eligible action start. Each prediction selects its causal readout at
the current time; future observations cannot affect that prediction. Action
head observation windows, eligible targets, padding masks and memory insertion
cadence retain their existing behavior.

State resets between independently sampled segments. Training does not carry
state across episodes or unrelated random crops. Longer within-segment context
is retained beyond the head's history window. `history_size=1` now means a
single-observation **action-head window**; Mamba still accumulates earlier
observations in the segment. Disabling intent tokens still omits Mamba entirely.

The implementation uses batched PyTorch projections/convolution and functional
SSM updates, rather than the inference-only in-place update kernel. This keeps
gradients through all observation states: the official selective scan's final
cache output does not supply those gradients for forked readouts. The existing
native full-sequence `forward` remains available for a single terminal readout.

## Streaming deployment

`forward_step` uses the native Mamba one-step kernel. Intent tokens run on
cloned conv/SSM states, fixing contamination of the carried observation cache.
`allocate_state` now correctly moves both tensors in its returned tuple.

`encode_step` now uses the same per-camera raw-image encoding and compressed,
state-conditioned patch features as the batch path. Its head inputs previously
contained uncompressed patches, with a different dimensional contract.

`IntentionStream` holds episode-local conv/SSM caches and encoded visual/state
windows. Initial head-window padding repeats encoded features; Mamba processes
only real observations. Construct a fresh stream after every environment reset.

The synchronous LIBERO V4 trajectory evaluator updates recurrence on every
environment observation and reads intent only when planning a new action chunk.
It encodes/transfers only the current image instead of replaying raw windows.
The asynchronous evaluator queues every observation on CPU in chronological
order, drains available observations, and coalesces action planning. It retains
the single-slot latest-window behavior for models without an intention encoder.
The recurrent CPU backlog can grow if inference cannot keep up; its maximum is
the finite rollout length. This preserves observation coverage but does not
guarantee real-time latency. No simulator success-rate evaluation was run.

## Verification

GPU tests check each causal readout against the native appended-token forward
on the corresponding full prefix, including one/two tokens, streaming parity,
cache isolation, future-observation perturbations, gradient parity across
checkpoint boundaries, and BF16 finite outputs/gradients. FP32 mathematical
parity checks disable TF32 temporarily because different batched GEMM shapes
can round differently. Production retains its existing precision settings.
Additional tests cover per-camera batch/step feature parity and asynchronous
observation ordering. Existing memory, padding, ablation, distributed trainer
and evaluator dispatch regressions were also run.

Production LIBERO Goal cached-data smoke tests completed one optimizer update
and validation over all 44 held-out episodes for intent with/without memory.
Artifacts: `checkpoints/mamba_recurrence_smoke_20261008/`.

Encoder-only CUDA forward/backward benchmark (batch 2, two cameras, raw CLS 768,
state 64, one intent token of width 128, BF16 autocast, FP32 SSM state):

| Segment | Head history | Action chunk | Previous median ms | New median ms | Ratio | Previous/new peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 8 | 8 | 12.22 | 8.97 | 1.36x | 178.46 / 167.50 |
| 40 | 20 | 8 | 23.35 | 15.14 | 1.54x | 213.51 / 180.70 |
| 80 | 20 | 8 | 115.20 | 34.23 | 3.37x | 355.22 / 221.02 |

These are eight measured rounds after warmup on an RTX 5060 Ti shared with an
existing unrelated training job. They are local encoder measurements, not
whole-policy throughput guarantees. The old/new state horizons differ by
design. Reproduce with `scripts/benchmark_mamba_recurrence.py --output PATH`.
Raw samples: `checkpoints/mamba_recurrence_smoke_20261008/encoder_benchmark.json`.

Previous ablation artifacts remain intact and reflect the old window-reset
training semantics. New recurrence requires retraining and rerunning the
ablation comparison before drawing policy-quality conclusions.
