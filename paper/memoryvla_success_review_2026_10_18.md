# MemoryVLA success review — 2026-10-18

Scope: original MemoryVLA / OpenVLA codebase, not MemoryVLA+ or MemoryVLA++.
Reference source commit: `d732ea9072bc063399ccc817aed74ab172eb50be`.
Paper: arXiv 2508.19236v2 (ICLR 2026).

## Sources inspected

- https://arxiv.org/html/2508.19236 — Sections 3, 4.6; Appendices C, D, G.
- https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/vla/memory_vla.py — cognition/perception, bank management, inference.
- https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/vla/datasets/rlds/dataset.py — `group_sample` sorts sampled timestamps.
- https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/action_model/models.py — perception cross-attention in DiT blocks.
- https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/train.py — actual defaults.
- https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/script/train/libero/train_libero_goal.sh — released training recipe.

## Verified distinctions and interpretation

The released model stores detached history and trains retrieval through its action
objective. A current-query residual is also present in retrieval. Neither detached
writes nor such a residual alone explains ALIGN's weak historical-content benefit.
Paper diagrams describe fused writes, while released `update_fused=False` stores
raw features; EMA is optional and defaults off. Do not claim fused writes, full
bank BPTT, auxiliary memory supervision, or EMA are required by this reference.

ALIGN already implements episodic timestamps, detached/raw writes, gates,
consolidation, optional patch retrieval, and optional visual attention. Adding
these mechanisms again is not a useful priority. Its default retrieval and tested
flat-memory models differ from the token architecture in the reference.

The semantic-conditioning difference needs to be interpreted against ALIGN's
instruction-free goal. ALIGN's cognitive encoder reads visual CLS/state, while
optional task text enters `action_condition` after memory retrieval. The intended
policy infers objectives from causal experience through its intention encoder;
external instructions are diagnostic scaffolding, not a required deployed input.
An appropriate experiment is to condition visual/state retrieval queries on the
inferred intention token. That token already queries cognitive memory and enters
the action head, but does not directly condition visual/state retrieval queries.
This is a recommendation, not an established cause of ALIGN failure.

A second gap is how retrieved visual details reach the action head. Compare token
access across multiple denoising blocks with the current flat conditioning, keeping
capacity comparable. Increasing compression width without changing flat retrieval
can greatly increase memory parameters.

A third experiment should compare sparse, ordered episode sampling with dense
bank accumulation under matched observation cadence and supervision budgets.
MemoryVLA's grouped training is not evidence that continuous full-history BPTT is
necessary. Batch units differ: its frame batch and ALIGN's episode batch cannot
be compared directly by their numeric sizes.

Our earlier normal/shuffled-bank position MSE was 0.023611/0.023665, while complete
module bypass also removes learned transformations. Retain correct/wrong/empty
history interventions, independent closed-loop selection/testing, and tasks where
a past cue changes the correct action despite identical current observations.
Strong no-memory pretrained policies and comparable-capacity controls are needed
to separate representation quality from retrieval benefit.

Research-only review: no production model changes or new training runs performed.
Existing uncommitted XVLA integration changes were left intact.
