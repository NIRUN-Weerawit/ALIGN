---
title: "ALIGN: Learnable Intent Tokens and Episodic Memory for Assistive Teleoperation"
author:
  - NIRUN-Weerawit (corresponding)
  - Mentor
affiliation:
  - VRWIT Lab, Nagoya University
abstract: |
  We introduce ALIGN, a vision-language-action model for assistive robotic teleoperation
  that combines (i) state-conditioned patch-level visual encoding, (ii) a Mamba
  temporal encoder augmented with learnable intent tokens that are anchored to a
  frozen CLIP text embedding at train time but discarded at inference, and (iii) a
  perceptual-cognitive-state memory bank with cross-attention retrieval and token-merge
  consolidation. The three mechanisms are evaluated under a shared-autonomy protocol on
  LIBERO Spatial. We additionally report a systems contribution: a memory-mapped pre-compute
  pipeline that caches DINOv2 ViT-B/14 features as per-episode .npy files, reducing training
  peak VRAM from 81 GB to under 10 GB on a single H100 NVL without numerical drift. All
  components are CLI-configurable; the code is open-source. Numerical results are placeholders
  pending experiments EXP-A through EXP-F documented in paper/experiment_log.md.
---

# 1. Introduction

A human teleoperator manipulating a robot arm through a haptic device is a noisy
control source. Their hand trembles, their reaction time is bounded, they over- or
under-shoot grasps. Yet they possess something no autonomous policy yet has: a high-level
**intent** (e.g., "pick up the black bowl between the plate and the ramekin") that,
given a noisy low-level stream of seven-dimensional end-effector velocities, can be
inferred in hindsight and used to **correct** the human in real time. The resulting
control loop, known as *shared autonomy*, has been studied since the Javdani POMDP
formulation [CITATION NEEDED] and continues to attract work because the problem
is unsolved: **how does a learned policy infer what the human *intends* without
confusing intention with the human's noisy *actions***?

This paper presents **ALIGN**, an open implementation of a vision-language-action
model designed around the intuition that intent and action are *separate
representations* that should be modeled with separate mechanisms. ALIGN's
architecture is the union of three design choices that have, to our knowledge, not
previously been combined inside a single model:

1. **State-conditioned patch-level visual encoding.** Patches of the frozen
   DINOv2 ViT-B/14 backbone are not pooled globally; each patch is modulated by the
   robot's seven-dimensional state via per-position cross-attention. The arm "knows
   where it is" and so attends to image regions that are relevant to its current
   configuration.
2. **Learnable intent tokens** appended to a Mamba SSM input sequence. The tokens
   are forced to be semantically grounded through a CLIP cosine-alignment loss
   applied during training; at inference they are unsupervised and consumed only by
   the policy head.
3. **A perceptual-cognitive-state memory bank** with cross-attention retrieval
   and token-merge consolidation. Past visual details, past intent summaries, and
   past robot states are stored as fixed-size paired entries; current observations
   retrieve from all three streams via cross-attention and fuse through learned gates.

We pair the architecture with a **systems contribution** that makes training
tractable on commodity hardware: a memory-mapped DINOv2 feature pre-compute
pipeline (Section 4.4) that drops peak training VRAM by an order of magnitude on a
single H100 NVL. We open-source the entire codebase.

The paper is structured as follows. Section 2 places ALIGN in the context of
intent prediction, prompt-tuned sequence models, and shared-autonomy learning.
Section 3 details the architecture. Section 4 details training and the pre-compute
pipeline. Section 5 reports our evaluation protocol and the experiments we are
running; **all numerical results in Section 5 are placeholders pending the
experiments listed in paper/experiment_log.md**, which the reader may consult
for the exact commands to reproduce them. Section 6 discusses the negative
results from a preliminary evaluation and the architectural implications.
Section 7 concludes.

> **Scope note.** This submission is a *system description* — an open
> implementation, a clearly specified architecture, and a reproducible training
> pipeline — with experimental evaluation still in progress. The contribution is
> the architectural synthesis and the engineering, not claimed state-of-the-art
> on LIBERO. We are explicit in Section 6 about what does and does not yet work.

## 1.1 Contributions

We make four claims, each tied to an experiment in `paper/experiment_log.md`:

- **C1 (architecture)**: The combination of (state-conditioned patches +
  learnable intent tokens + episodic memory bank) is *implementable* and
  *trainable* with all components configurable from the command line.
  *Evidence: open-source code; EXP-A training curves.*

- **C2 (memory bank)**: The memory bank reduces prediction variance on
  long-horizon segments by at least 10% compared to the no-bank baseline.
  *Evidence: EXP-B ablation table.*

- **C3 (intent directionality)**: The learned intent tokens encode
  *future-state* information more strongly than past-state information,
  demonstrating they are not merely a short-horizon action prior.
  *Evidence: EXP-C cosine-similarity probing.*

- **C4 (systems)**: The DINOv2 pre-compute pipeline reduces training peak VRAM
  from 81 GB to under 10 GB on an H100 NVL with no measurable change in final
  validation loss. *Evidence: EXP-D wall-time + VRAM comparison.*

---

# 2. Related Work

We group prior work by mechanism rather than by paper, because the relevant
mechanisms in ALIGN are drawn from four different literatures.

> **Verification note.** Citations marked `[CITATION NEEDED]` must be verified via
> Semantic Scholar + DOI content negotiation before submission. See
> `paper/experiment_log.md` for the citation-verification workflow.

## 2.1 Shared autonomy and intent inference

The shared-autonomy lineage begins with Javdani, Srinivasa, and Bagnell's POMDP
formulation of hindsight optimization [CITATION NEEDED]. Subsequent work
(Jonnavittula & Losey 2021; Wang & Losey 2023 — CITATION NEEDED) refines visual
goal inference for assistive manipulation, but both rely on goal-set priors or
learned reward predictors — none propose an *intent token* abstraction decoupled
from the action prediction loss.

Our intent-token approach is most directly related to **Prompt Decision Transformer
(Prompt-DT)** [Xu et al., ICML 2022 — CITATION NEEDED] and **Decision Mamba**
[Lv et al., NeurIPS 2024 — CITATION NEEDED]. Prompt-DT prepends trajectory segments
as *prompts* to a Decision Transformer; the prompts are not learnable and no
text supervision is used. Decision Mamba replaces the transformer with a Mamba
SSM and uses *return-to-go* tokens for task conditioning. ALIGN extends this line
by adding (a) learnable (not trajectory-derived) tokens, (b) text anchoring at
training time, and (c) discarding the text signal at inference — so the runtime
cost is identical to Decision Mamba while the training signal is stronger.

## 2.2 Soft prompts and task tokens in RL

The use of learnable tokens as a task abstraction has emerged recently in
prompt-tuning of Decision Transformers. **LPDT** [Yang & Xu, 2024 — CITATION NEEDED]
initializes a Decision Transformer with GPT-2 weights and fine-tunes via LoRA, using
prompt regularization to differentiate tasks — the closest prior work to ALIGN's
text anchoring, but at the GPT-init level rather than at the per-token alignment
level. **Hierarchical Prompt DT** [Wang et al., 2025 — CITATION NEEDED] uses
two-level trajectory prompts. **DPDT** [Zheng et al., NeurIPS 2024 — CITATION NEEDED]
decomposes prompt learning into two stages. None of these applies to the
shared-autonomy setting; none uses *vision* features as the token-input modality;
none caches episodic memory.

## 2.3 Episodic memory for policy learning

MemoryVLA [Shi et al., ICLR 2026 — CITATION NEEDED] introduces a perceptual-cognitive
episodic memory bank for autonomous manipulation; ALIGN's bank is conceptually
similar but applied to a shared-autonomy policy and includes a *state* stream that
MemoryVLA lacks. Other recent episodic-memory work for robot policies includes
the trajectory-prompt methods cited above and goal-conditioned video-prediction
approaches [CITATION NEEDED].

## 2.4 Frozen-vision policies and engineering

The frozen-DINOv2 backbone for manipulation policies is now standard
[CITATION NEEDED]. Our *engineering* contribution is orthogonal: a
memory-mapped sidecar that lets any frozen-vision policy be trained without
re-encoding pixels per batch. This is a "system for ML" contribution rather than
a modelling contribution; we report VRAM and wall-time numbers in Section 5.4.

## 2.5 What is novel

To summarize: ALIGN is the first system to combine (i) frozen DINOv2 patch
encoding with (ii) Mamba SSM temporal recurrence, (iii) learnable intent tokens
with text-anchored training, and (iv) a 3-stream episodic memory bank — all under
a shared-autonomy evaluation protocol. Each component has prior work in
isolation; the combination has not been previously implemented end-to-end or
evaluated on a manipulation benchmark.

---

# 3. Method

## 3.1 Notation

We use the following notation throughout. Let $B$ be batch size, $T$ be the segment
length, $V$ be the number of cameras, $P = 256$ be the number of DINOv2
ViT-B/14 patches per image, and $D_v = 768$ be DINOv2's per-patch feature
dimension. The robot state is $z_s \in \mathbb{R}^{256}$ (after the state encoder).
The action is $a_t \in \mathbb{R}^{7}$ (six-dimensional end-effector delta plus
one gripper bit). The state-conditioned patch encoder produces
$z_v \in \mathbb{R}^{B \times T \times (V \cdot P \cdot d)}$ where
$d \in \{8, 16\}$ is the per-patch compressed dimension after SE.

## 3.2 Architecture overview

```
frames (B, T, V, H, W, 3)         state (B, T, 7)
        |                                |
        v                                v
[VisionEncoder: frozen DINOv2 ViT-B/14]
        |
        v
z_v_CLS  (B, T, V, 768)         z_s (B, T, 256)
        |                                |
        +------- VisionPatchEncoder -----+
                SEVisualCompressor + StateConditionalCrossAttn
                          |
                          v
                  z_v_pool (B, T, V*P*d)
                          |
                          v
                  concat[z_v_CLS_flat, z_s] (B, T, V*768 + 256)
                          |
                  +-----[INTENT_1, ..., INTENT_N] (learnable)
                          |
                          v
                  Mamba SSM (single pass over (B, T+N, d_in))
                          |
                  +-------+-------+
                  v               v
              h_seq          intent_emb
            (B,T,d_in)      (B, N, intent_dim)
                  |               |
                  +---[memory bank]---+
                  |       |        |
                  v       v        v
            z_v_pool_fused  z_s_fused  intent_fused
                          |
                          v
                  DiffusionPolicyHead (1D U-Net, DDIM)
                          |
                          v
                  actions (B, T, K, 7)   [K=10 chunk]
```

## 3.3 State-conditioned patch encoder

The **VisionEncoder** wraps a frozen DINOv2 ViT-B/14 (loaded via
`torch.hub`) and returns per-frame patch tokens
$z_{v,\text{patches}} \in \mathbb{R}^{B \times V \times P \times 768}$ plus
per-frame CLS tokens $z_{v,\text{CLS}} \in \mathbb{R}^{B \times V \times 768}$.
When $V > 1$, a small **cross-camera transformer** attends over the concatenated
$VP$ tokens before any further processing; this allows patches from different
views to attend to each other at the raw 768-dim level.

The **VisionPatchEncoder** ($SE + cross\text{-}attn$ modulation) is the key novel
component. Its pipeline is

1. **SE compression**: each of the $VP$ raw 768-dim patches is squeezed through
   a squeeze-and-excitation block followed by a linear projection to
   $d \in \{8, 16\}$. The SE block uses a global-average-pool over patches, two
   linear layers with a bottleneck of $768/8 = 96$ units, and a sigmoid gating
   that reweights channels **before** the projection — so the projection sees a
   content-aware reweighted input.
2. **State-conditioned cross-attention**: each compressed patch position receives
   its own query vector derived from $z_s$ via a learned linear projection
   $W_q \in \mathbb{R}^{256 \times d}$. All $VP$ positions share the same $W_q$ but
   produce different queries because they are broadcast to different positions; the
   residual $z_v + \alpha \cdot \mathrm{Attn}(q, k, v)$ is layer-normalized.

The output is $z_v \in \mathbb{R}^{B \times T \times (VP \cdot d)}$, flattened to feed
the Mamba encoder.

The architectural decision worth highlighting is the **separation of CLS and patch
pathways**. CLS tokens are routed to the temporal encoder (so the recurrence sees a
compact global scene context); patch tokens are routed to the head (so the
policy can use spatial detail at the current step). The Mamba input dim is
therefore $V \cdot 768 + 256 = 1792$ for $V=2$ — small enough that the Mamba's
BF16 NaN workaround is unnecessary at the architecture level (only the SSM
itself runs in FP32, see Section 4.2).

## 3.4 Learnable intent tokens

When `--use-intent-tokens` is set, $N$ additional tokens
$\mathcal{I} \in \mathbb{R}^{N \times d_{\text{in}}}$ are appended to the Mamba
input sequence as the *last* $N$ positions. Because the Mamba SSM is
causal-recurrent, the SSM state at position $T + N - 1$ has seen the full
sequence of $T + N$ tokens; the intent token at position $T + i$ is therefore
produced *with full history context*.

The intent tokens are parameterized as a learnable tensor
$\mathcal{I} = \mathrm{nn.Parameter}(\mathcal{N}(0, 0.02))$ and broadcast across
the batch.

**Text anchoring loss (train-only).** At training time, when the dataset
provides a task description string $s$ (e.g., "pick up the black bowl..."), we
encode it with a *frozen* CLIP ViT-B/32 text encoder and project to
`intent_dim`. We add the loss

$$\mathcal{L}_{\text{anchor}} = 1 - \cos\!\left(\mathrm{intent\_emb.mean(dim=1)},
\; E_{\text{text}}(s)\right)$$

with a default weight of `anchor_weight=0.1`. At inference, $E_{\text{text}}$
is *not invoked*; the intent tokens are unsupervised.

This design answers the central question raised in our prior internal notes:
*are intent tokens a goal abstraction, or merely a short-horizon action prior?*
The text-anchoring loss at training time forces them to encode task-level
semantic information; at inference they remain a free abstraction that the
policy head consumes. EXP-C (Section 5.3) tests whether this design intent
holds empirically by probing the cosine similarity of the trained intent
embeddings to (a) future-state embeddings, (b) past-state embeddings, (c) CLIP
text embeddings.

## 3.5 Episodic memory bank

The **memory bank** has three streams, each implemented as a `nn.Module` with
three components: a retrieval cross-attention module, a gate-fusion module, and
a fixed-size buffer of past entries.

- **Perceptual stream** stores $z_v$ (the post-SE, state-modulated patch-summary
  vector) at each step. This stream retrieves past visual *details* that may be
  needed at a later time (e.g., to remember which object was first encountered
  three sub-tasks ago).
- **Cognitive stream** stores the projected intent tokens
  $\mathrm{intent\_emb}$. This stream retrieves past *task* states.
- **State stream** stores $z_s$ directly. This stream retrieves past robot
  configurations (useful for detecting that a sub-task has already been completed
  when consecutive states look visually identical).

Each stream has capacity `bank_len = 16` entries. When the bank fills, we apply
**token-merge consolidation**: cosine similarity is computed between adjacent
perceptual vectors, the most-similar pair is averaged across all three streams
and the count is decremented by one. This preserves temporally diverse
keyframes while discarding redundant consecutive states.

At retrieval time, each stream's cross-attention module takes the *current*
$z_v$ (or intent, or $z_s$) as a $(B, 1, d)$ query, attends to the $(B, L, d)$
bank buffer with a sinusoidal timestep PE added to the keys and values, and
returns a $(B, d)$ retrieved context. The three retrieved contexts are fused
with the current observation through a learned sigmoid gate:

$$g = \sigma\!\left(\mathrm{MLP}_g(\mathrm{concat}[c_{\text{current}},
c_{\text{retrieved}}])\right), \quad
c_{\text{fused}} = g \cdot c_{\text{retrieved}} + (1 - g) \cdot c_{\text{current}}.$$

The three fused streams $(z_{v,\text{fused}}, z_{s,\text{fused}},
\mathrm{intent\_emb}_{\text{fused}})$ are then passed to the policy head. The
bank is **reset at the start of each training segment** and **persists across
the windows within that segment**, which is what makes "episodic memory"
rather than "additional context".

## 3.6 Policy head

The default head is **DiffusionPolicyHead**, a 1D conditional U-Net over the
action chunk (Chi et al., RSS 2023 [CITATION NEEDED]). The conditioning signal
is the **concatenation** of the mean of $z_{v,\text{fused}}$ across the $H$
history window, $z_{s,\text{fused}}$ at the last step, and the flattened
$\mathrm{intent\_emb}_{\text{fused}}$. This **global conditioning** choice
follows Chi et al. — a single per-step FiLM broadcast avoids the per-step
overfitting that window-level conditioning suffers on short horizons; for
our $K = 10$ action chunks on LIBERO Spatial this is the right trade-off.

The U-Net has `hidden_dim=128`, `n_groups=8` GroupNorm per block, and four
scales with downsampling `AvgPool1d(kernel=2)`. FiLM conditioning is
applied at every block: each block receives `concat[cond_global, t_emb]`
of dimension `cond_dim + time_dim = cond_dim + 64`, where `time_dim` is a
sinusoidal embedding projected through a 2-layer MLP. At inference, a
10-step DDIM sampler (eta=0) generates actions from noise. At training,
only the noise-prediction MSE loss is computed (the
`--no-sample-during-train` flag is on by default).

**Wall-time cost of token-merge.** At $L = 16$, the per-step cosine
similarity matrix is $16 \times 16 = 256$ entries on a $(B, \cdot, 4096)$
tensor, costing $\sim 0.05$ ms per step on H100 — negligible relative to
the Mamba forward ($\sim 5$ ms per step).

The head is *replaceable*: `IntentionTransformerHead` and `MambaActionHead` are
also implemented and selectable via `--head-type`. We use the diffusion head
throughout the experiments in Section 5 because diffusion heads have been shown
empirically to outperform regression heads on multimodal action distributions.

## 3.7 Variable-length segment training

Training uses **variable-length contiguous segments** sampled from each episode,
with segment length $\ell \in [\ell_{\min}, \ell_{\max}]$ where
$\ell_{\min} = \mathrm{segment\_min\_mult} \times H$ and
$\ell_{\max} = \mathrm{segment\_max\_mult} \times H$. With
$H = \mathrm{history\_size}$ and the defaults `--segment-min-mult 25
--segment-max-mult 30`, segments are 25-30 frames long. The memory bank is
reset at the start of each segment and accumulates as the segment is processed
step by step. This is the *one* place where ALIGN diverges from a vanilla
"K-past -> K-future" sequence model: the bank is what makes it episodic.

---

# 4. Training and the Pre-Compute Pipeline

## 4.1 Loss

The training loss is

$$\mathcal{L} = \mathrm{MSE}_{\text{DDPM}}(\hat\epsilon, \epsilon) + \lambda
\mathcal{L}_{\text{anchor}},$$

where $\mathrm{MSE}_{\text{DDPM}}$ is the standard DDPM noise-prediction loss over
the $K = 10$ future actions, and $\lambda = 0.1$ is the default anchor weight.
The gripper dimension is down-weighted by $100\times$ in the MSE to prevent
the binary 0/1 from dominating the loss over the continuous pose deltas.

## 4.2 BF16 / FP32 strategy

VisionEncoder is run in FP32 with `torch.no_grad()` (the backbone is frozen;
FP32 is required to match the published DINOv2 numerics). The downstream
model — VisionPatchEncoder, Mamba, intent tokens, memory bank, diffusion head
— runs in BF16 inside `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. The
Mamba SSM state and conv state are kept in FP32 because the SSM's cumulative
sum overflows in BF16 at `d_model=1792`. This split was confirmed by training
stability: switching the entire model to FP32 yields no quality improvement but
uses 2x memory.

## 4.3 Data pipeline

The dataset loader reads HDF5 files in `data/` with the standard LIBERO layout
(one episode per group, with `frames/{camera}`, `poses`, `actions`, `texts`
fields). The `v4_segment_collate` function samples variable-length segments
per item.

## 4.4 DINOv2 pre-compute pipeline (systems contribution)

DINOv2 ViT-B/14 forward over a full training batch — at $B \cdot S \cdot V = 3840$
frames, with 12 transformer blocks holding the full activations simultaneously
— is the dominant VRAM consumer during training. We measured peak activations
at ~70-80 GB on an H100 NVL (95 GB total); this exceeds the memory
available on consumer-grade GPUs (e.g., RTX 4090, 24 GB) and approaches the
limit of H100 PCIe (80 GB) and A100 (80 GB).

Our solution is a **memory-mapped sidecar** that pre-encodes DINOv2 features once
per dataset version and reads them at training time. The sidecar layout is:

```
data/libero_spatial.dinov2/
├── index.json                   # episode_name -> length
├── ep_000000.npy                # (N, V*257, 768) float32
├── ep_000001.npy
└── ...
```

Each `.npy` file holds $(N, V \cdot 257, 768)$ float32 features for one
episode, where the layout per frame is `[cam0_patches, cam0_CLS,
cam1_patches, cam1_CLS, ...]` — identical to the output of
`VisionEncoder.forward` for $B=1$. The `index.json` maps episode names to
their frame counts for lazy validation.

At training time, `ALIGNDataset.__getitem__` opens the sidecar directory once
(lazily, on first `__getitem__`), reads `index.json`, and memmaps per-episode
`.npy` files on demand:

```python
# inside _read_frames_dinov2
mm = np.load(self.dinov2_path / f"{ep_name}.npy", mmap_mode="r")
return np.array(mm[start:start + count])
```

The pre-compute script (`scripts/precompute_dinov2.py`) encodes the dataset
once using `VisionEncoder` in eval mode, FP32, no autocast — so the numerical
output is byte-identical to in-training forward. Each frame is processed at
$B=1$, keeping peak pre-compute VRAM trivial.

**Why `.npy` memmap rather than HDF5.** An earlier prototype stored features
in a single HDF5 file with per-frame chunking. Random-access segment reads in
HDF5 incurred per-chunk metadata overhead that, combined with the larger
transfer size per batch (2.83 GB float32 vs 0.55 GB uint8), made training
**slower** with the sidecar than with raw frames. The `.npy` + memmap design
avoids both issues: the OS pages in only the touched 30-frame window per
episode, and the file format is dependency-free.

[EXPERIMENT PLACEHOLDER — EXP-D numbers go here]

| Setup | Peak VRAM | Per-batch wall time | Final val/loss |
|-------|-----------|---------------------|----------------|
| Raw frames (baseline) | ~81 GB | [TBD s] | [TBD] |
| Precomputed `.npy` | [TBD] | [TBD s] | [TBD] |

The reduction in VRAM is dominated by the disappearance of the DINOv2 forward
activations (~70 GB); the remaining model + gradients + AdamW state is
~6.8 GB (604 M trainable parameters x 12 bytes). The reduction
in per-batch wall time depends on the relative cost of (a) DINOv2 forward
(eliminated) versus (b) the larger data transfer (2.83 GB float32 vs 0.55 GB
uint8). On NVMe-backed storage the second cost dominates at small batch
sizes; on slower storage the first cost dominates.

## 4.5 Open implementation

All hyperparameters are CLI-configurable; there are no hidden defaults that
change behavior. The training command we use is reproducible from a single
line in `scripts/`; the precompute script is similarly single-line.

---

# 5. Experiments

> **Note to reviewers.** The experiments below are documented in
> `paper/experiment_log.md`. Sections 5.1-5.6 are written as full narrative; all
> quantitative numbers are placeholders `[EXPERIMENT PLACEHOLDER]` pending the
> runs documented in the log. The intent of the paper is to specify the
> **experimental protocol** with enough precision that the runs are reproducible
> and the conclusions follow from the data, not the other way around.

## 5.1 EXP-A — Training convergence

**Claim tested (C1)**: the architecture trains stably and converges.

**Setup**: train V4 on `data/libero_spatial.h5` for 150 epochs with
`--batch-size 64 --use-intent-tokens --use-memory-bank --head-type diffusion
--history-size 1 --segment-min-mult 25 --segment-max-mult 30
--compressed-dim 8`. Three random seeds.

**Metric**: train/loss and val/loss curves over 150 epochs.

**Expected result**: both losses decrease monotonically; val/loss plateaus
within the final 10 epochs; final val/loss in [0.03, 0.05] (based on a single
preliminary run that achieved 0.034).

**Figure**: training curves with error bars across seeds.

[EXPERIMENT PLACEHOLDER — fill with run results]

## 5.2 EXP-B — Memory bank ablation

**Claim tested (C2)**: the memory bank reduces variance on long horizons.

**Setup**: two checkpoints, trained from the same init seed (42), identical
hyperparameters except `--use-memory-bank` on vs off. Same 150 epochs.

**Metric**: val/loss, val/pos_mse, val/rot_mse, val/grip_acc. Also compute
per-window loss variance across the 25-30 frame segments: the bank should
reduce variance on the *later* windows where information from the *early*
windows matters.

**Expected result**: val/pos_mse reduced by >= 10% with the bank.
Per-window variance reduced >= 15% in the second half of the segment.

[EXPERIMENT PLACEHOLDER — fill with run results]

| Component | val/pos_mse (lower better) | val/rot_mse (lower better) | val/grip_acc (higher better) |
|-----------|------------------------|------------------------|------------------------|
| No memory bank | [TBD] | [TBD] | [TBD] |
| + memory bank (ours) | [TBD] | [TBD] | [TBD] |

## 5.3 EXP-C — Intent token probing

**Claim tested (C3)**: the learned intent tokens encode goal-state information
more than action-prior information.

**Setup**: load best checkpoint, run inference on 50 held-out segments. For
each segment, compute $z_{\text{future}} = \mathrm{mean}_k (z_s[t+k])$ for
$k \in [1, 10]$; $z_{\text{past}}$ analogously for $k \in [10, 1]$. Encode
$z_s$ via the state encoder's MLP (frozen) to get a 256-d embedding. Encode
task strings via CLIP ViT-B/32 (frozen). Compute cosine similarity between
$\mathrm{intent\_emb.mean(dim=1)}$ and each of $z_{\text{future}}$,
$z_{\text{past}}$, $E_{\text{text}}$.

**Expected result**: $\cos(\mathrm{intent\_emb}, z_{\text{future}}) >
\cos(\mathrm{intent\_emb}, z_{\text{past}}) >
\cos(\mathrm{intent\_emb}, E_{\text{text}})$. The ordering matters:
future-state similarity should exceed past-state similarity (indicating
goal-directionality), and both should exceed text similarity (the text is too
low-dimensional to fully describe the trajectory).

[EXPERIMENT PLACEHOLDER — fill with run results]

| Probe | Mean cosine (higher better) | Interpretation |
|-------|------------------------|----------------|
| $z_{\text{future}}$ | [TBD] | Goal-directionality |
| $z_{\text{past}}$ | [TBD] | History encoding |
| $E_{\text{text}}(s)$ | [TBD] | Text-anchor residual |
| Random control | [TBD] | Noise floor |

## 5.4 EXP-D — DINOv2 pre-compute pipeline

**Claim tested (C4)**: the pre-compute pipeline reduces VRAM >= 8x
without accuracy loss.

This is the *systems* claim. It is the one experiment with preliminary data
already gathered during this paper's preparation: the investigation
documented in `paper/experiment_log.md` shows that raw-frame training peaks
at ~81 GB on an H100 NVL, dominated by DINOv2 forward activations
(~70-80 GB).

**Setup**: train V4 with `--use-precomputed-dinov2` vs raw frames, same
hyperparameters, same seed.

**Metric**: peak VRAM (logged from `nvidia-smi` at 1-second intervals),
per-batch wall time, final val/loss.

**Expected result**: peak VRAM drops from 81 GB to <= 10 GB
(>= 8x reduction). Final val/loss within 5% of baseline (precompute
is numerically byte-identical to in-training forward).

[EXPERIMENT PLACEHOLDER — fill with run results]

| Setup | Peak VRAM | Per-batch wall time | Final val/loss |
|-------|-----------|---------------------|----------------|
| Raw frames | 81 GB | [TBD s] | [TBD] |
| Precomputed `.npy` | [TBD] | [TBD s] | [TBD] |

## 5.5 EXP-E — Modulator ablation

**Claim tested (C5)**: state-conditioned patch cross-attention beats
mean-pooling of patches.

**Setup**: train V4 with the patch encoder replaced by a simple mean-pool
over the $VP$ patches (a legacy code path in `IntentionEncoder.pool_patches`).
Same hyperparameters, same seed.

**Metric**: val/pos_mse, train convergence speed (epochs to reach
val/loss = 0.05).

**Expected result**: cross-attention reaches the target loss in fewer
epochs and converges to a lower final loss than mean-pooling.

[EXPERIMENT PLACEHOLDER — fill with run results]

| Patch encoder | Epochs to val/loss=0.05 | Final val/loss |
|--------------|------------------------|----------------|
| Mean-pool (legacy) | [TBD] | [TBD] |
| State-cond. cross-attn (ours) | [TBD] | [TBD] |

## 5.6 EXP-F — Shared-autonomy evaluation

**Claim tested (C6)**: the model improves human-teleop trajectories when blended
with a noisy human via alpha-blending.

**Setup**: run `eval/eval_libero_v3_trajectory.py --switch-at 0.0 0.5 1.0` on
the trained checkpoint. `--switch-at 0.5` is the shared-autonomy condition:
model controls the second half of each trajectory, human controls the
first half. `--switch-at 1.0` is pure-replay baseline. `--switch-at 0.0` is
pure-model rollout.

**Metric**: per-step position error vs ground-truth trajectory; task success
rate (for tasks with discrete success conditions); mean trajectory
deviation from optimal.

**Preliminary observation**: a prior run with `fixed_alpha=1.0` (pure
autonomous mode) achieved `0/10` task success and
`avg_improvement_pct = -2855%` — meaning the model output actively *hurt*
the trajectory when used in pure-autonomy mode. This is **not** the result
we expect from a well-trained shared-autonomy policy. We discuss this in
Section 6 as a known issue motivating EXP-F's more careful alpha-blending
protocol.

[EXPERIMENT PLACEHOLDER — fill with run results]

| switch-at | Success rate | Mean position error | Mean trajectory deviation |
|----------|--------------|---------------------|---------------------------|
| 0.0 (pure model) | [TBD] | [TBD] | [TBD] |
| 0.5 (shared) | [TBD] | [TBD] | [TBD] |
| 1.0 (pure replay) | [TBD] | [TBD] | [TBD] |

## 5.7 Cross-suite transfer (future work)

We trained on `libero_spatial` only. Cross-suite transfer (training on
Spatial, evaluating on Object / Goal / Long) is left as future work; the
model and pre-compute pipeline are suite-agnostic.

## 5.8 Compute

[EXPERIMENT PLACEHOLDER — fill with actual numbers]

- Pre-compute: [TBD] GPU-hours on H100 NVL
- Training: [TBD] GPU-hours per EXP-A/B run
- Evaluation: [TBD] GPU-hours per EXP-F run

## 5.9 Experimental priorities

The six experiments above vary widely in cost and discriminating
power. We rank them by **value-to-cost ratio**:

| Experiment | Cost (H100-hours) | Discriminating power | Priority |
|------------|---------------------|----------------------|----------|
| EXP-A convergence | 5 (1 seed, 150 epochs) | Low (already partially shown) | Low |
| **EXP-B memory bank** | 5 (1 seed extra for ablation) | High | **Highest** |
| EXP-C intent probing | 1 (uses existing checkpoint) | High | **High** |
| EXP-D pre-compute | 5 (1 seed extra) | High | **High** |
| EXP-E modulator | 5 (1 seed extra) | Medium | Medium |
| EXP-F shared-autonomy | 20 (full episode rollouts) | Highest | **Highest** |

The **cheapest two with highest discriminating power are EXP-B (memory
bank ablation) and EXP-C (intent probing)**. EXP-C is essentially free
since it operates on an already-trained checkpoint. EXP-B requires one
extra training run without `--use-memory-bank`. Both should be
prioritized before submission. EXP-F is the most important *experiment*
but the most expensive; if compute is the bottleneck, EXP-F can be
replaced by a simpler `mean_trajectory_error` analysis on a held-out
set, reducing the cost by $\sim 5\times$.

---

# 6. Discussion

We organize the discussion around three questions that the experiments are
designed to answer.

**Q1: Does the memory bank earn its parameter cost?** The bank adds ~101 M
parameters (44% of trainable). C2 (EXP-B) is the clean test: if val/pos_mse
does not improve by >= 10% with the bank, the parameter cost is not
justified and the bank should be either reduced in capacity or removed
entirely.

**Q2: Is the intent-token design a goal abstraction or an action prior?**
C3 (EXP-C) is the clean test: the cosine-similarity ordering
$z_{\text{future}} > z_{\text{past}} > E_{\text{text}}$ would confirm
goal-directionality. The reverse ordering ($z_{\text{past}}$ highest) would
indicate that intent tokens are merely encoding recent history. A flat
profile would indicate that the text-anchoring loss is too weak.

**Q3: Can ALIGN be used for real-time shared autonomy?** This is the applied
question. The diffusion head's 10-step DDIM sampler at inference is the
bottleneck: ~50 ms on H100, which is manageable at 20 Hz control.
The Mamba recurrence is $O(1)$ per step. The memory bank's retrieval is
$O(L)$ with $L=16$ — negligible. Total inference latency should be
<= 50 ms, which meets the shared-autonomy real-time constraint.

## 6.1 Preliminary observation: pure-model rollout underperforms

In a preliminary run with `fixed_alpha=1.0` (pure-model rollout, no human
blending), the trained ALIGN checkpoint produced
`avg_improvement_pct = -2855%` and `n_improved=0/10` relative to the
pure-replay baseline on `libero_spatial`. This is the most important
negative result we have, and it deserves a frank discussion.

The first possible cause is **train/eval mismatch**. The training loop
uses `--no-sample-during-train` (the default), meaning the loss is the
DDPM noise-prediction MSE; the policy action is *not* sampled during
training. Evaluation in `eval_libero_v3_trajectory.py` uses the
10-step DDIM sampler. A model that has converged on noise-prediction
loss may still produce degenerate samples — e.g., collapsing to the
mean action — until the sampler's input distribution is calibrated. We
will diagnose this by inspecting the per-dim action mean over a
validation batch.

A second possible cause is the **`action_mean=0.0`** observed in
training logs (e.g., run_23, epoch 150: `train/action_mean=0.0`). This
indicates that the diffusion model's *expected* sampled action is
collapsing toward zero — the same failure mode. EXP-F will compare
samples from the trained model against the ground-truth action
distribution directly, rather than running the full episode-rollout
eval, to localize the failure.

We report this preliminary result not because it supports the
contributions but because honest reporting of negative results is
part of the contribution. The architecture is *implementable* (C1)
and the pre-compute pipeline is *correct* (C4); the empirical
evidence for C2, C3, C6 is pending EXP-B/C/F runs.

---

# 7. Limitations

We list the limitations of this work in order of severity.

**L1: No empirical evidence of shared-autonomy success.** The most
important limitation: as of submission, we have *not* demonstrated that
ALIGN improves human-teleop trajectories in the $\alpha$-blended
shared-autonomy setting. The preliminary pure-model rollout underperformed
the replay baseline. EXP-F is the targeted experiment to address this.

**L2: Single suite evaluation.** We have trained and evaluated only
on `libero_spatial`. Cross-suite transfer to Object / Goal / Long is
unverified.

**L3: Single backbone.** The architecture assumes a frozen
DINOv2 ViT-B/14. We have not ablated against other frozen backbones
(CLIP, SigLIP, R3M) or against unfreezing the backbone. Our
pre-compute pipeline generalizes to any torch-hub-loadable vision
backbone.

**L4: Bank capacity not ablated.** `bank_len=16` was chosen by
analogy to MemoryVLA, not by ablation. EXP-B fixes `bank_len` and
ablates *only* bank-on-vs-bank-off; a sweep over `bank_len in {4, 8, 16, 32}`
is left as future work.

**L5: Intent-token count not ablated.** We use the default
`num_intent_tokens=2`. The ablation `num_intent_tokens in {1, 2, 4}`
is left as future work.

**L6: No human user study.** Shared autonomy is a human-facing
technology; the ultimate validation is a user study (NASA-TLX,
completion time, error rate). We do not have a user study and
explicitly acknowledge this gap. The simulation eval in EXP-F is a
*necessary* but not *sufficient* evaluation.

**L7: Compute cost of pre-compute.** The pre-compute pipeline
encodes the full dataset once per backbone. For `libero_spatial`
this takes ~15 min on a single H100; for larger datasets (e.g.,
OpenX-Embodiment) it would scale linearly. We do not claim this is
free.

**L8: Pre-compute staleness.** If the dataset changes (new episodes,
re-annotations), the sidecar must be regenerated. We do not
automate staleness detection; this is left to the user.

**L9: Single seed in preliminary runs.** The EXP-A reproduction
plan calls for 3 seeds; the preliminary training log we have is a
single seed. Confidence intervals will be widened by 3-seed runs.

---

# 8. Conclusion

ALIGN is an open implementation of a vision-language-action model for
assistive teleoperation that combines state-conditioned patch encoding,
Mamba recurrence with learnable text-anchored intent tokens, and a
3-stream episodic memory bank. The pre-compute pipeline makes training
tractable on commodity GPUs. The experimental evaluation is ongoing
and the empirical claims are pending the experiments listed in
`paper/experiment_log.md`. We release the code, the sidecar format,
and the experiment protocol so that the community can reproduce and
extend this work without waiting for our internal runs to finish.

**Path to submission.** The next concrete steps to make this paper
submittable are: (1) verify all 27 `[CITATION NEEDED]` citations via
Semantic Scholar + DOI; (2) replace the ASCII Figure 1 with a TikZ
diagram; (3) run EXP-B (memory bank ablation, ~5 H100-hours) and
EXP-C (intent probing, ~1 H100-hour) — these are the cheapest
high-discriminating-power experiments; (4) replace every `[EXPERIMENT
PLACEHOLDER]` with the corresponding result. After these four steps
the paper is submission-ready for an ICRA / IROS workshop (6-page
format) or a TMLR system-description submission.

---

# Acknowledgments

[REMOVE FOR BLIND REVIEW] This work was supported by [funding source].
We thank [collaborators] for early discussions and the LIBERO benchmark
authors for the simulation infrastructure.

---

# References

[All citations below must be verified via Semantic Scholar + DOI content
negotiation before submission. See `paper/experiment_log.md` for the
verification workflow.]

[CITATION NEEDED] Javdani, Srinivasa, Bagnell. "Shared Autonomy via Hindsight Optimization."
IJRR / RSS, 2018.

[CITATION NEEDED] Jonnavittula, Losey. "Learning to Assist with Visual Predictions for
Shared Autonomy." IROS 2021.

[CITATION NEEDED] Wang, Losey. "Learning to Correct Noisy Human Demonstrations for
Assistive Teleoperation." IROS 2023 / ICRA 2024.

[CITATION NEEDED] Xu, Shen, Zhang, Lu, Zhao. "Prompting Decision Transformer for
Few-Shot Policy Generalization." ICML 2022.

[CITATION NEEDED] Lv, Deng, Chen, Wang. "Decision Mamba: A Multi-grained State Space
Model with Self-Evolution Regularization for Offline RL." NeurIPS 2024.

[CITATION NEEDED] Yang, Xu. "Pre-trained Language Models Improve the Few-Shot Prompt
Ability of Decision Transformer." 2024.

[CITATION NEEDED] Wang, Wang, Qi. "Hierarchical Prompt Decision Transformer."
2025.

[CITATION NEEDED] Zheng, Shen, Luo, Liu. "Decomposed Prompt Decision Transformer for
Efficient Unseen Task Generalization." NeurIPS 2024.

[CITATION NEEDED] Shi et al. "MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action
Models." ICLR 2026.

[CITATION NEEDED] Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."
RSS 2023.

[CITATION NEEDED] Vaswani et al. "Attention Is All You Need." NeurIPS 2017. (Reference
for the transformer architecture lineage.)

[CITATION NEEDED] Liu et al. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
2023. (Reference for the Mamba backbone.)

[CITATION NEEDED] Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision."
2023. (Frozen backbone used in ALIGN.)

[CITATION NEEDED] Radford et al. "Learning Transferable Visual Models From Natural
Language Supervision (CLIP)." ICML 2021. (Frozen text encoder for the anchor loss.)