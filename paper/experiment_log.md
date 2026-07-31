# ALIGN Paper — Experiment Log

This log defines the experiments that the paper will reference. Placeholders
in the paper (`[EXPERIMENT PLACEHOLDER]`) point here. Run these, fill in
the numbers, replace placeholders with the result.

## Contribution (one sentence)

ALIGN is a vision-language-action model for assistive teleoperation that
augments a Mamba temporal encoder with (i) learnable intent tokens aligned
to a frozen CLIP text embedding at train time but discarded at inference,
and (ii) a perceptual-cognitive-state memory bank with cross-attention
retrieval and token-merge consolidation.

## Claims → Experiments

| Claim | Experiment ID | Status |
|-------|--------------|--------|
| C1: Architecture trains stably and converges | EXP-A (train loss curves) | TO RUN |
| C2: Memory bank reduces variance on long horizons | EXP-B (memory on/off ablation) | TO RUN |
| C3: Intent tokens encode goal state, not action prior | EXP-C (intent probing) | TO RUN |
| C4: Pre-compute pipeline reduces VRAM ≥10× without accuracy loss | EXP-D (raw vs precompute) | DONE (this conversation) |
| C5: State-conditioned cross-attention beats mean-pool on vision | EXP-E (modulator ablation) | TO RUN |
| C6: α-blending helps human operator in shared autonomy | EXP-F (shared-autonomy eval) | TO RUN |

## EXP-A — Training convergence

- **Setup**: train V4 on libero_spatial, 150 epochs, B=64, history=1, segments=[25,30]
- **Metric**: train/loss, val/loss curves
- **Expected**: both decrease monotonically; val plateaus
- **Source files**: `checkpoints/v4/libero_spatial/run_*/intention_log.jsonl`
- **Reference data**: run_23, epoch 150: train_loss=0.032, val_loss=0.034

## EXP-B — Memory bank ablation

- **Setup**: train V4 with `--use-memory-bank` vs without (B=64 same as C1)
- **Metric**: val/loss, val/pos_mse, val/rot_mse, val/grip_acc
- **Expected**: memory bank reduces val/pos_mse by ≥10% on LIBERO Spatial
- **Action**: run `python training/train_intention.py --data data/libero_spatial.h5 ... --use-memory-bank` AND without `--use-memory-bank`; ensure same seed (42)
- **Config to add**: train two variants, write results to `results/exp_b_memory.json`

## EXP-C — Intent token probing

- **Setup**: load best checkpoint, run inference on held-out segments
- **Metric**: cosine similarity between `intent_emb` and (a) future-state embedding (t+10), (b) past-state embedding (t-10), (c) CLIP text embedding of task description
- **Expected**: similarity to future state > similarity to past state > similarity to CLIP text (this would confirm goal-directionality)
- **Null hypothesis**: similarity uniform → intent encodes action prior, not goal
- **Script**: write `tools/probe_intent.py` to extract embeddings, write `paper/results/exp_c_intent.json`

## EXP-D.5 — Intent embedding task clustering

- **Setup**: load best checkpoint, run `forward_with_probe` on 50-100 held-out episodes per task (500-1000 episodes total), collect intent_emb per episode
- **Metric**: silhouette score, 5-fold CV logistic regression accuracy, 5-fold CV KNN accuracy, K-means cluster purity (K=10), within-task and cross-task cosine similarity
- **Expected (H1)**: silhouette > 0.1, logistic > 50% (5× chance), clear t-SNE clusters — intent tokens discover task structure without explicit supervision
- **Expected (H2)**: clustering by sub-task phase rather than task identity — finer-grained abstraction
- **Null (H3)**: silhouette ~ 0, classifier ~ 10% — opaque, requires CLIP anchor to force interpretable structure
- **Script**: `tools/probe_intent_clustering.py` (IMPLEMENTED), reads from `--checkpoint`, outputs `results.json`, `intent_tsne.pdf`, `confusion_matrix.pdf`, `intent_embeddings.npz`
- **Model change**: `forward_with_probe` added to `ALIGNIntentionModel` (no training impact, runs under `torch.no_grad()`)
- **Priority**: HIGH — produces a strong figure for the paper, cheap to run

## EXP-D — DINOv2 pre-compute pipeline

- **Setup**: train V4 with `--use-precomputed-dinov2` vs raw frames, same B/S
- **Metric**: peak VRAM (nvidia-smi), per-batch wall time, final val/loss
- **Expected**: VRAM drops from ~81 GB to ~6-8 GB; wall-time similar or slightly faster; final val/loss within 5% of baseline
- **Status**: PARTIAL — investigation in this conversation showed:
  - Peak VRAM at training: 81 GB with raw frames (H100 NVL, 95 GB total)
  - Memory profile: 70-80 GB DINOv2 forward activations (B·S·V = 3840 images, 12 transformer blocks)
  - Model + grad + optim: 6.8 GB (604M trainable × 12 bytes)
  - Memory bank + others: ~1 GB
  - Memory model: per-batch GPU memory = DINOv2 activations + bank + head + activations
- **Action**: run the comparison end-to-end on helios, fill numbers in Table 1

## EXP-E — Modulator ablation

- **Setup**: train V4 with mean-pool over patches (legacy `pool_patches`) vs state-conditioned cross-attention
- **Metric**: val/pos_mse, val/rot_mse, train convergence speed
- **Expected**: cross-attention converges faster (lower epochs to target loss) and reaches lower final loss
- **Action**: add CLI flag `--use-pool-patches` (legacy mode) vs default cross-attention

## EXP-F — Shared-autonomy eval (the key claim)

- **Setup**: run `eval/eval_libero_v3_trajectory.py --switch-at 0.0 0.5 1.0` on trained model
- **Metric**: task success rate, mean trajectory deviation from optimal
- **Expected**: switch-at=0.5 (half human, half model) > switch-at=0.0 (pure model) OR switch-at=1.0 (pure human), showing the model adds value to a noisy human
- **Baseline**: pure-replay (switch-at=1.0) — must succeed in some episodes
- **Status**: TO RUN — this is the make-or-break experiment for the paper's claim
- **Note**: prior run with `fixed_alpha=1.0` showed `n_improved=0/10`. Need to (a) use α-blending, (b) possibly use a stronger base model, before this experiment succeeds.

## Figures to generate

| Figure | Description | Source |
|--------|-------------|--------|
| Fig 1 | Architecture diagram (TikZ) | Hand-drawn → TikZ |
| Fig 2 | Training curves (loss vs epoch) | EXP-A results |
| Fig 3 | Memory bank ablation bar chart | EXP-B results |
| Fig 4 | Intent token UMAP/probe | EXP-C results |
| Fig 5 | VRAM comparison: raw vs precomputed | EXP-D results |
| Fig 6 | LIBERO success rate by switch-at | EXP-F results |

## Failed / negative results (to document)

- Run 1 evaluation (`eval/libero_traj_results/summary.json`): 0/10 success
  with `fixed_alpha=1.0`, `avg_improvement_pct=-2855%`. This means the current
  model output actively hurts trajectory tracking when used in pure-autonomy
  mode. The paper MUST address this in Limitations and not present it as
  positive evidence.

## Open questions

- Is `avg_action_mean=0.0` in run_23 logs a training bug (model collapsed to
  predict zero)? This would explain the eval failure. Need to inspect
  saved checkpoint and a single forward pass to verify.
- The `--no-sample-during-train` flag is set ON by default — does this
  cause train/eval mismatch because eval uses sampling?