# Causal intention-token task probe

This probe freezes the policy encoders and fits small task classifiers. It does
not train the policy or provide task labels/text to the intention encoder.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/whinnoy/.venvs/align-ablation/bin/python tools/probe_intent_clustering.py \
  --checkpoint checkpoints/piper_v4_ablation_no_text_2/intent_memory/intention_best.pt \
  --data data/libero_goal.h5 \
  --cache data/libero_goal.dinov2.external \
  --output reports/intent_task_probe_2026_10_18/no_text_memory \
  --device cuda --sample-points 21 --random-control
```

Use an intact cache path; the default `data/libero_goal.dinov2` link can point to a
missing local copy. Dependencies beyond the training environment are matplotlib,
scikit-learn and its dependencies. Mamba must be installed to extract tokens.

The checkpoint must record disjoint `train_episodes` and `val_episodes`. Encoder
weights load strictly from `model_state_dict`; corrupt/mismatched files fail
instead of falling back to random weights. The extraction metadata records an
encoder weight hash, checkpoint epoch, data path, and gripper convention.
Old ablation checkpoints without a gripper convention reproduce `legacy_zero`;
new checkpoints use their recorded convention. Explicitly overriding this changes
the input distribution and should be reported.

Each episode starts with empty Mamba state. All causal observation prefixes are
processed using per-frame robot states; learned token readouts fork that state.
Twenty-one equally spaced episode positions are saved by default. The output is
the **raw intention encoder embedding before memory fusion**. Memory retrieval
and the action head are not executed. Multiple token slots are concatenated.
Extraction defaults to full precision to reduce sensitivity of geometry and
linear decoding to quantization. `--precision bfloat16` provides an optional
inference-precision check; the selected precision is saved in metadata.

Controls include independently resetting recurrence at each sampled frame,
current DINO CLS features plus state, raw state alone, and the causal history
mean plus current state. `--random-control` adds a randomly initialized
Mamba/readout with the same trained state encoder. It tests whether task cues
survive arbitrary temporal projections, rather than testing a fully random
policy. Features have different dimensions, so compare their scores cautiously.

The linear classifier and scaling fit training episodes only. Each progress
position has a separate classifier, evaluated on held-out whole episodes. The
additional fixed final-position decoder is evaluated across all progress points
to test temporal consistency of task directions without refitting each phase. The
script reports balanced accuracy, a final-position stratified episode-bootstrap
interval, and a shuffled-training-label null test. Chance is 1 / number of tasks.
All model comparisons should use the same saved episode split.

Clustering uses a scaler fitted on training tokens across progress positions.
K-means fits training tokens at each position and predicts held-out clusters.
Silhouette, ARI/NMI, within-task distance, between-centroid distance, and distance
to fixed late-training-task centroids use the original feature dimensions.
PCA fits training tokens only. t-SNE jointly projects train and held-out points
without labels and is a visualization, not a quantitative generalization test.

Outputs include `REPORT.md`, `results.json`, `extraction.json`, raw
`intent_embeddings.npz`, task labels, warnings, classification curves,
PCA trajectories/stages, t-SNE, geometry curves, and final confusion matrix.
To redo analysis without GPU extraction, rerun the command with
`--reuse-embeddings`. Keep the checkpoint argument identical.

Interpret task decoding, compact clusters, convergence, and useful control as
different claims. Task information may already exist in the initial scene.
Phase and uncertainty can legitimately spread a task across multiple clusters.
Successful decoding does not prove future-goal prediction or that the action
head/memory retrieval uses the token. Those require future-state probes and
controlled token replacement/ablation during policy evaluation.
