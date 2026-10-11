# Training stage profile

Profile recorded on 2026-10-11 from the no-intent-token, memory-enabled ablation, warm-started from the completed no-text run. It measured the first 32 training batches with batch size 2, then ran full validation over 44 held-out episodes (22 batches). Another training job was using the GPU concurrently, so absolute GPU times are noisy. This is a partial epoch sample; episode lengths and memory-update counts vary across batches.

## Measured time

| Stage | Seconds | Notes |
|---|---:|---|
| Training wall time | 62.35 | 32 batches |
| DataLoader wait | 30.48 | Host wall time around `next(loader)`; includes data read/collation stalls |
| Memory observe-only updates | 10.92 | GPU event time for observations without a supervised action window |
| Memory conditioning and retrieval | 8.32 | GPU event time at supervised anchors |
| Backward and optimizer | 5.38 | GPU event time |
| Diffusion loss | 2.24 | GPU event time |
| Feature transfer/parse and state encoder | 1.69 | GPU event time |
| Patch encoder | 0.58 | GPU event time |
| Other measured segment work | 2.67 | Residual GPU event time |
| Total measured segment GPU time | 31.84 | Sum of segment intervals |
| Full validation | 71.27 | 22 batches; action sampling included |
| Checkpoint writes | 1.12 | |
| Total profile epoch wall time | 134.74 | |

The two memory stages consumed 19.25 seconds, about 60% of measured segment GPU time. DataLoader stalls were about 0.95 seconds per training batch and around 49% of training wall time. Validation took longer than the 32-batch training sample because it evaluated the full held-out set and sampled actions. In this sample, diffusion loss itself was a smaller cost than memory updates and data input.

## Interpretation and limits

The next useful optimization targets are reducing HDF5/DataLoader stalls and profiling the sequential memory observe/retrieval path. This sample does not justify changing the diffusion head. The first 32 batches are not a complete epoch, and another GPU workload was active, so use the per-epoch `timing` fields in the training `metrics.jsonl` for stable comparisons across complete runs.

The ablation runner now records wall time for training, DataLoader waits, validation, checkpoint saves, and total epoch time. `train_v4_epoch` records CUDA-event timings for feature encoding, patch encoding, intention scan, memory reset/update/conditioning, head conditioning, loss assembly, diffusion loss, backward/optimizer, and total segment work. The CUDA events synchronize only once at the end of the training epoch.
