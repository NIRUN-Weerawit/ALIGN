# Training stage profile

Profile recorded on 2026-10-11 from the no-intent-token, memory-enabled ablation, warm-started from the completed no-text run. It measured the first 32 training batches with batch size 2, then ran full validation over 44 held-out episodes (22 batches). Another training job was using the GPU concurrently, so absolute GPU times are noisy. This is a partial epoch sample; episode lengths and memory-update counts vary across batches.

## Measured time

| Stage | Seconds | Notes |
|---|---:|---|
| Training wall time | 62.35 | 32 batches |
| DataLoader wait | 30.48 | Host wall time around `next(loader)`; includes feature memmap reads/copies, HDF5 state/action reads, and collation |
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

The next useful optimization targets are reducing input-pipeline wait and profiling the sequential memory observe/retrieval path. The profile uses precomputed per-episode `.npy` memmaps for visual features; HDF5 supplies state, gripper, action, and text data. The wait measurement cannot separate memmap page faults, array copies, HDF5 reads, and collation. This sample does not justify changing the diffusion head. The first 32 batches are not a complete epoch, and another GPU workload was active, so use the per-epoch `timing` fields in the training `metrics.jsonl` for stable comparisons across complete runs.

The ablation runner now records wall time for training, DataLoader waits, validation, checkpoint saves, and total epoch time. `train_v4_epoch` records CUDA-event timings for feature encoding, patch encoding, intention scan, memory reset/update/conditioning, head conditioning, loss assembly, diffusion loss, backward/optimizer, and total segment work. The CUDA events synchronize only once at the end of the training epoch.

## Input-pipeline follow-up

The ablation runner now defaults to four persistent spawned workers with one prefetched batch per worker. HDF5 handles and DINOv2 mmap caches are created per process. The ablation collator copies samples directly into the final padded batch allocation, and the ablation dataset reads DINOv2 features as mmap views to avoid an extra per-episode copy. Training episodes are length-bucketed, with randomized ordering and randomized `drop_last` selection when the episode count is not divisible by batch size.

On 2026-10-11, a 20-batch loader-only comparison used two disjoint, length-matched sets of LIBERO Goal episodes (about 5.8 GiB of feature tensors per setting). Zero workers took 11.84 seconds total, with a 0.57-second mean wait after the first batch. Four workers took 12.58 seconds including 2.78 seconds of worker startup, then 0.52 seconds per batch on average. The steady-state wait was about 10% lower with workers; persistent workers amortize the startup across epochs. This was one CPU/storage sample rather than a complete training throughput comparison.

A collator microbenchmark on an 80-frame, two-camera feature batch produced identical tensors and reduced collation time from 0.058 to 0.028 seconds (2.05x), avoiding about 0.25 GiB of duplicate temporary padded features. These loader changes preserve all per-episode observations and targets. Length bucketing changes which episodes share a batch while retaining epoch coverage, so the first full training run should still be checked for validation quality as well as epoch time.

## Memory and validation follow-up

The measured memory time is a good optimization target. Memory retrieval previously branched on GPU reductions to detect empty banks and subset the batch, forcing device-to-host synchronization during retrieval. It now makes empty rows safe with a tensor mask and returns the query unchanged for those rows. Episodic timestamp validation now uses an asynchronous CUDA assertion rather than synchronizing every observation. Validation also keeps anchor-validity filtering on CPU, aggregates prediction metrics before transferring them to the host, and samples diffusion/flow actions for all valid anchors in a segment in one batched sampler call. Memory updates still run sequentially so each anchor sees only earlier observations, and the sampler retains its configured denoising-step count.

These changes remove avoidable synchronization and improve sampling batch utilization, but a GPU timing comparison has not yet been recorded: the RTX 5060 Ti was at 86% utilization with another process using 7.3 GiB during implementation. The direct CPU checks confirm empty-bank retrieval remains finite and exact for empty rows, validation metrics still exclude padded samples, and generative validation makes one sampler call per segment. Re-profile on an otherwise idle GPU before assigning a numeric speedup.

## Acceleration commit verification

Reviewed `3dcbd65` (grouped training losses), `46bad06` (worker-safe cached data loading), and `54d84db` (memory retrieval and batched validation). The review found and fixed two regressions in `54d84db`:

- The legacy memory path returned its mutable bank directly when it was empty. Computing attention for empty rows caused autograd to save that bank before the subsequent in-place write, breaking backward with a tensor-version error. Empty-bank positional encoding now returns a snapshot, matching its documented copy contract.
- Batched validation sampling had moved outside CUDA autocast. Sampling now retains the original BF16 autocast context; diffusion still explicitly uses its FP32 denoiser, while flow matching retains its previous precision behavior and supports BF16 visual-attention conditioning.

After the fixes, 91 focused tests passed, including spawned HDF5 workers, persistent-worker epoch updates, collation and padding, legacy and episodic memory backward, timestamp contracts, gripper metrics, distributed training/validation, four CUDA diffusion/flow validation cases with/without memory, and deterministic grouped-loss update/memory-order equivalence. Pytest and plotting dependencies were installed only under `/tmp/align-acceleration-testdeps`; the training virtualenv was unchanged. A local tests package marker also prevents collisions with a third-party package named `tests`.

This verifies correctness under the tested configurations, not an end-to-end throughput gain. The GPU remained saturated by another process. Batched validation can use more peak VRAM than per-window sampling, and grouping stochastic losses/sampling changes random-number assignment, so historical loss/metric values are not guaranteed to match bitwise. Length bucketing also changes which episodes share optimizer steps. An idle-GPU comparison should record epoch time, validation time, peak VRAM, and validation quality. CUDA event intervals in the original stage profile measure elapsed stream time, which may include host submission gaps and GPU contention; they do not establish exclusive GPU compute cost.
