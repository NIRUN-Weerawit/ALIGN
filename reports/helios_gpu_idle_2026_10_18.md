# Helios GPU idle diagnosis — 2026-10-18

Read-only inspection of active PID 461455, running from
`/mnt/bigdata/00_students/wee_ucl/ALIGN`, output
`checkpoints/piper_v4_ablation_no_text_4_cp_32_helios`.

GPU 0 showed 0% SM activity throughout a 12-second sample, retaining 78,743 MiB.
The trainer was in Linux D state with `folio_wait_bit_common`. This process has
zero swap usage; host available RAM was about 751 GiB. `/mnt/bigdata` is an NFS4
mount (100 TiB, 97% used), not local disk. Host I/O pressure was high (about 49%
full stall over the preceding minute); that is a host-wide metric, not exclusively
attributable to this job. Read-only manifest/stat requests also showed long delays.

Recent completed `intent_memory/metrics.jsonl` records:

| Epoch | Train s | Validation s | Checkpoint s | Total s | Save fraction |
|---|---:|---:|---:|---:|---:|
| 13 | 295.0 | 36.0 | 4858.9 | 5192.4 | 93.6% |
| 14 | 342.7 | 18.5 | 6967.3 | 7328.7 | 95.1% |
| 15 | 473.5 | 26.9 | 5585.6 | 6088.9 | 91.7% |

Epoch 15 data wait was 233.6 seconds within training, a secondary storage-related
bottleneck. Stage CUDA-event intervals include stream/launch gaps and should not
be interpreted as hardware SM utilization measurements.

File sizes observed: `training_last.pt` 30.603 GiB; `intention_best.pt` 10.554 GiB.
The saved configuration has compressed width 32, memory patch retrieval disabled,
microbatch 2, effective batch 32, two loader workers, and no saved multiscale
visual-context/retrieval-depth options. It is an older flat-memory run, not a
training run of the newly implemented patch-token architecture.

Priorities:

1. Stage checkpoint writes on local SSD/NVMe, then mirror completed files to NFS
   without blocking GPU work. Bound asynchronous writes and use immutable CPU
   snapshots/atomic publication so training cannot mutate an in-flight checkpoint.
2. Exclude reconstructible frozen vision from best-model saves as already done
   for resume saves; retain complete optimizer state for resumability.
3. Stage cached features locally if data wait remains large. Increasing workers
   alone does not cure a congested shared storage server.
4. Train the patch-preserving architecture in a new output directory. Changing
   architecture is a separate experiment and cannot preserve the old optimizer
   as an exact continuation. Keep its performance claims separate from I/O fixes.

No process was interrupted and no remote training/configuration files were edited.
