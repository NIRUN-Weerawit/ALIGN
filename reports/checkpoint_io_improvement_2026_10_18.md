# Checkpoint I/O acceleration — 2026-10-18

**Latest status:** the old-model recovery transfer and auto-resume controller
were stopped at the user's request. The old 30.6 GiB checkpoint is abandoned
for this experiment. A fresh patch-memory model passed a local SSD checkpoint
and optimizer-resume test under `/data/wee_ucl/align-fast-checkpoint-20261018-v2`
on Helios. Local saves took 0.87–1.81 seconds; recovery is 619 MiB. See
[completed test](helios_fast_checkpoint_test_2026_10_18.md).
The abandoned local sparse partial was removed; original NFS saves remain.
The handoff sections below describe earlier attempts, not active processes.

Implemented `training/checkpoint_io.py` and integrated it into the ablation
runner with `--checkpoint-staging-dir`. See
[training instructions](../scripts/README_intention_ablation.md#checkpoints-on-shared-storage).

The training thread saves immutable local files; a background thread mirrors
these to NFS with atomic replacement. Pending latest/best generations coalesce.
Explicit epoch archives apply bounded backpressure instead of being dropped.
CPU snapshots preserve parameter/optimizer dtypes and isolate later mutations.
Frozen vision is omitted from best saves as well as recovery saves. Recovery
uses the highest complete checkpoint epoch from local/output candidates with
CPU/mmap loading, and never treats summary metadata as proof of saved progress.
Mirroring failure is surfaced and leaves completed local files recoverable.

Controlled overlap benchmark (five identical 40 ms compute intervals, 4 MiB
checkpoints, 400 ms simulated storage latency): synchronous 2.212 s; staging
foreground 0.222 s; total including final mirror drain 0.848 s. Final epoch was
preserved and two generations were mirrored. This is a 2.61x end-to-end speedup
in that controlled test, not a measured Helios epoch speedup. Raw measurements:
[JSON](checkpoint_io_overlap_2026_10_18.json).

Regression run: 163 passed in 22.23 s, including the earlier architecture,
training, gripper and evaluation checks. Additional CUDA fused-AdamW continuation
coverage is included in `tests/test_checkpoint_io.py`.

## Helios deployment and pending handoff

The existing job uses an older flat-memory architecture. A checkpoint-only copy
of its runner and the helper were installed under:
`/home/wee_ucl/.local/share/align-training-io/` on Helios. The repository and model
architecture there were not replaced with the separate patch-token changes.

The original command/environment were captured in `restart_state.json`.
Recovery checkpoint epoch 15, including optimizer state, was verified with CPU
mmap loading. Its best loss is 0.29340880002939346. A copy of recovery/best files
is being staged to `/dev/shm/wee_ucl-align-checkpoints` before switching. A
one-time controller (`handoff.py`, PID 2371511 at launch) verifies the staged
checkpoint and waits for old PID 461455 to exit before starting the same run
with the new staging argument. It does not launch a duplicate CUDA trainer.

The original trainer was sent SIGINT, then SIGKILL after remaining blocked in
`folio_wait_bit_common`. At the last observation it was still in D state; kernel
I/O had not returned. The copy process was also waiting on that filesystem.
Thus the new training run has **not yet resumed**, and there is no measured
post-change Helios epoch. Local tested changes are complete; live handoff is
waiting for external storage/kernel progress. No other user's process was altered.

Status can be read without accessing NFS:

```bash
ssh helios 'cat /home/wee_ucl/.local/share/align-training-io/handoff_status.json'
```

After phase `resumed`, `resume.log` in that directory records the new PID and
training output. NFS mirroring may lag while training proceeds. `/dev/shm` data
survives process restarts, but not a host reboot; earlier completed NFS checkpoints
remain available until atomically replaced. Normal run completion waits for the
final mirror drain.

## Follow-up: original trainer exited; parallel recovery active

After the user confirmed that the earlier process was killed, verified that
PID 461455 no longer existed. The sequential recovery copy was only about
2.7/30.6 GiB through the resume checkpoint and was limited by NFS reads. Replaced
only that diagnostic copy and waiting controller with a bounded 16-thread
`pread`/`pwrite` transfer, preserving its completed prefix after recording source
inode, size and mtime. The transfer journals completed chunks and verifies local
ZIP CRCs and checkpoint metadata before publishing. Local prefix-transfer and
restart checks passed byte-for-byte.

New remote processes at launch: parallel copy PID 2564686 and handoff controller
PID 2564687. `copy_progress.json` and `handoff_status.json` under the same Helios
home directory expose live progress. The controller resumes the unchanged run
from saved epoch 15 (next epoch 16), only after validated staging is complete.
Training was not yet resumed at this update. The named/IP storage-server SSH
shortcuts were unavailable under existing strict host-key verification; host-key
checks were not bypassed. Remaining delay is the one-time shared-storage read.
