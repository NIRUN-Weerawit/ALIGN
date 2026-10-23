# Teacher/student refit on the Piper 384/44 split

Completed from scratch with seed 42 using exactly the episode assignments in
`checkpoints/piper_v4_ablation_no_text_2/manifest.json`. Both training and held-out
membership match all 20 available Piper ablation manifests and the shared
`piper_v4_ablation_val_episodes.txt` list. No Piper held-out episode is in training.
The policy's validation keys are named `confirmation` in the representation manifest;
this naming does not change episode membership. Development is empty.

| Stage | Training episodes | Epochs | Optimizer updates |
|---|---:|---:|---:|
| Full-episode teacher, quarters pooling, temporal decoder | 384 | 80 | 3,840 |
| Causal full-prefix student, selector weight 0.2 | 384 | 100 | 4,800 |

Both models were trained from scratch. Normalization was fitted only on Piper's
384 training episodes. New teacher tokens and prefix summaries were rebuilt.
No previous teacher/student checkpoint or fitted normalization was reused.
Checkpoints are fixed-budget final weights: held-out metrics were computed only
after both checkpoints were fixed, without early stopping or checkpoint selection.
This is Piper's existing validation set, not a newly collected untouched test set.

## Artifacts and provenance

- [Observer bundle](../../checkpoints/task_context_piper_384_20261023/observer_frozen.pt)
- [Frozen teacher](../../checkpoints/task_context_piper_384_20261023/teacher_frozen.pt)
- [Frozen student](../../checkpoints/task_context_piper_384_20261023/student_frozen.pt)
- [Exact split and target manifest](targets/manifest.json)
- [Reference policy manifest](policy_split_reference.json)
- [All-Piper split verification](piper_split_verification.json)
- [Training protocol](protocol.json)
- [Held-out diagnostics and episode results](heldout.json)
- [Observer streaming verification](observer_verification.json)
- [Final integrity audit](integrity_verification.json)

## Held-out representation diagnostics

Student results average seven causal prefixes (0%, 10%, 25%, 40%, 50%, 70%, 90%)
per episode. Teacher diagnostics use three action anchors per full demonstration.

- Teacher position MSE: 0.029275; rotation MSE: 0.002084.
- Student position MSE: 0.037917; rotation MSE: 0.002758.
- Student normalized token MSE: 0.449031.
- Future prediction error, +16 frames: 0.263272; persistence: 0.467909.
- Future prediction error, +64 frames: 0.545927; persistence: 1.060031.

These diagnostics are not action-policy simulator completion rates. The new
teacher's latent space, normalization, and held-out membership differ from the
previous context refit, so those representation errors are not directly comparable.

Cached versus streaming inference passed on all 44 held-out episodes /
5,567 frames. Maximum absolute error: 0.00001276; tolerance: 0.0002.
Observer step latency on precomputed features: median 1.044 ms,
p95 1.100 ms, excluding DINO and the action policy.

## Downstream use

Use this observer for new action-policy training with the same Piper episode
membership. Its latent space requires new action-head training and new task-label
estimators; existing policies are unchanged. The representation manifest uses
384 train, zero development and 44 confirmation keys. Policy scripts requiring
a nonempty development set need an explicit compatible training/evaluation protocol.
No action policy or task-label estimator was trained in this run.

The export records fixed-budget refitting, without claiming a new pass of the
original controlled teacher/student information gates. Previous refit artifacts
remain unchanged.
