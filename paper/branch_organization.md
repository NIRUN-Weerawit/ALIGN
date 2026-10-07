# Branch and worktree organization — 2026-10-08

## Layout

| Worktree | Branch | Purpose |
| --- | --- | --- |
| `/home/whinnoy/ALIGN` | `main` | Shared ALIGN model, training, evaluation, data utilities, and documentation |
| `/home/whinnoy/ALIGN-semantic` | `feat/semantic-intent-hysteresis` | Semantic schemas, VLM labeling interface, sidecar lookup, dataset integration |
| `/home/whinnoy/ALIGN-xvla` | `feat/xvla-finetuning` | X-VLA baseline, Piper collection, controls, and inference |

Both feature branches descend from updated main. Semantic implementation stays
on the semantic branch and is no longer inherited by XVLA. Semantic hysteresis
itself remains WIP; the branch currently provides label preparation/loading.

## Work promoted from XVLA to main

Original commit IDs identify the source history, retained in local archive tags.

| Original commit | Work | Decision |
| --- | --- | --- |
| `2043eb5` | Piper replay to ALIGN HDF5 converter | Main: converts data for ALIGN without importing the XVLA pipeline |
| `3c2d596` | DINOv2 cache/training consistency | Main: core data and vision contract |
| `cae19b8` | Precompute disk-space check | Main: general cache utility fix |
| `58e0f82` | uv installation support | Main: project setup |
| `60137d5` | Intention training performance controls | Main: core trainer |
| `92703fc` | Distributed intention training | Main: core trainer and CPU distributed regression |
| `495f179` | Earlier multi-GPU changes | No separate replay: the merged trainer equals `92703fc` |
| `590af47` (partial) | ALIGN evaluator defaults and paper/reference updates | Main: only `eval/eval_libero_v4_trajectory.py`, `paper/main_paper.md`, and `paper/references.bib`; Piper UI/collection changes stay on XVLA |
| `954f6d5` (partial) | Generic intention evaluation head dispatch | Main: only `eval/eval_intention.py` and its regression test; Piper integration and XVLA training stay on XVLA |

Other XVLA compatibility/configuration/cache/training commits stay on XVLA.
ALIGN-specific Piper live inference and master/slave controls also stay there:
they depend on the shared Piper package, schema, guard, and UI. Extracting them
would require a separate package refactor rather than moving an independent
commit. Relative to main, the XVLA branch changes only `baselines/piper_xvla/`
and its ignore rules.

## Foundational fixes

Main contains causal gripper fallback and rollout state, padding exclusion for
both generative losses, valid-window normalization, optional intention encoder
construction, memory retrieval without intent tokens, empty-memory attention
handling, and supported-head/WIP documentation. Conflicts were resolved to
preserve the newer distributed synchronization and training performance controls.
The encoder implementation is retained; its construction is controlled by the
intent-token/history flags.

The proposed explicit camera-fusion disabling change was excluded because the
historical V4 docs specify enabled cross-camera attention. The existing model
fusion configuration and versioned per-camera cache path are preserved. A
separate train/cache/inference fusion audit remains necessary.

## Validation

- Main: 26 targeted tests passed, including two-process CPU training, causal
  gripper state, loss masking, optional memory, cache validation, converter math,
  and evaluation dispatch. Compilation and whitespace checks passed.
- Semantic: 37 targeted feature and foundational regression tests passed.
- XVLA/Piper: 41 selected tests passed; 12 tests failed at dependency imports
  because FastAPI or LeRobot is unavailable. No successful full XVLA validation
  is claimed. All `baselines/piper_xvla/` files match the pre-rebase branch exactly.
- No full GPU training or hardware/simulator rollout was performed.

## Preserved local work and recovery

Existing uncommitted `requirements.txt` and `scripts/decode_libero_to_hdf5.py`
edits were restored byte-for-byte in the XVLA worktree. They are not included in
this integration. The intention audit is committed on main.

Local recovery tags retain the original main, semantic, and XVLA tips:
`archive/pre-organize-main-20261008`,
`archive/pre-organize-semantic-20261008`, and
`archive/pre-organize-xvla-20261008`. `archive/intention-fixes-20261008` retains
the initial fixes commit before main integration. The named pre-organization
stash is retained as an additional copy of the existing edits. Archive tags and
the stash are local; feature branch updates use explicit force-with-lease checks.
