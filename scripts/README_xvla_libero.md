# XVLA LIBERO through ALIGN's simulator evaluator

Use `lerobot/xvla-libero` as a policy backend in
`eval/eval_libero_v4_trajectory.py`. ALIGN remains the default backend.
Both backends share task matching, HDF5 episode selection, per-episode reset
seeds, control frequency, rollout limit, action horizon, temporal ensembling,
expert-prefix handover, videos, timing and success reporting.

## Runtime

The tested optional runtime is `.venv-xvla-eval`, with LeRobot 0.6.1 and
Transformers 5.5.4. Existing training packages are unchanged. It reuses CUDA
PyTorch and the simulator packages from the existing align-ablation environment.
To recreate this layout from the repository root:

```bash
/home/whinnoy/.venvs/align-ablation/bin/python -m venv --system-site-packages .venv-xvla-eval
.venv-xvla-eval/bin/python - <<'PY'
import pathlib, site
(pathlib.Path(site.getsitepackages()[0]) / 'align_simulator_dependencies.pth').write_text(
    '/home/whinnoy/.venvs/align-ablation/lib/python3.12/site-packages\n')
PY
.venv-xvla-eval/bin/python -m pip install -r scripts/requirements_xvla_eval.txt
```

## Model cache and command

The downloaded checkpoint is in the standard HF cache:

`/media/whinnoy/Extreme-Linux/huggingface/hub/models--lerobot--xvla-libero`

Pinned model revision: `12e8783e996944f5c97e490d37d4c145484ed70a`.
The adapter accepts that cache directory, a specific snapshot directory, or a
Hub repo ID. A cache directory with no main ref is accepted when it contains
exactly one snapshot. Snapshot paths provide the strongest reproducibility. BART tokenizer
files are cached separately on the same drive; no BART model weights are needed.

```bash
export PYTHONPATH=/home/whinnoy/.local/share/align/LIBERO
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_CACHE=/media/whinnoy/Extreme-Linux/huggingface/hub

.venv-xvla-eval/bin/python eval/eval_libero_v4_trajectory.py \
  --policy-type xvla \
  --checkpoint "$HF_HUB_CACHE/models--lerobot--xvla-libero/snapshots/12e8783e996944f5c97e490d37d4c145484ed70a" \
  --hf-cache-dir "$HF_HUB_CACHE" \
  --xvla-tokenizer "$HF_HUB_CACHE/models--facebook--bart-large/snapshots/cb48c1365bd826bd521f650dc2e0940aee54720c" \
  --data data/libero_goal.h5 \
  --device cuda --xvla-dtype bfloat16 \
  --n-episodes 1 --seed 42 --max-steps 300 \
  --switch-at 0 --alpha 1 --action-horizon 1 \
  --out-dir eval/libero_results/xvla_goal
```

For the same held-out comparison, add `--val-episodes <episode-key-file>` and
`--n-episodes 44`, using the same episode file and reset options as the ALIGN run.
Use `--no-video --no-plot` for faster evaluation. To run on CPU, use
`--device cpu --xvla-dtype float32`. Native sampling uses 10 flow steps;
`--xvla-denoising-steps` is an explicit experimental override. The native model
predicts 30 actions; the shared evaluator executes only the configured horizon
before making a fresh prediction. `--action-horizon 1` matches ALIGN's usual
comparison setting; native XVLA's longer chunk execution is a separate experiment.

## Verified policy contracts

- Native XVLA LIBERO proprioception is 20-D: current XYZ, the first two rotation
  matrix columns concatenated, zero gripper, and ten inactive-arm zeros. This
  follows the policy-specific XVLA environment processor. The checkpoint's 8-D
  `input_features` metadata is legacy; the generic LeRobot 8-D LIBERO state and
  the Piper adapter are different contracts.
- Raw simulator agentview is flipped in both H/W; wrist stays unflipped. Inputs
  receive ImageNet normalization. The missing third view is padded and masked by
  XVLAPolicy. The shared evaluator's display flips do not change native policy
  preprocessing.
- Saved checkpoint tokenizer settings are used: BART, right padding, length 50.
  Saved domain ID is 3. State/action normalization is identity. Unsupported
  normalization/configurations fail instead of silently applying another policy's
  conventions.
- Native actions are absolute XYZ + rotation-6D + close probability for the active
  arm, plus inactive-arm outputs. Rotation-6D uses consecutive columns, unlike the
  Piper branch's interleaved convention. Absolute targets stay in the shared
  horizon/ensemble queue. At execution, convert against the current actual pose,
  invert the live OSC controller's scaling, and respect its input limits. This
  keeps the same delta controller as ALIGN. Inactive-arm outputs are unused.
- XVLA close probability is converted to ALIGN's 0=closed, 1=open convention;
  the shared loop then performs the single simulator +/-1 conversion. Each episode
  resets the policy's queues and cached instruction tokens.

Result JSONs are written in `--out-dir` (never inside the model cache) and include
checkpoint revision, package versions, native contracts, precision, sampling
steps, and all evaluation arguments. The same evaluator still uses seeded random
resets rather than official LIBERO fixed initial states; compare both policies
under identical settings and describe this protocol explicitly. The pretrained
model's training data may overlap LIBERO tasks/demonstrations, so this is a
pretrained baseline rather than a matched-data ablation.

Sources:
- https://huggingface.co/lerobot/xvla-libero
- https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/xvla/processor_xvla.py
- https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/libero_client.py

## Verification on this machine (2026-10-16)

- Downloaded all six model repository files; weights are 3,519,073,692 bytes and
  contain 903 safetensors entries. Real weights load strictly through LeRobot.
- 37 tests passed, covering camera/pose conversion, controller scaling, rotation
  column layout, gripper polarity, queue execution against the current pose,
  cache-root resolution, headless cleanup, and existing ALIGN inference contracts.
- Real CPU/FP32 LIBERO smoke test completed with offline Hub access, one episode,
  three simulator steps, horizon two, two native 30-action predictions, an expert
  replay, three-panel MP4, JSON output, and clean exit. This checks integration;
  three steps do not measure task success. GPU inference was not exercised because
  other jobs left only about 1.9 GiB free.
- Artifacts: `reports/xvla_libero_adapter_2026_10_16/cpu_smoke_final/` and
  `reports/xvla_libero_adapter_2026_10_16/download_manifest.json`.
