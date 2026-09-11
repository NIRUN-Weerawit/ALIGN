# Piper X-VLA single-task fine-tuning

**Summary:** This runs `xvla-libero` on the retained Piper task **“grab an object and put in a cup”** using 90 train and 11 held-out validation episodes. It is an offline training procedure only; it does not send any Piper command.

## What was prepared

Source data remains immutable:

```text
/home/ucluser/ALIGN/baselines/data/piper_replay/dataset
```

The source has intentionally deleted episodes. Rather than duplicate or rewrite its 21 GB of embedded image Parquet data, the training manifest selects only files that exist:

```text
101 episodes
24,352 frames
20 Hz
```

Training manifest:

```text
/home/ucluser/ALIGN/baselines/data/piper_replay/xvla_training_manifest.json
```

The adapter in `piper_xvla/xvla_dataset.py` maps each raw source sample to the installed checkpoint contract:

| Field | Source | X-VLA training view |
|---|---:|---:|
| Global camera | RGB 480×640 | `observation.images.image` |
| Wrist camera | RGB 480×640 | `observation.images.image2` |
| Third view | absent | black `224×224` `empty_camera_0` |
| Proprioception | xyz + rotation-6D + gripper (10-D) | xyz + quaternion + normalized gripper (8-D) |
| Action | future Piper state (10-D) | active-arm EE6D + zero inactive arm (20-D) |

Gripper values are mapped monotonically using retained-data limits:

```text
-0.042400 m -> 0.0
+0.047900 m -> 1.0
```

## Checkpoint compatibility preflight

Checkpoint:

```text
/media/ucluser/PortableSSD/hf_models/xvla-libero
```

A real GPU forward pass used a retained Piper frame and returned finite loss:

```text
loss            22.3881
position_loss   20.8573
rotate6D_loss    0.8497
gripper_loss     0.6811
```

The checkpoint's JSON advertises a 1024-token language limit, but its transformer has a 512-token total sequence limit after image tokens. This pipeline therefore fixes:

```text
tokenizer_max_length = 64
```

## Train/validation split

The training script deterministically reserves the final 11 retained source episode IDs for validation:

```text
train: 90 episodes
validation: 11 episodes
```

The exact IDs are written alongside checkpoints to `split.json`. Best checkpoint selection uses `val/loss`, never train loss.

## Launch procedure

All run settings are in:

```text
piper_xvla/config/piper_xvla_single_task.json
```

The cache contains only converted numeric tensors plus source frame/episode indices; it does **not** duplicate camera images. Its actual size for this dataset is 3.34 MiB. Build it once before training:

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.prepare_xvla_cache \
  --config piper_xvla/config/piper_xvla_single_task.json
```

The configured artifact is:

```text
data/piper_replay/piper_xvla_prepared_cache.pt
```

The training script validates that the cache refers to the exact manifest/source episode set before opening images. It prints six explicit lifecycle stages: config, cache validation, image datasets, model/optimizer, artifact writing, and training. At 1,000-step intervals it prints a JSON train/validation loss record.

This follows the official [X-VLA new-embodiment fine-tuning guidance](https://huggingface.co/docs/lerobot/en/xvla): BF16 full adaptation, with neither VLM encoder frozen and both the policy transformer and soft prompts trainable. The settings are explicit rather than inherited from library defaults:

```text
steps = 20,000
batch_size = 4
freeze_vision_encoder = false
freeze_language_encoder = false
train_policy_transformer = true
train_soft_prompts = true
```

It also uses X-VLA's official optimizer preset: differential AdamW learning rates (VLM at 1/10 the base rate) and a 1,000-step warmup followed by cosine decay across the 20,000-step run. The Piper target remains `action_mode = ee6d`, not generic `auto`, because the data has a defined EE6D layout and binary gripper targets.

Edit the file to change output path, steps, batch size, validation count, checkpoint, or device. The CLI takes only an optional config path:

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.train_xvla_piper \
  --config piper_xvla/config/piper_xvla_single_task.json
```

Artifacts:

```text
outputs/piper_xvla_single_task/split.json
outputs/piper_xvla_single_task/best.pt
outputs/piper_xvla_single_task/last.pt
```

`best.pt` is selected by held-out validation loss. `last.pt` is overwritten every 1,000 steps and at the final step, so it contains optimizer and scheduler state suitable for inspecting the most recent run state.

The default uses one-step action supervision at 20 Hz because each replay label is measured `state[t+1]`. The script explicitly sets:

```text
chunk_size = 1
n_action_steps = 1
```

This avoids fabricating 30-step targets from data that only records one-step measured future state.

## Operational checks

Before treating a resulting checkpoint as deployable:

1. Confirm `best.pt` exists and validation loss is finite.
2. Plot train versus validation loss from the printed JSON step records; rising validation loss indicates overfit.
3. Run offline action-range inspection on held-out episodes. Confirm predicted xyz stays within the Piper workspace, quaternion norm is valid after conversion, and gripper output stays in `[0,1]`.
4. Run a **read-only** camera/Piper observation pass with the saved checkpoint before enabling any motion path.
5. Only after offline and read-only checks should a separate, explicit physical execution procedure be considered.

This is a single-task policy. It should not be used as a general Piper manipulation model or evaluated on the deleted/unusable episodes.
