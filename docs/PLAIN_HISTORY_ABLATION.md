# Plain history diffusion ablation

The `plain_history` variant has no intention encoder and no memory bank. It
keeps the last `--plain-history-size` encoded observations in temporal order.
Each observation contributes its 4,096-dimensional compressed visual feature
and 64-dimensional encoded robot state in the current LIBERO Goal setup.
The diffusion U-Net conditions its FiLM blocks on the current observation and
uses cross-attention at the bottleneck to read the full ordered history.
Task text, when enabled, is projected into the condition width and added to
each history step.

Training reads cached DINOv2 features once per episode segment. The first
observation is repeated on the left until a complete history is available;
this preserves the same action supervision anchors as a one-frame model.
MuJoCo evaluation encodes each new camera frame once and retains encoded
visual and state features in `IntentionStream`. No past raw frame is encoded
again at later action calls.

The first comparison is text-conditioned `plain_history` with four frames
against `checkpoints/piper_v4_ablation_use_text_2/intent_memory`. It uses the
same LIBERO Goal data, whole-episode split, seed 42, 200 epochs, batch size
10, learning rate 0.0002, and diffusion settings. The history head adds a
projection and cross-attention parameters; this comparison tests the complete
history-head architecture rather than isolating memory capacity alone.

Run directory: `checkpoints/piper_v4_ablation_use_text_plain_history_h4`.
The `queue.sh` there waits for the four current held-out MuJoCo evaluations
before starting training. Training progress and metrics are written to
`run.log` and `plain_history/metrics.jsonl`.

The reproducible training command is:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_intention_ablation.py \
  --data data/libero_goal.h5 --cache data/libero_goal.dinov2 \
  --output checkpoints/piper_v4_ablation_use_text_plain_history_h4 \
  --variants plain_history --use-task-text \
  --history-size 1 --plain-history-size 4 \
  --temporal-sampling episode --epochs 200 --batch-size 10 \
  --num-workers 0 --compressed-dim 8 --lr 0.0002 \
  --gpu-memory-fraction 1.0 --seed 42
```

Zero DataLoader workers avoids exhausting host shared memory on this 32 GB
machine. Expandable CUDA segments reduces fragmentation across variable-length
episode batches. Neither change alters the model or effective batch size.

After training, evaluate the same 44 held-out episodes with `--switch-at 0.0`,
`--val-episodes checkpoints/piper_v4_ablation_val_episodes.txt`, and
`--n-episodes 44`. Compare task success, per-task failures, gripper behavior,
and inference time with the text-conditioned intent-plus-memory model.
