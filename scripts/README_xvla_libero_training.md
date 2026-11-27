## Train an X-VLA action transformer on the ALIGN Goal split

Use the canonical ALIGN split with 384 training episodes and 44 held-out
episodes. The trainer loads only the Florence-2 weights from the encoder source,
freezes that entire module, and initializes a fresh action transformer and soft
prompts. With the LIBERO checkpoint as the encoder source, the frozen encoder may
already have seen LIBERO data.

```bash
/home/whinnoy/ALIGN/.venv-xvla-eval/bin/python scripts/train_xvla_libero.py \
  --data /home/whinnoy/ALIGN/data/libero_goal.h5 \
  --encoder-source /path/to/local/xvla-libero/snapshot \
  --tokenizer /path/to/local/bart-large/snapshot \
  --output runs/xvla_goal --epochs 100 --batch-size 4 --chunk-size 4
```

Training uses the two recorded RGB views and the first recorded instruction.
EE6D chunks use the measured pose at frame t+1 as the target for each command at
frame t, with the recorded gripper command converted to X-VLA polarity. The
loader includes only complete chunks within each episode. The held-out episodes
select the best checkpoint by mean validation loss. Point the LIBERO X-VLA
evaluator at the output directory's `best/` subdirectory.

For a smoke test, append `--max-batches 1` and set `--epochs 1 --batch-size 1`.
