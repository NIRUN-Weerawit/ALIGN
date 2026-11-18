# Bag-of-words context replacement

Use `--task-context-bag-text --use-task-text --text-encoder-type bag` with context variants. Start a new output directory: this changes the architecture and cannot resume an ordinary student-context checkpoint.

The training vocabulary comes only from training episodes. Word embeddings are mean pooled and a trainable linear projection produces five separate context slots (four task slots and the former execution slot). All five are instruction-only embeddings, constant across an episode; none predicts execution progress. They receive action-loss gradients. The pretrained student encoder is removed. The existing pretrained observation summary remains to feed the independent trainable temporal Mamba; summary fine-tuning remains opt-in. Vision, state, memory, and ACI routing are retained.

Keep the observer checkpoint and split manifest arguments: they initialize the retained observation summary and preserve the held-out split. Do not enable student fine-tuning or teacher-source conditioning. Do not use task-label probes fitted to the former student token space. Inference requires episode task text; the simulator reads it from the dataset. Existing checkpoints without the new flag retain their architecture.

For the best Context 8 architecture, use the original training arguments with a new output directory and add:

```bash
--task-context-bag-text --use-task-text --text-encoder-type bag --text-dim 128
```

Keep `--head-type diffusion --action-backbone unet --head-d-model 256 --compressed-dim 32 --state-dim 64 --chunk-size 4 --supervision-points 16 --policy-temporal-mamba --policy-temporal-width 256 --policy-temporal-layers 2`, as well as the original memory settings. This is an explicit-instruction ablation; replacing execution with text deliberately removes the pretrained live execution representation.

Tests cover token/text dependence, gradients, frozen summary, sequence/streaming equivalence, checkpoint reconstruction, and production train/validation on CPU and CUDA for diffusion and flow matching. No full training or rollout-success result exists yet.
