# Transformer action backbone

Select the denoising network independently from the objective:

```bash
--head-type diffusion --action-backbone transformer \
--head-d-model 256 --transformer-layers 4 --transformer-heads 8
```

Use `--head-type flow_matching` for the same network with the existing flow objective and Euler/Heun solver. These options are available in both `scripts/run_intention_ablation.py` and `training/train_intention.py`. Add them to an otherwise valid training command and use a new output directory. `--resume` restores saved architecture settings; it does not convert U-Net weights into Transformer weights.

Defaults remain `--action-backbone unet`. Historical checkpoints missing backbone settings load as U-Net. The legacy direct-regression `--head-type transformer` is a separate, unchanged head; do not use it to select the new backbone.

## Architecture

Each noisy action is projected into one token with a sinusoidal action-position embedding. Each block applies bidirectional action self-attention, conditioning cross-attention, and a four-times-width feed-forward network. Adaptive LayerNorm supplies timestep conditioning. Action horizon and observation-history length are independent. At least two blocks are required, and width must divide the positive attention head count. The training runner retains its width-multiple-of-eight requirement.

With `--task-context-fusion aci`, blocks alternate context and observation attention. Context contains task slots, execution, and optional temporal-Mamba tokens, with slot embeddings and true presence masks. Visual patches use camera and 2-D spatial embeddings. State and noise/flow time drive adaptive LayerNorm; visual/context values do not enter global modulation. Context and visual K/V are cached across sampler steps and shared across repeated noise draws during training.

Other conditioning modes retain their existing semantics: the established packed/additive condition becomes a cross-attention token per observation frame, with temporal position embeddings. This supports baseline, legacy intention, text, and full-history variants without silently changing their conditioning definitions. Existing optional inline visual attention and multiscale visual-context settings add a separate patch bank. Plain history preserves all frame conditions; it does not average them in the Transformer.

This implementation does **not** change legacy text into a separately encoded text-token bank or expose raw memory slots to the head. Those are separate conditioning changes. Memory retrieval and temporal recurrence remain upstream, as before. Context freezing/fine-tuning remains controlled by existing flags.

The existing diffusion/flow losses, padding filters, gripper target convention, action normalization, clipping and sampling schedules are reused. Internally the backbone retains the owning head's `unet` attribute name to preserve existing sampler and intervention interfaces; its module is `ActionTransformer`.

## Validation

`tests/test_action_transformer.py` covers masked losses, history/chunk independence, both objectives, optional visual paths, context masks, memory and temporal gradients, frozen/fine-tuned context, production checkpoint reload and production training/validation on CPU and CUDA. Existing ACI U-Net tests remain unchanged.

No full training or simulator success benchmark has been run for this new backbone. Smoke tests establish integration and numerical operation, not improved task success or speed.
