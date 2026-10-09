# ALIGN: Assistive Latent Intention-Guided Network

**ALIGN** is a shared autonomy framework for robotic manipulation. It learns to infer human intent from visual observations and robot state, then assists by predicting corrective actions during teleoperation.

The core idea: a human leads, the model observes. Over time, the model builds an understanding of the task through temporal context (Mamba), explicit intent tokens, and an episodic memory bank. It then generates smooth, task-appropriate actions via a diffusion policy head.

### V4 implementation status

- Historical V4 docs specify active cross-camera attention. This update preserves the existing camera-fusion configuration; reconciling training, cached features, and inference fusion is separate work.
- Diffusion and flow matching are the supported V4 action heads. Transformer, Mamba, and hybrid action heads are future work; their legacy implementations are retained for research.
- When `use_intent_tokens=False` (the CLI default; omit `--use-intent-tokens`), the model does not construct the intention Mamba encoder. Its implementation remains in the codebase and is constructed when intent tokens and history are enabled. Observation history can still condition the action head. Visual/state memory retrieval remains active when `--use-memory-bank` is enabled without intent tokens.
- Semantic labels and semantic hysteresis are WIP. Labels can be loaded by the dataset, but semantic supervision and inference hysteresis are not connected to the policy. `--anchor-weight` and `--use-text` do not currently add semantic supervision.
- State inputs use measured gripper observations when available. Otherwise they carry the previous executed command, never the current target command. Episode initialization uses the `initial_gripper` HDF5 attribute, defaulting to a neutral 0.0 only when no initial state is recorded. Rollouts carry the last executed command instead of resetting gripper state to zero every step.
- Diffusion/flow losses exclude padded samples before reduction. Validation accumulates errors across all valid prediction windows.

Legacy feature caches must be regenerated with the versioned per-camera precompute path. Existing checkpoints may contain unused cross-camera or disabled-intent encoder weights; checkpoint loading must report those differences explicitly.

---

## Architecture

```
frames (B, T, V, H, W, 3)          states (B, T, 7)
  │                                       │
  ▼                                       ▼
DINOv2 ViT-B/14 (frozen)           StateEncoder (MLP)
  │                                       │
  ▼                                       ▼
CLS tokens (B, T, V, 768)          z_s (B, T, 256)
patch tokens (B, T, V*P, 768)
  │                                       │
  ▼                                       ▼
VisionPatchEncoder
  ├─ SEVisualCompressor (768 → comp_dim)
  └─ StateConditionalCrossAttn
  │
  ▼
z_v_pooled (B, T, pool_out_dim)
```

### Temporal Encoding (Mamba, optional)

When `--use-history` and `--use-intent-tokens` are enabled, CLS tokens and states are fed through a Mamba SSM for temporal recurrence:

```
z_v_CLS (B, T, V, 768) + z_s (B, T, 256)
  │
  ▼
Flatten + concat → (B, T, V*768 + 256)
  │
  ▼
Mamba SSM (d_model = V*768 + state_dim)
  │
  ▼
h_seq (B, T, d_model)
```

### Intent Tokens (optional)

Learnable tokens appended to the Mamba input sequence. The SSM processes them with full temporal context, producing intent embeddings that encode the model's understanding of the current task:

```
Input: [h_0, h_1, ..., h_T, INTENT_1, ..., INTENT_N]
  │
  ▼
Mamba → h_seq (B, T, d_model) + intent_emb (B, N, intent_dim)
```

### Memory Bank (optional)

A fixed-size episodic memory that stores past (visual, state, intent) triplets. At each step, the current observation retrieves relevant context via cross-attention, which is then fused through learned gates:

- **Perceptual stream**: past visual features
- **Cognitive stream**: past intent embeddings
- **State stream**: past robot states

The bank uses a circular buffer with token-merge consolidation when full.

Experimental episodic modes are opt-in: `--memory-patch-temporal` retrieves each camera/grid slot across time; `--memory-value-preserving` returns weighted raw historical values. `--memory-pre-state-visual` stores compressed vision before robot-state modulation and applies current state after retrieval, avoiding replay of old state-conditioned visual context. It requires episodic raw writes. `--memory-field-masks` separately excludes zero-marked missing fields from lookup, preserves a valid field when merged with a missing one, and tracks the field’s actual timestamp. In this opt-in mode an exactly zero feature vector is the missing-field sentinel. `--memory-perceptual-recency` optionally penalizes visual attention logits by feature age; zero preserves the old selection. It requires raw-value retrieval; pair it with field masks when packets can be missing. Existing checkpoint behavior stays unchanged. See [the experiment ledger](paper/memory_adjustment_work_log.md) for measured benefits and remaining failures.

### Diffusion Policy Head

A 1D U-Net that denoises random noise into a chunk of K future actions via DDPM/DDIM. Conditioning is global (mean-pooled over the window) rather than per-step, using FiLM modulation:

```
noise (B, K, action_dim) + cond (B, 1, cond_dim)
  │
  ▼
U-Net (10 DDIM steps)
  │
  ▼
actions (B, K, action_dim)
```

Condition = `concat[mean(z_v_pooled), z_s_last, intent_pooled]`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **DINOv2 frozen** | 86M-param ViT-B/14 is expensive to fine-tune; frozen features generalize across scenes |
| **CLS tokens for Mamba** | CLS tokens carry global scene context; patch tokens go to the head for spatial detail |
| **SE compression** | Squeeze-excitation reweights DINOv2 channels before projection, suppressing noisy dimensions |
| **State-conditioned modulation** | Each patch token is modulated by robot state via cross-attention, not concatenation |
| **Mamba (not Transformer)** | O(1) inference per step vs O(K) for attention; critical for 30-100Hz teleoperation |
| **Intent tokens (not pooling)** | Learnable tokens let the SSM decide what to compress, rather than averaging all history |
| **Circular memory bank** | Fixed-size avoids unbounded growth; token-merge preserves diversity |
| **Diffusion head (not regression)** | DDPM produces multimodal action distributions; regression collapses to mean |
| **Global conditioning** | Mean-pool over the window gives a single task context; avoids per-step overfitting |

---

## Training

### Data Format

Episodes are stored as HDF5 files with the following structure:

```
ep_000000/
├── frames/
│   ├── image        (N, H, W, 3) uint8
│   └── wrist_image  (N, H, W, 3) uint8
├── poses           (N, 6) float32
├── actions         (N, 7) float32  [dx, dy, dz, drx, dry, drz, gripper]
├── gripper         (N,) float32
└── texts           JSON string
```

### Training Loop (V4)

Each batch samples a variable-length segment (2-5× history_size) from each episode:

1. **Pre-encode**: Batch DINOv2 over all frames → split CLS/patch tokens
2. **Encode patches**: SE compress + state modulate patches → `z_v_pooled`
3. **Window loop**: For each valid window in the segment:
   - Forward CLS tokens through Mamba → `h_seq` + `intent_emb`
   - Retrieve from memory bank → fused features
   - Diffusion head → predict K future actions
   - Compute DDPM noise-prediction loss
4. **Optimizer step**: Sum all window losses, backward, clip, step

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | Path to HDF5 dataset |
| `--cameras` | `["wrist_image"]` | Camera names to use |
| `--head-type` | `diffusion` | Supported V4 heads: diffusion, flow_matching; regression heads are future work |
| `--action-dim` | 7 | Action dimensions (6 pose + 1 gripper) |
| `--chunk-size` | 10 | Future action prediction horizon (K) |
| `--history-size` | 1 | Mamba temporal window (H) |
| `--compressed-dim` | 8 | Per-patch dimension after SE compression |
| `--use-intent-tokens` | False | Enable learnable intent tokens |
| `--use-memory-bank` | False | Enable episodic memory bank |
| `--use-history` | True | Enable Mamba temporal encoder |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 100 | Number of epochs |

### Example

```bash
# Train with intent tokens + memory bank + diffusion head
python training/train_intention.py \
    --data data/libero_spatial.h5 \
    --cameras image wrist_image \
    --output-dir checkpoints/v4 \
    --action-dim 7 \
    --use-intent-tokens \
    --use-memory-bank \
    --epochs 100 \
    --head-type diffusion \
    --batch-size 16 \
    --compressed-dim 8 \
    --lr 1e-4

# Train without history (no Mamba, no intent tokens)
python training/train_intention.py \
    --data data/libero_spatial.h5 \
    --cameras image wrist_image \
    --output-dir checkpoints/baseline \
    --action-dim 7 \
    --head-type diffusion \
    --batch-size 16 \
    --no-history
```

### Multiple GPUs

Launch one intention training process per GPU with `torchrun`:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 training/train_intention.py \
    --data data/libero_spatial.h5 \
    --cameras image wrist_image \
    --output-dir checkpoints/v4 \
    --epochs 100 \
    --batch-size 16
```

`--batch-size` is per GPU, so this example uses a global batch of 32.
Training samples are split across processes, validation metrics are combined,
and only the main process writes logs and checkpoints. Use
`CUDA_VISIBLE_DEVICES` to choose which GPUs participate.

---

## Evaluation

### Offline (dataset replay)

```bash
python eval/eval_intention.py \
    --data data/libero_spatial.h5 \
    --checkpoint checkpoints/v4/libero_spatial/run_15/intention_best.pt \
    --n-batches 20
```

### MuJoCo Simulator

```bash
python eval/eval_libero_v4_trajectory.py \
    --data data/libero_spatial.h5 \
    --checkpoint checkpoints/v4/libero_spatial/run_15/intention_best_fixed.pt \
    --cameras image wrist_image \
    --n-episodes 5 \
    --switch-at 0.5
```

The `--switch-at` flag controls when the model takes over:
- `0.0` = model from the start (fully autonomous)
- `0.5` = expert controls first half, model second half (intent observation)
- `1.0` = expert only (replay baseline)

Outputs per-episode metrics, trajectory plots, and a 3-panel video (dataset recording | expert replay | model inference).

---

## Project Structure

```
ALIGN/
├── data/               # Dataset loading and collation
│   └── align_dataset.py
├── models/
│   ├── align_intention.py    # Main model (ALIGNIntentionModel)
│   ├── align_model.py        # Vision encoder, state encoder
│   ├── intention_encoder.py  # Mamba + SE compression + state modulation
│   ├── intention_head.py     # Diffusion policy head (1D U-Net)
│   └── memory_bank.py        # Episodic memory bank
├── training/
│   └── train_intention.py    # Training loop
├── eval/
│   ├── eval_intention.py              # Offline evaluation
│   └── eval_libero_v4_trajectory.py   # MuJoCo sim evaluation
├── inference/
│   └── align_inference.py    # Real-time inference engine
├── scripts/
│   └── test_checkpoint_inference.py   # Checkpoint compatibility test
└── docs/
    ├── V4_PLAN.md
    ├── V4_SYSTEM_OVERVIEW.md
    └── V4_TRAINING_PSEUDOCODE.md
```

---

## Installation

Clone the repository first:

```bash
git clone https://github.com/NIRUN-Weerawit/ALIGN.git
cd ALIGN
```

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate align
python scripts/check_deps.py
```

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
./setup.sh uv
source .venv/bin/activate
python scripts/check_deps.py
```

The uv option creates a Python 3.12 `.venv` for current LeRobot releases and
installs the CUDA 12.8 PyTorch wheels, `requirements.txt`, and the Mamba CUDA
extensions. Use `./setup.sh uv --minimal` to skip optional data collection
dependencies. Run installation from a machine with a compatible NVIDIA CUDA
driver, CUDA toolkit (`nvcc`), and C++ build tools for the Mamba extensions.
For a different CUDA version, adjust the PyTorch index and matching wheel
versions in `setup.sh` before installing.

Requires PyTorch 2.x, DINOv2, Mamba SSM, and LIBERO (for sim evaluation).

---

## Citation

If you use this code, please cite the project repository.

---

## License

MIT
