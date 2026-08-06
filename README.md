# ALIGN: Assistive Latent Intention-Guided Network

**ALIGN** is a shared autonomy framework for robotic manipulation. It learns to infer human intent from visual observations and robot state, then assists by predicting corrective actions during teleoperation.

The core idea: a human leads, the model observes. Over time, the model builds an understanding of the task through temporal context (Mamba), explicit intent tokens, and an episodic memory bank. It then generates smooth, task-appropriate actions via a diffusion policy head.

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

When `--use-history` is enabled, CLS tokens and states are fed through a Mamba SSM for temporal recurrence:

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
| `--head-type` | `diffusion` | Head architecture |
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
    --no-use-history
```

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

```bash
git clone https://github.com/NIRUN-Weerawit/ALIGN.git && cd ALIGN

# Conda (recommended)
conda env create -f environment.yml
conda activate align

# Verify
python scripts/check_deps.py
```

Requires PyTorch 2.x, DINOv2, Mamba SSM, and LIBERO (for sim evaluation).

---

## Citation

If you use this code, please cite the project repository.

---

## License

MIT
