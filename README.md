# ALIGN: Assistive Latent Intention-Guided Network

**Assistive Latent Intention-Guided Network (ALIGN)** is a shared autonomy framework for assistive teleoperation of robotic manipulators. It learns *when* to assist (decision) and *what* to do (assistant) from a single shared **3-modal** visual-motion-language representation — using contrastive vision-trajectory-language alignment as a natural gating signal.

## Getting Started

```bash
# 1. Clone and install dependencies
git clone <repo-url> && cd ALIGN

# Option A — Conda (recommended for GPU training)
conda env create -f environment.yml
conda activate align

# Option B — pip / venv
python3 -m venv align-env
source align-env/bin/activate
pip install -r requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# Option C — auto-setup script (detects conda/pip)
./setup.sh

# 2. Run streaming pretraining (zero disk — pulls data from Hugging Face Hub)
python training/pretrain_streaming.py --epochs-pretrain 10
```

See [`environment.yml`](environment.yml) for the full conda env, [`requirements.txt`](requirements.txt) for pip, or [`setup.sh`](setup.sh) for the auto-installer.

### Core Idea

A 3-way contrastive pretraining step aligns vision embeddings (from egocentric camera frames), trajectory embeddings (from noisy teleoperation poses), and language embeddings (from task descriptions) into a shared space. The pairwise cosine similarities between these embeddings serve as a confidence score that:

1. **Gates assistance** — when all three modalities agree (vision, motion, and task description), assistance activates. When any disagrees (wrong object, uncertain intention, novel scene), it defers to the human.
2. **Disambiguates objects** — text explicitly specifies the target ("the left mug", "the red one") when vision alone can't distinguish.
3. **Detects out-of-distribution inputs** — low alignment across any pair → safe fallback to pure teleoperation.
4. **Filters tremor** — the trajectory encoder learns to suppress noise that doesn't correlate with visual features or task semantics.

### Implementation Status

| Component | Status |
|-----------|--------|
| Phase 0: Data Collection | ✅ Complete — `scripts/align_data_recorder.py`, `collect_episodes.py`, `align_noise.py` |
| Phase 0: Ground Truth | ✅ Complete — `scripts/generate_ground_truth.py` (SavGol + Quintic/DMP/CHOMP) |
| Model Architecture | ✅ Complete — `models/align_model.py` (DINOv2 + CLIP + Transformer + dual heads) |
| Contrastive Loss | ✅ Complete — `training/contrastive_loss.py` (3-way InfoNCE) |
| Training Pipeline (local) | ✅ Complete — `training/pretrain.py`, `training/train_heads.py` |
| Training Pipeline (streaming) | ✅ Complete — `training/pretrain_streaming.py` |
| Training Pipeline (full) | ✅ Complete — `training/train_full_pipeline.py` |
| Inference Runtime | ✅ Complete — `inference/align_inference.py` (30Hz loop) |
| Open Dataset Adapters | ✅ Complete — `data/open_dataset.py` (Robomimic, DROID, Bridge, LeRobot v3) |
| DMP/CHOMP Planners | ✅ Complete — `scripts/align_dmp.py`, `scripts/align_chomp.py` |
| Phase 1: Data Collection | ⬜ Next |
| Phase 2: Contrastive Pretraining | ⬜ Needs data or `pip install lerobot` |
| Phase 3: Head Training | ⬜ Depends on Phase 2 |

### Verified Open Datasets

| Dataset | Robot | Frames | EEF Pose | Text | Wrist Camera | Match |
|---------|-------|--------|----------|------|-------------|-------|
| **nvidia/LIBERO_LeRobot_v3** | Franka Panda (sim) | 130K eps | ✅ 8D [x,y,z,ax3,grip2] | ✅ Multi-step tasks | ✅ `observation.images.wrist_image` (256×256) | **Perfect** |
| nvidia/BridgeData2_LeRobot_v3 | WidowX 250 (real) | 50K+ traj | ✅ EEF | ✅ Language | ⚠️ Front view | Good |

LIBERO is the ideal match: same Franka Panda robot, egocentric wrist camera, rich language tasks ("put the white mug on the left plate and put the yellow and white mug on the right plate"), 20fps video in AV1 codec. Requires `lerobot` + `torchcodec` for streaming decode.

### Three Training Pathways

```bash
# 1. STREAMING (zero disk, recommended for pretraining)
# Requires: pip install lerobot torchcodec
python training/pretrain_streaming.py --epochs-pretrain 50

# 2. LOCAL DATA (own Phase 1 collection + converted open datasets)
python -m data.open_dataset --dataset robomimic --data-dir ./robomimic_data --task lift
python training/pretrain.py --data align.h5 --epochs 50
python training/train_heads.py --data align.h5 --pretrained checkpoints/pretrain/best.pt

# 3. FULL PIPELINE (open datasets → synthetic noise → heads)
python training/train_full_pipeline.py --robomimic-dir ./robomimic_data

# Inference
python inference/align_inference.py --checkpoint checkpoints/heads/joint_best.pt --task "pick up the red mug"
```

### Key Design Decisions

- **Hybrid shared encoder** — single vision + trajectory + text encoder with two lightweight heads (Decision, Assistant). Text is computed once per task (~5ms) and cached. Runs at 25Hz on G1's Jetson Orin.
- **3-way contrastive alignment** — InfoNCE on all three pairs (vision↔trajectory, vision↔language, trajectory↔language). Gating uses min of all three for maximum safety.
- **No object detector** — frozen DINOv2 + CLIP encoders work on novel objects without retraining.
- **No hand-tuned α** — the contrastive alignment provides a principled, data-driven gating signal from three modalities.
- **Text-conditioned corrections** — same visual scene + same motion but different task description → different Δpose. The assistant is task-aware.
- **Graceful degradation** — any alignment score drops on OOD inputs → min drops → α drops → human stays in control.

### Text Specificity Spectrum

The system does NOT require strict directions. It works across the full spectrum:
- **Specific** ("pick up the red mug") → highest α, object disambiguation via CLIP
- **Descriptive** ("pick up the mug") → high α if only one mug is visible
- **Neutral** ("pick and place") → moderate α, general smoothing, no disambiguation
- **None** → vision-only fallback, cos_sim_vt still gates

Training uses multiple text variants per episode so the model learns to calibrate confidence to specificity. Neutral text helps less because it *should* — the system is appropriately uncertain.

### Future Work

- **Enriched trajectory encoder input** — expand from (K,6) EEF pose to (K,13) by adding
  EEF velocity, angular velocity, and gripper state. Already recorded, trivially computed.
  Gives the Transformer explicit motion dynamics without the joint-level noise and 60% zero-padding
  of RDT-1B's full 128-dim unified action space.
- **DMP-based approach phase** — replace quintic interpolation with Dynamic Movement Primitives.
  One expert demo per object type → learned forcing term encodes approach style (arc, wrist rotation).
  DMP adapts to arbitrary start/goal poses via attractor scaling. Human-like paths by construction,
  deterministic, instant in deployment. Bi-RRT + smoothing as collision-fallback only.

### Development Plan

| Phase | Platform | Scope | Status |
|-------|----------|-------|--------|
| 0 | Isaac Sim + Franka Panda | Simulated pick-and-place, data collection, offline training | ✅ Complete |
| 1 | Franka Panda (real) | Real hardware validation, user studies | ⬜ |
| 2 | Unitree G1 arm-only | Full humanoid, fixed-base pick-and-place | ⬜ |
| 3 | Unitree G1 full-body | Add locomotion coordination | ⬜ |

### Venue Target

ICRA 2026 / RA-L. Core contribution: using contrastive vision-trajectory alignment as a learned assist gating mechanism — the first unified framework where the same embedding space drives both the decision to assist and the correction itself.

### Directory Structure

```
ALIGN/
├── README.md                  ← This file
├── models/
│   └── align_model.py         ← DINOv2 + CLIP + Transformer + Decision + Assistant heads
├── training/
│   ├── contrastive_loss.py    ← 3-way InfoNCE loss
│   ├── pretrain.py            ← Contrastive pretraining (HDF5 data)
│   ├── train_heads.py         ← Staged head training (Decision → Assistant → Joint)
│   ├── pretrain_streaming.py  ← Zero-disk streaming pretraining from LeRobot v3 Hub
│   └── train_full_pipeline.py ← Full pipeline: convert → noise → pretrain → heads
├── inference/
│   └── align_inference.py     ← 30Hz runtime: vision+traj+text → α + Δpose
├── data/
│   ├── align_dataset.py       ← HDF5 converter + PyTorch Dataset + collate
│   └── open_dataset.py        ← Adapters for Robomimic, DROID, Bridge, LeRobot v3
├── scripts/
│   ├── align_data_recorder.py ← Episode recording (frames + poses + text + metadata)
│   ├── align_noise.py         ← Noise injection: Gaussian, tremor, fatigue ramp
│   ├── collect_episodes.py    ← Isaac Sim Franka + VR teleop + data collection
│   ├── generate_ground_truth.py ← SavGol + Quintic/DMP/CHOMP + α/Δpose targets
│   ├── align_dmp.py           ← DMP approach planner (Ijspeert 2013)
│   └── align_chomp.py         ← CHOMP trajectory optimizer (Ratliff 2009)
```
