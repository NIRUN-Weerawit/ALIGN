# ALIGN: Assistive Latent Intention-Guided Network

**ALIGN** is a shared autonomy framework for assistive teleoperation of robotic
manipulators. It learns a shared **3-modal** vision–trajectory–language
representation and uses it to drive two heads:

- a **Decision head** that decides *when* to assist (the gating signal α)
- an **Assistant head** that decides *what* to do (a chunk of corrective Δposes)

The framework additionally supports an **action-conditioned world model** and a
**GAIL discriminator** for counterfactual α computation (a separate gating
paradigm from the prediction-error α), and ships a **deployment-time
calibrator** for new robots/cameras.

> **Status:** End-to-end code pipeline runs on LIBERO (Franka Panda, 6-DoF
> OSC_POSE) and supports both streaming and HDF5 training. Real-hardware
> validation is the next milestone.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/NIRUN-Weerawit/ALIGN.git && cd ALIGN

# Option A — Conda (recommended for GPU training)
conda env create -f environment.yml
conda activate align

# Option B — pip / venv
python3 -m venv align-env && source align-env/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 torchcodec==0.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install xformers --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Option C — auto-setup (detects conda/pip)
./setup.sh

# 2. Sanity check
python scripts/check_deps.py

# 3. Run the streaming pretraining + head pipeline (zero local disk)
python training/pretrain_streaming.py \
    --epochs-pretrain-encoder 40 \
    --epochs-pretrain-mixer 10 \
    --epochs-heads 30
```

## Core Idea

ALIGN's three encoders are pretrained with a **3-way InfoNCE** objective that
aligns vision, trajectory, and task-language into one shared space. From that
shared representation, two heads consume it:

```
              ┌──────────────┐
   frame ────►│ Vision       │ (DINOv2, frozen)
              ├──────────────┤
   pose K ───►│ Trajectory   │ (Transformer, trained)
              ├──────────────┤
   task  ────►│ Text         │ (CLIP, frozen)
              └──────┬───────┘
                     │ 3-way cross-attention mixer
                     ▼
        ┌────────────┴────────────┐
        │                         │
  Decision head              Assistant head
  (when to assist)           (what to do)
        │                         │
        ▼                         ▼
   α ∈ [0, 1]              chunk of K Δposes
```

**Inference blending rule** (default, current production code):

```python
final_pose = (1 - α) * human_pose + α * (human_pose + chunk[0])
           = human_pose + α · chunk[0]
```

α is small when any modality disagrees (novel scene, wrong task, OOD input) →
graceful degradation to pure teleoperation.

## Two α-pipelines (code supports both)

| Pipeline | Source of α | Training signal | Status |
|---|---|---|---|
| **Prediction-error α** | `FuturePredictionHead` (Decision) | MSE on predicted future (z_v, z_s) | ✅ Default; ships in `train_heads.py` |
| **Counterfactual α** | `WorldModel` + `ValueHead` + `GAIL` discriminator | TD(λ) returns on GAIL reward | ✅ Code complete; runs after world model + GAIL trained |

The two are alternatives, not stacked. Counterfactual α requires the
world model + GAIL to be trained first (`train_world_model.py`,
`train_gail.py`, `train_value.py`); prediction-error α needs only the
encoder pretraining + head stages.

## Multi-Camera Support

The vision encoder fuses **V camera views** through a learned linear layer
(`Linear(V·256 → 256)`). Single-camera input is a special case (V=1) and
requires no code change. All training/eval scripts accept `--cameras wrist
agent` (or any combination of LIBERO camera keys).

**Important:** Camera *selection* (wrist vs. front) at training time is
sensitive to the `PYTHONNOUSERSITE` env flag (see `FIXES.md`). Be consistent
between training and eval, or the frozen DINOv2 features will not match.

## Repository Layout

```
ALIGN/
├── README.md                     ← This file
├── FIXES.md                      ← Bug-by-bug fix log (grep-friendly)
│
├── models/
│   ├── align_model.py            ← DINOv2 + Transformer + CLIP, 3 encoders + 2 heads
│   ├── cross_attention_mixer.py  ← Bidirectional gated cross-attention (Flamingo-style)
│   ├── world_model.py            ← Action-conditioned single-step transition f(s,a)→s'
│   ├── value_head.py             ← V(s) head, TD(λ) on GAIL reward
│   ├── gail_discriminator.py     ← D(s,a) expert-vs-rollout classifier
│   └── sinusoidal_pos_emb.py
│
├── training/
│   ├── contrastive_loss.py       ← 3-way InfoNCE + helpers
│   ├── pretrain.py               ← Contrastive pretraining (HDF5 data)
│   ├── pretrain_streaming.py     ← Zero-disk streaming pretraining (LeRobot Hub)
│   ├── train_heads.py            ← Decision + Assistant head training (HDF5/streaming)
│   ├── train_world_model.py      ← World model f(s,a)→s'
│   ├── train_gail.py             ← GAIL discriminator
│   ├── train_value.py            ← Value head with TD(λ) + PPO/DQN/DDPG stability tricks
│   ├── train_full_pipeline.py    ← End-to-end: open dataset → noise → pretrain → heads
│   └── wandb_utils.py
│
├── data/
│   ├── align_dataset.py          ← HDF5 dataset + collate
│   ├── open_dataset.py           ← Adapters for Robomimic, DROID, Bridge, LeRobot v3
│   └── (HDF5 cache written here by scripts/decode_libero_to_hdf5.py)
│
├── inference/
│   ├── align_inference.py        ← 30Hz control loop
│   └── deployment_calibrator.py  ← 5-10s axis/scale/hand-eye calibration per session
│
├── scripts/
│   ├── decode_libero_to_hdf5.py  ← LIBERO LeRobot → HDF5 (faster than streaming)
│   ├── align_data_recorder.py    ← Episode recording (frames + poses + text)
│   ├── align_noise.py            ← Gaussian + tremor + fatigue noise injection
│   ├── collect_episodes.py       ← Isaac Sim Franka + VR teleop
│   ├── generate_ground_truth.py  ← SavGol + Quintic/DMP/CHOMP ground truth
│   ├── align_dmp.py              ← DMP approach planner (Ijspeert 2013)
│   ├── align_chomp.py            ← CHOMP trajectory optimizer (Ratliff 2009)
│   ├── optuna_search.py          ← H100-scale hyperparam search (encoders/decision/assistant)
│   ├── cache_libero_meta.py      ← Cache LIBERO task descriptions locally
│   ├── replay_libero_in_sim.py   ← Replay a trajectory in LIBERO sim
│   └── check_deps.py             ← Verify env is trainable
│
├── eval/
│   ├── eval_contrastive.py       ← InfoNCE alignment scores
│   ├── eval_world_model.py       ← World-model rollouts (incl. copy-baseline diagnostic)
│   ├── eval_gail.py              ← Discriminator quality
│   ├── eval_value.py             ← V(s) fit
│   ├── eval_heads.py             ← Decision + Assistant accuracy
│   ├── eval_assistant_head.py    ← Per-timestep assistant Δpose error
│   ├── eval_libero.py            ← LIBERO success-rate eval
│   ├── eval_libero_trajectory.py ← Replay-and-compare (no_align vs with_align)
│   ├── eval_alpha.py             ← α signal analysis
│   ├── compute_alpha.py          ← One-off α computation
│   ├── flip_sim_frames.py        ← Fix LIBERO upside-down camera frame
│   ├── print_sample_images.py    ← QA: print dataset sample images
│   ├── verify_action_to_pose_mapping.py ← Sanity-check action semantics
│   └── test_calibrator_lerobot.py ← Unit test the deployment calibrator
│
└── docs/                         ← Local-only design notes (not pushed to git)
```

## Training & Evaluation Pathways

### A. Streaming (zero local disk)

```bash
# Full pipeline
python training/pretrain_streaming.py \
    --epochs-pretrain-encoder 40 --epochs-pretrain-mixer 10 --epochs-heads 30

# Phase 1 only (encoder + mixer)
python training/pretrain_streaming.py --stages pretrain

# Phase 2 only (heads, given pretrain ckpt)
python training/pretrain_streaming.py \
    --stages heads --pretrained ./checkpoints/streaming/pretrain/best.pt
```

### B. HDF5 (faster training, recommended for H100)

```bash
# 1. Convert LIBERO LeRobot → HDF5 (5-10 min, then forever cached)
python scripts/decode_libero_to_hdf5.py \
    --data-dir ~/.cache/huggingface/lerobot/nvidia/LIBERO_LeRobot_v3/libero_10

# 2. Pretrain + heads on HDF5
python training/pretrain.py --data h5_data/libero_10.h5
python training/train_heads.py \
    --data h5_data/libero_10.h5 \
    --pretrained checkpoints/pretrain/best.pt
```

> **Why HDF5?** LeRobot's MP4 decoders don't support multi-worker
> DataLoaders. Pre-decoding to HDF5 enables `num_workers > 0` and ~3-5×
> faster training. See `FIXES.md` and the `align_data` H5 path.

### C. Counterfactual α (optional, after A or B)

```bash
python training/train_world_model.py --data h5_data/libero_10.h5
python training/train_gail.py       --data h5_data/libero_10.h5
python training/train_value.py      --data h5_data/libero_10.h5
```

### D. Inference

```bash
# 30Hz runtime
python inference/align_inference.py \
    --checkpoint checkpoints/heads_libero/best.pt \
    --task "pick up the red mug"

# New robot/camera: 5-10s calibration first
python -c "from inference.deployment_calibrator import DeploymentCalibrator; \
    c = DeploymentCalibrator(); c.run(robot, obs_fn); c.save('calib.yaml')"
```

### E. Optuna hyperparameter search

```bash
# Full search (H100, 30 trials)
python scripts/optuna_search.py --n-trials 30 --epochs 30

# Single-stage search
python scripts/optuna_search.py \
    --search-decision --skip-encoder-training \
    --encoder-checkpoint checkpoints/pretrain/run_3/best.pt
```

## Key Design Decisions

- **Frozen DINOv2 + CLIP backbones** — no retraining, works on novel objects
  out of the box. DINOv2 ViT-B/14 for vision, CLIP ViT-B/32 for text.
- **Cross-attention mixer, not concat** — Flamingo-style gated cross-attention
  (identity-initialized gates near sigmoid(1.0) ≈ 0.7) so pretrained features
  pass through cleanly during early training.
- **Two head architectures** — both MLP (default) and Transformer (K-window)
  variants for Decision and Assistant heads. Transformer variant uses
  per-step loss weighting and reads K past frames separately
  (`encode_raw_vision_window`).
- **Text is computed once per task** — ~5 ms, cached. Enables 25–30 Hz control
  on Jetson Orin.
- **No hand-tuned α** — every gating signal is learned. The decision head
  derives α from world-model prediction error; the value head derives it
  from TD(λ) returns on GAIL reward.
- **Pose-relative goals, not actions** — `chunk[k] = where EEF should be at
  step k+1` relative to current pose. This is a *planning* quantity, not a
  recovery correction, and composes cleanly with the α-blend at inference.

## Verified Datasets

| Dataset | Robot | Frames | EEF | Text | Wrist | Status |
|---|---|---|---|---|---|---|
| **nvidia/LIBERO_LeRobot_v3** | Franka Panda (sim) | 130K eps | ✅ 8D | ✅ Multi-step | ✅ 256×256 | Primary |
| nvidia/BridgeData2_LeRobot_v3 | WidowX 250 (real) | 50K+ traj | ✅ | ✅ | ⚠️ Front | Adapter ready |
| Robomimic (lift/can/pick_place) | MuJoCo sim | varies | ✅ | partial | ⚠️ | Adapter ready |
| DROID | Franka (real) | large | ✅ | ✅ | ✅ | Adapter ready |

LIBERO is the primary target: same Franka, egocentric wrist, rich language,
20 fps AV1. Requires `lerobot` + `torchcodec` for streaming; pre-decode to
HDF5 for production training.

## Text Specificity Spectrum

ALIGN calibrates confidence to text specificity during training by sampling
multiple variants per episode:

| Text style | Expected α | Why |
|---|---|---|
| Specific ("pick up the red mug") | highest | Object disambiguated by CLIP |
| Descriptive ("pick up the mug") | high (if unambiguous) | Single mug visible → no ambiguity |
| Neutral ("pick and place") | moderate | General smoothing, no disambiguation |
| None (text-free mode) | cos_sim_vt-driven | Vision-trajectory only |

## Development Plan

| Phase | Platform | Scope | Status |
|---|---|---|---|
| 0 | Isaac Sim + Franka | Sim data collection, offline training | ✅ Complete |
| 0.5 | LIBERO LeRobot v3 | Open-dataset validation, trajectory-replay eval | ✅ Complete |
| 1 | Franka (real) | Real hardware, user studies | ⬜ Next |
| 2 | Unitree G1 arm-only | Humanoid, fixed base | ⬜ |
| 3 | Unitree G1 full-body | + locomotion coordination | ⬜ |

## Venue Target

ICRA 2026 / RA-L. Core contribution: a shared vision–trajectory–language
embedding that simultaneously drives the **gating decision** (when to assist)
and the **assistive action** (what to do) — plus an action-conditioned world
model + GAIL-trained value head as an alternative counterfactual gating
paradigm.

## Known Quirks (read before debugging)

- `PYTHONNOUSERSITE=1` is required when running in the `align` conda env to
  prevent the user-site pip from shadowing torch.
- Camera selection (wrist vs. agent) is sensitive to `PYTHONNOUSERSITE` — be
  consistent between training and eval.
- LIBERO MP4 decoders deadlock at `num_workers > 0` → use the HDF5 pipeline
  for multi-worker training.
- HDF5 pose field: legacy `noisy_poses` (misnomer-clean) and new `poses` are
  both supported transparently.
- Eval reads `poses` from sim observations, not `noisy_poses` from the
  replay buffer — only actions are noised.
- `docs/` is local-only and intentionally not tracked in git.

## Citation

Internal project, no paper yet. Pin to commit SHA when referencing.
