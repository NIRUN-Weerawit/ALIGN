# ALIGN: Task Plan & Checklists

**Assistive Latent Intention-Guided Network (ALIGN)**

**Total phases:** 7 (0–6)
**Estimated duration:** 10–16 weeks
**Hardware prerequisites:** GPU (RTX 3090+), VR headset (Quest 2/3), Franka Panda arm, Unitree G1

---

## Phase 0: Development Environment & Simulation Setup (Week 1–2)

### 0.1 Isaac Sim Franka Environment

- [ ] Install Isaac Sim (2023.1+) on workstation
- [ ] Load Franka Panda USD model in Isaac Sim
- [ ] Verify Franka joints move via Python API
- [ ] Place a table in the scene at appropriate height
- [ ] Attach RGB-D camera to Franka wrist link (`panda_hand` or `panda_link8`)
- [ ] Verify camera stream at 224×224×3, 30Hz
- [ ] Add 5–10 YCB objects on table (mug, bowl, can, box, bottle, etc.)
- [ ] Randomize object positions per episode via script
- [ ] Implement `reset_scene()` function for rapid episode cycling
- [ ] Test: run 10 episodes end-to-end without crashes

### 0.2 VR Teleoperation Pipeline

- [ ] Set up Quest 2/3 hand tracking in Isaac Sim
- [ ] Establish streaming connection (ROS 2 or direct TCP) from VR to sim
- [ ] Map human hand pose → Franka EEF pose (position + orientation)
- [ ] Verify hand pose stream at 30Hz with <10ms latency
- [ ] Implement synthetic noise injection (Gaussian + tremor + fatigue)
- [ ] Switchable: raw teleop vs. noise-injected teleop
- [ ] Test: reach and touch objects with noisy teleop — record video
- [ ] Test: smooth teleop (no noise) for expert demonstration collection

### 0.3 Data Recorder

- [x] Define episode data schema (camera frame, noisy pose, smooth pose, gripper, text) — align_data_recorder.py
- [x] Implement `DataRecorder` class recording at 30Hz — align_data_recorder.py
- [x] Store episodes as structured directory (frames/ + data.npz + meta.json) — align_data_recorder.py
- [x] Auto-naming: `episode_{id}_{object}_{variant}` — align_data_recorder.py
- [x] Record object poses alongside trajectory for ground truth generation — align_data_recorder.py
- [x] Record task description text per episode — align_data_recorder.py
- [x] Test: synthetic 50-frame test built into module — align_data_recorder.py (__main__)
- [ ] Implement episode viewer (play back frames + trajectory overlay) — future
- [x] Loader functions: load_episode() and list_episodes() — align_data_recorder.py

### 0.4 Text Annotation Pipeline

- [x] Annotation integrated into episode recording — task description in meta.json
- [x] Multiple text variants generated per episode — collect_episodes.py (get_episode_config)
- [x] Store text in episode metadata — meta.json in DataRecorder
- [ ] Implement text variant sampling for training (random per epoch) — Phase 2
- [ ] Verify: CLIP text encoder produces different embeddings for different objects — Phase 2

### 0.5 Noise Injection (New — Phase 0 enabler)

- [x] Gaussian jitter, tremor, fatigue ramp — align_noise.py
- [x] 3 presets + 6D/7D support — align_noise.py
- [-] Synthetic noise NOT used in default collection (real human teleop noise is sufficient)
      See: DESIGN_DECISIONS.md Decision 13 for rationale

### 0.6 Standalone Collection Script (New)

- [x] VR teleop via MQTT (same as sim_vr_panda_single.py) — collect_episodes.py
- [x] Isaac Sim Franka scene setup — collect_episodes.py
- [x] Object spawning with randomization — collect_episodes.py
- [x] Episode lifecycle: trigger start → record → trigger release → save — collect_episodes.py
- [x] Noise injection before IK (recorded as noisy) — collect_episodes.py
- [x] Original clean pose saved as absolute_pose for smooth reference — collect_episodes.py
- [x] Auto-finalize on timeout — collect_episodes.py
- [x] Button A: cancel/reset, Button B: force-finalize — collect_episodes.py
- [x] CLI: --num-episodes, --noise, --smooth-only, --operator, --output-dir — collect_episodes.py

### Phase 0 Deliverables

- [x] Reproducible Isaac Sim Franka environment with 5+ objects
- [x] VR teleop working with noise injection
- [x] Data recorder saving complete episodes
- [x] Text annotations integrated into episode metadata
- [ ] Episode viewer (low priority)

---

## Phase 1: Data Collection & Ground Truth (Week 2–4)

### 1.1 Training Data Collection

- [ ] Recruit 2–3 operators
- [ ] Train operators: 10 practice episodes each
- [ ] Collect 200–250 episodes (see TRAINING_PIPELINE.md for targets)
- [ ] Balance: equal episodes per object, per operator
- [ ] Record both noisy teleop AND expert smooth teleop for each configuration
- [ ] Annotate each episode with task description text
- [ ] Quality check: no corrupted frames, no dropped timestamps
- [ ] Split: 80% training / 10% validation / 10% test

### 1.2 Ground Truth Smooth Trajectory Generation

- [x] Implement Savitzky-Golay filter for transit phase (window=11, polyorder=3) — generate_ground_truth.py
- [x] Implement approach phase detection (hand < 8cm from object) — generate_ground_truth.py
- [x] Quintic polynomial interpolation for approach phase (no external motion planner needed) — generate_ground_truth.py
- [x] Blend transit + approach with smooth step — generate_ground_truth.py
- [x] Detect and handle orientation via SLERP over keyframes — generate_ground_truth.py
- [x] Compute α_target per timestep: need × capability (capability placeholder=1.0) — generate_ground_truth.py
- [x] Compute Δpose chunk targets (K=5) — generate_ground_truth.py
- [x] Save smooth_poses into existing data.npz, chunk_targets to separate file — generate_ground_truth.py
- [x] Batch processing: process_all() for entire episode directories — generate_ground_truth.py
- [x] CLI: --episode, --input-dir, --output-dir, --visualize, --chunk-size — generate_ground_truth.py
- [x] Built-in synthetic test + verification — generate_ground_truth.py (__main__)
- [ ] Visualize 10 random episodes: overlay noisy vs. smooth vs. α over time — future
- [ ] Fix any episodes where ground truth looks wrong — Phase 1

### 1.3 Held-Out / Novel Object Collection

- [ ] Collect 50–60 test episodes with held-out objects (not in training set)
- [ ] Collect 20 episodes with novel objects (never seen in training)
- [ ] Annotate text appropriately (generic descriptions for novel objects)
- [ ] Verify: held-out episodes have the same schema as training

### Phase 1 Deliverables

- [ ] 200+ training episodes with ground truth smooth trajectories
- [ ] α_target and Δpose_target computed for all timesteps
- [ ] Held-out and novel object test sets ready
- [ ] Data verified — visual overlay inspection, no errors

---

## Phase 2: Contrastive Pretraining (Week 4–5)

### 2.1 Encoder Implementation

- [ ] Load DINOv2 ViT-B (frozen) + vision projection head
- [ ] Load CLIP ViT-B/32 text tower (frozen) + text projection head
- [ ] Implement trajectory Transformer encoder
- [ ] Implement all three projection heads (768→256, 128→256, 512→256)
- [ ] Verify forward pass shapes: z_v (B,256), z_t (B,256), z_text (B,256)
- [ ] Implement L2-normalization after each projection

### 2.2 3-Way InfoNCE Loss

- [ ] Implement pairwise InfoNCE (z_a, z_b)
- [ ] Implement 3-way `contrastive_loss_3way(z_v, z_t, z_text)`
- [ ] Positive pair construction: within-episode window W_pos=5
- [ ] Negative pair construction: in-batch cross-episode
- [ ] Batch sampler: 8 episodes × 8 frames per episode = 64
- [ ] Verify: loss decreases over mini-batches on synthetic data
- [ ] Add temperature τ as learnable parameter (initial 0.07)

### 2.3 Pretraining Loop

- [ ] Implement training loop with AdamW (lr=1e-4, wd=1e-4)
- [ ] Log: loss, avg cos_sim per pair (vt, vl, tl)
- [ ] Validation loop every 5 epochs
- [ ] Early stopping: validation loss plateau for 10 epochs
- [ ] Gradient clipping: max norm 1.0
- [ ] Save checkpoints every 10 epochs

### 2.4 Convergence Checks

- [ ] Verify: within-episode cos_sim > 0.7 for all three pairs
- [ ] Verify: cross-episode cos_sim < 0.2 for all three pairs
- [ ] Verify: text-mismatch (frame="mug", text="bowl") cos_sim < 0.3
- [ ] Verify: novel object + general text cos_sim < 0.4
- [ ] Verify: trajectory-only (random noise) → z_t doesn't align with anything
- [ ] If any check fails: debug (batch size too small? Temperature wrong?)
- [ ] Freeze encoders after successful convergence

### Phase 2 Deliverables

- [ ] Trained 3-modal encoders with verified convergence
- [ ] Frozen backbone ready for head training
- [ ] Alignment scores behaving as expected on held-out data

---

## Phase 3: Head Training (Week 5–6)

### 3.1 Decision Head

- [ ] Implement 3-layer MLP: 774 → 256 → 64 → 1 (+Sigmoid)
- [ ] Implement binary cross-entropy loss
- [ ] Train for 10 epochs (frozen encoders)
- [ ] Log: α_pred vs. α_target scatter plot, BCE loss per epoch
- [ ] Validate: α ≈ 0 for novel objects, α ≈ 0.7–1.0 for clear targets
- [ ] Validate: α drops within 200ms when trajectory starts diverging from text
- [ ] Save checkpoint

### 3.2 Assistant Head

- [ ] Implement 3-layer MLP: 774 → 256 → 128 → K×6 (default K=5)
- [ ] Implement MSE loss over all K outputs
- [ ] Train for 20 epochs (frozen encoders)
- [ ] Log: training MSE, validation MSE per epoch
- [ ] Visualize: predicted chunk vs. ground truth chunk for 10 random timesteps
- [ ] Validate: predicted chunk is smooth (no discontinuities between adjacent timesteps)
- [ ] Save checkpoint

### 3.3 Joint Fine-Tuning

- [ ] Unfreeze both heads
- [ ] Train for 10–20 epochs with combined loss: BCE + 0.5×MSE
- [ ] Monitor: losses shouldn't diverge
- [ ] Early stopping on validation combined loss
- [ ] Save final checkpoint

### 3.4 Ablation Checkpoints

- [ ] Save checkpoint: Decision head only (no Assistant)
- [ ] Save checkpoint: Assistant head only (no Decision, α=1)
- [ ] Save checkpoint: both heads with K=1 (single-step mode, no chunk)
- [ ] Save checkpoint: both heads with K=5 (default)
- [ ] Save checkpoint: 2-modal version (no text encoder) for ablation comparisons

### Phase 3 Deliverables

- [ ] Trained Decision + Assistant heads
- [ ] Ablation checkpoints saved
- [ ] α behavior validated on held-out data
- [ ] Chunk predictions visually verified as smooth

---

## Phase 4: Simulation Evaluation (Week 6–8)

### 4.1 Evaluation Framework

- [ ] Build evaluation harness in Isaac Sim (episodic runs, no human in loop? or with human?)
- [ ] Define metrics:
  - `success_rate`: grasp success (binary)
  - `completion_time`: seconds from start to grasp
  - `jerk`: mean jerk over trajectory (cm/s³)
  - `max_deviation`: max distance from smooth trajectory (cm)
  - `alpha_mean / alpha_var`: mean and variance of α over episode
  - `human_effort`: NASA TLX (for user studies) or proxy (alpha_total = ∫α dt)
- [ ] Log all metrics per episode to CSV
- [ ] Implement metric aggregation (mean, std across episodes)

### 4.2 Ablation Experiments

| Experiment | α | Assistant | Text | Chunk |
|-----------|----|-----------|------|-------|
| 1. Pure teleop (baseline) | 0 | — | — | — |
| 2. Distance-gated | f(d) | ✅ | — | — |
| 3. Always-on assist | 1 | ✅ | — | — |
| 4. Ours (2-modal, no text) | learned | ✅ | — | ✅ |
| 5. Ours (3-modal, full) | learned | ✅ | ✅ | ✅ |
| 6. Ours (K=1, no chunk) | learned | ✅ | ✅ | — |
| 7. Ours (full, novel objects) | learned | ✅ | ✅ | ✅ |

- [ ] Run 50 episodes per experiment configuration
- [ ] Ensure same object placements across runs (seeded)
- [ ] Record all metrics for each experiment
- [ ] Generate comparison plots (bar charts, error bars)

### 4.3 Generalization Tests

- [ ] **Novel objects**: 20 episodes with unseen objects
- [ ] **Held-out objects**: 20 episodes with objects in test set only
- [ ] **Text variation**: 3 text specificity levels per episode ("red mug" vs. "mug" vs. "pick and place")
- [ ] **Wrong text**: 20 episodes with deliberately wrong task description (text says "bowl", operator reaches for mug)
- [ ] **Mid-reach switch**: 20 episodes where operator starts toward one object, switches mid-way
- [ ] **Lighting change**: 20 episodes with different lighting in simulation
- [ ] **Camera angle change**: 20 episodes with slightly perturbed wrist camera angle

### 4.4 Ablation Analysis

- [ ] Generate table: method vs. all metrics
- [ ] Statistical significance: paired t-test between best baseline and ALIGN
- [ ] Plot: α trajectory over time for each scenario (normal, switch, novel)
- [ ] Plot: deviation from smooth trajectory over time, with/without ALIGN
- [ ] Plot: jerk comparison across methods
- [ ] Identify: which ablation(s) cause the biggest performance drop
- [ ] Write up: "which component matters most" analysis

### Phase 4 Deliverables

- [ ] Full evaluation results in simulation
- [ ] Ablation table with statistical significance
- [ ] Generalization results across all test scenarios
- [ ] Written analysis: what works, what doesn't, surprising findings

---

## Phase 5: Real Franka Panda Deployment (Week 8–11)

### 5.1 Hardware Setup

- [ ] Set up Franka Panda with wrist RGB-D camera
- [ ] Install Franka ROS 2 drivers / franka_ros2
- [ ] Verify: joint control at 30Hz from workstation
- [ ] Verify: camera stream at 30Hz
- [ ] Set up table with physical objects matching YCB set (3D-printed or purchased)
- [ ] Safety: emergency stop, velocity limits, force limits
- [ ] Calibrate: camera intrinsics, hand-eye calibration (wrist cam → EEF)
- [ ] Verify: IK solver matches real Franka configuration

### 5.2 Real-to-Sim Gap

- [ ] Port model weights from simulator to real deployment
- [ ] Adjust DINOv2 preprocessing to match real camera (white balance, exposure)
- [ ] Adjust CLIP text encoder (same, no change needed)
- [ ] Test: tensor shapes, forward pass on real camera frames
- [ ] Test: latency — should match simulation (~40ms total)
- [ ] If latency exceeds 50ms: benchmark each component, optimize

### 5.3 Real Robot Evaluation

- [ ] Run same ablation experiments as Phase 4 (50 episodes each)
- [ ] Record same metrics (success_rate, completion_time, jerk, etc.)
- [ ] Add: real-world specific metrics (grasp success rate, collision count)
- [ ] Verify: sim-to-real transfer works (metrics should be similar)
- [ ] Debug any degradation: identify root cause, adjust preprocessing
- [ ] Iterate: fine-tune vision projection head on ~10 real frames if needed

### 5.4 User Study (Real Robot)

- [ ] Recruit N=5–8 operators (mix of experienced and novice)
- [ ] Define study protocol:
  - 5 practice trials with pure teleop (baseline)
  - 5 practice trials with ALIGN
  - 10 test trials per condition (randomized order)
  - NASA TLX survey after each condition
  - Post-study interview: qualitative feedback
- [ ] Implement: condition blinding (operator doesn't know which mode is active)
- [ ] Collect: success rate, time, jerk, TLX per operator per condition
- [ ] Analyze: paired t-test ALIGN vs. pure teleop on all metrics
- [ ] Interview quotes: collect for paper

### Phase 5 Deliverables

- [ ] ALIGN running on real Franka at 25Hz
- [ ] Sim-to-real validation results
- [ ] User study results (N=5–8)
- [ ] Statistical significance of improvements
- [ ] Qualitative feedback from operators

---

## Phase 6: G1 Humanoid Port (Week 11–14)

### 6.1 Arm-Only Setup

- [ ] Set up G1 in lab with locked lower body (zero gait)
- [ ] Install G1 SDK / ROS 2 bridge
- [ ] Mount RGB-D camera at shoulder/chest height (operator's egocentric view)
- [ ] Verify: arm joint control at 30Hz
- [ ] Verify: camera stream at 30Hz
- [ ] Calibrate: camera intrinsics, camera → torso transform
- [ ] Test: IK matches G1 arm kinematics

### 6.2 Model Transfer

- [ ] Load same model weights as Franka (no retraining needed — arm-agnostic)
- [ ] Adjust: vision encoder preprocessing if camera differs
- [ ] Test: forward pass, latency budget (~40ms target)
- [ ] If latency >50ms: switch to DINOv2 ViT-S (smaller, faster, slightly worse features)
- [ ] Benchmark end-to-end latency: camera → control → motor command

### 6.3 G1 Evaluation

- [ ] Run core ablation (pure teleop vs. ALIGN) — 30 episodes each
- [ ] Run generalization tests: novel objects, wrong text, mid-reach switch
- [ ] Record: success_rate, completion_time, jerk
- [ ] Compare: Franka vs. G1 metrics — identify humanoid-specific challenges
- [ ] Document: any issues with base stability during arm movement

### 6.4 G1 User Study (Optional — If Time Permits)

- [ ] Study protocol same as Franka user study
- [ ] Recruit N=3–5 operators
- [ ] Collect same metrics: success rate, TLX, time, jerk
- [ ] Compare across platforms: Franka vs. G1 assist effectiveness
- [ ] This becomes a strong paper result (cross-platform validation)

### Phase 6 Deliverables

- [ ] ALIGN running on G1 at 25Hz
- [ ] Cross-platform validation results (Franka + G1)
- [ ] G1-specific challenges documented
- [ ] Optional: G1 user study results

---

## Phase 7: Paper Writing & Submission (Ongoing from Week 8)

### 7.1 Paper Outline

- [ ] Title + abstract (draft early, iterate often)
- [ ] Introduction (1 page)
- [ ] Related work (1 page)
- [ ] Method — ALIGN architecture (2 pages)
- [ ] Experiments (3 pages) — setup, ablations, generalization, user study
- [ ] Results (1 page)
- [ ] Discussion + limitations (0.5 page)
- [ ] Conclusion (0.5 page)
- [ ] References

### 7.2 Figures to Prepare

- [ ] Architecture diagram (trilinear, 3-modal)
- [ ] Ablation results bar chart (method vs. metrics)
- [ ] α trajectory over time for normal / switch / novel scenarios
- [ ] Smooth trajectory vs. noisy trajectory vs. ALIGN output (overlay plot)
- [ ] User study results (TLX box plot)
- [ ] Generalization results table
- [ ] Real robot setup photo (Franka + G1)
- [ ] Sim-to-real comparison scatter plot

### 7.3 Writing Milestones

- [ ] **First draft** (messy, all sections filled)
- [ ] **Review with co-authors** (if any)
- [ ] **Revision cycle 1**: fix experiments, add missing analysis
- [ ] **Revision cycle 2**: figures, polish writing, clarity
- [ ] **Internal read**: have someone outside the project read for clarity
- [ ] **Final polish**: consistent notation, cross-references, supplementary material
- [ ] **Format check**: venue template, page limit, bibliography style

### 7.4 Supplementary Material

- [ ] Video of ALIGN in action (Franka + G1)
- [ ] Code repository (GitHub)
- [ ] Trained model weights
- [ ] Dataset description (DOI or static link)
- [ ] Detailed ablation results (all runs, not just averages)
- [ ] User study raw data (anonymized)

### 7.5 Submission

- [ ] Target venue: ICRA 2026 (deadline ~Sep 2025) or RA-L
- [ ] Check page limit, format, author guidelines
- [ ] Upload to conference system
- [ ] Prepare response to reviewers (if desk rejected, salvage for next venue)

### Phase 7 Deliverables

- [ ] Conference-ready paper
- [ ] Supplementary video, code, data
- [ ] Submitted to target venue

---

## Summary Timeline

```
Week 1–2    Phase 0: Sim setup + VR teleop
Week 2–4    Phase 1: Data collection + ground truth
Week 4–5    Phase 2: Contrastive pretraining
Week 5–6    Phase 3: Head training
Week 6–8    Phase 4: Simulation evaluation
Week 8–11   Phase 5: Real Franka deployment + user study
Week 11–14  Phase 6: G1 port + evaluation
Week 8–14   Phase 7: Paper writing (parallel)
```

## Dependencies Between Phases

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6
                 │                                                 │
                 └────────────────── Phase 7 ◀──────────────────────┘
                                     (paper writing, parallel from Phase 4)
```

## Critical Path Items (Longest Chain)

1. Phase 0: Isaac Sim Franka + VR teleop working → **2 weeks**
2. Phase 1: 200 episodes collected → **2 weeks**
3. Phase 2: Contrastive pretraining → **1 week**
4. Phase 3: Head training → **1 week**
5. Phase 4: Simulation evaluation complete → **2 weeks**
6. Phase 5: Real Franka study complete → **3 weeks**
7. Phase 7: Paper written + submitted → **6 weeks (parallel)**

**Total critical path: ~11 weeks if paper writing starts at Week 8 (Phase 4 complete) and submission is at Week 14.**

## Risk Items

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VR teleop latency >30ms | Medium | High — can't collect clean data | Use keyboard/joint GUI as fallback for Phase 1 |
| Contrastive pretraining doesn't converge | Low | High — architecture failure | Smaller batch? W_pos too strict? Debug with 10 episodes first |
| Sim-to-real gap large | Medium | Medium — need recalibration | Test IRL as early as Week 5 with single episodes |
| User study recruits unavailable | Medium | Medium — weakens paper | Start recruiting in Week 6, run with lab members as fallback |
| G1 SDK unavailable / broken | Low | High — Phase 6 blocked | Phase 6 is optional; paper works with Franka only |
| DINOv2 too slow on Jetson Orin | Medium | Low — can switch to ViT-S | Benchmark early (Phase 4 at latest) |
| Chunk output oscillates in real robot | Low | Medium | Clamp per-step Δpose, reduce α, or reduce K |

## Quick-Start Path (If You Want Results Fast)

```
Skip to Phase 5 as fast as possible:
  1. Phase 0.1 (Franka sim) + Phase 0.3 (data recorder)  → 1 week
  2. Phase 1.1 (100 episodes, not 200)                     → 1 week
  3. Phase 1.2 (ground truth with SavGol only, no planner) → 2 days
  4. Phase 2 (contrastive, 50 epochs)                      → 1 week
  5. Phase 3 (heads, 30 epochs)                            → 3 days
  6. Phase 4 (3 ablations only: baseline / distance / full) → 1 week
  7. Phase 5 (real Franka, no user study, 20 episodes)     → 1 week
  Total: ~5 weeks — enough for a workshop paper or proof-of-concept
```