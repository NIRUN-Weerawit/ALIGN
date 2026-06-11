# ALIGN: Initial Plan

**Assistive Latent Intention-Guided Network (ALIGN)**

## Problem

Teleoperating a humanoid robot (Unitree G1) for pick-and-place tasks is challenging because human hand motion is inherently noisy, oscillatory, and imprecise. VR-based teleoperation amplifies this: hand tremor, jitter from tracking, and the absence of haptic feedback all contribute to unstable robot arm motion. Yet the human's motion *is* meaningful — they know what they want to do, but their motor output is imperfect.

Running the noisy commands directly through the robot's IK produces jerky, inefficient, and sometimes failed grasps.

## Goal

Develop an **assistive teleoperation system** that:

1. **Understands the operator's intention** — what object are they reaching for, and what grasp is appropriate
2. **Corrects the trajectory** — smoothening, snapping the gripper into position near the target, adjusting speed
3. **Gates assistance by confidence** — only intervenes when it is confident about the intention, otherwise stays hands-off
4. **Degrades gracefully** — novel objects, unclear intent, or changing goals → defers to the human

## Core Insight

The operator's visual feed (egocentric camera on the robot's wrist or head) contains rich information about their intended target. When the operator reaches toward an object, the camera sees that object. The correlation between **what the camera sees** and **how the hand moves** is a natural signal for confidence in the operator's intention.

**Contrastive learning** can align vision and motion into a shared embedding space, where the cosine similarity between the two embeddings directly reflects how well the visual scene "explains" the current motion. This similarity becomes a learned, continuous, OOD-aware gating signal — eliminating hand-tuned thresholds.

## High-Level Approach

### Contrastive Vision-Trajectory Alignment

```
Camera → Vision Encoder → z_v
                              → cos_sim(z_v, z_t) → α (assist confidence)
Trajectory → Traj Encoder → z_t
```

- **Positive pairs**: camera frame + trajectory window when the human is moving toward the visible object
- **Negative pairs**: frame + trajectory from different episodes, or frame + random motion
- **InfoNCE loss**: forces the encoders to learn that the same goal-directed behavior produces similar embeddings across both modalities

### Shared Encoder, Dual Heads

Rather than training two separate models (decision + assistant), we use:

- **One frozen vision encoder** (DINOv2) producing z_v
- **One trainable trajectory encoder** (Transformer) producing z_t
- **Decision Head (MLP)**: [z_v, z_t, cos_sim] → α ∈ [0,1]
- **Assistant Head (MLP)**: [z_v, z_t, noisy_pose] → Δpose (correction offset)

Both heads are lightweight and train on the same frozen backbone.

### Final Command Blending

```
final_command = raw_human_pose + α · Δpose
```

When α = 0, the system is transparent. When α = 1, full correction is applied. α is a continuous blend.

## Why This Works

| Situation | cos_sim | α | Behavior |
|-----------|---------|---|----------|
| Familiar object, clear reach | High | High | Strong assist |
| Novel object, uncertain reach | Low | Low | Hands off |
| Human sees one object but reaches toward another | Low | Low | Hands off |
| Mid-reach target switch | Drops | Drops | Human regains control |
| Tremor noise (doesn't correlate with vision) | Suppressed by encoder | — | Encoder learns to ignore it |

## Development Roadmap

| Phase | What | Duration |
|-------|------|----------|
| **0** | Isaac Sim Franka Panda environment, VR teleop data collection pipeline | 1-2 weeks |
| **1** | 200+ teleop episodes, ground truth generation (SavGol + motion planner) | 1-2 weeks |
| **2** | Contrastive pretraining of vision + trajectory encoders | 1 week |
| **3** | Train Decision + Assistant heads, evaluate in sim | 1-2 weeks |
| **4** | Real Franka deployment, user studies, ablation experiments | 2-3 weeks |
| **5** | Port to G1 arm, generalizaton experiments | 2-3 weeks |
| **6** | Paper writing, ICRA/RA-L submission | Ongoing |

## Starting Point

Phase 0 starts on a **Franka Panda arm in Isaac Sim** (fixed mount, wrist camera, 5-10 YCB objects on a table). The arm reduces DOF from 30+ to 7, simplifies IK, and removes locomotion complexity. The core contribution (learned assist gating via contrastive alignment) is identical on Franka and G1.