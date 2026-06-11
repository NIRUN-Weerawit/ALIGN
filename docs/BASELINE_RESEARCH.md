# ALIGN: Baseline & Competitor Research

Comprehensive survey of related works for comparison, baselines, and positioning.
Updated: June 2026.

---

## 1. FOUNDATIONAL SHARED AUTONOMY (Must Cite)

### Javdani et al. 2017 — "Shared Autonomy via Hindsight Optimization for Teleoperation and Teaming"
- arXiv: 1706.00155 | ~251 citations
- **Core idea**: Bayesian inference over discrete goal sets to infer human intention.
  Uses hindsight optimization to compute assistance policy toward most likely goal.
- **Limitation**: Requires pre-defined discrete set of possible goals
- **Relevance to ALIGN**: We remove the discrete goal requirement — continuous embeddings instead.
- **Baseline use**: Goal-set baseline in our ablation experiments

### Reddy, Dragan, Levine 2018 — "Shared Autonomy via Deep Reinforcement Learning"
- arXiv: 1802.01744 | RSS 2018
- **Core idea**: RL-based shared autonomy where the agent learns to infer the goal from user input and assist. No prior knowledge of environment dynamics or user policy.
- **Limitation**: Still relies on a known discrete goal space
- **Relevance**: Early learning-based shared autonomy without knowing dynamics

### Oh, Toussaint, Mainprice 2019 — "Learning Arbitration for Shared Autonomy by Hindsight Data Aggregation"
- arXiv: 1906.12280
- **Core idea**: Learns the arbitration function (when to let autonomous agent take over) from
  human demonstrations using hindsight. Framework for pick-and-place teleoperation.
- **Relevance**: Directly addresses the same question — when to assist? Uses DAgger-style
  learning, not contrastive alignment.

### Jeon, Losey, Sadigh 2020 — "Shared Autonomy with Learned Latent Actions"
- arXiv: 2005.03210 | ~89 citations
- **Core idea**: Learns a low-dimensional latent action space from human teleoperation.
  Maps noisy human input to learned latent actions representing meaningful behaviors.
- **Limitation**: No visual context, latent actions are task-specific, no learned assist gating
- **Relevance to ALIGN**: Similar motivation (learn better intent representation),
  but ALIGN adds vision + trajectory alignment + language

### Schaff & Walter 2020 — "Residual Policy Learning for Shared Autonomy"
- arXiv: 2004.05097 | RSS 2020
- **Core idea**: Learns a residual policy on top of human commands, using RL. No restrictive
  assumptions about known goal space or environment dynamics.
- **Limitation**: Continuous RL can be unstable; no explicit confidence gating
- **Relevance**: Another learning-based correction approach; residual Δpose is similar to
  ALIGN's Assistant head

### Jonnavittula & Losey 2021 — "Learning to Share Autonomy Across Repeated Interaction"
- arXiv: 2107.09650
- **Core idea**: Robot learns personalized α weighting from repeated interactions with
  a specific user. As confidence increases, automation increases.
- **Relevance**: Personalization aspect; but α is learned per-user, not from visual alignment

### Zurek, Bobu, Brown, Dragan 2021 — "Situational Confidence Assistance for Lifelong Shared Autonomy"
- arXiv: 2104.06556 | ICRA 2021
- **Core idea**: Robot detects when its repertoire of intents is insufficient to explain user
  input, and asks for a new demonstration rather than fighting the human.
- **Relevance**: Very similar safety goal (detect OOD and back off). Our approach uses
  contrastive alignment as OOD detector, theirs uses Bayesian intent model entropy.

### Li et al. 2020 — "A General Arbitration Model for Robust Human-Robot Shared Control with Multi-Source Uncertainty Modeling"
- arXiv: 2003.05097
- **Core idea**: Formal arbitration model accounting for multiple uncertainty sources
  (human, robot, environment). Arbitration policy learned from multi-source uncertainty.
- **Relevance**: Formal uncertainty framework for arbitration; complementary to our
  contrastive approach

### Owan, Garbini, Devasia 2015 — "Uncertainty-based Arbitration of Human-Machine Shared Control"
- arXiv: 1511.05996
- **Core idea**: Level of autonomy determined by uncertainty of the autonomous system.
  High uncertainty → more human control.
- **Relevance**: Early uncertainty-based arbitration. Our learned α is a modern
  generalization of this principle.

---

## 2. DIFFUSION-BASED SHARED AUTONOMY (Closest Competitors)

### Yoneda et al. 2023 — "To the Noise and Back: Diffusion for Shared Autonomy"
- arXiv: 2302.12244 | Project: diffusion-for-shared-autonomy.github.io | ~30 citations
- **Core idea**: Uses conditional diffusion to blend human input with autonomous control.
  The diffusion denoising process naturally produces smooth trajectories.
- **Key differences from ALIGN**:
  - Yoneda uses diffusion for *blending* (always-on assist). ALIGN uses contrastive for *gating* (learned when-to-assist).
  - Yoneda has no explicit OOD detection. ALIGN's min(cos_sims) provides natural OOD safety.
  - Yoneda needs multiple diffusion steps (~10-50). ALIGN: single forward pass (~2ms).
  - Yoneda assumes assistance is always beneficial. ALIGN gates it out when uncertain.
- **Baseline use**: Primary comparison point. "Always-on" assistant baseline.

### Chi et al. 2023 — "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
- arXiv: 2303.04137 | RSS 2023 | 500+ citations
- **Core idea**: Robot visuomotor policy as conditional denoising diffusion process.
  Outputs action chunks, state-of-the-art on 12 tasks.
- **Limitation**: Designed for full autonomy, not shared autonomy / human-in-the-loop.
- **Relevance**: Foundational for diffusion-based action prediction. Our chunk output (K=5)
  follows the same action-chunking paradigm.

### HITL-D: Zilka et al. 2026 — "Human In The Loop Diffusion Assisted Shared Control"
- arXiv: 2605.21460 | Accepted ICRA 2026 ⚠️ **CONCURRENT WORK**
- **Core idea**: Shared control framework using diffusion-based policies with novel combination
  of human input and autonomous diffusion for multi-step insertion/fine manipulation.
- **Relevance**: Direct competitor — diffusion + shared control at ICRA 2026.
  Need to differentiate: ALIGN gates assistance via contrastive alignment, HITL-D uses
  blending of human+diffusion.

### DiSCo: Wang et al. 2026 — "Diffusion Sequence Copilots for Shared Autonomy"
- arXiv: 2603.22787 | Accepted HRI 2026 ⚠️ **CONCURRENT WORK**
- **Core idea**: Shared autonomy using diffusion sequence copilots. Corrects user actions
  consistent with user goals in high-dimensional control.
- **Relevance**: Very close — diffusion + shared autonomy + trajectory correction.
  ALIGN's differentiator: 3-modal contrastive alignment for gating (not just diffusion
  blending), language conditioning, and explicit OOD handling.

### Set-Supervised Diffusion Policy: Li et al. 2026
- arXiv: 2606.01865 | June 2026
- **Core idea**: Diffusion policies that learn from human corrections. Paired supervision
  of undesired actions + teacher corrections. On-policy correction data.
- **Relevance**: Human corrections as training signal for diffusion.

### CFG-DP: Lu et al. 2025 — "Enhancing Diffusion Policy with Classifier-Free Guidance for Temporal Robotic Tasks"
- arXiv: 2510.09786
- **Core idea**: Adds classifier-free guidance to diffusion policy for temporal task awareness.
- **Relevance**: Improving action chunking for temporal tasks on humanoids.

---

## 3. RECENT ASSISTIVE TELEOPERATION (2025-2026)

### Adaptor: Liu et al. 2026 — "Advancing Assistive Teleoperation with Few-Shot Learning and Cross-Operator Generalization"
- arXiv: 2604.09462 | Accepted ICRA 2026 ⚠️
- **Core idea**: Few-shot framework for cross-operator intent recognition. Bridges domain
  gap via preprocessing + few-shot adaptation to new operators.
- **Relevance**: Intent recognition for assistive teleop. Shows the community is actively
  working on the same problem space.

### Sha et al. 2026 — "Efficient and Reliable Teleoperation through Real-to-Sim-to-Real Shared Autonomy"
- arXiv: 2603.17016 | Project: residual-copilot.github.io
- **Core idea**: Learns shared autonomy assistance in simulation using residual policies.
  Real→sim→real loop for fine-grained contact-rich teleoperation.
- **Relevance**: Residual policy approach to shared autonomy; sim-to-real focus.

### AssistDLO: Guler et al. 2026 — "Assistive Teleoperation for Deformable Linear Object Manipulation"
- arXiv: 2605.06323
- **Core idea**: Assistive teleop for DLOs with multi-view state estimation + visual assistance.
- **Relevance**: Assistive teleop for a specific domain (DLOs). Shows assistive teleop is
  an active research area.

### Chen et al. 2026 — "Shared Autonomy Assisted by Impedance-Driven Anisotropic Guidance Field"
- arXiv: 2605.02410 | IEEE RA-L
- **Core idea**: Focuses on *mutual* understanding — not just robot inferring human intent,
  but human understanding robot's intent too. Uses impedance-driven guidance fields.
- **Relevance**: Interesting angle on bidirectional transparency. Our approach could
  complement this with learned gating.

### MIRAGE: Sun et al. 2025 — "Multimodal Intention Recognition and Admittance-Guided Enhancement in VR-based Multi-object Teleoperation"
- arXiv: 2509.01996 | ISMAR 2025
- **Core idea**: Multimodal-CNN for human intention perception + virtual admittance model
  for shared control in VR multi-object teleoperation.
- **Relevance**: Multi-modal intent recognition (like our 3-modal), but uses CNN+admittance,
  not contrastive alignment. No language modality.

### Casper: Liu et al. 2025 — "Inferring Diverse Intents for Assistive Teleoperation with Vision Language Models"
- arXiv: 2506.14727
- **Core idea**: Uses VLMs to infer diverse human intentions from control inputs for
  assistive teleoperation.
- **Relevance**: VLM-based intent inference. Our approach is much faster (CLIP embeddings
  cached per-episode vs online VLM inference).

### Tao et al. 2024 — "Incremental Learning for Robot Shared Autonomy"
- arXiv: 2410.06315
- **Core idea**: Incremental learning so shared autonomy adapts over time without
  catastrophic forgetting. Handles obstacles and environmental changes.
- **Relevance**: Lifelong learning for shared autonomy.

---

## 4. INTENTION PREDICTION & RECOGNITION

### IntentVLM: Rahimi et al. 2026 — "Open-Vocabulary Intention Recognition through Forward-Inverse Modeling with Video-Language Models"
- arXiv: 2604.24002
- **Core idea**: VLM-based intention recognition in open-vocabulary settings for HRI.
- **Relevance**: Open-vocabulary intent recognition via language grounding.

### Belsare et al. 2025 — "Toward Zero-Shot User Intent Recognition in Shared Autonomy"
- arXiv: 2501.08389 | HRI 2025
- **Core idea**: Zero-shot intent recognition without prior demonstrations or known intent set.
- **Relevance**: Same goal as ALIGN's OOD handling — recognize when intent is unknown.

### TATIC: Song et al. 2026 — "Task-Aware Temporal Learning for Human Intent Inference from Physical Corrections"
- arXiv: 2603.11077
- **Core idea**: Extracts task-level semantic intent from physical corrections during collaboration.
- **Relevance**: Intent inference from physical interaction.

### Sticky-Glance: Lai et al. 2026 — "Robust Intent Recognition for Human Robot Collaboration via Single-Glance"
- arXiv: 2603.06121
- **Core idea**: Gaze-based intent recognition robust to noise, micro-saccades, viewpoint changes.
- **Relevance**: Alternative modality for intent (gaze). Our wrist camera serves similar purpose.

### GUIDER: Contreras et al. 2025 — "Probabilistic Human Intent Prediction for Mobile Manipulation"
- arXiv: 2507.10131
- **Core idea**: Probabilistic framework estimating both navigation goals and manipulation intents.
  Two coupled belief layers.
- **Relevance**: Multi-level intent estimation.

### Xie et al. 2026 — "Learning Human-Intention Priors from Large-Scale Human Demonstrations for Robotic Manipulation"
- arXiv: 2604.24681 | MoT-HRA framework, HA-2.2M dataset
- **Core idea**: Hierarchical VLA framework learning intention priors from 2.2M human demos.
- **Relevance**: Large-scale human intention learning. Our approach is much lighter-weight
  with 1.6M params.

---

## 5. CONTRASTIVE LEARNING FOR ROBOTICS

### R3M: Nair et al. 2022 — "A Universal Visual Representation for Robot Manipulation"
- arXiv: 2203.12601 | CoRL 2022
- **Core idea**: Time-contrastive learning + video-language alignment on Ego4D human video.
  Pre-trained visual representation for downstream manipulation.
- **Relevance**: Shows contrastive pretraining on egocentric video produces useful features
  for manipulation. ALIGN applies similar principle but aligns vision with trajectory
  (not just time-contrastive).

### CLAMP: Liu et al. 2026 — "Contrastive Learning for 3D Multi-View Action-Conditioned Robotic Manipulation Pretraining"
- arXiv: 2602.00937
- **Core idea**: Contrastive learning over 3D multi-view representations for robot
  manipulation pretraining.
- **Relevance**: Contrastive learning specifically for robot manipulation.

### ConLA: Dai et al. 2026 — "Contrastive Latent Action Learning from Human Videos for Robotic Manipulation"
- arXiv: 2602.00557
- **Core idea**: Contrastive latent action learning from human videos. Bridges human
  video data to robot action space.
- **Relevance**: Contrastive action learning from human data. Shares motivation of
  aligning vision and action through contrastive learning.

### DynaFLIP: Lee et al. 2026 — "Rethinking Robotics Perception via Tri-Modal-Dynamics Guided Representation"
- arXiv: 2605.30350
- **Core idea**: Tri-modal (RGB + depth + motion) dynamics-guided representation for
  robot perception. Motion understanding built into the encoder instead of downstream.
- **Relevance**: Multi-modal dynamics representation. Similar motivation — preserve
  action-relevant information in the representation.

### Action-based Contrastive Learning: Halawa et al. 2022 — "Action-based Contrastive Learning for Trajectory Prediction"
- arXiv: 2207.08664
- **Core idea**: Contrastive loss using pedestrian action information to improve trajectory
  prediction embeddings.
- **Relevance**: The closest prior use of contrastive learning on trajectory data.

### RS-Contrast: Kim et al. 2025 — "Contrastive Representation Regularization for Vision-Language-Action Models"
- arXiv: 2510.01711 | ICML 2026
- **Core idea**: Robot State-aware Contrastive Loss (RS-CL) that makes VLA representations
  sensitive to control actions and proprioception.
- **Relevance**: Contrastive learning to improve VLA representations with action-awareness.

---

## 6. ACTION CHUNKING & TRAJECTORY PREDICTION

### ACT: Zhao et al. 2023 — "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT)
- The original Action Chunking with Transformers. Foundational.
- **Relevance**: ALIGN's Assistant head outputs action chunks (K=5) like ACT. K=5 is
  smaller than ACT's K but follows the same principle.

### Q-chunking: Li, Zhou, Levine 2025 — "Reinforcement Learning with Action Chunking"
- arXiv: 2507.07969 | NeurIPS 2025
- **Core idea**: Action chunking for RL. Improves exploration and value estimation in
  long-horizon sparse-reward tasks.
- **Relevance**: Action chunking validated in RL context.

### InterACT: Lee et al. 2024 — "Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation"
- arXiv: 2409.07914 | CoRL 2024
- **Core idea**: Hierarchical attention for bimanual action chunking.
- **Relevance**: Advanced action chunking for multi-arm coordination.

### ReMAC: Wang et al. 2026 — "Real-Time Robot Execution with Masked Action Chunking"
- arXiv: 2601.20130 | ICLR 2026
- **Core idea**: Masked action chunking for asynchronous real-time execution.
  Predict next chunk while executing current one.
- **Relevance**: Real-time chunk execution pattern. ALIGN's cached chunk blending
  is similar in spirit.

### VQ-ACE: Yang et al. 2024 — "Efficient Policy Search for Dexterous Robotic Manipulation via Action Chunking Embedding"
- arXiv: 2411.03556
- **Core idea**: Compresses hand motion into quantized latent space for efficient
  dexterous manipulation.
- **Relevance**: Action chunking for dexterous manipulation.

---

## 7. LANGUAGE-CONDITIONED MANIPULATION (VLA)

### Yao et al. 2023 — "Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation"
- arXiv: 2312.10807
- **Core idea**: Comprehensive survey of language-conditioned robot manipulation.
- **Relevance**: Survey paper — useful for positioning and related work section.

### Stepputtis et al. 2020 — "Language-Conditioned Imitation Learning for Robot Manipulation Tasks"
- arXiv: 2010.12083
- **Core idea**: Early work combining language, vision, motion for imitation learning.
  Multimodal approach for task generalization.
- **Relevance**: Shows the value of language conditioning for manipulation.

### Stepputtis et al. 2019 — "Imitation Learning of Robot Policies by Combining Language, Vision and Demonstration"
- arXiv: 1911.11744
- **Core idea**: End-to-end imitation learning combining natural language, vision,
  and motion. Abstract task representation synthesizes motion controllers at runtime.
- **Relevance**: Very early multimodal (language+vision+motion) — precursor to our 3-modal approach.

### Dexora: Zhang et al. 2026 — "Open-source VLA for High-DoF Bimanual Dexterity"
- arXiv: 2605.18722
- **Core idea**: Open-source VLA for bimanual dexterous manipulation.
- **Relevance**: VLA for high-DoF manipulation.

### 3DThinkVLA: Shi et al. 2026 — "Endowing VLA Models with Latent 3D Priors via 3D-Thinking-Guided Co-training"
- arXiv: 2606.04436
- **Core idea**: 3D spatial reasoning integrated into VLA action prediction.
- **Relevance**: Enhancing VLA with spatial awareness.

### Language-Driven Grasp: Vuong et al. 2024 — "Language-driven Grasp Detection"
- arXiv: 2406.09489 | CVPR 2024
- **Core idea**: Language-conditioned grasp detection with Grasp-Anything++ dataset (1M samples).
- **Relevance**: Shows language-conditioned grasping works well. ALIGN uses language not
  just for grasp selection but for gating.

### Command Grasp: Chen et al. 2021 — "A Joint Network for Grasp Detection Conditioned on Natural Language Commands"
- arXiv: 2104.00492 | ICRA 2021
- **Core idea**: Single network for both object localization from language AND grasp detection.
- **Relevance**: Early language-conditioned grasping.

---

## 8. GAZE / VR TELEOPERATION

### Luo et al. 2024 — "User-customizable Shared Control for Robot Teleoperation via Virtual Reality"
- arXiv: 2403.13177 | IROS 2024
- **Core idea**: VR teleop with shared control. User can customize arbitration process.
- **Relevance**: VR shared control with user customization.

### Xu et al. 2022 — "Shared-Control Robotic Manipulation in Virtual Reality"
- arXiv: 2205.10564
- **Core idea**: VR-based teleop interface for robotic manipulators with iterative
  human-in-the-loop waypoint setting.
- **Relevance**: VR shared control implementation.

### Clever et al. 2021 — "Assistive Tele-op: Leveraging Transformers to Collect Robotic Task Demonstrations"
- arXiv: 2112.05129
- **Core idea**: VR system for data collection that displays autonomous trajectory
  predictions alongside user input.
- **Relevance**: VR assistive teleop for data collection.

### Cui et al. 2025 — "End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy"
- arXiv: 2511.00139
- **Core idea**: VR teleop augmented by autonomous hand VLA policy for efficient data collection.
- **Relevance**: VR + VLA shared autonomy for dexterous hands.

### GAMMA: Tay et al. 2026 — "Intent at a Glance: Gaze-Guided Robotic Manipulation via Foundation Models"
- arXiv: 2601.05336 | RSS 2025 Workshop
- **Core idea**: Eye gaze as intent input, foundation models for manipulation.
- **Relevance**: Gaze as alternative intent modality.

---

## 9. HUMANOID TELEOPERATION

### RHINO: Chen et al. 2025 — "Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations"
- arXiv: 2502.13134
- **Core idea**: Humanoid interacting with humans and objects from demonstrations.
- **Relevance**: Humanoid teleoperation with interaction.

### H2O: He et al. 2024 — "Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation"
- arXiv: 2403.04436
- **Core idea**: RL-based real-time whole-body humanoid teleoperation from RGB camera.
- **Relevance**: Humanoid teleop framework.

### OmniH2O: He et al. 2024 — "Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning"
- arXiv: 2406.08858
- **Core idea**: Whole-body humanoid teleop via VR, verbal instruction, and RGB. Also enables autonomy.
- **Relevance**: Most comprehensive humanoid teleop system.

### HumanPlus: Fu et al. 2024 — "Humanoid Shadowing and Imitation from Humans"
- arXiv: 2406.10454
- **Core idea**: Humanoid learns from human shadowing and imitation.
- **Relevance**: Human-to-humanoid transfer.

### Generalizable Humanoid Manipulation with 3D Diffusion Policies: Ze et al. 2024
- arXiv: 2410.10803
- **Core idea**: 3D diffusion policies for humanoid manipulation with generalization.
- **Relevance**: Diffusion + humanoid manipulation.

### Mixed Reality Teleoperation: Penco et al. 2024
- arXiv: 2411.01014
- **Core idea**: Mixed reality + assistive autonomy for humanoid teleop.
- **Relevance**: Assistive autonomy for humanoid teleop in MR.

### HuMI: Nai et al. 2026 — "Humanoid Manipulation Interface"
- arXiv: 2602.06643
- **Core idea**: Portable whole-body humanoid manipulation without robot-specific hardware.
- **Relevance**: Robot-free humanoid data collection.

### Sanjar Atamuradov 2025 — "Learning Adaptive Neural Teleoperation for Humanoid Robots: From IK to End-to-End Control"
- arXiv: 2511.12390
- **Core idea**: Neural teleop replacing IK + PD controllers for humanoids. Learns to handle
  external forces and produce natural motions.
- **Relevance**: Directly addresses humanoid teleop control problems.

---

## 10. TREMOR & NOISE SUPPRESSION

### Steady Hand Eye Robot: Esfandiari et al. 2024 — "Bimanual Manipulation of Steady Hand Eye Robots with Adaptive Sclera Force Control"
- arXiv: 2402.18088
- **Core idea**: Surgical robot for hand tremor cancellation during retinal surgery.
- **Relevance**: Tremor cancellation in teleoperation. Our approach is learning-based
  instead of mechanical/control-based.

### WAKE: Shahtalebi et al. 2017 — "Wavelet Decomposition Coupled with Adaptive Kalman Filtering for Pathological Tremor Extraction"
- arXiv: 1711.06815
- **Core idea**: Wavelet + Kalman filtering for tremor extraction and cancellation.
- **Relevance**: Signal processing approach to tremor. ALIGN's trajectory encoder learns
  to suppress tremor implicitly via contrastive pretraining.

---

## 11. UNIQUE POSITIONING OF ALIGN

Based on this survey, here's what makes ALIGN novel:

| Feature | ALIGN | Yoneda 2023 | HITL-D 2026 | DiSCo 2026 | Adaptor 2026 | Casper 2025 |
|---------|-------|-------------|-------------|------------|-------------|-------------|
| **Learned assist gating** | ✅ Contrastive | ❌ Always-on | ❌ Diffusion blend | ❌ Diffusion blend | ❌ Probabilistic | ❌ VLM-based |
| **No object detector needed** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ (VLM) |
| **Language as gating modality** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **3-way contrastive alignment** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **OOD detection via alignment** | ✅ | ❌ | ❌ | ❌ | ❌ | Partially |
| **Action chunk output** | ✅ K=5 | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Humanoid deployment target** | ✅ G1 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **<50ms inference** | ✅ ~40ms | ❌ ~100ms | ❌ ~100ms+ | ❌ ~100ms+ | ✅ | ❌ ~500ms |
| **Single unified model** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |

---

## 12. RECOMMENDED BASELINES (for ablation table)

1. **Pure teleop** (α=0) — lower bound
2. **Distance-gated** — α = f(distance), common simple baseline
3. **Always-on assist** (α=1) — equivalent to Yoneda-style blending
4. **2-modal ALIGN** (ablate text) — vision+trajectory only
5. **ALIGN without contrastive pretraining** (MLP heads only) — ablate contrastive
6. **ALIGN single-step** (K=1, ablate chunking) — ablate action chunking
7. **Diffusion-based blending** — Yoneda 2023 or HITL-D style
8. **VLM-based intent** — Casper or similar (if compute allows)

---

## 13. KEY PAPERS TO TRACK

- **ICRA 2026**: HITL-D (2605.21460), Adaptor (2604.09462), AssistDLO (2605.06323)
- **HRI 2026**: DiSCo (2603.22787)
- **RSS 2023**: Diffusion Policy (2303.04137)
- **CoRL 2024**: InterACT (2409.07914)
- **NeurIPS 2025**: Q-chunking (2507.07969)
- **ICML 2026**: RS-Contrast (2510.01711)
- **AAAI 2026**: AC3 (2508.11143)
- **ICLR 2026**: ReMAC (2601.20130)

---

## 14. GAP STATEMENT (for paper)

> "Existing shared autonomy systems either (1) gate assistance using hand-tuned thresholds or discrete goal sets, (2) always blend human and autonomous control without learning *when* assistance is beneficial, (3) lack language grounding for object disambiguation, or (4) cannot run at control frequency on humanoid hardware. ALIGN is the first system to use 3-way contrastive vision-trajectory-language alignment as a learned, continuous, OOD-aware gating signal — in a unified architecture that simultaneously learns when to assist and what correction to apply, from a single dataset with per-episode text annotations."
