#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN inference — runs the full shared autonomy pipeline at 30Hz.

Usage:
    python -m inference.align_inference \
        --checkpoint checkpoints/heads/joint_best.pt \
        --task "pick up the red mug"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel


class ALIGNInference:
    """Real-time ALIGN inference engine.

    Runs a 30Hz control loop:
        1. Capture camera frame → z_v
        2. Buffer K noisy EEF poses → z_t
        3. Compute z_text once per task (cached)
        4. Decision head → α
        5. Assistant head → chunk of Δposes
        6. Blend: final = raw_pose + α · chunk[0]
        7. IK → motor commands
    """

    def __init__(
        self,
        checkpoint_path: str,
        task_description: str,
        traj_window: int = 10,
        chunk_size: int = 5,
        device: Optional[str] = None,
    ):
        self.traj_window = traj_window
        self.chunk_size = chunk_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ALIGNModel(
            embed_dim=256,
            chunk_size=chunk_size,
            use_text=True,
            device=self.device,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.eval()

        # Precompute text embedding (cached for episode)
        self.z_text = self.model.encode_text([task_description])
        print(f"[ALIGN] Loaded: {checkpoint_path}")
        print(f"[ALIGN] Task:   {task_description}")

        # Running buffers
        self.pose_buffer: list = []  # last K noisy poses
        self.chunk_cache: Optional[np.ndarray] = None  # previous chunk for blending

        # Stats
        self.step_count = 0
        self.alpha_history: list[float] = []

    def reset(self, task_description: Optional[str] = None):
        """Reset for a new episode."""
        self.pose_buffer = []
        self.chunk_cache = None
        self.alpha_history = []
        if task_description is not None:
            self.z_text = self.model.encode_text([task_description])

    @torch.no_grad()
    def step(self, frame: np.ndarray, raw_pose: np.ndarray) -> dict:
        """One control step (call at 30Hz).

        Args:
            frame: (H, W, 3) uint8 RGB wrist camera image.
            raw_pose: (6,) noisy teleoperation EEF pose [x,y,z,rx,ry,rz].

        Returns:
            dict with 'commanded_pose' (6,), 'alpha', 'chunk' (K,6), computed in <40ms.
        """
        if self.step_count == 0:
            # Pre-fill buffer on first frame with copies of first pose
            self.pose_buffer = [raw_pose.copy() for _ in range(self.traj_window)]

        # Update buffer (ring)
        self.pose_buffer.append(raw_pose.copy())
        if len(self.pose_buffer) > self.traj_window:
            self.pose_buffer.pop(0)

        # Prepare tensors
        frame_t = torch.from_numpy(frame).unsqueeze(0).to(self.device, non_blocking=True)  # (1, H, W, 3)
        traj_t = torch.from_numpy(np.stack(self.pose_buffer, axis=0)).unsqueeze(0).float().to(self.device)  # (1, K, 6)

        # Encode
        z_v = self.model.encode_vision(frame_t)
        z_t = self.model.encode_trajectory(traj_t)
        z_text = self.z_text

        # Decision head (no external distances — learned from visual features)
        alpha = self.model.decision_head(z_v, z_t, z_text)  # (1, 1)
        alpha_val = float(alpha.squeeze().cpu())

        # Assistant head
        noisy_t = torch.from_numpy(raw_pose).unsqueeze(0).float().to(self.device)  # (1, 6)
        chunk = self.model.assistant_head(z_v, z_t, z_text, noisy_t)  # (1, K, 6)
        chunk_np = chunk.squeeze(0).cpu().numpy()  # (K, 6)

        # Blend with cached chunk for temporal smoothness
        if self.chunk_cache is not None:
            # Exponential blend: prev chunk[-1] weights decayed
            commanded_pose = raw_pose + alpha_val * (0.7 * chunk_np[0] + 0.3 * self.chunk_cache[-1])
        else:
            commanded_pose = raw_pose + alpha_val * chunk_np[0]

        self.chunk_cache = chunk_np
        self.step_count += 1
        self.alpha_history.append(alpha_val)

        return {
            "commanded_pose": commanded_pose,
            "alpha": alpha_val,
            "chunk": chunk_np,
        }

    @property
    def mean_alpha(self) -> float:
        return float(np.mean(self.alpha_history)) if self.alpha_history else 0.0


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ALIGN Inference Runtime")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--task", default="pick and place", help="Task description for the episode")
    parser.add_argument("--device", default=None, help="Inference device")

    args = parser.parse_args()

    engine = ALIGNInference(
        checkpoint_path=args.checkpoint,
        task_description=args.task,
    )

    # Quick smoke test with synthetic data
    print("[ALIGN] Running 100 synthetic steps...")
    N = 100
    for i in range(N):
        # Synthetic frame
        frame = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        # Synthetic noisy pose
        raw = np.array([0.5 + 0.01 * np.sin(i * 0.3), 0.0, 0.25, 0.0, 0.0, 0.0])
        result = engine.step(frame, raw)
        if i < 3 or i % 30 == 0:
            print(f"  Step {i:3d}: α={result['alpha']:.3f}  "
                  f"chunk={result['chunk'].shape}  cmd={result['commanded_pose'][:3].round(3)}")

    print(f"\n[ALIGN] Smoke test complete. Mean α: {engine.mean_alpha:.3f}")
    print("[ALIGN] Pipeline: ✓ Vision + Trajectory + Text → Decision + Assistant → Control")


if __name__ == "__main__":
    main()
