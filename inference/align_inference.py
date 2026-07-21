#!/usr/bin/env python3
"""ALIGN v3 inference — runs the intention-estimation pipeline at 30Hz.

Uses ALIGNIntentionModel with Mamba recurrence. Maintains persistent
mamba state (conv_state, ssm_state) across control cycles.

Usage:
    # Quick smoke test with synthetic data
    python inference/align_inference.py \\
        --checkpoint checkpoints/intention/libero_spatial/intention_best.pt \\
        --num-cameras 1

    # Real deployment (provide a camera callback)
    python inference/align_inference.py \\
        --checkpoint checkpoints/intention/libero_spatial/intention_best.pt \\
        --camera 0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_intention import ALIGNIntentionModel


class ALIGNIntentionInference:
    """Real-time intention-estimation inference engine.

    Runs a 30Hz control loop:
        1. Capture camera frame(s) → encode via VisionEncoder → patch tokens
        2. Read current robot state (7-D) → encode via StateEncoder
        3. State-Conditioned Attention Pool → z_v_pooled
        4. Mamba step (with persistent state) → h(t)
        5. Buffer K past (z_v_pooled, z_s) for the head
        6. IntentionTransformerHead → K future actions
        7. Use first predicted action as the next command
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})

        # Extract model config
        self.chunk_size = cfg.get("chunk_size", 10)
        self.state_dim = cfg.get("state_dim", 256)
        self.mamba_output_dim = cfg.get("mamba_output_dim", 512)
        self.action_dim = cfg.get("action_dim", 6)
        self.num_cameras = cfg.get("num_cameras", 1)
        self.compressed_dim = cfg.get("compressed_dim", 16)
        print(f"[ALIGN] Loaded config: chunk={self.chunk_size}, "
              f"state={self.state_dim}, "
              f"mamba={self.mamba_output_dim}, action={self.action_dim}, "
              f"cams={self.num_cameras}")

        # Build model
        self.model = ALIGNIntentionModel(
            state_dim=self.state_dim,
            mamba_output_dim=self.mamba_output_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            num_cameras=self.num_cameras,
            compressed_dim=self.compressed_dim,
        ).to(self.device)

        # Load weights
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

        print(f"[ALIGN] Loaded weights: {checkpoint_path}  "
              f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')})")

        # Buffers
        self.z_v_pooled_buffer: List[torch.Tensor] = []  # K past pooled visions
        self.z_s_buffer: List[torch.Tensor] = []         # K past states
        self.h_states: Optional[tuple] = None            # (conv_state, ssm_state)
        self.step_count = 0

    def reset(self):
        """Reset for a new episode."""
        self.z_v_pooled_buffer = []
        self.z_s_buffer = []
        self.h_states = None
        self.step_count = 0

    @torch.no_grad()
    def step(self, frames: np.ndarray, robot_state: np.ndarray) -> dict:
        """One control step (call at 30Hz).

        Args:
            frames: (H, W, 3) or (V, H, W, 3) uint8 RGB camera image(s)
            robot_state: (7,) — [pos(3), euler(3), gripper(1)]

        Returns:
            dict with:
              - action: (action_dim,) — predicted current action
              - chunk: (K, action_dim) — predicted K future actions
              - z_v_pooled: (vision_dim,) — current pooled visual
              - z_s: (state_dim,) — current state
              - h: (mamba_output_dim,) — current mamba state
        """
        # Preprocess
        if frames.ndim == 3:
            # Single camera
            frames_t = torch.from_numpy(frames).unsqueeze(0).to(
                self.device, non_blocking=True
            )  # (1, H, W, 3)
        else:
            # Multi-camera
            frames_t = torch.from_numpy(frames).unsqueeze(0).to(
                self.device, non_blocking=True
            )  # (1, V, H, W, 3)
        state_t = torch.from_numpy(robot_state).unsqueeze(0).float().to(
            self.device
        )  # (1, 7)

        # One step encoding
        z_v_pooled, z_s, h_new, h_states_new = self.model.encode_step(
            frames_t, state_t, self.h_states,
        )
        self.h_states = h_states_new

        # Update buffers (rolling window)
        self.z_v_pooled_buffer.append(z_v_pooled)
        self.z_s_buffer.append(z_s)
        if len(self.z_v_pooled_buffer) > self.chunk_size:
            self.z_v_pooled_buffer.pop(0)
            self.z_s_buffer.pop(0)

        # First step: not enough history for head, return zero action
        if len(self.z_v_pooled_buffer) < self.chunk_size:
            self.step_count += 1
            return {
                "action": np.zeros(self.action_dim, dtype=np.float32),
                "chunk": np.zeros((self.chunk_size, self.action_dim), dtype=np.float32),
                "z_v_pooled": z_v_pooled.squeeze(0).cpu().numpy(),
                "z_s": z_s.squeeze(0).cpu().numpy(),
                "h": h_new.squeeze(0).cpu().numpy(),
            }

        # Build head input: K past (z_v_pooled, z_s) + 1 latest h
        z_v_pooled_window = torch.stack(
            self.z_v_pooled_buffer[-self.chunk_size:], dim=1
        )  # (1, K, vision_dim * num_cameras)
        z_s_window = torch.stack(
            self.z_s_buffer[-self.chunk_size:], dim=1
        )  # (1, K, state_dim)
        h_current = h_new  # (1, mamba_output_dim)

        # Predict actions
        chunk = self.model.predict_actions(
            z_v_pooled_window, z_s_window, h_current
        )  # (1, K, action_dim)
        chunk_np = chunk.squeeze(0).cpu().numpy()  # (K, action_dim)
        action = chunk_np[0]  # use first action

        self.step_count += 1
        return {
            "action": action,
            "chunk": chunk_np,
            "z_v_pooled": z_v_pooled.squeeze(0).cpu().numpy(),
            "z_s": z_s.squeeze(0).cpu().numpy(),
            "h": h_new.squeeze(0).cpu().numpy(),
        }


# ================================================================
# CLI / Smoke test
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ALIGN v3 Intention Inference")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to intention_best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device ID for live feed (optional)")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="Number of synthetic steps for smoke test")
    args = parser.parse_args()

    engine = ALIGNIntentionInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.camera is None:
        # Synthetic smoke test
        H, W = 256, 256
        print(f"\n  Running {args.n_steps} synthetic steps...")
        for t in range(args.n_steps):
            # Random frame
            frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            # Random state
            state = np.random.randn(7).astype(np.float32)
            state[6] = float(np.random.randint(0, 2))  # gripper
            out = engine.step(frame, state)
            print(f"  Step {t+1}: action mean={out['action'].mean():.4f}, "
                  f"chunk shape={out['chunk'].shape}, h shape={out['h'].shape}")
        print(f"\n  Smoke test passed. Final step_count={engine.step_count}")
    else:
        print(f"  Live camera mode: device={args.camera}")
        print("  Not yet implemented — please use a custom camera callback.")


if __name__ == "__main__":
    main()
