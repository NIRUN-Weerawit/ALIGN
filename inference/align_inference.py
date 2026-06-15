#!/usr/bin/env python3
"""ALIGN inference — runs the full shared autonomy pipeline at 30Hz.

Usage:
    # Quick smoke test with synthetic data
    python inference/align_inference.py \
        --checkpoint checkpoints/heads_libero_haruka/heads_best.pt \
        --task "pick up the red mug"

    # Real deployment (provide a camera callback)
    python inference/align_inference.py \
        --checkpoint checkpoints/heads_libero_haruka/heads_best.pt \
        --task "pick up the red mug" \
        --camera 0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel


class ALIGNInference:
    """Real-time ALIGN inference engine.

    Runs a 30Hz control loop:
        1. Capture camera frame → encode via mixer → z_v
        2. Buffer K noisy EEF poses → encode via mixer → z_t
        3. Compute z_text once per task (cached)
        4. Decision head → α (from z_v, z_t, z_text + internal cosines)
        5. Assistant head → chunk of K corrective Δposes
        6. Blend: final = raw_pose + α · chunk[0]
    """

    def __init__(
        self,
        checkpoint_path: str,
        task_description: str,
        traj_window: int = 20,
        chunk_size: int = 10,
        device: Optional[str] = None,
        encoder_checkpoint: Optional[str] = None,
    ):
        self.traj_window = traj_window
        self.chunk_size = chunk_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load heads checkpoint first to detect chunk_size
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt.get("config", {})
        if cfg.get("chunk_size"):
            self.chunk_size = cfg["chunk_size"]
            print(f"[ALIGN] Detected chunk_size={self.chunk_size} from checkpoint")

        # Build model with correct chunk_size
        self.model = ALIGNModel(
            embed_dim=256,
            chunk_size=self.chunk_size,
            use_text=True,
            device=self.device,
        ).to(self.device)

        # Load encoder backbone (Phase 1 checkpoint) if provided
        if encoder_checkpoint:
            enc_ckpt = torch.load(encoder_checkpoint, map_location=self.device)
            if "trainable_state_dict" in enc_ckpt:
                self.model.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
            print(f"[ALIGN] Loaded encoder backbone: {encoder_checkpoint}")
        else:
            print("[ALIGN] WARNING: No encoder checkpoint provided. Encoders are randomly initialized!")

        # Load heads on top (overwrites head params from encoder checkpoint if any)
        if "trainable_state_dict" in ckpt:
            self.model.load_trainable_state_dict(ckpt["trainable_state_dict"])
        elif "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)

        self.model.eval()
        print(f"[ALIGN] Loaded heads: {checkpoint_path}  (phase={ckpt.get('phase','?')}, "
              f"epoch={ckpt.get('epoch','?')})")

        # Precompute text embedding (cached for episode)
        self.z_text = self.model.encode_text([task_description])
        print(f"[ALIGN] Task:   {task_description}")

        # Running buffers
        self.pose_buffer: list = []  # last K noisy poses
        self.chunk_cache: Optional[np.ndarray] = None

        # Stats
        self.step_count = 0
        self.alpha_history: list[float] = []

    def reset(self, task_description: Optional[str] = None):
        """Reset for a new episode."""
        self.pose_buffer = []
        self.chunk_cache = None
        self.alpha_history = []
        self.step_count = 0
        if task_description is not None:
            self.z_text = self.model.encode_text([task_description])

    @torch.no_grad()
    def step(self, frame: np.ndarray, raw_pose: np.ndarray) -> dict:
        """One control step (call at 30Hz).

        Args:
            frame: (H, W, 3) uint8 RGB camera image (ideally wrist).
            raw_pose: (6,) noisy teleoperation EEF pose [x,y,z,rx,ry,rz].

        Returns:
            dict with 'commanded_pose' (6,), 'alpha' (float), 'chunk' (K,6).
        """
        if self.step_count == 0:
            self.pose_buffer = [raw_pose.copy() for _ in range(self.traj_window)]

        # Update buffer (ring buffer)
        self.pose_buffer.append(raw_pose.copy())
        if len(self.pose_buffer) > self.traj_window:
            self.pose_buffer.pop(0)

        # Prepare tensors
        frame_t = torch.from_numpy(frame).unsqueeze(0).to(self.device, non_blocking=True)  # (1, H, W, 3)
        traj_t = torch.from_numpy(np.stack(self.pose_buffer, axis=0)).unsqueeze(0).float().to(self.device)  # (1, K, 6)

        # Encode through mixer (Phase 1b/2: frozen encoders + mixer)
        mixed = self.model.encode_mixed(frame_t, traj_t, [""])  # text used from cache below
        z_v = mixed["z_v"]
        z_t = mixed["z_t"]
        # Use precomputed text embedding (overwrite the dummy empty string)
        z_text = self.z_text

        # Decision head (computes cosines internally)
        alpha = self.model.decision_head(z_v, z_t, z_text)
        alpha_val = float(alpha.squeeze().cpu())

        # Assistant head
        noisy_t = torch.from_numpy(raw_pose).unsqueeze(0).float().to(self.device)
        chunk = self.model.assistant_head(z_v, z_t, z_text, noisy_t)
        chunk_np = chunk.squeeze(0).cpu().numpy()

        # Blend: commanded = raw + α × corrective delta
        if self.chunk_cache is not None:
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
# CLI / Smoke test
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ALIGN Inference Runtime")
    parser.add_argument("--checkpoint", required=True, help="Path to heads checkpoint (.pt)")
    parser.add_argument("--task", default="pick and place", help="Task description")
    parser.add_argument("--device", default=None, help="Inference device")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device ID for live feed (optional)")
    args = parser.parse_args()

    engine = ALIGNInference(
        checkpoint_path=args.checkpoint,
        task_description=args.task,
        device=args.device,
    )

    if args.camera is not None:
        # ── Live deployment ──
        try:
            import cv2
        except ImportError:
            print("[ALIGN] OpenCV not installed. Install: pip install opencv-python")
            sys.exit(1)

        cap = cv2.VideoCapture(args.camera)
        print(f"[ALIGN] Live camera {args.camera} opened. Press Ctrl+C to stop.")

        try:
            from pynput import keyboard
            # Integrate with robot control here
        except ImportError:
            pass

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            # Convert BGR → RGB, resize to 224×224
            frame = cv2.cvtColor(cv2.resize(frame_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
            # raw_pose comes from the teleoperation device (e.g., VR controller)
            raw_pose = np.zeros(6, dtype=np.float32)

            result = engine.step(frame, raw_pose)
            print(f"  α={result['alpha']:.3f}  cmd=[{result['commanded_pose'][:3].round(3)}]")
    else:
        # ── Smoke test with synthetic data ──
        print("[ALIGN] Running 100 synthetic steps...")
        N = 100
        for i in range(N):
            frame = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            raw = np.array([0.5 + 0.01 * np.sin(i * 0.3), 0.0, 0.25, 0.0, 0.0, 0.0])
            result = engine.step(frame, raw)
            if i < 3 or i % 30 == 0:
                print(f"  Step {i:3d}: α={result['alpha']:.3f}  "
                      f"Δ={np.linalg.norm(result['chunk'][0]):.3f}  "
                      f"cmd=[{result['commanded_pose'][:3].round(3)}]")

        print(f"\n[ALIGN] Smoke test complete. Mean α: {engine.mean_alpha:.3f}")
        print(f"[ALIGN] Pipeline: ✓ Encode(mixed) → Decision(α) → Assistant(Δ) → Blend")


if __name__ == "__main__":
    main()