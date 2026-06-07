#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full ALIGN training pipeline using open datasets + synthetic noise.

Pipeline:
  1. Convert open datasets (Robomimic, DROID, Bridge) → ALIGN HDF5
  2. Inject synthetic noise to create pseudo-noisy pairs for head training
  3. 3-way contrastive pretraining on clean data (vision↔trajectory↔language)
  4. Head training: Decision (BCE) + Assistant (MSE) on noisy data

Usage:
    # Run full pipeline
    python training/train_full_pipeline.py \\
        --robomimic-dir ./robomimic_data \\
        --output-dir ./training_output \\
        --epochs-pretrain 50

    # Resume from checkpoint
    python training/train_full_pipeline.py \\
        --droid-dir ./droid_data \\
        --pretrained checkpoints/pretrain/best.pt \\
        --stages heads
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.open_dataset import DROIDAdapter, RobomimicAdapter, create_align_dataset
from data.align_dataset import convert_raw_to_hdf5


# ================================================================
# Synthetic noise injection for head training
# ================================================================

class SyntheticNoiseInjector:
    """Adds synthetic noise to clean trajectories for head training.

    Generates noisy versions of clean demonstrations to create
    supervision pairs (noisy, smooth) for Decision + Assistant heads.

    Noise types:
      - Gaussian jitter: σ = 1-3 cm pos, σ = 2-5° orientation
      - Physiological tremor: 8-12 Hz sinusoidal oscillation ±5mm
      - Fatigue ramp: noise magnitude grows 2× over episode
      - Stick-slip: random 2-5mm jumps at low probability
    """

    def __init__(
        self,
        pos_noise_std: float = 0.015,    # 1.5 cm
        orn_noise_std: float = 3.0,      # 3 degrees
        tremor_amplitude: float = 0.005,  # 5 mm
        tremor_freq: float = 10.0,       # 10 Hz
        fatigue_factor: float = 2.0,      # noise amplification over episode
        slip_prob: float = 0.02,          # probability of stick-slip event
        slip_magnitude: float = 0.003,    # 3 mm
        seed: int = 42,
    ):
        self.pos_noise_std = pos_noise_std
        self.orn_noise_std = orn_noise_std
        self.tremor_amplitude = tremor_amplitude
        self.tremor_freq = tremor_freq
        self.fatigue_factor = fatigue_factor
        self.slip_prob = slip_prob
        self.slip_magnitude = slip_magnitude
        self.rng = np.random.RandomState(seed)

    def inject(self, poses: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        """Add synthetic noise to clean poses.

        Args:
            poses: (N, 6) or (N, 7) clean EEF poses.
                Expected format: [x, y, z, rx, ry, rz] or [x, y, z, qx, qy, qz, qw]
            dt: Timestep for frequency calculations.

        Returns:
            (N, 6) or (N, 7) noisy poses with same shape as input.
        """
        N, D = poses.shape
        noisy = poses.copy().astype(np.float64)

        # 1. Gaussian jitter on position
        noisy[:, :3] += self.rng.randn(N, 3) * self.pos_noise_std

        # 2. Tremor (sinusoidal in random direction)
        t = np.arange(N) * dt
        tremor_dir = self.rng.randn(N, 3)
        tremor_dir /= np.linalg.norm(tremor_dir, axis=1, keepdims=True) + 1e-10
        tremor = self.tremor_amplitude * np.sin(2 * np.pi * self.tremor_freq * t)
        noisy[:, :3] += tremor[:, None] * tremor_dir

        # 3. Fatigue ramp
        fatigue_scale = 1.0 + (self.fatigue_factor - 1.0) * np.linspace(0, 1, N)
        fatigue_noise = self.rng.randn(N, 3) * self.pos_noise_std * fatigue_scale[:, None] * 0.5
        noisy[:, :3] += fatigue_noise

        # 4. Stick-slip events (random jumps)
        for i in range(1, N):
            if self.rng.rand() < self.slip_prob:
                slip = self.rng.randn(3) * self.slip_magnitude
                noisy[i:i + 5, :3] += slip

        # 5. Orientation noise
        if D >= 6:
            orn_noise = self.rng.randn(N, 3) * np.deg2rad(self.orn_noise_std)
            if D == 6:
                noisy[:, 3:6] += orn_noise
            elif D == 7:
                # Add small random quaternion perturbation
                from scipy.spatial.transform import Rotation
                r_noise = Rotation.from_rotvec(orn_noise)
                r_orig = Rotation.from_quat(noisy[:, 3:7])
                noisy[:, 3:7] = (r_orig * r_noise).as_quat()

        return noisy


def create_noisy_hdf5(
    clean_h5_path: str,
    output_path: str,
    noise_seed: int = 42,
    noise_configs: Optional[list[dict]] = None,
) -> str:
    """Create a noisy version of a clean HDF5 dataset.

    Generates multiple noise variants per episode for robustness.

    Args:
        clean_h5_path: Path to clean ALIGN HDF5 file.
        output_path: Path for output noisy HDF5 file.
        noise_seed: Base random seed.
        noise_configs: List of noise config dicts for different variants.
            If None, uses default config with 3 noise levels.

    Returns:
        Path to created noisy HDF5 file.
    """
    import h5py

    if noise_configs is None:
        noise_configs = [
            {"pos_noise_std": 0.010, "orn_noise_std": 2.0, "label": "light"},
            {"pos_noise_std": 0.020, "orn_noise_std": 4.0, "label": "medium"},
            {"pos_noise_std": 0.030, "orn_noise_std": 6.0, "label": "heavy"},
        ]

    print(f"[noise] Generating {len(noise_configs)} noise variants...")

    with h5py.File(clean_h5_path, "r") as clean_h5:
        ep_keys = sorted([k for k in clean_h5.keys() if k.startswith("ep_")])
        n_eps = len(ep_keys)

        with h5py.File(output_path, "w") as noisy_h5:
            for ep_idx, ep_key in enumerate(ep_keys):
                clean_group = clean_h5[ep_key]
                frames = clean_group[f"frames/{clean_h5['meta/camera'][()]}"][()]
                clean_poses = clean_group["noisy_poses"][()]  # clean in open data
                gripper = clean_group.get("gripper", np.zeros(len(clean_poses)))[()]
                texts = json.loads(clean_group["texts"][()])
                meta = json.loads(clean_group["meta"][()])

                # Create noisy variants
                for cfg_idx, cfg in enumerate(noise_configs):
                    injector = SyntheticNoiseInjector(
                        pos_noise_std=cfg["pos_noise_std"],
                        orn_noise_std=cfg["orn_noise_std"],
                        seed=noise_seed + ep_idx * 100 + cfg_idx,
                    )

                    noisy_poses = injector.inject(clean_poses)
                    N = len(noisy_poses)

                    # Compute α_target = need × capability (capability=1.0 for open data)
                    d_max = 0.10
                    pos_error = np.linalg.norm(noisy_poses[:, :3] - clean_poses[:, :3], axis=1)
                    alpha_target = np.clip(pos_error / d_max, 0.0, 1.0)

                    # Compute chunk targets
                    chunk_targets = _compute_chunk_targets(noisy_poses, clean_poses, chunk_size=5)

                    # Write to HDF5
                    variant_name = f"{ep_key}_{cfg['label']}"
                    group = noisy_h5.create_group(variant_name)
                    group.create_dataset(
                        f"frames/{clean_h5['meta/camera'][()]}",
                        data=frames.astype(np.uint8),
                    )
                    group.create_dataset("noisy_poses", data=noisy_poses.astype(np.float32))
                    group.create_dataset("gripper", data=gripper.astype(np.float32))
                    group.create_dataset("texts", data=json.dumps(texts))
                    group.create_dataset("smooth_poses", data=clean_poses.astype(np.float32))
                    group.create_dataset("alpha_target", data=alpha_target.astype(np.float32))
                    group.create_dataset("chunk_targets", data=chunk_targets.astype(np.float32))
                    group.create_dataset("meta", data=json.dumps({
                        **meta,
                        "noise_config": cfg,
                        "source": meta.get("source", "open") + "_noisy",
                    }))

                if (ep_idx + 1) % 50 == 0:
                    print(f"  {ep_idx + 1}/{n_eps} episodes processed × {len(noise_configs)} variants")

            noisy_h5["meta/total_episodes"] = n_eps * len(noise_configs)
            noisy_h5["meta/source"] = "synthetic_noise"
            noisy_h5["meta/camera"] = clean_h5["meta/camera"][()]

    output_abs = str(Path(output_path).absolute())
    print(f"  Done: {output_abs} ({n_eps * len(noise_configs)} episodes)")
    return output_abs


def _compute_chunk_targets(noisy_poses, smooth_poses, chunk_size=5):
    """Compute delta chunk targets for Assistant head."""
    N = len(noisy_poses)
    if N <= chunk_size:
        return np.array([])

    n_chunks = N - chunk_size
    chunks = np.zeros((n_chunks, chunk_size, 6))

    for t in range(n_chunks):
        for i in range(1, chunk_size + 1):
            chunks[t, i - 1, :3] = smooth_poses[t + i, :3] - noisy_poses[t, :3]
            chunks[t, i - 1, 3:6] = smooth_poses[t + i, 3:6] - noisy_poses[t, 3:6]

    return chunks


# ================================================================
# Full Training Pipeline
# ================================================================

def run_full_pipeline(
    output_dir: str,
    robomimic_dir: Optional[str] = None,
    droid_dir: Optional[str] = None,
    bridge_dir: Optional[str] = None,
    own_data_dir: Optional[str] = None,
    epochs_pretrain: int = 50,
    epochs_heads: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    stages: str = "all",
    pretrained_checkpoint: Optional[str] = None,
    noise_only: bool = False,
):
    """Run the full ALIGN training pipeline.

    Args:
        output_dir: Base output directory.
        robomimic_dir: Path to Robomimic data.
        droid_dir: Path to DROID data.
        bridge_dir: Path to Bridge data.
        own_data_dir: Path to ALIGN-recorded data (Phase 1).
        epochs_pretrain: Epochs for contrastive pretraining.
        epochs_heads: Epochs for head training.
        batch_size: Training batch size.
        lr: Learning rate.
        stages: Which stages to run: 'all', 'pretrain', 'heads'.
        pretrained_checkpoint: Resume from existing pretrained checkpoint.
        noise_only: Only create noisy datasets, skip training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 0: Convert open datasets ──
    clean_h5 = output_dir / "align_clean.h5"
    noisy_h5 = output_dir / "align_noisy.h5"

    if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
        print(f"[pipeline] Using existing pretrained checkpoint: {pretrained_checkpoint}")
        # Skip dataset conversion
        pass
    else:
        # Convert clean data
        dataset_dirs = []
        dataset_names = []

        if robomimic_dir:
            dataset_dirs.append(robomimic_dir)
            dataset_names.append("robomimic")
        if droid_dir:
            dataset_dirs.append(droid_dir)
            dataset_names.append("droid")
        if bridge_dir:
            dataset_dirs.append(bridge_dir)
            dataset_names.append("bridge")

        if not dataset_dirs and not own_data_dir:
            raise ValueError("At least one dataset directory required")

        if dataset_dirs:
            print("\n[pipeline] Stage 0a: Converting open datasets → clean HDF5")
            create_align_dataset(
                dataset_names=dataset_names,
                data_dirs=dataset_dirs,
                output_path=str(clean_h5),
                max_episodes_per_dataset=5000,
            )

        # Merge with own data if available
        if own_data_dir:
            print("\n[pipeline] Stage 0b: Merging own data")
            import h5py
            own_h5 = output_dir / "align_own.h5"
            convert_raw_to_hdf5(own_data_dir, str(own_h5))

            if clean_h5.exists() and own_h5.exists():
                # Merge — append own data to clean
                with h5py.File(str(clean_h5), "a") as h5:
                    with h5py.File(str(own_h5), "r") as own:
                        offset = len([k for k in h5.keys() if k.startswith("ep_")])
                        for key in own.keys():
                            if key.startswith("ep_"):
                                own.copy(key, h5, name=f"ep_{offset + int(key.split('_')[1]):05d}")
                        h5["meta/total_episodes"] = len([k for k in h5.keys() if k.startswith("ep_")])

        if noise_only:
            print("\n[pipeline] Noise-only mode. Skipping training.")
            # Still generate noisy version
            create_noisy_hdf5(str(clean_h5), str(noisy_h5))
            return

        # ── Stage 0c: Generate noisy variants for head training ──
        print("\n[pipeline] Stage 0c: Generating synthetic noise variants")
        create_noisy_hdf5(str(clean_h5), str(noisy_h5))

    # ── Stage 1: Contrastive Pretraining ──
    if stages in ("all", "pretrain"):
        print(f"\n{'='*60}")
        print(f"[pipeline] Stage 1: 3-Way Contrastive Pretraining ({epochs_pretrain} epochs)")
        print(f"{'='*60}")

        pretrain_checkpoint = output_dir / "checkpoints" / "pretrain" / "best.pt"
        train_cmd = [
            sys.executable, str(Path(__file__).resolve().parent / "pretrain.py"),
            "--data", str(clean_h5),
            "--output-dir", str(pretrain_checkpoint.parent),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs_pretrain),
            "--lr", str(lr),
        ]
        import subprocess
        result = subprocess.run(train_cmd)
        if result.returncode != 0:
            print("[pipeline] ERROR: Pretraining failed")
            sys.exit(1)
        pretrained_path = str(pretrain_checkpoint)
    else:
        if pretrained_checkpoint:
            pretrained_path = pretrained_checkpoint
        else:
            pretrained_path = str(output_dir / "checkpoints" / "pretrain" / "best.pt")
        if not Path(pretrained_path).exists():
            print(f"[pipeline] ERROR: No pretrained checkpoint at {pretrained_path}")
            sys.exit(1)
        print(f"[pipeline] Using existing pretrained checkpoint: {pretrained_path}")

    # ── Stage 2: Head Training ──
    if stages in ("all", "heads"):
        print(f"\n{'='*60}")
        print(f"[pipeline] Stage 2: Head Training ({epochs_heads} epochs)")
        print(f"{'='*60}")

        head_checkpoint = output_dir / "checkpoints" / "heads" / "joint_best.pt"
        train_cmd = [
            sys.executable, str(Path(__file__).resolve().parent / "train_heads.py"),
            "--data", str(noisy_h5),
            "--pretrained", pretrained_path,
            "--output-dir", str(head_checkpoint.parent),
            "--batch-size", str(batch_size),
            "--epochs-decision", str(epochs_heads // 3),
            "--epochs-assistant", str(2 * epochs_heads // 3),
            "--epochs-joint", str(epochs_heads // 3),
            "--lr", str(lr),
        ]
        import subprocess
        result = subprocess.run(train_cmd)
        if result.returncode != 0:
            print("[pipeline] ERROR: Head training failed")
            sys.exit(1)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("[pipeline] Training Complete!")
    print(f"[pipeline]")
    print(f"[pipeline] Checkpoints:")
    print(f"[pipeline]   Pretrain: {pretrained_path}")
    print(f"[pipeline]   Heads:    {head_checkpoint}")
    print(f"[pipeline]")
    print(f"[pipeline] To run inference:")
    print(f"[pipeline]   python inference/align_inference.py \\")
    print(f"[pipeline]       --checkpoint {head_checkpoint} \\")
    print(f"[pipeline]       --task \"pick up the red mug\"")
    print(f"{'='*60}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ALIGN Full Training Pipeline")
    parser.add_argument("--output-dir", default="./training_output", help="Base output directory")
    parser.add_argument("--robomimic-dir", help="Path to Robomimic data")
    parser.add_argument("--droid-dir", help="Path to DROID data")
    parser.add_argument("--bridge-dir", help="Path to Bridge data")
    parser.add_argument("--own-data-dir", help="Path to ALIGN-recorded episodes (Phase 1)")
    parser.add_argument("--epochs-pretrain", type=int, default=50)
    parser.add_argument("--epochs-heads", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stages", default="all", choices=["all", "pretrain", "heads"])
    parser.add_argument("--pretrained", help="Resume from existing pretrained checkpoint")
    parser.add_argument("--noise-only", action="store_true", help="Only generate noisy datasets")

    args = parser.parse_args()

    run_full_pipeline(
        output_dir=args.output_dir,
        robomimic_dir=args.robomimic_dir,
        droid_dir=args.droid_dir,
        bridge_dir=args.bridge_dir,
        own_data_dir=args.own_data_dir,
        epochs_pretrain=args.epochs_pretrain,
        epochs_heads=args.epochs_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        stages=args.stages,
        pretrained_checkpoint=args.pretrained,
        noise_only=args.noise_only,
    )


if __name__ == "__main__":
    main()
