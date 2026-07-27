#!/usr/bin/env python3
"""Pre-compute DINOv2 features for an ALIGN HDF5 dataset.

Encodes every (episode, camera, frame) once with VisionEncoder and writes
the raw features to a directory of per-episode .npy files + an index.json.
Layout matches VisionEncoder.forward output for B=1 exactly (same numerical
result as the in-training forward — no autocast, FP32 weights and activations).

Output structure:
    <output_dir>/
        index.json                       -- {ep_name: {"length": N, "shape": [...]}, ...}
        ep_000000.npy                    -- (N, V*257, 768) float32
        ep_000001.npy
        ...

Why per-episode .npy (not HDF5):
- The dataset reads random segments, each spanning one episode. Per-episode
  .npy files let the DataLoader memmap one file at a time and let the OS
  evict pages we don't need.
- HDF5 random reads of chunked datasets incur per-chunk metadata overhead
  that dominates when each segment = ~30 chunk reads of (1, 514, 768).

Usage:
    python scripts/precompute_dinov2.py \
        --data data/libero_spatial.h5 \
        --cameras image wrist_image \
        --output data/libero_spatial.dinov2 \
        --device cuda

Notes:
- FP32 storage is ~3x larger than fp16 but guarantees zero precision drift.
- Pre-encode uses no_grad + FP32 (no autocast) to match training exactly.
- Per-frame processing (B_eff=1) keeps pre-encode VRAM trivial.
- Cross-camera attention is applied per-frame (B=1, V*P=512), matching training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from models.align_model import VisionEncoder


def build_encoder(device: torch.device, num_cameras: int) -> VisionEncoder:
    """Build VisionEncoder matching ALIGNIntentionModel defaults.

    Default: dinov2_vitb14, use_patch_tokens=True, fusion_type="transformer".
    `num_cameras` must match the V dimension of the input (set to len(--cameras)).
    """
    return VisionEncoder(
        backbone="dinov2_vitb14",
        embed_dim=256,           # unused for v2 patch mode, kept for API compat
        num_cameras=num_cameras,
        use_patch_tokens=True,
        fusion_type="transformer",
    ).to(device).eval()


def encode_frame(encoder: VisionEncoder, frame: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """Encode one frame with all cameras through VisionEncoder.

    Args:
        encoder: VisionEncoder (frozen, eval mode)
        frame:   (V, H, W, 3) uint8 — all cameras for one timestep
        device:  torch device

    Returns:
        (V*257, 768) float32 numpy — same layout as VisionEncoder.forward(B=1, V, ...)
    """
    # (V, H, W, 3) uint8 → (1, V, H, W, 3) float on device
    x = torch.from_numpy(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        # VisionEncoder handles resize to 224, normalize, backbone forward,
        # and cross-camera attention internally. No autocast — FP32 to match
        # the in-training pre-encode path exactly.
        features = encoder(x)  # (1, V*257, 768)
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute DINOv2 features for an ALIGN HDF5 dataset"
    )
    parser.add_argument("--data", required=True,
                        help="Path to ALIGN HDF5 dataset (input)")
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="Camera names to encode (e.g. 'image wrist_image')")
    parser.add_argument("--output", required=True,
                        help="Output DIRECTORY for per-episode .npy files + index.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    args = parser.parse_args()

    src_path = Path(args.data)
    dst_dir = Path(args.output)

    if dst_dir.exists():
        if not args.overwrite:
            print(f"  Output exists: {dst_dir} (use --overwrite to replace)")
            return
        # Wipe old contents
        for child in dst_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DINOv2 Pre-compute ===")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_dir}")
    print(f"  Cameras: {args.cameras}")
    print(f"  Device:  {args.device}")

    device = torch.device(args.device)
    V = len(args.cameras)
    encoder = build_encoder(device, num_cameras=V)
    # Freeze all params (already in eval mode, but make requires_grad explicit)
    for p in encoder.parameters():
        p.requires_grad = False

    # Discover episodes and probe shape
    with h5py.File(src_path, "r") as src:
        episodes = sorted(k for k in src.keys() if k.startswith("ep_"))
        sample_ep = src[episodes[0]]
        sample_frames = sample_ep["frames"][args.cameras[0]]
        H, W = sample_frames.shape[1:3]
        print(f"  Episodes: {len(episodes)}")
        print(f"  Frame size: {H}x{W}")
        print(f"  Camera shape (per frame): ({V}, {H}, {W}, 3)")

    out_tokens = V * 257
    out_dim = 768

    # Build index as we go (don't hold all episodes in memory)
    index = {}
    total_size_bytes = 0

    with h5py.File(src_path, "r") as src:
        for ep_name in tqdm(episodes, desc="  Episodes", unit="ep"):
            ep_src = src[ep_name]
            n_frames = ep_src["frames"][args.cameras[0]].shape[0]

            # Allocate in-memory buffer for this episode and encode all frames
            ep_features = np.empty(
                (n_frames, out_tokens, out_dim), dtype=np.float32,
            )
            # Load all frames for this episode (per camera) into memory once
            frames_all = np.stack([
                ep_src["frames"][cam][:] for cam in args.cameras
            ], axis=1)  # (N, V, H, W, 3) uint8

            for t in range(n_frames):
                ep_features[t] = encode_frame(encoder, frames_all[t], device)

            # Save as per-episode .npy
            npy_path = dst_dir / f"{ep_name}.npy"
            np.save(npy_path, ep_features)
            total_size_bytes += npy_path.stat().st_size

            index[ep_name] = {
                "length": n_frames,
                "tokens": out_tokens,
                "dim": out_dim,
            }

    # Write index.json
    index_path = dst_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n  Done. Wrote {len(episodes)} .npy files + index.json")
    print(f"  Shape per frame: ({out_tokens}, {out_dim}) float32")
    print(f"  Total size: {total_size_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()