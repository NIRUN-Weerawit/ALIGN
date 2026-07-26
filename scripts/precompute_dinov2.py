#!/usr/bin/env python3
"""Pre-compute DINOv2 features for an ALIGN HDF5 dataset.

Encodes every (episode, camera, frame) once with VisionEncoder and writes
the raw features (V*257, 768) float32 to a sidecar .h5 file. Layout matches
VisionEncoder.forward output for B=1 exactly (same numerical result as the
in-training forward — no autocast, FP32 weights and activations).

Usage:
    python scripts/precompute_dinov2.py \
        --data data/libero_spatial.h5 \
        --cameras image wrist_image \
        --output data/libero_spatial.dinov2.h5 \
        --device cuda \
        [--batch-size 64]

Sidecar layout (per episode, mirrors main HDF5):
    ep_000000/
        dinov2/         (N, V*257, 768) float32   -- VisionEncoder forward output

Notes:
- FP32 storage is ~3x larger than fp16 but guarantees zero precision drift.
- Pre-encode uses no_grad + FP32 (no autocast) to match training exactly.
- Per-frame processing (B_eff=1) keeps pre-encode VRAM trivial.
- Cross-camera attention is applied per-frame (B=1, V*P=512), matching training.
"""
from __future__ import annotations

import argparse
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
                        help="Output HDF5 path for sidecar features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per VisionEncoder forward (default 1 = "
                             "lowest VRAM; larger values batch frames within "
                             "an episode for speed)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    src_path = Path(args.data)
    dst_path = Path(args.output)

    if dst_path.exists() and not args.overwrite:
        print(f"  Output exists: {dst_path} (use --overwrite to replace)")
        return

    print(f"\n=== DINOv2 Pre-compute ===")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    print(f"  Cameras: {args.cameras}")
    print(f"  Device:  {args.device}")

    device = torch.device(args.device)
    V = len(args.cameras)
    encoder = build_encoder(device, num_cameras=V)
    # Freeze all params (already in eval mode, but make requires_grad explicit)
    for p in encoder.parameters():
        p.requires_grad = False

    # Discover episodes
    with h5py.File(src_path, "r") as src:
        episodes = sorted(k for k in src.keys() if k.startswith("ep_"))
        # Probe one episode for shape sanity check
        sample_ep = src[episodes[0]]
        sample_frames = sample_ep["frames"][args.cameras[0]]
        H, W = sample_frames.shape[1:3]
        print(f"  Episodes: {len(episodes)}")
        print(f"  Frame size: {H}x{W}")
        print(f"  Camera shape (per frame): ({len(args.cameras)}, {H}, {W}, 3)")
        V = len(args.cameras)
        out_tokens = V * 257
        out_dim = 768

    # Build output file with same episode structure
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for ep_name in tqdm(episodes, desc="  Episodes", unit="ep"):
            ep_src = src[ep_name]
            ep_dst = dst.create_group(ep_name)
            n_frames = ep_src["frames"][args.cameras[0]].shape[0]

            # Pre-allocate output for this episode
            dinov2_dst = ep_dst.create_dataset(
                "dinov2",
                shape=(n_frames, out_tokens, out_dim),
                dtype=np.float32,
                chunks=(1, out_tokens, out_dim),  # chunk per-frame for fast writes
            )

            # Load all frames for this episode (per camera) and encode.
            # Per-frame VisionEncoder forward: V*257 tokens, 768 dim.
            # At B=1, peak VRAM is ~86M params + tiny activations.
            frames_all = np.stack([
                ep_src["frames"][cam][:] for cam in args.cameras
            ], axis=1)  # (N, V, H, W, 3) uint8

            for t in range(n_frames):
                feat = encode_frame(encoder, frames_all[t], device)
                dinov2_dst[t] = feat

    print(f"\n  Done. Wrote {dst_path}")
    print(f"  Shape per frame: ({out_tokens}, {out_dim}) float32")
    print(f"  File size: {dst_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()