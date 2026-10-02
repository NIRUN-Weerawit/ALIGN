#!/usr/bin/env python3
"""Pre-compute DINOv2 features for an ALIGN HDF5 dataset.

Encodes every (episode, camera, frame) once with VisionEncoder and writes
the raw features to a directory of per-episode .npy files + an index.json.
Layout matches train_intention.py's per-camera _vision_forward calls. The
training path flattens cameras into the batch before calling VisionEncoder,
so cross-camera attention is not applied to these frozen backbone features.

Output structure:
    <output_dir>/
        index.json                       -- cache metadata + per-episode lengths
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
- FP32 storage is 2x larger than fp16 but avoids precision reduction.
- Pre-encode uses no_grad + FP32 (no autocast) to match training precision.
- `--batch-size` controls the number of image frames per backbone call.
- Camera order is preserved as [cam0 patches, cam0 CLS, cam1 patches, cam1 CLS].
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from models.align_model import VisionEncoder


def build_encoder(device: torch.device) -> VisionEncoder:
    """Build the single-camera encoder used by the training forward path."""
    return VisionEncoder(
        backbone="dinov2_vitb14",
        embed_dim=256,           # unused for v2 patch mode, kept for API compat
        num_cameras=1,
        use_patch_tokens=True,
        fusion_type="transformer",
    ).to(device).eval()


def encode_frames(encoder: VisionEncoder, frames: np.ndarray,
                  device: torch.device, batch_size: int) -> np.ndarray:
    """Encode (time, camera) frames exactly as training flattens them.

    Args:
        encoder: VisionEncoder (frozen, eval mode)
        frames:  (T, V, H, W, 3) uint8
        device:  torch device
        batch_size: number of camera images per backbone call

    Returns:
        (T, V*257, 768) float32 numpy
    """
    T, V, H, W, C = frames.shape
    flat = frames.reshape(T * V, H, W, C)
    features = np.empty((T * V, 257, 768), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            x = torch.from_numpy(flat[start:start + batch_size]).to(device)
            features[start:start + len(x)] = encoder(x).cpu().numpy()
    return features.reshape(T, V * 257, 768)


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
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Camera images per DINOv2 call (default: 16)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

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
                shutil.rmtree(child)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DINOv2 Pre-compute ===")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_dir}")
    print(f"  Cameras: {args.cameras}")
    print(f"  Device:  {args.device}")

    device = torch.device(args.device)
    V = len(args.cameras)
    # Match train_intention.py's vision settings. Precision is kept at FP32.
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True

    # Discover episodes and check capacity before writing a large FP32 cache.
    with h5py.File(src_path, "r") as src:
        episodes = sorted(k for k in src.keys() if k.startswith("ep_"))
        if not episodes:
            raise ValueError(f"No episodes in {src_path}")
        sample_ep = src[episodes[0]]
        sample_frames = sample_ep["frames"][args.cameras[0]]
        H, W = sample_frames.shape[1:3]
        total_frames = sum(src[ep]["frames"][args.cameras[0]].shape[0]
                           for ep in episodes)
        print(f"  Episodes: {len(episodes)}")
        print(f"  Frame size: {H}x{W}")
        print(f"  Camera shape (per frame): ({V}, {H}, {W}, 3)")

    out_tokens = V * 257
    out_dim = 768
    expected_bytes = total_frames * out_tokens * out_dim * np.dtype(np.float32).itemsize
    available_bytes = shutil.disk_usage(dst_dir).free
    print(f"  Expected cache size: {expected_bytes / 1e9:.2f} GB")
    if expected_bytes + 2 * 1024**3 > available_bytes:
        raise OSError(
            f"Insufficient free space at {dst_dir}: need about "
            f"{(expected_bytes + 2 * 1024**3) / 1e9:.2f} GB including "
            f"safety margin, have {available_bytes / 1e9:.2f} GB"
        )
    encoder = build_encoder(device)
    # Freeze all params (already in eval mode, but make requires_grad explicit)
    for p in encoder.parameters():
        p.requires_grad = False

    # Build index as we go (don't hold all episodes in memory)
    index = {}
    total_size_bytes = 0

    with h5py.File(src_path, "r") as src:
        for ep_name in tqdm(episodes, desc="  Episodes", unit="ep"):
            ep_src = src[ep_name]
            n_frames = ep_src["frames"][args.cameras[0]].shape[0]

            # Load all frames for this episode (per camera) into memory once
            frames_all = np.stack([
                ep_src["frames"][cam][:] for cam in args.cameras
            ], axis=1)  # (N, V, H, W, 3) uint8

            ep_features = encode_frames(encoder, frames_all, device, args.batch_size)

            # Save as per-episode .npy
            npy_path = dst_dir / f"{ep_name}.npy"
            np.save(npy_path, ep_features)
            total_size_bytes += npy_path.stat().st_size

            index[ep_name] = {
                "length": n_frames,
                "tokens": out_tokens,
                "dim": out_dim,
            }

    # Metadata lets ALIGNDataset reject legacy multi-camera caches that were
    # produced with cross-camera attention and do not match training inputs.
    index["__meta__"] = {
        "format": "align-dinov2-per-camera-v2",
        "cameras": args.cameras,
        "source": str(src_path.resolve()),
        "dtype": "float32",
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
