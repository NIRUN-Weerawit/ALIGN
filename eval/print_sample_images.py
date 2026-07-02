#!/usr/bin/env python3
"""Print sample images from the dataset to verify camera + orientation.

This script:
  1. Opens a specified HDF5 dataset
  2. Loads N sample frames from each available camera
  3. Saves them as a side-by-side comparison image
  4. Prints metadata (camera names, shapes, dtypes)
  5. Reports whether images look "upside down" (sky at bottom)

Run:
  python eval/print_sample_images.py \\
      --data /path/to/libero_10.h5 \\
      --episode 0 \\
      --frame-idx 50 \\
      --output /tmp/sample_images.png
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_image_sample(h5_path: str, episode: int, frame_idx: int):
    """Load all camera images for a given episode + frame."""
    with h5py.File(h5_path, "r") as f:
        ep_key = f"ep_{episode:06d}"
        if ep_key not in f:
            raise ValueError(f"Episode {ep_key} not found in {h5_path}")
        ep = f[ep_key]
        if "frames" not in ep:
            raise ValueError(f"No frames group in {ep_key}")
        frames_group = ep["frames"]
        if isinstance(frames_group, h5py.Dataset):
            # Single camera, no subgroups
            frames = frames_group[frame_idx]
            return {"image": frames}

        # Multi-camera: dict of camera_name -> (H, W, 3) image
        result = {}
        for cam_name in frames_group.keys():
            result[cam_name] = frames_group[cam_name][frame_idx]
        return result


def analyze_orientation(image: np.ndarray) -> str:
    """Heuristic: check if image appears upside-down.

    The convention for LIBERO MuJoCo:
      - 'image' / 'agentview': top of frame = far side of workspace, bottom = near side
        This is the standard "looking down" view, not upside down.
      - 'wrist_image': the gripper-mounted camera, oriented by robot pose

    Heuristic: check if the top row is much darker than the bottom row.
    In a typical scene with the table below, the top is sky/bg (brighter) and
    bottom is the table surface (darker). If this is REVERSED (top dark, bottom
    bright), the image is upside down.

    Returns: "normal", "upside_down", or "indeterminate"
    """
    if image.ndim == 2:
        # Grayscale
        top_mean = image[: image.shape[0] // 4].mean()
        bottom_mean = image[3 * image.shape[0] // 4 :].mean()
    else:
        # RGB — convert to luminance
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        top_mean = lum[: lum.shape[0] // 4].mean()
        bottom_mean = lum[3 * lum.shape[0] // 4 :].mean()

    diff = top_mean - bottom_mean
    # If top is much brighter than bottom (>20 luminance), it's normal
    # If bottom is much brighter than top (<-20 luminance), it's upside down
    if diff > 20:
        return "normal (top brighter than bottom)"
    elif diff < -20:
        return "UPSIDE DOWN (bottom brighter than top)"
    else:
        return f"indeterminate (top-bottom diff: {diff:.1f})"


def print_image_stats(image: np.ndarray, cam_name: str):
    """Print detailed image statistics."""
    print(f"  [{cam_name}]")
    print(f"    shape: {image.shape}, dtype: {image.dtype}")
    print(f"    min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
    # Corner colors
    h, w = image.shape[:2]
    corners = {
        "top-left":     image[0, 0],
        "top-right":    image[0, w - 1],
        "bottom-left":  image[h - 1, 0],
        "bottom-right": image[h - 1, w - 1],
        "center":       image[h // 2, w // 2],
    }
    for name, val in corners.items():
        if image.ndim == 3:
            print(f"    {name}: RGB={val.tolist()}")
        else:
            print(f"    {name}: {val}")
    # Orientation analysis
    print(f"    orientation: {analyze_orientation(image)}")


def save_comparison_grid(images: dict, output_path: str, frame_idx: int, episode: int):
    """Save a grid of all camera views for visual inspection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib not installed — skipping grid save")
        return False

    n_cams = len(images)
    fig, axes = plt.subplots(1, n_cams, figsize=(5 * n_cams, 5))
    if n_cams == 1:
        axes = [axes]
    for ax, (cam_name, image) in zip(axes, images.items()):
        ax.imshow(image)
        ax.set_title(f"{cam_name}\nshape={image.shape}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Episode {episode}, Frame {frame_idx}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Print sample images from the dataset to verify camera + orientation"
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (0-based)")
    parser.add_argument("--frame-idx", type=int, default=50, help="Frame index in episode")
    parser.add_argument(
        "--n-frames", type=int, default=4,
        help="Number of frames to sample evenly across the episode"
    )
    parser.add_argument("--output", default="/tmp/sample_images.png",
                        help="Output image path for grid view")
    parser.add_argument("--print-stats", action="store_true", default=True,
                        help="Print per-image statistics (default: True)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: HDF5 file not found: {args.data}")
        sys.exit(1)

    print(f"\n=== Sample Images from {args.data} ===")
    print(f"  Episode:    {args.episode}")
    print(f"  Frame idx:  {args.frame_idx}")

    # Get episode length
    with h5py.File(args.data, "r") as f:
        ep_key = f"ep_{args.episode:06d}"
        if ep_key not in f:
            print(f"ERROR: {ep_key} not found")
            sys.exit(1)
        ep_group = f[ep_key]
        if "frames" in ep_group:
            frames_group = ep_group["frames"]
            if isinstance(frames_group, h5py.Group):
                n_frames = len(list(frames_group.values())[0])
                cameras = list(frames_group.keys())
            else:
                n_frames = len(frames_group)
                cameras = ["single"]
        else:
            print("ERROR: no frames group")
            sys.exit(1)
    print(f"  Episode length: {n_frames} frames")
    print(f"  Cameras:        {cameras}")
    print()

    # Load a sample frame
    print(f"=== Frame {args.frame_idx} ===")
    images = load_image_sample(args.data, args.episode, args.frame_idx)
    for cam_name, image in images.items():
        if args.print_stats:
            print_image_stats(image, cam_name)

    # Save comparison grid
    print()
    if save_comparison_grid(images, args.output, args.frame_idx, args.episode):
        print(f"[OK] Saved grid to {args.output}")
    else:
        print(f"[!] Skipped grid save (matplotlib not available)")

    # Sample N frames evenly across the episode
    print()
    print(f"=== Sampling {args.n_frames} frames evenly across episode ===")
    frame_indices = np.linspace(0, n_frames - 1, args.n_frames, dtype=int)
    for fidx in frame_indices:
        imgs = load_image_sample(args.data, args.episode, int(fidx))
        for cam_name, image in imgs.items():
            orient = analyze_orientation(image)
            print(f"  frame {fidx:3d}, [{cam_name:15s}]: {orient}")


if __name__ == "__main__":
    main()