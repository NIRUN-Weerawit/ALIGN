#!/usr/bin/env python3
"""Render a video from dataset frames for a given episode.

Usage:
    # Single camera
    python scripts/render_dataset_video.py \
        --data data/libero_spatial.h5 \
        --episode ep_000000 \
        --camera image \
        --output ep_000000.mp4

    # Multi-camera side-by-side
    python scripts/render_dataset_video.py \
        --data data/libero_spatial.h5 \
        --episode ep_000000 \
        --cameras image wrist_image \
        --output ep_000000.mp4
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Render a video from dataset frames")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--episode", required=True, help="Episode key (e.g. ep_000000)")
    parser.add_argument("--camera", default=None, help="Single camera name")
    parser.add_argument("--cameras", nargs="+", default=["image"],
                        help="Camera names (side-by-side if >1)")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = all)")
    args = parser.parse_args()

    cameras = [args.camera] if args.camera else args.cameras

    try:
        import imageio
    except ImportError:
        print("Error: pip install imageio[ffmpeg]")
        sys.exit(1)

    with h5py.File(args.data, "r") as f:
        if args.episode not in f:
            print(f"Episode '{args.episode}' not found")
            sys.exit(1)
        ep = f[args.episode]
        cam_frames = []
        for cam in cameras:
            if cam not in ep["frames"]:
                print(f"Camera '{cam}' not found, skipping")
                continue
            frames = ep["frames"][cam][:]
            if args.max_frames > 0:
                frames = frames[:args.max_frames]
            cam_frames.append(frames)
        if not cam_frames:
            print("No frames found")
            sys.exit(1)

    n = min(len(f) for f in cam_frames)
    print(f"Frames: {n}, Cameras: {cameras}")

    if args.output is None:
        args.output = f"{args.episode}_{'_'.join(cameras)}.mp4"

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)
    for i in range(n):
        if len(cam_frames) == 1:
            frame = cam_frames[0][i].astype(np.uint8)
        else:
            panels = [cf[i].astype(np.uint8) for cf in cam_frames]
            min_h = min(p.shape[0] for p in panels)
            frame = np.concatenate([p[:min_h] for p in panels], axis=1)
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
