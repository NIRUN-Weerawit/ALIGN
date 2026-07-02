#!/usr/bin/env python3
"""Flip sim frames before encoding in eval_libero_trajectory.py.

The simulation renders images with a coordinate convention that's
flipped relative to the training data. Without flipping, the encoder
receives upside-down images at eval time and produces bad embeddings.

This script:
  1. Applies a vertical flip (np.flipud) to all sim frames
  2. Optionally applies a horizontal flip (np.fliplr) via --horizontal
  3. Verifies the flip by saving before/after images

Usage:
  python eval/flip_sim_frames.py \
      --input /tmp/before.png \
      --output /tmp/after.png
  # or to flip all .png files in a directory:
  python eval/flip_sim_frames.py \
      --input-dir /tmp/frames/ \
      --output-dir /tmp/frames_flipped/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def flip_image(image: np.ndarray, vertical: bool = True, horizontal: bool = False) -> np.ndarray:
    """Flip an image vertically and/or horizontally.

    Args:
        image: (H, W, 3) uint8 RGB image
        vertical: if True, flip upside-down (np.flipud)
        horizontal: if True, flip left-right (np.fliplr)

    Returns:
        Flipped image of same shape and dtype
    """
    if vertical:
        image = np.flipud(image)
    if horizontal:
        image = np.fliplr(image)
    return image


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(image: np.ndarray, path: str) -> None:
    Image.fromarray(image).save(path)


def process_directory(input_dir: str, output_dir: str, vertical: bool, horizontal: bool):
    """Flip all images in a directory and write to output_dir."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in in_path.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])
    print(f"Found {len(image_files)} images in {input_dir}")
    for f in image_files:
        img = load_image(str(f))
        flipped = flip_image(img, vertical=vertical, horizontal=horizontal)
        out_file = out_path / f.name
        save_image(flipped, str(out_file))
        print(f"  {f.name} -> {out_file}")


def demo_before_after(input_path: str, output_path: str, vertical: bool, horizontal: bool):
    """Flip a single image and show before/after stats."""
    img = load_image(input_path)
    print(f"Input:  {input_path}")
    print(f"  shape={img.shape}, dtype={img.dtype}")
    print(f"  top-left RGB:     {img[0, 0].tolist()}")
    print(f"  bottom-left RGB:  {img[-1, 0].tolist()}")
    print(f"  center RGB:       {img[img.shape[0] // 2, img.shape[1] // 2].tolist()}")

    flipped = flip_image(img, vertical=vertical, horizontal=horizontal)
    print(f"Output: {output_path}")
    print(f"  shape={flipped.shape}, dtype={flipped.dtype}")
    print(f"  top-left RGB:     {flipped[0, 0].tolist()}")
    print(f"  bottom-left RGB:  {flipped[-1, 0].tolist()}")
    print(f"  center RGB:       {flipped[flipped.shape[0] // 2, flipped.shape[1] // 2].tolist()}")

    save_image(flipped, output_path)
    print(f"[OK] Saved flipped image to {output_path}")

    # Sanity check
    if vertical:
        # After vertical flip, top of output = bottom of input
        if np.array_equal(flipped[0], img[-1]):
            print("[OK] Vertical flip verified: top == input's bottom row")
        else:
            print("[WARN] Vertical flip does NOT match expected transformation")
    if horizontal:
        if np.array_equal(flipped[:, 0], img[:, -1]):
            print("[OK] Horizontal flip verified: left == input's right column")
        else:
            print("[WARN] Horizontal flip does NOT match expected transformation")


def main():
    parser = argparse.ArgumentParser(
        description="Flip sim frames (vertical/horizontal) to match training data orientation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single image to flip")
    group.add_argument("--input-dir", help="Directory of images to flip")

    parser.add_argument("--output", help="Output image path (for --input)")
    parser.add_argument("--output-dir", help="Output directory (for --input-dir)")
    parser.add_argument("--vertical", action="store_true", default=True,
                        help="Apply vertical flip (default: True)")
    parser.add_argument("--no-vertical", dest="vertical", action="store_false")
    parser.add_argument("--horizontal", action="store_true", default=False,
                        help="Apply horizontal flip (default: False)")
    parser.add_argument("--no-horizontal", dest="horizontal", action="store_false")
    args = parser.parse_args()

    if args.input and not args.output:
        parser.error("--output is required when --input is used")
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when --input-dir is used")

    if args.input:
        demo_before_after(args.input, args.output, args.vertical, args.horizontal)
    else:
        process_directory(args.input_dir, args.output_dir, args.vertical, args.horizontal)


if __name__ == "__main__":
    main()