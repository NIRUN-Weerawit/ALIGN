#!/usr/bin/env python3
"""Download LIBERO LeRobot v3 dataset for local training.

Pre-downloads the full dataset so you don't need to stream from HF Hub.
Requires roughly 15-25GB disk space.

Usage:
    python scripts/download_libero.py
    python scripts/download_libero.py --subsets libero_10 libero_90  # pick subsets

After download, training automatically picks up the local data.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "nvidia/LIBERO_LeRobot_v3"
DEFAULT_SUBSETS = ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"]
HOME = Path.home()
DEFAULT_DEST = HOME / ".cache" / "huggingface" / "lerobot" / REPO_ID


def main():
    parser = argparse.ArgumentParser(description="Download LIBERO for local training")
    parser.add_argument("--subsets", nargs="*", default=DEFAULT_SUBSETS,
                        help="Subsets to download (default: all 5)")
    parser.add_argument("--dest", default=str(DEFAULT_DEST),
                        help=f"Download destination (default: {DEFAULT_DEST})")
    parser.add_argument("--skip-meta", action="store_true",
                        help="Skip meta download (if already cached)")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} to {dest}")
    print(f"Subsets: {args.subsets}")

    if not args.skip_meta:
        patterns = ["*/meta/*"]
    else:
        patterns = []

    for subset in args.subsets:
        patterns.extend([
            f"{subset}/data/**/*.parquet",
            f"{subset}/videos/**/*.mp4",
        ])

    local_dir = Path(snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision="main",
        allow_patterns=patterns,
        local_dir=dest,
        local_dir_use_symlinks=False,
    ))

    # Print sizes
    total = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file())
    print(f"\nDownloaded to: {local_dir}")
    print(f"Total size: {total / 1e9:.1f} GB")

    # Each subset lives at local_dir/<subset>/ with its own meta/info.json
    # StreamingLeRobotDataset(root=dir) loads dir/meta/info.json directly
    print(f"\nDone. Train with (point at subset dir):")
    for subset in args.subsets:
        subset_path = local_dir / subset
        if subset_path.exists():
            print(f"  python training/pretrain_streaming.py \\")
            print(f"      --data-dir {subset_path}")
            break  # show first as example
    else:
        print(f"  python training/pretrain_streaming.py --data-dir {local_dir}")


if __name__ == "__main__":
    main()