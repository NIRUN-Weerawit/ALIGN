#!/usr/bin/env python3
"""ALIGN HDF5 Conversion — batch-convert episode directories into per-split HDF5 files.

Takes a directory of episodes (each with frames/, data.npz, meta.json),
runs ground truth generation if needed, and writes train.h5 / val.h5 / test.h5.

Usage:
    # Convert all episodes with default 80/10/10 split
    python convert_to_h5.py --input-dir ./align_data --output-dir ./h5_data

    # Custom split ratios
    python convert_to_h5.py --input-dir ./align_data --train-ratio 0.7 --val-ratio 0.15

    # Skip ground truth (already generated)
    python convert_to_h5.py --input-dir ./align_data --skip-gt

    # Dry run — show what would be done
    python convert_to_h5.py --input-dir ./align_data --dry-run

HDF5 structure per split file:
    /ep_0000/
        frames          (N, H, W, 3) uint8, chunked + lzf compressed
        noisy_poses     (N, 7) float64
        smooth_poses    (N, 7) float64
        alpha_target    (N,) float32
        chunk_targets   (N-K, K, 6) float32
        gripper_states  (N,) float64
        timestamps      (N,) float64
        is_approach     (N,) bool
    Attributes on each group:
        task_description, target_object, operator_id, num_frames, duration_s
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Try to import h5py — it's the only non-stdlib dependency
try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)

# Import ground truth generator
GT_SCRIPT = Path(__file__).parent / "generate_ground_truth.py"


# ================================================================
# Helpers
# ================================================================

def load_episode_data(episode_dir: Path) -> dict:
    """Load all data from an episode directory into a dict of arrays."""
    # Frames
    frames_dir = episode_dir / "frames"
    camera_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
    if camera_dirs:
        label_dir = camera_dirs[0]
        frame_files = sorted(label_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    else:
        frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = []
    for f in frame_files:
        img = Image.open(f)
        frames.append(np.array(img))
    frames_arr = np.stack(frames, axis=0) if frames else np.array([])

    # NPZ data
    npz_path = episode_dir / "data.npz"
    data = dict(np.load(npz_path))

    # Metadata
    meta_path = episode_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "frames": frames_arr,
        "noisy_poses": data.get("noisy_poses", np.array([])),
        "smooth_poses": data.get("smooth_poses", None),
        "alpha_target": data.get("alpha_target", None),
        "gripper_states": data.get("gripper_states", np.array([])),
        "timestamps": data.get("timestamps", np.array([])),
        "meta": meta,
    }


def has_ground_truth(episode_dir: Path) -> bool:
    """Check if ground truth has already been generated for this episode."""
    npz_path = episode_dir / "data.npz"
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path) as npz:
            return "smooth_poses" in npz and "alpha_target" in npz
    except Exception:
        return False


def run_ground_truth(episode_dir: Path) -> bool:
    """Run generate_ground_truth.py on a single episode. Returns True on success."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(GT_SCRIPT), "--episode", str(episode_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  [GT] FAILED: {result.stderr.strip()}")
        return False
    # Verify output exists
    return has_ground_truth(episode_dir)


def write_episode_to_h5(h5_group, data: dict, chunk_size: int = 5):
    """Write one episode's data into an HDF5 group.

    Args:
        h5_group: h5py Group to write into.
        data: dict from load_episode_data().
        chunk_size: K for chunk targets.
    """
    frames = data["frames"]
    noisy = data["noisy_poses"]
    smooth = data.get("smooth_poses")
    alpha = data.get("alpha_target")
    gripper = data["gripper_states"]
    timestamps = data["timestamps"]
    meta = data["meta"]

    N = len(frames)

    # ── Frames: chunked + compressed ──
    # Chunk shape: 1 frame per chunk for efficient random access
    H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
    ds = h5_group.create_dataset(
        "frames", shape=(N, H, W, C), dtype=np.uint8,
        chunks=(1, H, W, C),  # one frame per chunk → O(1) random access
        compression="lzf",     # fast compression, good for uint8
        shuffle=True,
    )
    ds[:] = frames

    # ── Poses ──
    h5_group.create_dataset("noisy_poses", data=noisy, compression="gzip", compression_opts=4)
    if smooth is not None:
        h5_group.create_dataset("smooth_poses", data=smooth, compression="gzip", compression_opts=4)
    if alpha is not None:
        h5_group.create_dataset("alpha_target", data=alpha.astype(np.float32), compression="gzip", compression_opts=4)

    # ── Chunk targets (if available) ──
    ep_dir = data.get("episode_dir")
    chunk_path = Path(ep_dir) / "chunk_targets.npz" if ep_dir else None
    if chunk_path and chunk_path.exists():
        chunk_data = np.load(chunk_path)["chunk_targets"]
        h5_group.create_dataset("chunk_targets", data=chunk_data, compression="gzip", compression_opts=4)
    elif smooth is not None and len(noisy) > chunk_size:
        # Compute on the fly
        from generate_ground_truth import compute_chunk_targets
        chunk_data = compute_chunk_targets(noisy, smooth, chunk_size)
        if len(chunk_data) > 0:
            h5_group.create_dataset("chunk_targets", data=chunk_data, compression="gzip", compression_opts=4)

    # ── Other arrays ──
    h5_group.create_dataset("gripper_states", data=gripper, compression="gzip", compression_opts=4)
    h5_group.create_dataset("timestamps", data=timestamps, compression="gzip", compression_opts=4)

    # ── Attributes (metadata) ──
    for key in ["task_description", "target_object", "operator_id", "notes"]:
        val = meta.get(key, "")
        if val:
            h5_group.attrs[key] = val
    h5_group.attrs["num_frames"] = N
    h5_group.attrs["duration_s"] = meta.get("duration_s", 0.0)
    h5_group.attrs["pose_format"] = meta.get("pose_format", "xyz_quat")
    h5_group.attrs["camera_label"] = str(meta.get("camera_label", "wrist"))


# ================================================================
# Main conversion
# ================================================================

def convert(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    skip_gt: bool = False,
    dry_run: bool = False,
    chunk_size: int = 5,
    seed: int = 42,
):
    """Convert all episodes in input_dir to per-split HDF5 files."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # ── Discover episodes ──
    episode_dirs = sorted([
        p for p in in_path.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "data.npz").exists()
    ])

    if not episode_dirs:
        print(f"No episodes found in {input_dir}")
        return

    print(f"Found {len(episode_dirs)} episodes in {input_dir}")
    print()

    # ── Run ground truth if needed ──
    if not skip_gt:
        gt_needed = [ep for ep in episode_dirs if not has_ground_truth(ep)]
        if gt_needed:
            print(f"Ground truth needed for {len(gt_needed)} episodes:")
            for ep in gt_needed:
                print(f"  {ep.name}...", end=" ", flush=True)
                if dry_run:
                    print("(dry run, skipped)")
                else:
                    ok = run_ground_truth(ep)
                    print("OK" if ok else "FAILED")
            print()
        else:
            print("All episodes have ground truth. Skipping GT generation.")
            print()

    # ── Split episodes ──
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(episode_dirs))
    n_train = int(len(episode_dirs) * train_ratio)
    n_val = int(len(episode_dirs) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": [episode_dirs[i] for i in train_idx],
        "val": [episode_dirs[i] for i in val_idx],
        "test": [episode_dirs[i] for i in test_idx],
    }

    print(f"Split: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")
    print()

    if dry_run:
        print("=== DRY RUN — no files written ===")
        for split_name, eps in splits.items():
            print(f"\n{split_name}.h5 would contain:")
            for ep in eps:
                n_frames = len(list((ep / "frames").glob("*.jpg")))
                print(f"  {ep.name} ({n_frames} frames)")
        print()
        print(f"Output directory: {out_path}")
        return

    # ── Write HDF5 files ──
    out_path.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    for split_name, eps in splits.items():
        if not eps:
            print(f"{split_name}.h5: no episodes, skipping")
            continue

        h5_path = out_path / f"{split_name}.h5"
        print(f"Writing {split_name}.h5 ({len(eps)} episodes)...")

        with h5py.File(h5_path, "w") as f:
            # Global attributes
            f.attrs["split"] = split_name
            f.attrs["num_episodes"] = len(eps)
            f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
            f.attrs["chunk_size"] = chunk_size

            for ep_idx, ep_dir in enumerate(eps):
                ep_name = ep_dir.name
                print(f"  [{ep_idx + 1}/{len(eps)}] {ep_name}...", end=" ", flush=True)

                # Load data
                data = load_episode_data(ep_dir)
                data["episode_dir"] = ep_dir  # for chunk target path lookup

                N = len(data["frames"])
                if N == 0:
                    print("SKIP (no frames)")
                    continue

                # Write to HDF5
                grp = f.create_group(ep_name)
                write_episode_to_h5(grp, data, chunk_size)

                total_frames += N
                total_episodes += 1
                print(f"{N} frames")

        # Print file size
        size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"  → {h5_path.name}: {size_mb:.1f} MB")
        print()

    elapsed = time.time() - start_time
    print("=" * 50)
    print(f"Conversion complete: {total_episodes} episodes, {total_frames} frames")
    print(f"Time: {elapsed:.1f}s ({total_frames / max(elapsed, 0.1):.0f} frames/s)")
    print(f"Output: {out_path}")
    print("=" * 50)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALIGN HDF5 Conversion — batch-convert episodes to per-split HDF5 files"
    )
    parser.add_argument("--input-dir", type=str, default="./align_data",
                        help="Directory containing episode directories (default: ./align_data)")
    parser.add_argument("--output-dir", type=str, default="./h5_data",
                        help="Output directory for HDF5 files (default: ./h5_data)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training split ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="Chunk size K for Assistant head (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--skip-gt", action="store_true",
                        help="Skip ground truth generation (assume already done)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    args = parser.parse_args()

    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print(f"ERROR: train ({args.train_ratio}) + val ({args.val_ratio}) must be < 1.0")
        sys.exit(1)

    convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        skip_gt=args.skip_gt,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
