#!/usr/bin/env python3
"""Pre-decode LIBERO LeRobot dataset into HDF5 for fast multiprocess training.

Converts local LIBERO data (MP4 videos + Parquet metadata) into HDF5 files
with pre-decoded JPEG frames, so the DataLoader can read frames directly
without FFmpeg/video decoding. This:
  - Enables num_workers > 0 (no video decoder contention)
  - Speeds up training ~3-5x (no per-step video decode)
  - Adds ~5-15GB per LIBERO subtask (libero_10, libero_90, etc.)

Usage:
    # Convert one subtask
    python scripts/decode_libero_to_hdf5.py \
        --data-dir /path/to/libero_10 \
        --output ./data/libero_10.h5

    # Convert multiple subtasks (merged into one HDF5)
    python scripts/decode_libero_to_hdf5.py \
        --data-dir /path/to/libero_10 --data-dir /path/to/libero_90 \
        --output ./data/libero_all.h5

    # Limit episodes or frames per episode
    python scripts/decode_libero_to_hdf5.py \
        --data-dir /path/to/libero_10 --max-eps 50 --max-frames 100

After conversion, train with:
    python training/pretrain.py --data ./data/libero_10.h5 --epochs 50
    python training/pretrain_streaming.py \
        --data-dir /path/to/data/libero_10.h5 \
        --epochs-pretrain-encoder 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# ================================================================
# Constants
# ================================================================

DEFAULT_SIZE = (224, 224)  # resize frames to this
DEFAULT_CAMERA = "wrist_image"
TRAJ_WINDOW = 20  # K — trajectory window size used during training
MAX_FRAMES_PER_EP = 2000  # cap episode length


def decode_libero_to_hdf5(
    data_dirs: list[str],
    output_path: str,
    camera: str = DEFAULT_CAMERA,
    image_size: tuple = DEFAULT_SIZE,
    max_eps: int = 0,
    max_frames: int = MAX_FRAMES_PER_EP,
    skip_existing: bool = True,
) -> str:
    """Decode LIBERO episodes from multiple data directories into HDF5.

    Each data_dir should be a LIBERO sub-task directory (e.g. libero_10/)
    containing the standard LeRobot v3 structure:
        meta/info.json, meta/tasks.parquet, meta/episodes/
        data/chunk-XXX/file-XXX.parquet
        videos/observation.images.wrist_image/chunk-XXX/file-XXX.mp4
        videos/observation.images.image/chunk-XXX/file-XXX.mp4

    Args:
        data_dirs: List of paths to LIBERO sub-task directories.
        output_path: Output .h5 file path.
        camera: Camera view ('wrist_image' or 'image').
        image_size: Resize frames to (H, W).
        max_eps: Max episodes to process (0 = all).
        max_frames: Max frames per episode.
        skip_existing: Skip if output file already exists.

    Returns:
        Path to created HDF5 file.
    """
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"Output exists: {output_path}")
        reply = input("Overwrite? [y/N] ")
        if reply.lower() != "y":
            print("Aborted.")
            return str(output_path)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_eps = 0

    with h5py.File(str(output_path), "w") as h5:
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            print(f"\nProcessing: {data_path}")
            print(f"  Camera:   {camera}")

            # Load LeRobot dataset (reads MP4s + parquet, uses pyav)
            ds = LeRobotDataset(
                repo_id="nvidia/LIBERO_LeRobot_v3",
                root=str(data_path.parent),
                video_backend="pyav",
            )
            n_total = len(ds)
            n_process = min(n_total, max_eps) if max_eps > 0 else n_total
            print(f"  Episodes: {n_process} / {n_total}")

            # Camera key in LeRobot dataset
            cam_key = f"observation.images.{camera}"
            if cam_key not in ds.features:
                # Fall back to front camera
                alt = f"observation.images.image"
                print(f"  WARNING: '{cam_key}' not found, using '{alt}'")
                cam_key = alt

            # Track current episode boundaries
            ep_indices = {}  # ep_idx → list of sample indices
            for idx in range(n_process):
                sample = ds[idx]
                ep_idx = sample.get("episode_index", idx // 200)
                if isinstance(ep_idx, np.ndarray):
                    ep_idx = ep_idx.item()
                elif isinstance(ep_idx, torch.Tensor):
                    ep_idx = ep_idx.item()
                ep_indices.setdefault(int(ep_idx), []).append(idx)

            ep_keys = sorted(ep_indices.keys())
            print(f"  Unique episodes: {len(ep_keys)}")

            start_time = time.time()
            for ep_i, ep_key in enumerate(ep_keys):
                indices = ep_indices[ep_key]
                ep_name = f"ep_{total_eps + ep_i:05d}"
                group = h5.create_group(ep_name)
                n_frames = min(len(indices), max_frames)

                frames_list = []
                poses_list = []
                text = "pick and place"

                for j, idx in enumerate(indices[:max_frames]):
                    sample = ds[idx]

                    # --- Frame ---
                    img = sample.get(cam_key)
                    if img is None:
                        continue
                    if hasattr(img, "dim"):
                        if img.dim() == 4:
                            img = img[-1]
                        if img.dim() == 3 and img.shape[0] in (1, 3):
                            img = img.permute(1, 2, 0)
                        if img.dtype == torch.float32 or img.dtype == torch.float16:
                            img = img.mul(255).clamp(0, 255).to(torch.uint8)
                        elif img.dtype != torch.uint8:
                            img = img.to(torch.uint8)
                    img_np = img.cpu().numpy() if hasattr(img, "cpu") else np.array(img)

                    # Resize if needed
                    if img_np.shape[:2] != image_size:
                        from PIL import Image as PILImage
                        img_np = np.array(PILImage.fromarray(img_np).resize(image_size))

                    frames_list.append(img_np)

                    # --- Pose (EEF position + axis-angle) ---
                    state = sample.get("observation.state")
                    if state is not None:
                        if hasattr(state, "cpu"):
                            state = state.float().cpu().numpy()
                        if state.ndim == 2:
                            state = state[-1]
                        pose = state[:6] if len(state) >= 6 else np.pad(state, (0, 6 - len(state)))
                    else:
                        pose = np.zeros(6, dtype=np.float32)
                    poses_list.append(pose)

                    # --- Text ---
                    task = sample.get("task", sample.get("language_instruction", None))
                    if task is not None:
                        if isinstance(task, bytes):
                            task = task.decode("utf-8")
                        elif isinstance(task, np.ndarray):
                            task = str(task.item()) if task.ndim == 0 else str(task[0])
                        elif hasattr(task, "item"):
                            task = str(task.item())
                        else:
                            task = str(task)
                        text = task

                if not frames_list:
                    continue

                # Write to HDF5
                frames_arr = np.stack(frames_list).astype(np.uint8)
                poses_arr = np.stack(poses_list).astype(np.float32)

                group.create_dataset(
                    f"frames/{camera}", data=frames_arr,
                    chunks=(1, *frames_arr.shape[1:]),
                    compression="lzf",
                )
                group.create_dataset(
                    "noisy_poses", data=poses_arr,
                    compression="gzip",
                )

                # Text variants
                text_variants = [
                    text,
                    f"pick and place",  # generic
                    f"grasp and move",
                    "complete the task",
                ]
                group.create_dataset("texts", data=json.dumps(text_variants))
                group.create_dataset("meta", data=json.dumps({"task": text}))

                if (ep_i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    eps_per_sec = (ep_i + 1) / elapsed
                    remaining = (len(ep_keys) - ep_i - 1) / eps_per_sec
                    print(f"  {ep_i + 1}/{len(ep_keys)}  "
                          f"{eps_per_sec:.1f} eps/s  "
                          f"ETA {remaining:.0f}s")

            total_eps += len(ep_keys)

        # Write metadata
        h5["meta/total_episodes"] = total_eps
        h5["meta/camera"] = camera
        h5["meta/source"] = json.dumps(data_dirs)

    # Report size
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nDone: {output_path}")
    print(f"  Episodes: {total_eps}")
    print(f"  Size:     {size_mb:.0f} MB")
    print(f"\nTrain with:")
    print(f"  python training/pretrain.py --data {output_path} --epochs 50")
    print(f"  python training/train_heads.py --data {output_path} "
          f"--pretrained checkpoints/pretrain/best.pt")

    return str(output_path)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decode LIBERO LeRobot data to HDF5 for fast training"
    )
    parser.add_argument("--data-dir", required=True, action="append",
                        help="LIBERO sub-task directory (repeatable)")
    parser.add_argument("--output", required=True,
                        help="Output .h5 file path")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        choices=["wrist_image", "image"],
                        help="Camera view")
    parser.add_argument("--image-size", type=int, nargs=2,
                        default=list(DEFAULT_SIZE),
                        help="Resize to H W (default: 224 224)")
    parser.add_argument("--max-eps", type=int, default=0,
                        help="Max episodes per sub-task (0 = all)")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_EP,
                        help="Max frames per episode")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Overwrite existing output without prompt")
    args = parser.parse_args()

    decode_libero_to_hdf5(
        data_dirs=args.data_dir,
        output_path=args.output,
        camera=args.camera,
        image_size=tuple(args.image_size),
        max_eps=args.max_eps,
        max_frames=args.max_frames,
        skip_existing=not args.yes,
    )


if __name__ == "__main__":
    # Torch is imported lazily (inside the function) to avoid import order
    # issues with the rdt/align conda environment
    import torch  # noqa: F401 — ensures torch is available for lerobot
    main()
