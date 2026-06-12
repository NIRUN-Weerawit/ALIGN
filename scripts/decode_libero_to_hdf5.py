#!/usr/bin/env python3
"""Pre-decode LIBERO LeRobot dataset into HDF5 for fast multiprocess training.

Converts local LIBERO data (MP4 videos + Parquet metadata) into HDF5 files
with pre-decoded frames, so the DataLoader can read frames directly
without FFmpeg/video decoding. This:
  - Enables num_workers > 0 (no video decoder contention)
  - Speeds up training ~3-5x (no per-step video decode)
  - Adds ~5-15GB per LIBERO subtask (libero_10, libero_90, etc.)

Both wrist and front cameras are saved by default.

Usage:
    # Convert one subtask (both cameras)
    python scripts/decode_libero_to_hdf5.py \
        --data-dir /path/to/libero_10 \
        --output ./data/libero_10.h5

    # Limit to 50 episodes
    python scripts/decode_libero_to_hdf5.py \
        --data-dir /path/to/libero_10 --max-eps 50

After conversion, train with:
    python training/pretrain_streaming.py \
        --data-dir ./data/libero_10.h5 \
        --epochs-pretrain-encoder 10
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


DEFAULT_SIZE = (224, 224)
MAX_FRAMES_PER_EP = 2000


def decode_libero_to_hdf5(
    data_dirs: list[str],
    output_path: str,
    camera: str = "both",
    image_size: tuple = DEFAULT_SIZE,
    max_eps: int = 0,
    max_frames: int = MAX_FRAMES_PER_EP,
    skip_existing: bool = True,
) -> str:
    """Decode LIBERO episodes from multiple data directories into HDF5."""
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"Output exists: {output_path}")
        reply = input("Overwrite? [y/N] ")
        if reply.lower() != "y":
            print("Aborted.")
            return str(output_path)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch  # noqa: F401

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_eps = 0

    with h5py.File(str(output_path), "w") as h5:
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            print(f"\nProcessing: {data_path}")

            ds = LeRobotDataset(
                repo_id="nvidia/LIBERO_LeRobot_v3",
                root=str(data_path.parent),
                video_backend="pyav",
            )
            n_total = len(ds)
            n_process = min(n_total, max_eps) if max_eps > 0 else n_total
            print(f"  Episodes: {n_process} / {n_total}")

            # Camera keys
            if camera == "both":
                cam_keys = [
                    k for k in ["observation.images.wrist_image", "observation.images.image"]
                    if k in ds.features
                ]
                if not cam_keys:
                    raise ValueError("No camera keys found in dataset features")
                print(f"  Cameras:  {cam_keys}")
            else:
                cam_key = f"observation.images.{camera}"
                if cam_key not in ds.features:
                    alt = "observation.images.image"
                    print(f"  WARNING: '{cam_key}' not found, using '{alt}'")
                    cam_key = alt
                cam_keys = [cam_key]
                print(f"  Camera:   {cam_key}")

            # Build episode index
            ep_indices = {}
            for idx in range(n_process):
                sample = ds[idx]
                ep_idx = sample.get("episode_index", idx // 200)
                if isinstance(ep_idx, np.ndarray):
                    ep_idx = ep_idx.item()
                elif hasattr(ep_idx, "item"):
                    ep_idx = ep_idx.item()
                ep_indices.setdefault(int(ep_idx), []).append(idx)

            ep_keys = sorted(ep_indices.keys())
            print(f"  Unique episodes: {len(ep_keys)}")

            start_time = time.time()
            for ep_i, ep_key in enumerate(ep_keys):
                indices = ep_indices[ep_key]
                ep_name = f"ep_{total_eps + ep_i:05d}"
                group = h5.create_group(ep_name)

                frames_per_cam = {ck: [] for ck in cam_keys}
                poses_list = []
                text = "pick and place"

                for idx in indices[:max_frames]:
                    sample = ds[idx]

                    # Frames from each camera
                    for ck in cam_keys:
                        img = sample.get(ck)
                        if img is None:
                            frames_per_cam[ck].append(
                                np.zeros((*image_size, 3), dtype=np.uint8)
                            )
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
                        if img_np.shape[:2] != image_size:
                            img_np = np.array(PILImage.fromarray(img_np).resize(image_size))
                        frames_per_cam[ck].append(img_np)

                    # Pose (EEF position + axis-angle)
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

                    # Text
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

                if not frames_per_cam[cam_keys[0]]:
                    continue

                # Write frames per camera
                for ck in cam_keys:
                    frames_arr = np.stack(frames_per_cam[ck]).astype(np.uint8)
                    cam_name = ck.split(".")[-1]
                    group.create_dataset(
                        f"frames/{cam_name}", data=frames_arr,
                        chunks=(1, *image_size[0], *image_size[1], 3),
                        compression="lzf",
                    )

                # Write poses
                poses_arr = np.stack(poses_list).astype(np.float32)
                group.create_dataset("noisy_poses", data=poses_arr, compression="gzip")

                # Text variants
                text_variants = [text, "pick and place", "grasp and move"]
                group.create_dataset("texts", data=json.dumps(text_variants))
                group.create_dataset("meta", data=json.dumps({"task": text}))

                if (ep_i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    eps_per_sec = (ep_i + 1) / elapsed
                    remaining = (len(ep_keys) - ep_i - 1) / eps_per_sec if eps_per_sec > 0 else 0
                    print(f"  {ep_i + 1}/{len(ep_keys)}  "
                          f"{eps_per_sec:.1f} eps/s  "
                          f"ETA {remaining:.0f}s")

            total_eps += len(ep_keys)

        h5["meta/total_episodes"] = total_eps
        h5["meta/camera"] = "both" if len(cam_keys) > 1 else cam_keys[0]
        h5["meta/source"] = json.dumps(data_dirs)
        h5["meta/state_dim"] = 6

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nDone: {output_path}")
    print(f"  Episodes: {total_eps}")
    print(f"  Cameras:  {len(cam_keys)}")
    print(f"  Size:     {size_mb:.0f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Decode LIBERO LeRobot data to HDF5 for fast training"
    )
    parser.add_argument("--data-dir", required=True, action="append",
                        help="LIBERO sub-task directory (repeatable)")
    parser.add_argument("--output", required=True, help="Output .h5 file path")
    parser.add_argument("--camera", default="both",
                        choices=["wrist_image", "image", "both"],
                        help="Camera view (default: both)")
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
    main()