#!/usr/bin/env python3
"""Convert local LeRobot v3 LIBERO dataset to ALIGN-compatible HDF5 format.

Reads locally cached LeRobot parquet/video data and exports it as a single HDF5
file for efficient training with `pretrain.py`. Progress is continuously updated
on-screen during the slow video decoding steps.

Usage:
    python scripts/decode_libero_to_hdf5.py --data-dir ~/.cache/huggingface/lerobot/nvidia/LIBERO_LeRobot_v3/libero_10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import h5py
import json
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("ERROR: `lerobot` is not installed. Run: pip install lerobot", file=sys.stderr)
    sys.exit(1)


def write_ep_to_hdf5(f, ep_idx, camera_key, frames, states, text):
    """Write a single episode buffer to the HDF5 file."""
    gr = f.create_group(f"ep_{ep_idx:06d}")

    # Stack frames (T, H, W, 3) and cast to uint8 if needed
    stacked = torch.stack(frames).permute(0, 2, 3, 1)
    if stacked.dtype != torch.uint8:
        max_val = stacked.max().item()
        if max_val <= 1.0:
            stacked = (stacked * 255.0).byte()
        else:
            stacked = stacked.byte()
    gr.create_dataset(f"frames/{camera_key}", data=stacked.numpy())

    # Stack states (T, D) and cast to float32
    s_stack = torch.stack(states).float()
    if s_stack.shape[-1] > 6:
        s_stack = s_stack[:, :6]
    elif s_stack.shape[-1] < 6:
        pad = torch.zeros(s_stack.size(0), 6 - s_stack.shape[-1], device=s_stack.device)
        s_stack = torch.cat([s_stack, pad], dim=1)
    gr.create_dataset("noisy_poses", data=s_stack.numpy().astype(np.float32))

    # Metadata for this episode
    variant_text = text.strip() if text else "pick and place"
    gr.create_dataset(
        "texts",
        data=json.dumps([variant_text, f"complete the {variant_text} task"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Decode LeRobot v3 subset → ALIGN HDF5")
    parser.add_argument("--data-dir", required=True, help="Path to local dataset (e.g., .../LIBERO_LeRobot_v3/libero_10)")
    parser.add_argument("--output", default="libero_align.h5", help="Destination HDF5 file path")
    args = parser.parse_args()

    print(f"[1/4] Loading LeRobot metadata from {args.data_dir}")
    ds = LeRobotDataset("nvidia/LIBERO_LeRobot_v3", root=args.data_dir)

    # Detect keys
    img_key = [k for k in ds.meta.features if "images" in k and "wrist" not in k]
    img_key = img_key[0] if len(img_key) > 0 else None
    state_key = [k for k in ds.meta.features if k.startswith("observation.state")]
    state_key = state_key[0] if len(state_key) > 0 else None

    if not img_key or not state_key:
        raise ValueError(f"Could not find image or state keys. Found: {list(ds.meta.features.keys())}")

    print(f"[2/4] Dataset keys -> Image: '{img_key}', State: '{state_key}'")

    # Group rows into episodes on disk
    ep_buffer = {"frames": [], "states": [], "task": None}
    cur_ep_id = None
    episodes_processed = 0
    total_rows = len(ds.hf_dataset)
    
    print(f"[3/4] Decoding videos & writing HDF5 (this takes a while)...")
    pbar = tqdm(total=total_rows, desc="Decoding frames", unit="frame", postfix={"ep": 0})

    with h5py.File(args.output, "w") as f:
        f.create_group("meta")
        f["meta/camera"] = img_key.split(".")[-1]
        f["meta/source"] = "nvidia/LIBERO_LeRobot_v3"
        
        for row_idx in range(total_rows):
            sample = ds[row_idx]

            # Find the episode index key (varies by LeRobot version)
            ep_keys = [k for k in sample.keys() if "episode" in k.lower()]
            if not ep_keys:
                raise ValueError("Could not find an 'episode' key in sample! Available keys: " + str(list(sample.keys())))
            
            # Handle both integer strings and floats
            raw_ep_id = sample[ep_keys[0]]
            if hasattr(raw_ep_id, 'item'):
                raw_ep_id = raw_ep_id.item()
            ep_id = int(float(raw_ep_id))

            # Detect episode boundary & flush previous 
            if cur_ep_id is not None and ep_id != cur_ep_id:
                write_ep_to_hdf5(f, cur_ep_id, img_key.split(".")[-1], ep_buffer["frames"], 
                                 ep_buffer["states"], ep_buffer["task"])
                
                episodes_processed += 1
                pbar.set_postfix({"episodes written": episodes_processed})
                ep_buffer = {"frames": [], "states": [], "task": None}

            # Accumulate data for current timestep
            frame_tensor = sample[img_key]
            if hasattr(frame_tensor, 'dim') and frame_tensor.dim() == 4:  # (T, C, H, W) -> take last T=1 or flatten
                frame_tensor = frame_tensor[-1]
            ep_buffer["frames"].append(frame_tensor)

            state_tensor = sample[state_key]
            if hasattr(state_tensor, 'dim') and state_tensor.dim() == 2:
                if state_tensor.size(0) > 1:
                     state_tensor = state_tensor[0] # typically T=1 here per row 
                else:
                     state_tensor = state_tensor[-1]
            ep_buffer["states"].append(state_tensor)

            # Grab task instruction once for the episode
            if "task" in sample and ep_buffer["task"] is None:
                t = sample["task"]
                ep_buffer["task"] = str(t) if not isinstance(t, str) else t

            cur_ep_id = ep_id
            pbar.update(1)
            pbar.set_postfix({"episodes written": episodes_processed})

        # Flush final episode
        write_ep_to_hdf5(f, cur_ep_id, img_key.split(".")[-1], ep_buffer["frames"], 
                         ep_buffer["states"], ep_buffer["task"])
        episodes_processed += 1

    pbar.close()
    print(f"\n[4/4] Done! Exported {episodes_processed} episodes to {args.output}")


if __name__ == "__main__":
    main()
