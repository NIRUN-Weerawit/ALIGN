#!/usr/bin/env python3
"""Convert local Piper LeRobot replays to train_intention.py HDF5.

Example:
    conda run -n align python scripts/decode_piper_to_hdf5.py \
        --data-dir baselines/data/piper_replay_new \
        --output data/piper_replay_new_align.h5

The source stores measured EE state and a commanded absolute EE target as
XYZ + row-major rotation-6D + physical gripper stroke. ALIGN expects an
absolute XYZ/Euler pose for state, but a *relative* 6D action and gripper.
Rotations are converted with the Piper controller's extrinsic xyz convention.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


CAMERAS = {
    "image": "observation.images.global_rgb",
    "wrist_image": "observation.images.wrist_rgb",
}
NUMERIC_COLUMNS = ["observation.state", "action", "episode_index", "frame_index"]


def dataset_roots(paths: list[Path]) -> list[Path]:
    """Accept a LeRobot root, the legacy /dataset wrapper, or baselines/data."""
    roots = []
    for path in paths:
        path = path.expanduser().resolve()
        candidates = [path, path / "dataset"]
        if not any((p / "meta/info.json").is_file() for p in candidates):
            candidates.extend(child for parent in (path, path / "dataset")
                              if parent.is_dir() for child in sorted(parent.iterdir()))
            candidates.extend(child / "dataset" for child in list(candidates))
        found = [p for p in candidates if (p / "meta/info.json").is_file()]
        if not found:
            raise FileNotFoundError(f"No LeRobot meta/info.json under {path}")
        for root in sorted(found):
            if root not in roots:
                roots.append(root)
    return roots


def episodes(root: Path) -> list[tuple[int, Path, str]]:
    """Use metadata to select intact episodes; deleted source files are skipped."""
    records = []
    for meta_file in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(meta_file, columns=[
            "episode_index", "data/chunk_index", "data/file_index", "tasks",
        ])
        for row in table.to_pylist():
            source = root / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
            if source.is_file():
                task = row["tasks"][0] if row["tasks"] else "Piper manipulation"
                records.append((int(row["episode_index"]), source, task))
    if not records:
        raise ValueError(f"No intact episodes under {root}")
    records.sort(key=lambda row: row[0])
    return records


def gripper_range(root: Path) -> tuple[float, float]:
    """Use the raw training set's calibration, shared by state and action."""
    manifest = root / "xvla_training_manifest.json"
    if manifest.is_file():
        values = json.loads(manifest.read_text())["gripper_normalization"]
        low, high = values["raw_meters_min"], values["raw_meters_max"]
    else:
        stats = json.loads((root / "meta/stats.json").read_text())
        low = min(stats[key]["min"][9] for key in ("observation.state", "action"))
        high = max(stats[key]["max"][9] for key in ("observation.state", "action"))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError(f"Invalid gripper calibration for {root}: {low}, {high}")
    return float(low), float(high)


def rotations(raw: np.ndarray) -> Rotation:
    """Decode row-major 3x2 rotations, orthonormalizing noisy measurements."""
    axes = raw[:, 3:9].reshape(-1, 3, 2).astype(np.float64)
    x = axes[:, :, 0]
    y = axes[:, :, 1]
    x_norm = np.linalg.norm(x, axis=1)
    if np.any(x_norm < 1e-6):
        raise ValueError("Invalid Piper rotation-6D first axis")
    x = x / x_norm[:, None]
    y = y - np.sum(x * y, axis=1)[:, None] * x
    y_norm = np.linalg.norm(y, axis=1)
    if np.any(y_norm < 1e-6):
        raise ValueError("Invalid Piper rotation-6D second axis")
    y = y / y_norm[:, None]
    matrix = np.stack((x, y, np.cross(x, y)), axis=-1)
    return Rotation.from_matrix(matrix)


def convert_vectors(states: np.ndarray, targets: np.ndarray,
                    grip_min: float, grip_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute 6D pose, relative 7D action, normalized gripper."""
    if states.ndim != 2 or states.shape[1] != 10 or targets.shape != states.shape:
        raise ValueError(f"Expected matched (N, 10) state/action, got {states.shape} and {targets.shape}")
    if not np.isfinite(states).all() or not np.isfinite(targets).all():
        raise ValueError("Piper state/action contains NaN or infinity")
    state_rot, target_rot = rotations(states), rotations(targets)
    poses = np.concatenate((states[:, :3], state_rot.as_euler("xyz")), axis=1).astype(np.float32)
    delta_rot = (target_rot * state_rot.inv()).as_euler("xyz").astype(np.float32)
    grip_scale = grip_max - grip_min
    grippers = np.clip((states[:, 9] - grip_min) / grip_scale, 0, 1).astype(np.float32)
    target_grip = np.clip((targets[:, 9] - grip_min) / grip_scale, 0, 1).astype(np.float32)
    actions = np.concatenate((targets[:, :3] - states[:, :3], delta_rot, target_grip[:, None]), axis=1).astype(np.float32)
    return poses, actions, grippers


def decode_image(value: dict, root: Path, size: int) -> np.ndarray:
    payload = value.get("bytes")
    if payload is None:
        image_path = root / value["path"]
        source = image_path
    else:
        source = BytesIO(payload)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def write_episode(h5: h5py.File, key: str, root: Path, record: tuple[int, Path, str],
                  size: int, max_frames: int, camera_names: list[str], compression: str) -> int:
    ep_id, source, task = record
    columns = NUMERIC_COLUMNS + [CAMERAS[camera] for camera in camera_names]
    table = pq.read_table(source, columns=columns)
    count = min(table.num_rows, max_frames) if max_frames else table.num_rows
    if count < 2:
        raise ValueError(f"Episode {ep_id} has fewer than two frames")
    numeric = table.select(NUMERIC_COLUMNS).slice(0, count)
    states = np.asarray(numeric["observation.state"].to_pylist(), dtype=np.float32)
    targets = np.asarray(numeric["action"].to_pylist(), dtype=np.float32)
    ids = np.asarray(numeric["episode_index"].to_pylist())
    frames = np.asarray(numeric["frame_index"].to_pylist())
    if not np.all(ids == ep_id) or not np.array_equal(frames, np.arange(count)):
        raise ValueError(f"Episode {ep_id} has mixed IDs or non-contiguous frame indices")
    grip_min, grip_max = gripper_range(root)
    poses, actions, grippers = convert_vectors(states, targets, grip_min, grip_max)

    group = h5.create_group(key)
    group.attrs["source_dataset"] = str(root)
    group.attrs["source_episode_index"] = ep_id
    group.attrs["action_semantics"] = "target position delta; target/current relative xyz rotation; normalized target gripper"
    group.create_dataset("poses", data=poses)
    group.create_dataset("actions", data=actions)
    group.create_dataset("gripper", data=grippers)
    group.create_dataset("texts", data=json.dumps([task]))
    for camera in camera_names:
        dataset = group.create_dataset(
            f"frames/{camera}", shape=(count, size, size, 3), dtype="uint8",
            chunks=(1, size, size, 3), compression=compression,
        )
        for index, value in enumerate(table[CAMERAS[camera]].slice(0, count).to_pylist()):
            dataset[index] = decode_image(value, root, size)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, nargs="+",
                        default=[Path("baselines/data/piper_replay_new")],
                        help="Piper LeRobot root(s), or baselines/data for all Piper datasets")
    parser.add_argument("--output", type=Path, required=True, help="Destination HDF5 file")
    parser.add_argument("--cameras", nargs="+", choices=list(CAMERAS),
                        default=list(CAMERAS), help="ALIGN camera names to export")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episodes", type=int, default=0, help="Limit for smoke conversion; 0 means all")
    parser.add_argument("--max-frames-per-episode", type=int, default=0,
                        help="Limit for smoke conversion; 0 means all")
    parser.add_argument("--compression", choices=["lzf", "gzip"], default="lzf")
    args = parser.parse_args()
    if args.image_size <= 0 or args.max_episodes < 0 or args.max_frames_per_episode < 0:
        parser.error("Image size must be positive and frame/episode limits must be non-negative")
    roots = dataset_roots(args.data_dir)
    work = [(root, record) for root in roots for record in episodes(root)]
    if args.max_episodes:
        work = work[:args.max_episodes]
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        parser.error(f"Partial output already exists: {temporary}")
    total = 0
    try:
        with h5py.File(temporary, "w") as h5:
            meta = h5.create_group("meta")
            meta.create_dataset("cameras", data=json.dumps(args.cameras))
            meta.create_dataset("source", data=json.dumps([str(root) for root in roots]))
            meta.attrs["pose_format"] = "XYZ meters + extrinsic xyz Euler radians"
            meta.attrs["action_format"] = "XYZ delta meters + relative extrinsic xyz Euler radians + normalized target gripper"
            for ep_num, (root, record) in enumerate(work):
                key = f"ep_{ep_num:06d}"
                count = write_episode(h5, key, root, record, args.image_size,
                                      args.max_frames_per_episode, args.cameras, args.compression)
                total += count
                print(f"{ep_num + 1}/{len(work)} {root.name} episode {record[0]}: {count} frames", flush=True)
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Wrote {len(work)} episodes, {total} frames to {args.output}")


if __name__ == "__main__":
    main()
