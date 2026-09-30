"""Build the one-time numeric cache for Piper X-VLA fine-tuning.

The cache contains converted state/action tensors and source frame indices only.
Camera images stay in the original embedded-PNG Parquet dataset, so this does
not duplicate the 21 GB source data.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from piper_xvla.train_xvla_piper import load_config
from piper_xvla.xvla_dataset import (
    PiperXVLAConversion,
    build_prepared_numeric_cache,
    discover_intact_replay_episodes,
    write_xvla_training_manifest,
)


def status(stage: int, total: int, message: str) -> None:
    print(f"[cache stage {stage}/{total}] {message}", flush=True)


def _read_numeric_source(root: Path, episode_ids: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, actions, indices, episodes = [], [], [], []
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no source Parquet files under {root / 'data'}")
    for number, path in enumerate(paths, start=1):
        table = pq.read_table(path, columns=["observation.state", "action", "index", "episode_index"])
        file_episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        keep = np.isin(file_episodes, list(episode_ids))
        if keep.any():
            states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[keep])
            actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32)[keep])
            indices.append(np.asarray(table["index"].to_numpy(), dtype=np.int64)[keep])
            episodes.append(file_episodes[keep])
        if number % 25 == 0 or number == len(paths):
            print(f"  read numeric columns from {number}/{len(paths)} Parquet files", flush=True)
    if not states:
        raise ValueError("no source frames matched the manifest episode IDs")
    return tuple(np.concatenate(items) for items in (states, actions, indices, episodes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="piper_xvla/config/piper_xvla_single_task.json")
    parser.add_argument(
        "--source-root",
        help="LeRobot replay dataset directory (defaults to <manifest directory>/dataset if present)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing prepared cache")
    args = parser.parse_args()
    total = 5

    status(1, total, "loading training config and manifest")
    config = load_config(args.config)
    import json
    manifest_path = Path(config["manifest"])
    preloaded_numeric = None
    if not manifest_path.exists():
        candidates = [Path(args.source_root)] if args.source_root else [manifest_path.parent / "dataset", manifest_path.parent]
        root = next((candidate.resolve() for candidate in candidates
                     if (candidate / "meta" / "episodes").is_dir() and (candidate / "data").is_dir()), None)
        if root is None:
            raise FileNotFoundError(
                f"training manifest not found at {manifest_path}; set --source-root to a LeRobot dataset directory"
            )
        status(1, total, f"manifest missing; discovering retained episodes under {root}")
        episode_ids = discover_intact_replay_episodes(root)
        preloaded_numeric = _read_numeric_source(root, set(episode_ids))
        states, actions, _, _ = preloaded_numeric
        gripper_min = float(min(states[:, 9].min(), actions[:, 9].min()))
        gripper_max = float(max(states[:, 9].max(), actions[:, 9].max()))
        write_xvla_training_manifest(
            root, manifest_path, episode_ids=episode_ids, frame_count=len(states),
            gripper_min_m=gripper_min, gripper_max_m=gripper_max,
        )
        print(f"  wrote training manifest: {manifest_path}", flush=True)
    manifest = json.loads(manifest_path.read_text())
    cache_path = Path(config["prepared_cache"])
    if cache_path.exists() and not args.force:
        raise FileExistsError(f"prepared cache already exists: {cache_path}; use --force to rebuild")

    status(2, total, "reading numeric source columns without decoding camera images")
    root = Path(manifest["source_dataset"]).resolve()
    if preloaded_numeric is not None:
        states, actions, indices, episodes = preloaded_numeric
    else:
        states, actions, indices, episodes = _read_numeric_source(root, set(manifest["source_episode_ids"]))
    if len(states) != manifest["frame_count"]:
        raise ValueError(f"manifest expects {manifest['frame_count']} frames, found {len(states)}")

    status(3, total, f"converting {len(states):,} state/action labels once")
    gripper = manifest["gripper_normalization"]
    conversion = PiperXVLAConversion(gripper["raw_meters_min"], gripper["raw_meters_max"])
    cache = build_prepared_numeric_cache(states, actions, indices, episodes, conversion)
    cache["source_dataset"] = str(root)
    cache["source_episode_ids"] = [int(x) for x in manifest["source_episode_ids"]]

    status(4, total, "atomically writing compact prepared cache")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, temporary_path)
    os.replace(temporary_path, cache_path)

    status(5, total, f"complete: {cache_path} ({cache_path.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
