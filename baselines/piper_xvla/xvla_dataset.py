"""Zero-copy X-VLA view over raw 10-D Piper replay demonstrations.

The legacy dataset embeds PNGs in Parquet and occupies about 21 GB. This adapter
keeps those images in place: it selects only files that still exist and converts
Piper vectors in memory for X-VLA's 20-D bimanual EE6D layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def discover_intact_replay_episodes(root: str | Path) -> list[int]:
    """Return original episode IDs whose referenced Parquet file still exists.

    The legacy capture directory intentionally has deleted source episodes. We
    keep it immutable and select the remaining files explicitly, avoiding a
    21-GB duplicate just to make indices contiguous.
    """
    root = Path(root)
    episodes_path = root / "meta" / "episodes" / "chunk-000"
    rows: list[dict[str, Any]] = []
    for path in sorted(episodes_path.glob("*.parquet")):
        rows.extend(pq.read_table(path, columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pylist())
    intact: list[int] = []
    for row in rows:
        data_path = root / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
        if data_path.is_file():
            intact.append(int(row["episode_index"]))
    return sorted(intact)


def _observed_gripper_limits(root: Path) -> tuple[float, float]:
    """Compute a deterministic calibration from all retained raw action labels."""
    lower, upper = np.inf, -np.inf
    for path in sorted((root / "data").rglob("*.parquet")):
        actions = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
            raise ValueError(f"invalid raw Piper action vectors in {path}")
        lower = min(lower, float(actions[:, 9].min()))
        upper = max(upper, float(actions[:, 9].max()))
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("could not derive a non-degenerate gripper calibration")
    return lower, upper


def open_piper_xvla_dataset(root: str | Path) -> tuple["PiperXVLAAdapterDataset", "PiperXVLAConversion"]:
    """Open only surviving legacy episodes as a zero-copy X-VLA training view."""
    root = Path(root).resolve()
    episodes = discover_intact_replay_episodes(root)
    if not episodes:
        raise ValueError(f"no intact replay episodes found under {root}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {"repo_id": "local/piper-replay", "root": root, "episodes": episodes}
    try:
        source = LeRobotDataset(**kwargs, return_uint8=True)
    except TypeError as exc:
        if "return_uint8" not in str(exc):
            raise
        source = LeRobotDataset(**kwargs)
    lower, upper = _observed_gripper_limits(root)
    conversion = PiperXVLAConversion(lower, upper)
    return PiperXVLAAdapterDataset(source, conversion), conversion


def write_xvla_training_manifest(
    source_root: str | Path,
    output_path: str | Path,
    *,
    episode_ids: list[int],
    frame_count: int,
    gripper_min_m: float,
    gripper_max_m: float,
) -> dict[str, Any]:
    """Persist the small, explicit training view without duplicating image data."""
    if not episode_ids or frame_count <= 0:
        raise ValueError("manifest requires at least one episode and frame")
    PiperXVLAConversion(gripper_min_m, gripper_max_m)
    manifest: dict[str, Any] = {
        "format": "piper-xvla-zero-copy-v1",
        "source_dataset": str(Path(source_root).resolve()),
        "source_episode_ids": [int(x) for x in episode_ids],
        "contiguous_training_episode_count": len(episode_ids),
        "frame_count": int(frame_count),
        "features": {
            "observation.images.global_rgb": {"shape": [480, 640, 3], "dtype": "uint8"},
            "observation.images.wrist_rgb": {"shape": [480, 640, 3], "dtype": "uint8"},
            "observation.state": {"shape": [8], "dtype": "float32"},
            "action": {"shape": [20], "dtype": "float32"},
        },
        "gripper_normalization": {"raw_meters_min": gripper_min_m, "raw_meters_max": gripper_max_m},
        "xvla": {
            "action_mode": "ee6d",
            "real_camera_views": 2,
            "empty_cameras": 1,
            "checkpoint": "xvla-libero",
            "note": "Matches the installed xvla-libero checkpoint: two real 224x224-resized views plus one black view.",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


@dataclass(frozen=True)
class PiperXVLAConversion:
    """Map raw Piper 10-D EE state into active-arm + zero-inactive X-VLA form."""

    gripper_min_m: float
    gripper_max_m: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.gripper_min_m, self.gripper_max_m]).all():
            raise ValueError("gripper calibration must be finite")
        if self.gripper_max_m <= self.gripper_min_m:
            raise ValueError("gripper_max_m must be strictly greater than gripper_min_m")

    def to_xvla20(self, vector: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Pad active arm and map physical gripper stroke monotonically to [0, 1]."""
        if tuple(vector.shape) != (10,):
            raise ValueError(f"raw Piper vector must have shape (10,), got {tuple(vector.shape)}")
        if isinstance(vector, torch.Tensor):
            out = torch.zeros(20, dtype=vector.dtype, device=vector.device)
            out[:9] = vector[:9]
            out[9] = ((vector[9] - self.gripper_min_m) / (self.gripper_max_m - self.gripper_min_m)).clamp(0, 1)
            return out
        raw = np.asarray(vector, dtype=np.float32)
        out = np.zeros(20, dtype=np.float32)
        out[:9] = raw[:9]
        out[9] = np.clip(
            (raw[9] - self.gripper_min_m) / (self.gripper_max_m - self.gripper_min_m), 0.0, 1.0
        )
        return out


class PiperXVLAAdapterDataset(Dataset):
    """Expose raw Piper LeRobot samples as X-VLA-compatible 20-D samples."""

    def __init__(self, source: Any, conversion: PiperXVLAConversion) -> None:
        self.source = source
        self.conversion = conversion

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.source[index])
        sample["observation.state"] = self.conversion.to_xvla20(sample["observation.state"])
        sample["action"] = self.conversion.to_xvla20(sample["action"])
        return sample


class PiperXVLALiberoAdapterDataset(PiperXVLAAdapterDataset):
    """Match the installed xvla-libero checkpoint's 8-D proprio and three views."""

    @staticmethod
    def _raw_state_to_libero8(raw: torch.Tensor, conversion: PiperXVLAConversion) -> torch.Tensor:
        if tuple(raw.shape) != (10,):
            raise ValueError(f"raw Piper state must have shape (10,), got {tuple(raw.shape)}")
        from scipy.spatial.transform import Rotation

        six = raw[3:9].detach().cpu().numpy().copy()
        col0 = six[[0, 2, 4]]
        col1 = six[[1, 3, 5]]
        col0 /= np.linalg.norm(col0)
        col1 -= col0 * np.dot(col0, col1)
        col1 /= np.linalg.norm(col1)
        matrix = np.column_stack((col0, col1, np.cross(col0, col1)))
        quat = torch.as_tensor(Rotation.from_matrix(matrix).as_quat(), dtype=raw.dtype, device=raw.device)
        gripper = conversion.to_xvla20(raw)[9:10]
        return torch.cat((raw[:3], quat, gripper))

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = dict(self.source[index])
        state = raw["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        raw["observation.state"] = self._raw_state_to_libero8(state, self.conversion)
        raw["action"] = self.conversion.to_xvla20(raw["action"])
        raw["observation.images.image"] = raw.pop("observation.images.global_rgb")
        raw["observation.images.image2"] = raw.pop("observation.images.wrist_rgb")
        raw["observation.images.empty_camera_0"] = torch.zeros(
            (3, 224, 224), dtype=raw["observation.images.image"].dtype,
            device=raw["observation.images.image"].device,
        )
        return raw


def build_prepared_numeric_cache(
    raw_states: np.ndarray,
    raw_actions: np.ndarray,
    global_indices: np.ndarray,
    episode_indices: np.ndarray,
    conversion: PiperXVLAConversion,
) -> dict[str, Any]:
    """Convert Piper numeric targets once, keyed by immutable dataset frame index.

    Images are deliberately excluded. They remain in the legacy embedded-PNG
    Parquet data, avoiding a large duplicate cache.
    """
    states = np.asarray(raw_states, dtype=np.float32)
    actions = np.asarray(raw_actions, dtype=np.float32)
    indices = np.asarray(global_indices, dtype=np.int64)
    episodes = np.asarray(episode_indices, dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != 10 or actions.shape != states.shape:
        raise ValueError("prepared cache requires matching raw state/action arrays with shape (N, 10)")
    if len(indices) != len(states) or len(episodes) != len(states) or len(indices) == 0:
        raise ValueError("prepared cache index arrays must be non-empty and match numeric rows")
    if indices.min() < 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("prepared cache global frame indices must be unique non-negative integers")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("prepared cache source vectors must be finite")

    capacity = int(indices.max()) + 1
    state_cache = np.zeros((capacity, 8), dtype=np.float32)
    action_cache = np.zeros((capacity, 20), dtype=np.float32)
    available = np.zeros(capacity, dtype=np.bool_)
    cached_episodes = np.full(capacity, -1, dtype=np.int64)

    col0 = states[:, [3, 5, 7]].copy()
    col1 = states[:, [4, 6, 8]].copy()
    col0_norm = np.linalg.norm(col0, axis=1, keepdims=True)
    if np.any(col0_norm <= 0):
        raise ValueError("prepared cache found degenerate rotation-6D first column")
    col0 /= col0_norm
    col1 -= col0 * np.sum(col0 * col1, axis=1, keepdims=True)
    col1_norm = np.linalg.norm(col1, axis=1, keepdims=True)
    if np.any(col1_norm <= 0):
        raise ValueError("prepared cache found degenerate rotation-6D second column")
    col1 /= col1_norm
    matrices = np.stack((col0, col1, np.cross(col0, col1)), axis=-1)
    from scipy.spatial.transform import Rotation

    state_cache[indices, :3] = states[:, :3]
    state_cache[indices, 3:7] = Rotation.from_matrix(matrices).as_quat().astype(np.float32)
    state_cache[indices, 7] = np.clip(
        (states[:, 9] - conversion.gripper_min_m) / (conversion.gripper_max_m - conversion.gripper_min_m), 0.0, 1.0
    )
    action_cache[indices, :9] = actions[:, :9]
    action_cache[indices, 9] = np.clip(
        (actions[:, 9] - conversion.gripper_min_m) / (conversion.gripper_max_m - conversion.gripper_min_m), 0.0, 1.0
    )
    available[indices] = True
    cached_episodes[indices] = episodes
    return {
        "format": "piper-xvla-prepared-cache-v1",
        "state": torch.from_numpy(state_cache),
        "action": torch.from_numpy(action_cache),
        "available": torch.from_numpy(available),
        "episode_index": torch.from_numpy(cached_episodes),
        "frame_count": len(indices),
    }


def load_prepared_numeric_cache(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load and validate a cache against the exact immutable source manifest."""
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"prepared cache not found: {cache_path}; run python -m piper_xvla.prepare_xvla_cache first"
        )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected_root = str(Path(manifest["source_dataset"]).resolve())
    expected_episodes = [int(x) for x in manifest["source_episode_ids"]]
    if cache.get("format") != "piper-xvla-prepared-cache-v1":
        raise ValueError("prepared cache format is unsupported")
    if cache.get("source_dataset") != expected_root:
        raise ValueError("prepared cache source dataset does not match manifest")
    if cache.get("source_episode_ids") != expected_episodes:
        raise ValueError("prepared cache episode IDs do not match manifest")
    if cache.get("frame_count") != manifest["frame_count"]:
        raise ValueError("prepared cache frame count does not match manifest")
    if int(cache["available"].sum()) != manifest["frame_count"]:
        raise ValueError("prepared cache availability mask does not match manifest frame count")
    return cache


class CachedPiperXVLALiberoDataset(Dataset):
    """Read images from source data while using precomputed X-VLA numeric tensors."""

    def __init__(self, source: Any, prepared_cache: dict[str, Any]) -> None:
        if prepared_cache.get("format") != "piper-xvla-prepared-cache-v1":
            raise ValueError("unrecognized prepared Piper X-VLA cache format")
        self.source = source
        self.cache = prepared_cache

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = dict(self.source[index])
        global_index = int(raw["index"])
        if global_index >= len(self.cache["available"]) or not bool(self.cache["available"][global_index]):
            raise KeyError(f"prepared cache lacks source frame index {global_index}")
        raw["observation.state"] = self.cache["state"][global_index]
        raw["action"] = self.cache["action"][global_index]
        raw["observation.images.image"] = raw.pop("observation.images.global_rgb")
        raw["observation.images.image2"] = raw.pop("observation.images.wrist_rgb")
        raw["observation.images.empty_camera_0"] = torch.zeros(
            (3, 224, 224), dtype=raw["observation.images.image"].dtype,
            device=raw["observation.images.image"].device,
        )
        return raw
