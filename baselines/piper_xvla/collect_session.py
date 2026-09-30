"""Reusable capture-and-write session for one LeRobot replay episode.

This is the hardware-facing core of `collect`, factored out so both the CLI
console and the web UI can drive it. It owns:

- the read-only camera + Piper snapshot adapter,
- the persistent LeRobot dataset (created on first use, resumed afterwards),
- the side-by-side global|wrist review video writer.

It is deliberately *command-free*: a caller decides when to trigger Piper replay
and when to stop capturing. `finalize()` flushes vectors + episode metadata and
releases the video; `close()` releases resources without finalizing (used on
error paths).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def ensure_writable_datasets_cache() -> Path:
    """Keep LeRobot's local Parquet cache off an unavailable HF_HOME mount."""
    from datasets import config

    configured = Path(config.HF_DATASETS_CACHE).expanduser()
    try:
        configured.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=configured):
            pass
        return configured
    except OSError:
        fallback = Path(os.environ.get(
            "PIPER_XVLA_DATASETS_CACHE", Path.home() / ".cache" / "piper_xvla" / "datasets"
        )).expanduser()
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=fallback):
                pass
        except OSError as exc:
            raise RuntimeError(f"no writable Hugging Face datasets cache at {configured} or {fallback}: {exc}") from exc
        os.environ["HF_DATASETS_CACHE"] = str(fallback)
        config.HF_DATASETS_CACHE = fallback
        if "HF_DATASETS_DOWNLOADED_DATASETS_PATH" not in os.environ:
            config.DOWNLOADED_DATASETS_PATH = fallback / "downloads"
        if "HF_DATASETS_EXTRACTED_DATASETS_PATH" not in os.environ:
            config.EXTRACTED_DATASETS_PATH = fallback / "downloads" / "extracted"
        print(f"Hugging Face datasets cache unavailable at {configured}; using {fallback}")
        return fallback


def _resume_legacy_dataset(repo_id: str, root: Path):
    """Open older LeRobot datasets for writing without loading every frame.

    Older LeRobot has no ``resume`` method: its constructor reads the complete
    dataset into a Hugging Face Arrow cache. Build the same writer state as its
    ``create`` method, using the existing on-disk metadata instead.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root)
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = meta
    obj.repo_id = repo_id
    obj.root = meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.batch_encoding_size = 1
    obj.episodes_since_last_encoding = 0
    obj.vcodec = "libsvtav1"
    obj._encoder_threads = None
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = None
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj._absolute_to_relative_idx = None
    obj.video_backend = None
    obj.writer = None
    obj.latest_episode = None
    obj._current_file_start_frame = None
    obj._lazy_loading = False
    obj._recorded_frames = meta.total_frames
    obj._writer_closed_for_reading = False
    obj._streaming_encoder = None
    return obj


@dataclass
class CollectSession:
    """One in-progress episode: capture sources + writers, not yet finalized."""

    piper: Any
    task: str
    data_dir: Path
    adapter: Any
    dataset: Any
    recorder: Any
    video_writer: Optional[Any]
    video_ok: bool
    episode_index: int
    video_path: Path
    video_work_path: Path
    width: int
    height: int

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def open(
        cls,
        piper: Any,
        task: str,
        data_dir: str | Path,
        camera_config: str | Path | None = None,
        dataset_root: str | Path | None = None,
    ) -> "CollectSession":
        """Open cameras + (create-or-resume) dataset + review-video writer.

        Raises RuntimeError with an operator-readable message on any failure;
        resources opened before the failure are released.
        """
        from piper_xvla.lerobot_adapter import PiperLeRobotEpisode
        from piper_xvla.lerobot_factory import piper_lerobot_features
        from piper_xvla.replay_collector import ReplayEpisodeRecorder
        from piper_xvla.snapshot_adapter import DEFAULT_CAMERA_CONFIG, PiperSnapshotAdapter

        cfg_path = Path(camera_config) if camera_config else DEFAULT_CAMERA_CONFIG
        data_dir = Path(data_dir)
        ensure_writable_datasets_cache()

        adapter = None
        try:
            adapter = PiperSnapshotAdapter.from_camera_config(piper, task, cfg_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not open cameras/Piper snapshot: {exc}") from exc

        config = json.loads(cfg_path.read_text())
        gcfg = config.get("global_camera", {})
        height, width = int(gcfg.get("height", 480)), int(gcfg.get("width", 640))

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        episode_dir = Path(dataset_root) if dataset_root is not None else data_dir / "dataset"
        meta_info = episode_dir / "meta" / "info.json"
        if meta_info.is_file():
            print(f"resuming existing dataset at {episode_dir} (appending a new episode)")
            try:
                resume = getattr(LeRobotDataset, "resume", None)
                if callable(resume):
                    dataset = resume(repo_id="local/piper-replay", root=episode_dir)
                else:
                    dataset = _resume_legacy_dataset("local/piper-replay", episode_dir)
            except Exception as exc:  # noqa: BLE001
                adapter.close()
                raise RuntimeError(
                    f"existing dataset at {episode_dir} could not be opened "
                    f"({type(exc).__name__}: {exc}). The dataset was left untouched; "
                    "check its metadata and data files before retrying."
                ) from exc
        else:
            try:
                # LeRobotDataset.create requires a nonexistent root, but an
                # operator may have already created the empty destination.
                if episode_dir.is_dir() and not any(episode_dir.iterdir()):
                    episode_dir.rmdir()
                dataset = LeRobotDataset.create(
                    repo_id="local/piper-replay",
                    root=episode_dir,
                    fps=20,
                    robot_type="piper",
                    features=piper_lerobot_features(height=height, width=width),
                    use_videos=False,
                )
            except Exception as exc:  # noqa: BLE001
                adapter.close()
                raise RuntimeError(f"could not create dataset at {episode_dir}: {exc}") from exc

        recorder = ReplayEpisodeRecorder(PiperLeRobotEpisode(dataset))

        import cv2
        episode_index = int(dataset.meta.total_episodes)
        review_dir = episode_dir / "images" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        video_path = review_dir / f"episode-{episode_index:06d}_review.mp4"
        # OpenCV's portable mp4v writer is used while capture is active. It is
        # transcoded to H.264 during finalize because browsers may not decode mp4v.
        video_work_path = review_dir / f".episode-{episode_index:06d}_review_capture.mp4"
        video_work_path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(str(video_work_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width * 2, height))
        video_ok = writer.isOpened()

        return cls(
            piper=piper, task=task, data_dir=data_dir, adapter=adapter, dataset=dataset,
            recorder=recorder, video_writer=writer if video_ok else None, video_ok=video_ok,
            episode_index=episode_index, video_path=video_path, video_work_path=video_work_path,
            width=width, height=height,
        )

    # -- per-frame ----------------------------------------------------------

    def snapshot(self):
        return self.adapter.snapshot()

    def write_frame(self, observation) -> int:
        """Append one observation to the dataset and the review video."""
        count = self.recorder.add_observation(observation)
        if self.video_ok:
            left = np.ascontiguousarray(observation.global_rgb[..., ::-1])   # RGB->BGR
            right = np.ascontiguousarray(observation.wrist_rgb[..., ::-1])
            self.video_writer.write(np.hstack((left, right)))
        return count

    @property
    def frames_written(self) -> int:
        return self.recorder._written  # noqa: SLF001 - same-package internal counter

    # -- teardown -----------------------------------------------------------

    def finalize(self) -> int:
        """Flush the episode and make its review video browser-playable."""
        total = self.recorder.finalize()
        try:
            self.dataset.finalize()
        except Exception:  # noqa: BLE001 - frames are already saved; report via caller if needed
            pass
        self._release_video()
        try:
            self.adapter.close()
        except Exception:  # noqa: BLE001
            pass
        if self.video_work_path.is_file():
            from piper_xvla.review_video import transcode_h264

            ok, _ = transcode_h264(self.video_work_path, self.video_path)
            if ok:
                self.video_work_path.unlink(missing_ok=True)
            else:
                # Preserve a review artifact even on a machine without ffmpeg.
                # The Web UI labels this codec and offers migration when ffmpeg is
                # installed later.
                self.video_work_path.replace(self.video_path)
        return total

    def close(self) -> None:
        """Release cameras + video without finalizing the episode (error path)."""
        try:
            self.adapter.close()
        except Exception:  # noqa: BLE001
            pass
        self._release_video()
        self.video_work_path.unlink(missing_ok=True)

    def _release_video(self) -> None:
        if self.video_ok and self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:  # noqa: BLE001
                pass
            self.video_writer = None
            self.video_ok = False
