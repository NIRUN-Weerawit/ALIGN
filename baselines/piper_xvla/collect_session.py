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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


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

        adapter = None
        try:
            adapter = PiperSnapshotAdapter.from_camera_config(piper, task, cfg_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not open cameras/Piper snapshot: {exc}") from exc

        config = json.loads(cfg_path.read_text())
        gcfg = config.get("global_camera", {})
        height, width = int(gcfg.get("height", 480)), int(gcfg.get("width", 640))

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        episode_dir = data_dir / "dataset"
        meta_info = episode_dir / "meta" / "info.json"
        if meta_info.is_file():
            print(f"resuming existing dataset at {episode_dir} (appending a new episode)")
            try:
                dataset = LeRobotDataset.resume(repo_id="local/piper-replay", root=episode_dir)
            except Exception as exc:  # noqa: BLE001
                adapter.close()
                raise RuntimeError(
                    f"existing dataset at {episode_dir} could not be opened ({type(exc).__name__}: {exc}). "
                    "It is probably a half-written episode. Delete it and re-collect:\n"
                    f"  rm -rf {episode_dir}"
                ) from exc
        else:
            try:
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
        episode_index = int(getattr(dataset, "num_episodes", 0))
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
