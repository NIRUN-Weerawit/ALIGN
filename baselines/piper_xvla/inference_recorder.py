"""Bounded background JPEG recording of camera frames already used by inference."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class InferenceImageRecorder:
    def __init__(self, directory: Path, queue_size: int = 8) -> None:
        self.directory = directory
        for role in ("global", "wrist"):
            (directory / role).mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._saved = self._dropped = 0
        self._error: str | None = None
        self._thread = threading.Thread(target=self._write, name="inference-image-writer", daemon=True)
        self._thread.start()

    def submit(self, global_rgb: np.ndarray, wrist_rgb: np.ndarray, metadata: dict) -> None:
        # SnapshotAdapter allocates independent RGB arrays each cycle. Retain
        # these immutable frames without a second full-image copy or disk wait.
        with self._lock:
            if self._error:
                self._dropped += 1
                return
            try:
                self._queue.put_nowait((global_rgb, wrist_rgb, dict(metadata)))
            except queue.Full:
                self._dropped += 1

    def status(self) -> dict:
        with self._lock:
            return {"enabled": True, "directory": str(self.directory), "saved": self._saved,
                    "dropped": self._dropped, "queued": self._queue.qsize(), "error": self._error}

    def _write(self) -> None:
        import cv2

        manifest = None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                global_rgb, wrist_rgb, metadata = item
                try:
                    if self._error:
                        raise RuntimeError(self._error)
                    if manifest is None:
                        manifest = (self.directory / "frames.jsonl").open("w")
                    paths = {}
                    for role, rgb in (("global", global_rgb), ("wrist", wrist_rgb)):
                        path = Path(role) / f"{metadata['frame']:08d}.jpg"
                        ok, jpeg = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            raise RuntimeError(f"JPEG encoding failed for {role}")
                        (self.directory / path).write_bytes(jpeg.tobytes())
                        paths[role] = str(path)
                    metadata["images"] = paths
                    manifest.write(json.dumps(metadata) + "\n")
                    manifest.flush()
                    with self._lock:
                        self._saved += 1
                except Exception as exc:
                    with self._lock:
                        first_error = self._error is None
                        self._error = str(exc)
                        self._dropped += 1
                    if first_error:
                        LOGGER.error("Image recording disabled after writer error: %s", exc)
        finally:
            if manifest is not None:
                manifest.close()

    def close(self) -> None:
        # Drain accepted frames only after inference/control has stopped.
        self._queue.put(None)
        self._thread.join()
        try:
            (self.directory / "recording.json").write_text(json.dumps(self.status(), indent=2) + "\n")
        except OSError as exc:
            LOGGER.error("Could not save recording summary: %s", exc)
