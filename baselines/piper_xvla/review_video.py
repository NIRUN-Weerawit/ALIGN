"""Browser-compatible review-video encoding and migration helpers."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_codec(path: str | Path) -> str | None:
    """Return the first video codec name, or None if ffprobe cannot read it."""
    if not ffmpeg_available() or not Path(path).is_file():
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        streams = json.loads(result.stdout).get("streams", [])
        return str(streams[0].get("codec_name")) if streams else None
    except Exception:  # noqa: BLE001
        return None


def transcode_h264(source: str | Path, destination: str | Path) -> tuple[bool, str]:
    """Encode source as browser-safe H.264/MP4 atomically.

    The input is commonly OpenCV's mp4v output. The destination is replaced only
    after ffmpeg produced a complete file, so an interrupted transcode never
    destroys the original review recording.
    """
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        return False, f"source video does not exist: {source}"
    if not ffmpeg_available():
        return False, "ffmpeg/ffprobe is not installed"
    temporary = destination.with_name(destination.stem + ".partial" + destination.suffix)
    try:
        temporary.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(source),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return False, (result.stderr.strip() or f"ffmpeg exited {result.returncode}")
        codec = probe_codec(temporary)
        if codec != "h264":
            return False, f"ffmpeg output codec was {codec!r}, expected h264"
        temporary.replace(destination)
        return True, "h264"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out after 600 seconds"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        temporary.unlink(missing_ok=True)


def ensure_h264(path: str | Path) -> tuple[bool, str]:
    """Convert an existing review video in place only when it is not H.264."""
    path = Path(path)
    codec = probe_codec(path)
    if codec == "h264":
        return True, "h264"
    if codec is None:
        return False, "could not probe video codec"
    source = path.with_suffix(path.suffix + ".source")
    try:
        source.unlink(missing_ok=True)
        path.replace(source)
        ok, message = transcode_h264(source, path)
        if ok:
            source.unlink(missing_ok=True)
            return True, message
        # Restore the original review file if conversion failed.
        if not path.exists() and source.exists():
            source.replace(path)
        return False, message
    except Exception as exc:  # noqa: BLE001
        if not path.exists() and source.exists():
            source.replace(path)
        return False, f"{type(exc).__name__}: {exc}"
