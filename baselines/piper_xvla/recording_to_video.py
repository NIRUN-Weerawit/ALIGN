"""Convert inference recordings to timestamp-paced H.264 videos.

From baselines/: python3 -m piper_xvla.recording_to_video [recording-or-parent-dir]
Defaults to all inference_recordings sessions, global view on the left and wrist
on the right. The output FPS duplicates frames to preserve the recorded timing;
it does not change playback speed. Requires ffmpeg with libx264.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

DEFAULT_RECORDINGS = Path(__file__).resolve().parent.parent / "outputs" / "piper_xvla_single_task" / "inference_recordings"


def _frames(directory: Path, roles: list[str]) -> tuple[list[dict], int]:
    frames = []
    missing = 0
    with (directory / "frames.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                timestamp = float(row["timestamp_monotonic_s"])
                if not math.isfinite(timestamp):
                    raise ValueError("nonfinite timestamp")
                images = {role: (directory / row["images"][role]).resolve() for role in roles}
                for path in images.values():
                    path.relative_to(directory)
                if not all(path.is_file() for path in images.values()):
                    missing += 1
                    continue
                frames.append({"timestamp": timestamp, "images": images})
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{directory}/frames.jsonl line {line_number}: {exc}") from exc
    if not frames:
        raise ValueError(f"no complete image frames in {directory}")
    if any(b["timestamp"] <= a["timestamp"] for a, b in zip(frames, frames[1:])):
        raise ValueError(f"frame timestamps must be strictly increasing in {directory}")
    return frames, missing


def convert_recording(directory: Path, destination: Path, *, view: str = "both", fps: float = 30.0, overwrite: bool = False) -> None:
    directory, destination = directory.resolve(), destination.resolve()
    if destination.exists() and not overwrite:
        print(f"Skipping existing video: {destination}")
        return
    roles = ["global", "wrist"] if view == "both" else [view]
    frames, missing = _frames(directory, roles)
    durations = [b["timestamp"] - a["timestamp"] for a, b in zip(frames, frames[1:])]
    durations.append(statistics.median(durations) if durations else 1.0 / fps)
    total_duration = sum(durations)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inference_video_", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        command = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
        for role in roles:
            concat = temporary / f"{role}.ffconcat"
            with concat.open("w") as stream:
                stream.write("ffconcat version 1.0\n")
                for frame, duration in zip(frames, durations):
                    escaped = str(frame["images"][role]).replace("'", "'\\''")
                    stream.write(f"file '{escaped}'\nduration {duration:.9f}\n")
                # Repeat the final image so ffmpeg honors its duration.
                escaped = str(frames[-1]["images"][role]).replace("'", "'\\''")
                stream.write(f"file '{escaped}'\n")
            command.extend(["-f", "concat", "-safe", "0", "-i", str(concat)])
        if view == "both":
            command.extend(["-filter_complex", "[0:v]scale=-2:480,setsar=1[left];[1:v]scale=-2:480,setsar=1[right];[left][right]hstack=inputs=2[out]", "-map", "[out]"])
        else:
            command.extend(["-vf", "scale=-2:480,setsar=1"])
        output = temporary / "video.mp4"
        command.extend(["-t", f"{total_duration:.9f}", "-r", str(fps), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
        print(f"Encoding {directory.name}: {len(frames)} frame pairs, {total_duration:.2f} s, {view}; {missing} missing frames skipped", flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-3000:]}")
        output.replace(destination)
    print(f"Saved: {destination}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_RECORDINGS, help="one recording session or a parent containing sessions")
    parser.add_argument("--view", choices=("both", "global", "wrist"), default="both")
    parser.add_argument("--fps", type=float, default=30.0, help="output frame rate; playback follows recorded timestamps (default: 30)")
    parser.add_argument("--output", type=Path, help="custom MP4 output for one session")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 0 < args.fps <= 240:
        parser.error("--fps must be finite and in (0, 240]")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    root = args.directory.resolve()
    sessions = [root] if (root / "frames.jsonl").is_file() else sorted(path.parent for path in root.glob("*/frames.jsonl"))
    if not sessions:
        parser.error(f"no recordings with frames.jsonl found in {root}")
    if args.output and (len(sessions) != 1 or args.output.suffix.lower() != ".mp4"):
        parser.error("--output requires exactly one session and an .mp4 extension")
    for directory in sessions:
        destination = args.output or directory / f"{args.view}.mp4"
        convert_recording(directory, destination, view=args.view, fps=args.fps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
