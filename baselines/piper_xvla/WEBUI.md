# Piper X-VLA Web UI

The web UI is a local control and collection surface for Piper pendant-replay demonstrations.

## Start

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.webui --can can0 --port 8050
```

Open `http://127.0.0.1:8050`.

For a no-hardware UI/API check:

```bash
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.webui --dry-run --port 8050
```

The server defaults to loopback-only (`127.0.0.1`). Keep it there for arm safety. To view it from another machine, use an SSH tunnel instead of exposing an unauthenticated arm-control server:

```bash
ssh -L 8050:127.0.0.1:8050 ucluser@ROBOT_HOST
```

Then browse to `http://127.0.0.1:8050` on the client machine.

## Operational workflow

1. Verify the **Live Cameras** and **Status** panels.
2. In **Runtime Settings**, verify/set camera devices, resolution/FPS, and replay speed. Saving persists camera settings to `piper_xvla/config/piper_cameras.json`.
3. Use the pendant to record and save the drag-teach trajectory, then reset the scene.
4. Set a task description and start **Collect Episode**. The web UI captures at 20 Hz, triggers the validated replay sequence, and stops on the selected cap or the UI Stop button.
5. Review the side-by-side global|wrist H.264 video and the state/action charts before retaining the episode.

## Control ownership

- **Host-owned:** Standby and CAN-control modes, driver enable/disable, reset, E-stop, replay trigger.
- **Pendant-owned:** teaching entry and offline/replay mode lifecycle. The web UI does not send host-side offline/replay `MotionCtrl_2` mode changes.
- Replay uses the verified controller sequence: move to stored start (`grag_teach_ctrl=0x07`) then execute (`0x03`).

## Dataset layout

```text
data/piper_replay/dataset/
├── data/                 # LeRobot state/action parquet
├── meta/                 # episode metadata, statistics, tasks
└── images/review/
    └── episode-000123_review.mp4  # browser-ready H.264 global|wrist review
```

Repeated collection appends a new LeRobot episode to the same dataset. It does not overwrite the dataset root.

## Review videos

New captures are written through OpenCV to a temporary file and finalized as **H.264/yuv420p MP4** with `faststart`, making them seekable in Chromium and the UI player. `ffmpeg` and `ffprobe` are required for this final conversion.

The UI’s **Convert legacy videos to H.264** button migrates older `mp4v` review files in place. The original is restored automatically if a conversion fails.

## Camera ownership

The UI suspends live camera preflight/preview while a collection is opening, recording, or finalizing. This prevents the preview from opening `/dev/video*` concurrently with the collector.

## Verification

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python -m pytest piper_xvla/tests/ -q
```
