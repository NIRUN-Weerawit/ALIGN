# Quest 3 → One-Arm Piper → LeRobot Data Collection

This package is the framework for collecting one-arm Piper demonstrations with a Quest 3, storing the raw demonstrations in LeRobot format, and later adapting them for X-VLA fine-tuning.

## Canonical contract

Raw LeRobot data preserves the physical Piper contract instead of hiding X-VLA assumptions:

```text
observation.images.global_rgb  uint8 H×W×3
observation.images.wrist_rgb   uint8 H×W×3
observation.state               float32[10]  # EE xyz + rot6d + gripper
action                          float32[10]  # commanded absolute EE target
```

The later X-VLA conversion adds a black third camera and zero-pads each 10-D tensor to X-VLA's 20-D two-arm layout. Piper always occupies the active/left slot; indices `10:20` must be exactly zero.

## Implemented, hardware-independent core

- `schema.py` validates image/state/action invariants and performs Piper EE pose → X-VLA 20-D encoding.
- `teleop.py` maps an anchored Quest controller delta into an **absolute** Piper EE target. It is derived from the dead-reckoning pattern in `ALIGN/scripts/vr_libero_panda.py`.
- `lerobot_adapter.py` writes validated raw Piper 10-D state/action plus the two RGB cameras into a `LeRobotDataset`-compatible `add_frame`/`save_episode` interface.

## Safety rules

- Raw state is measured Piper EE state.
- Raw action is the teleoperator-commanded **absolute** EE target, not the measured state.
- The physical bridge must pass target poses through a workspace, pose-step, rotation, and gripper safety filter before publishing to Piper.
- Never reinterpret X-VLA absolute targets as relative deltas.

## Real-arm readiness check

Before any collection run, use the read-only readiness probe. It opens the Piper SDK connection but never enables, moves, replays, homes, or stops the arm:

```bash
cd /home/ucluser/ALIGN/baselines
PYTHONPATH=. python -m piper_xvla.readiness --can can0 --min-stream-hz 10 --json
```

It checks SDK health, CAN frame rate, status/end-pose/joint/gripper feedback rates, arm fault status, and controller teaching/replay state. It exits `0` only when all required streams meet the threshold and the arm reports no fault; otherwise it exits `2` with an actionable JSON report.

## Interactive console (preferred for manual cycles)

`console.py` is a single long-lived REPL that wraps all of the above plus live
status/motion watching:

```bash
cd /home/ucluser/ALIGN/baselines
python3 -m piper_xvla.console --can can0          # live
python3 -m piper_xvla.console --dry-run           # no hardware, prints calls
```

Quick commands: `status` (includes driver enable state), `watch [seconds]`, `modes`,
`mode standby|can`, `raw-mode <hex>` (never use it for replay), `replay`,
`pause`, `resume`, `stop`, `move-start`, `enable`, `disable`, `reset`,
`cameras`, `estop`, `recover`, `help`, `quit`.

For manual drag-teach production data, use the Piper pendant for recording and
replay-mode selection. The console's `rec-start`, `rec-stop`, `record`, and
`discard` are SDK experiments—not the recommended collection path.

Driver power (`enable` / `disable`) mirrors the official `piper_enable.py` /
`piper_disable.py`: loop `EnableArm(7)` + `GripperCtrl(0,1000,0x01/0x02,0)` and
poll all six `driver_enable_status` flags up to 5 s. **CAN command control
(`mode can`) requires energized drivers first** — this is the prerequisite that
the official VR demo does with `EnableArm(7)` right after connecting.
`reset` mirrors `piper_reset.py`: clear the e-stop flag (`MotionCtrl_1(0x02)`)
then return to standby position-velocity mode.

Safety model: every command that can move the arm asks for a typed `yes`;
`estop` is immediate and needs nothing. `replay` sends only the documented
execute-trajectory command; it never changes control mode.

## Piper drag-teach / replay control (one-shot)

`replay_control.py` is dry-run by default. For the **manual pendant workflow**,
do not use the SDK recording/mode commands. Use the pendant to record a
trajectory and select replay mode, then use the host only to execute it:

```bash
cd /home/ucluser/ALIGN/baselines

# Shows the exact no-motion CAN command.
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.replay_control replay

# Actually execute the trajectory recorded through the pendant.
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.replay_control replay \
  --live --can can0 --i-understand-this-moves-arm
```

`replay` sends exactly:

```python
MotionCtrl_1(emergency_stop=0x00, track_ctrl=0x00, grag_teach_ctrl=0x03)
```

It does **not** send `MotionCtrl_2`, change `MOVE_J`/`MOVE_P`, or gate on
`ctrl_mode` / `teach_status`: the physical pendant owns that lifecycle and the
SDK continues to report teaching mode (`0x02`) after a successful replay.

Verify replay through copied feedback, not status fields:

```bash
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python \
  -m piper_xvla.motion_watch --can can0 --seconds 10 --poll-hz 20
```

Look for `MOVING` with nonzero joint deltas. `estop` only sends an emergency
stop; it never moves the arm. `recover-to-start` remains a separate deliberate
motion command after inspecting and clearing the emergency condition.

## Next implementation slice

Build the hardware bridge as a thin adapter around these pure modules:

```text
Piper manual drag-teach (pendant) → taught trajectory stored in controller
pendant replay mode → host sends direct `replay` execute command
read-only collector → global/wrist RGB + copied joint/EE/gripper feedback at fixed FPS
LeRobotDataset writer → raw 10-D one-arm episodes (20-D padding done at X-VLA conversion)
```

Cameras are confirmed in `config/piper_cameras.json`: global `/dev/video14`, wrist `/dev/video8`.
The LeRobot/PyAV recording environment is fixed and verified.

## Tests

```bash
cd /home/ucluser/ALIGN/baselines
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python -m pytest piper_xvla/tests -q
```
