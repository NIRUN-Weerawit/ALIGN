# Piper Manual Drag-Teach → Host-Triggered Replay SOP

**Verified on the real Piper (2026-08-31).** This is the data-collection
workflow before the camera/LeRobot collector is attached.

```text
read-only preflight
  → pendant: teach and save a trajectory
  → reset the scene (object + arm start area)
  → host: replay = move-to-start (0x07) + execute (0x03), like the pendant button
  → read-only feedback: confirm actual joint motion
```

The physical pendant owns recording. Replay is fully host-triggered: the console
sends the same two drag-teach commands the pendant's replay button uses, so no
manual priming is needed. The host does not change Piper control mode for this
workflow.

All commands below run from `~/ALIGN/baselines` using the `lerobot` Python:

```bash
PY=~/miniconda3/envs/lerobot/bin/python
```

---

## 1. Preflight — read-only

### Arm health

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. $PY -m piper_xvla.readiness \
  --can can0 --min-stream-hz 10 --warmup-s 1 --json
```

Pass criteria:

```text
ready: true
arm_status: 0
err_code: 0
status/end_pose/joint/gripper streams: at least 10 Hz
```

Do **not** use `ctrl_mode` or `teach_status` as a replay readiness gate. The
SDK can report `TEACHING_MODE (0x02)` even after a successful pendant replay.

### Cameras and workspace

- Global camera: `/dev/video14`
- Wrist/gripper camera: `/dev/video8`
- Confirm both views are non-black, correctly oriented, and show the task.
- Keep camera mounts, object position, lighting, and background fixed.
- Clear the workspace and keep the physical E-stop accessible.

---

## 2. Create a manual demonstration — pendant only

1. On the Piper pendant, select **teach/drag mode**.
2. Start recording with the pendant.
3. Drag one clean complete task trajectory by hand:

   ```text
   safe start → approach → grasp → lift → place → release → retreat
   ```

4. Stop/save recording with the pendant.
5. Reject and re-record any collision, missed grasp, or visibly poor pass.

Do not use `replay_control record-start`, `record-stop`, or `discard` for the
production collection path. They remain SDK experiments, but manual pendant
recording is the validated workflow.

---

## 3. Reset the scene (no pendant replay step needed)

1. Restore the object/basket to its recorded starting position.
2. Confirm the arm's start area is clear.
3. No pendant action is required: the host replay command now performs the
   full pendant sequence itself — **move to trajectory start** then **execute**.

The old workflow required pressing the pendant replay button first (which moved
the arm to the start pose); that priming step is now done in software.

---

## 4. Trigger replay from the host

First inspect the no-motion dry run:

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. $PY -m piper_xvla.replay_control replay
```

It must print exactly two logical control operations, in this order:

```text
MotionCtrl_1(emergency_stop=0x00, track_ctrl=0x00, grag_teach_ctrl=0x07)   # move to start
MotionCtrl_1(emergency_stop=0x00, track_ctrl=0x00, grag_teach_ctrl=0x03)   # execute
```

This mirrors the pendant replay button. The host waits for the arm to settle at
the start pose between the two commands. It does not send `MotionCtrl_2` and
does not change any control mode.

With the workspace clear, trigger physical replay:

```bash
PYTHONPATH=. $PY -m piper_xvla.replay_control replay \
  --live --can can0 --i-understand-this-moves-arm
```

or from the interactive console (typed `yes` required):

```bash
PYTHONPATH=. $PY -m piper_xvla.console --can can0
piper> replay
```

---

## 5. Verify movement — read-only

Run this in a second terminal before triggering replay:

```bash
cd ~/ALIGN/baselines
PYTHONPATH=. $PY -m piper_xvla.motion_watch \
  --can can0 --seconds 60 --poll-hz 20 --motion-threshold-deg 0.05
```

Successful replay is proved by lines such as:

```text
max_joint_delta=1.2120 deg MOVING
```

Use copied joint/EE feedback to prove a replay began. Status fields alone are
not enough.

For each production demonstration, visually confirm:

- global view contains the entire motion;
- wrist view captures gripper/object contact;
- grasp, placement, and release actually happened;
- no collision, drift, or changed scene layout occurred.

---

## 6. Stop / emergency handling

Terminate a replay early:

```bash
PYTHONPATH=. $PY -m piper_xvla.replay_control stop --live --can can0
```

Emergency stop:

```bash
PYTHONPATH=. $PY -m piper_xvla.replay_control estop --live --can can0
```

`estop` stops Piper only; it never commands a return motion. Inspect the arm
and workspace before any explicit recovery command.

---

## Troubleshooting

| Symptom | Meaning / action |
|---|---|
| Host `replay` exits 0 but `motion_watch` stays still | No usable taught trajectory, or pendant state is not ready. Confirm the manual record/save/replay sequence on the pendant and re-record if needed. |
| `ctrl_mode=0x02`, `teach_status=0x00` after a pendant replay | Expected SDK reporting; do not treat it as a replay failure. |
| CAN or feedback stream failure | Stop. Restore connectivity and rerun the read-only readiness check. A power loss may erase controller-RAM trajectories; re-record them. |
| Camera device number changed | Re-identify camera roles before collecting; `/dev/video14` and `/dev/video8` are not permanent USB identities. |

## Collection handoff

Once this manual replay loop is repeatable, the read-only collector should:

```text
start synchronized global/wrist camera + copied Piper feedback capture
→ send direct host replay command
→ detect measured movement and capture the full playback
→ save the raw LeRobot-format episode
```
