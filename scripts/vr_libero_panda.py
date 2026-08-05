#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vr_libero_panda.py -- VR Teleoperation for Franka Panda in LIBERO
=================================================================

OVERVIEW
--------
Drives a Franka Panda robot arm inside a LIBERO (MuJoCo) simulation
using a Meta Quest 3 VR controller. Pose data is received over MQTT
from the same WebXR browser client used by
`sim_vr_panda_single.py` (sora2.uclab.jp:1883, topic control/piper-wee),
so the **same browser page works unchanged**.

Key design choice: the existing Isaac Sim script runs a heavy USD-based
simulator; this script targets LIBERO's MuJoCo backend so the same VR
hardware can be used for data collection on tasks that are evaluated in
`eval/eval_libero_v3_trajectory.py` (libero_spatial / libero_goal).

WORKFLOW
--------
1. Connect to MQTT broker, subscribe to VR topic.
2. Initialize LIBERO environment for a chosen task (`--suite`, `--task`).
3. Reset the env. On the **first** trigger-down:
   - Record the controller's current world pose as `pos_start_ctrl`,
     `quat_start_ctrl`.
   - Read the robot's current EEF world pose as `pos_start_ee`,
     `quat_start_ee`.
4. While trigger held: dead-reckon the controller delta into a target
   EEF pose using the world-frame quaternion arithmetic from the
   existing script (no Euler).
5. Convert the absolute target EEF pose into a **LIBERO OSC_POSE
   action** (XYZ delta, axis-angle delta, gripper).
   - LIBERO's default controller is OSC_POSE, action_dim=7, input in
     [-1, 1] mapped to position deltas up to 0.05 m and rotation
     deltas up to 0.5 rad per step.
6. Step the env; capture `agentview` and `robot0_eye_in_hand` images.
7. Button A → reset to the same task.
8. Button B → cycle to the next task in the suite.

MQTT EXPECTED PAYLOAD
---------------------
Same payload as `sim_vr_panda_single.py`:
  {
    "controller_object": {
      "_x": float, "_y": float, "_z": float,
      "_qx": float, "_qy": float, "_qz": float, "_qw": float
    },
    "sending":    bool,
    "grip":       float,
    "trigger":    float,
    "buttonA":    bool,        # reset episode
    "buttonB":    bool,        # next task
    "thumbstick": {x, y} | null
  }

The browser is configured for the Isaac Sim's Z-up world; we reuse the
exact same payload format. VRState.update() converts the position+rot
into the LIBERO world frame (which is also Z-up, but with a different
axis alignment). See `libero_to_js_offset` below.

CONTROLS
---------
  Trigger (hold)        : Enable VR teleoperation (dead reckoning)
  Trigger (release)     : Hold current pose
  Grip button           : Close gripper
  Thumbstick UP/DOWN    : Increase/decrease alpha_pos (live-vs-saved pose blend)
  Thumbstick LEFT/RIGHT : Increase/decrease alpha_rot (live-vs-saved pose blend)
  Button A              : Reset episode (same task)
  Button B              : Cycle to next task in the suite

KEY CONFIGURATION
-----------------
  MQTT broker    : sora2.uclab.jp:1883
  MQTT topic     : control/piper-wee
  Camera outputs : agentview, robot0_eye_in_hand (128 x 128 default)
  Controller     : OSC_POSE (LIBERO default)
  Position tol   : input_max * 0.05 m = 0.05 m per step
  Orientation tol: input_max * 0.5 rad = 0.5 rad per step

DEPENDENCIES
------------
  libero (libero.libero.envs)
  NumPy, SciPy
  OpenCV (for video saving)
  paho-mqtt
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", message=".*paho.*")
import paho.mqtt.client as mqtt_client

# LIBERO imports (delayed to allow --help without libero installed)
LIBERO_AVAILABLE = False
OffScreenRenderEnv = None
get_benchmark = None
get_libero_path = None
try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    LIBERO_AVAILABLE = True
except ImportError as e:
    print(f"[warn] LIBERO not available: {e}", file=sys.stderr)

# --- VR coordinate frame conversion (Three.js -> LIBERO/MuJoCo world) ---
# The browser client sends Three.js coords (Y-up) for the controller.
# LIBERO's MuJoCo world is Z-up but with a different axis mapping than
# Isaac Sim. Empirically the alignment that matches the existing
# pipeline well is:
#   pos_libero = (x_js, z_js, y_js)
#   q_libero   = q_js rotated by a fixed -120 deg around (1,1,1)/sqrt(3)
# The same axis swap is applied to both position and rotation so axes
# stay consistent (this is what the existing script documents as
# "POSITION_OFFSET = R.from_euler('zy', [90,90], degrees=True)").
#
# In practice, LIBERO is also Y-up in world frame (mujoco convention).
# We map JS-Y to LIBERO-Y by negating one axis; the simplest mapping
# that produces sensible behaviour is:
#
#   pos_libero = (x_js,  y_js, -z_js)            # rotate 90 deg around +X
#   q_libero   = Rx(90 deg) * q_js * Rx(-90 deg)
#
# You may need to tweak this offset per scene; it is exposed as
# --js-to-libero-rot-deg / --js-to-libero-pos-axis CLI flags below.

DEFAULT_POS_TRANSFORM = np.array([1.0, 1.0, -1.0])  # multiply raw (x,y,z)
DEFAULT_ROT_OFFSET_DEG = -90.0                       # Rx by this many degrees


# ---------------------------------------------------------------
# VR state container (same shape as the Isaac script for easy comparison)
# ---------------------------------------------------------------
class VRState:
    """Holds the latest MQTT message, accessible from the main sim loop."""

    def __init__(self, pos_axis=DEFAULT_POS_TRANSFORM, rot_offset_deg=DEFAULT_ROT_OFFSET_DEG):
        self.lock = Lock()
        # Pose (in LIBERO world frame, after axis remap)
        self.goal_pos = np.zeros(3)
        self.goal_rot = np.array([0, 0, 0, 1])  # xyzw quaternion
        # Buttons
        self.sending    = False
        self.grip       = False
        self.buttonA    = False
        self.buttonB    = False
        self.thumbstick = None
        # Configurable transforms (so you can re-tune without editing code)
        self.pos_axis = pos_axis
        self.rot_offset_deg = rot_offset_deg
        # Cached rotation matrix for the JS -> LIBERO rotation remap
        self._rot_remap = R.from_euler(
            "x", rot_offset_deg, degrees=True
        )

    def update(self, data: dict):
        with self.lock:
            co = data.get("controller_object", {})
            # Position: multiply each axis by the configured sign
            raw_pos = np.array([
                co.get("_x", 0.0),
                co.get("_y", 0.0),
                co.get("_z", 0.0),
            ])
            self.goal_pos = raw_pos * self.pos_axis

            # Rotation: apply the rotation remap around X
            q_raw = np.array([
                co.get("_qx", 0.0),
                co.get("_qy", 0.0),
                co.get("_qz", 0.0),
                co.get("_qw", 1.0),
            ])
            nrm = np.linalg.norm(q_raw)
            if nrm > 1e-8:
                q_raw = q_raw / nrm
            q_remap = (self._rot_remap * R.from_quat(q_raw) *
                       self._rot_remap.inv()).as_quat()
            self.goal_rot = q_remap
            nrm2 = np.linalg.norm(self.goal_rot)
            if nrm2 > 1e-8:
                self.goal_rot = self.goal_rot / nrm2

            self.sending = bool(data.get("sending", False))
            self.grip    = bool(data.get("grip", False))
            self.buttonA = bool(data.get("buttonA", False))
            self.buttonB = bool(data.get("buttonB", False))
            ts = data.get("thumbstick")
            if ts is not None and isinstance(ts, dict):
                tx, ty = ts.get("x", 0), ts.get("y", 0)
                if abs(ty) > abs(tx):
                    self.thumbstick = 1 if ty > 0 else 3
                elif abs(tx) > abs(ty):
                    self.thumbstick = 0 if tx < 0 else 2
                else:
                    self.thumbstick = None
            else:
                self.thumbstick = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "goal_pos":   self.goal_pos.copy(),
                "goal_rot":   self.goal_rot.copy(),
                "sending":    self.sending,
                "grip":       self.grip,
                "buttonA":    self.buttonA,
                "buttonB":    self.buttonB,
                "thumbstick": self.thumbstick,
            }


# ---------------------------------------------------------------
# MQTT setup (same broker/topic as the Isaac script)
# ---------------------------------------------------------------
BROKER      = "sora2.uclab.jp"
PORT        = 1883
CLIENT_ID   = "LIBERO-vr-wee"
TOPIC       = "control/piper-wee"


def mqtt_on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("[MQTT] Connected to broker")
    else:
        print(f"[MQTT] Connect failed: rc={rc}")


def setup_mqtt(vr_state_arg: VRState) -> mqtt_client.Client:
    inst = mqtt_client.Client(
        client_id=CLIENT_ID,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
    )
    inst.on_connect = mqtt_on_connect

    def _on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            vr_state_arg.update(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[MQTT] Bad message: {e}")

    inst.on_message = _on_message
    inst.connect(BROKER, PORT)
    inst.subscribe(TOPIC)
    inst.loop_start()
    print(f"[MQTT] Subscribed: '{TOPIC}' on {BROKER}:{PORT}")
    return inst


# ---------------------------------------------------------------
# LIBERO action conversion
# ---------------------------------------------------------------
def world_pose_to_action(
    current_ee_pos: np.ndarray,
    current_ee_rot: np.ndarray,   # (3, 3) rotation matrix
    target_ee_pos: np.ndarray,
    target_ee_rot: np.ndarray,    # (3, 3) rotation matrix
    grip_closed: bool,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0,
    pos_max: float = 0.05,
    rot_max: float = 0.5,
) -> np.ndarray:
    """Convert absolute target EEF pose into a LIBERO OSC_POSE action.

    Args:
        current_ee_pos / target_ee_pos: (3,) world-frame position.
        current_ee_rot / target_ee_rot: (3, 3) world-frame rotation matrix.
        grip_closed: bool — whether to send gripper close signal.
        pos_scale: multiplier applied to the position delta before
            clipping (lets the operator trade off speed vs. accuracy).
        rot_scale: same for rotation.
        pos_max: LIBERO's OSC_POSE position delta cap (default 0.05 m).
        rot_max: LIBERO's OSC_POSE rotation delta cap (default 0.5 rad).

    Returns:
        action: (7,) np.ndarray in OSC_POSE format:
            [dx, dy, dz, drx, dry, drz, gripper]
            where the first 6 dims are in [-1, 1] and gripper is 1
            (close) / -1 (open).
    """
    # Position delta
    dp = (target_ee_pos - current_ee_pos) * pos_scale
    dp_norm = np.linalg.norm(dp)
    if dp_norm > pos_max and dp_norm > 0:
        dp = dp * (pos_max / dp_norm)
    dp_action = dp / pos_max  # map back to [-1, 1]
    # Rotation delta: target_rot * current_rot^-1 -> axis-angle
    delta_R = target_ee_rot @ current_ee_rot.T
    delta_rotvec = R.from_matrix(delta_R).as_rotvec() * rot_scale
    rot_norm = np.linalg.norm(delta_rotvec)
    if rot_norm > rot_max and rot_norm > 0:
        delta_rotvec = delta_rotvec * (rot_max / rot_norm)
    rot_action = delta_rotvec / rot_max
    # Gripper
    grip_action = -1.0 if grip_closed else 1.0
    return np.concatenate([dp_action, rot_action, [grip_action]]).astype(np.float32)


# ---------------------------------------------------------------
# SLERP helper
# ---------------------------------------------------------------
def slerp_quat(q0_xyzw, q1_xyzw, t):
    from scipy.spatial.transform import Slerp
    t = np.clip(t, 0.0, 1.0)
    if t <= 0.0:
        return q0_xyzw.copy()
    if t >= 1.0:
        return q1_xyzw.copy()
    interp = Slerp([0.0, 1.0], R.from_quat(np.array([q0_xyzw, q1_xyzw])))
    return interp(t).as_quat()


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_goal", "libero_object"],
                        help="LIBERO benchmark suite (default: libero_spatial)")
    parser.add_argument("--task", default=None,
                        help="Task name to run. If omitted, start at the first "
                             "task in the suite; use button B to cycle.")
    parser.add_argument("--render-size", type=int, default=128,
                        help="Camera render size (default: 128)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save rendered frames to ./videos/")
    parser.add_argument("--video-fps", type=int, default=20,
                        help="Output video frame rate")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max env steps per episode")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless rendering (no on-screen viewer)")
    parser.add_argument("--js-to-libero-rot-deg", type=float,
                        default=DEFAULT_ROT_OFFSET_DEG,
                        help="Rotation remap angle (degrees around X-axis) "
                             "for converting JS controller quaternion to "
                             "LIBERO world frame. Tune per scene.")
    parser.add_argument("--js-to-libero-pos-axis", default="1,1,-1",
                        help="Axis sign multiplier for converting JS controller "
                             "position to LIBERO world frame. Default '1,1,-1' "
                             "is a 90-deg rotation around X (JS-Y -> LIBERO-Y, "
                             "JS-Z -> LIBERO-Z negated). Tune per scene.")
    parser.add_argument("--pos-scale", type=float, default=2.0,
                        help="Position-delta multiplier before clipping "
                             "(1.0 = match LIBERO's natural 0.05 m/step cap; "
                             "2.0 = move twice as fast).")
    parser.add_argument("--rot-scale", type=float, default=2.0,
                        help="Rotation-delta multiplier before clipping "
                             "(1.0 = match LIBERO's natural 0.5 rad/step cap; "
                             "2.0 = rotate twice as fast).")
    parser.add_argument("--save-dataset", default=None,
                        help="If set, save each successful episode as an HDF5 "
                             "file compatible with ALIGN's data loader. Path is "
                             "the output directory.")
    args = parser.parse_args()

    if not LIBERO_AVAILABLE:
        print("[error] LIBERO is not installed in this Python environment.")
        print("        Activate the ALIGN conda env: conda activate align")
        sys.exit(1)

    # Parse pos axis
    pos_axis = np.array([float(x) for x in args.js_to_libero_pos_axis.split(",")])
    if pos_axis.shape != (3,):
        print("[error] --js-to-libero-pos-axis must be three comma-separated floats.")
        sys.exit(1)

    vr_state = VRState(pos_axis=pos_axis,
                       rot_offset_deg=args.js_to_libero_rot_deg)

    mqtt_inst = setup_mqtt(vr_state)

    # --- Initialize LIBERO benchmark + task list ---
    benchmark = get_benchmark(args.suite)()
    task_list = [t.name for t in benchmark.tasks]
    if args.task is not None:
        if args.task not in task_list:
            print(f"[error] --task '{args.task}' not in {args.suite}; choices: "
                  f"{task_list}")
            sys.exit(1)
        task_idx = task_list.index(args.task)
    else:
        task_idx = 0
    print(f"[LIBERO] Suite: {args.suite}, {len(task_list)} tasks available.")
    print(f"[LIBERO] Starting at task {task_idx}: {task_list[task_idx]}")

    # --- Pre-build env (one per task; rebuilt when cycling) ---
    def build_env(task_name):
        bddl_path = os.path.join(
            get_libero_path("bddl_files"),
            args.suite,
            # BDDL filename can have variant suffixes; get_benchmark already
            # gave us a normalized task_name but the file may have a
            # hash suffix, so we search the directory.
            "",  # filled below
        )
        # Find the BDDL file matching task_name (it may have a hash suffix)
        bddl_dir = bddl_path
        candidates = [
            f for f in os.listdir(bddl_dir)
            if f.startswith(task_name) and f.endswith(".bddl")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No BDDL file matching '{task_name}' in {bddl_dir}"
            )
        bddl_file = os.path.join(bddl_dir, candidates[0])
        return OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_widths=args.render_size,
            camera_heights=args.render_size,
            reward_shaping=False,
            control_freq=20,
            initialization_noise=None,
            has_renderer=not args.headless,
            has_offscreen_renderer=True,
        )

    # --- VR teleoperation state ---
    pos_start_ee   = None
    pos_start_ctrl = None
    quat_start_ctrl = None
    quat_start_ee   = None
    pos_save  = None  # last target pos (held pose)
    quat_save = None
    alpha_pos = 0.0
    alpha_rot = 0.0
    prev_trigger = False
    saved_videos = []
    saved_episodes = []

    # --- Episode loop ---
    try:
        while task_idx < len(task_list):
            task_name = task_list[task_idx]
            print(f"\n{'=' * 70}")
            print(f"[TASK {task_idx + 1}/{len(task_list)}] {task_name}")
            print(f"{'=' * 70}")

            # Build (or rebuild) env for the current task
            try:
                env = build_env(task_name)
            except Exception as e:
                print(f"[error] failed to build env: {e}")
                task_idx += 1
                continue

            obs = env.reset()
            task_desc = env.language_instruction
            print(f"[TASK] Description: {task_desc}")

            # Capture initial EEF pose (MuJoCo world frame)
            current_ee_pos = env.env._eef_xpos.copy()
            current_ee_rot = env.env._eef_xmat.copy()
            print(f"[INIT] EE pos: {current_ee_pos}")
            print(f"[INIT] EE rot:\n{current_ee_rot}")

            # Reset VR anchors
            pos_start_ee   = None
            pos_start_ctrl = None
            quat_start_ctrl = None
            quat_start_ee   = None
            pos_save  = current_ee_pos.copy()
            quat_save = R.from_matrix(current_ee_rot).as_quat()
            alpha_pos = 0.0
            alpha_rot = 0.0
            prev_trigger = False

            episode_frames = []  # list of (agentview_rgb, wrist_rgb) tuples

            for step in range(args.max_steps):
                d = vr_state.snapshot()
                goal_pos = d["goal_pos"]
                goal_rot = d["goal_rot"]
                sending  = d["sending"]
                grip     = d["grip"]
                buttonA  = d["buttonA"]
                buttonB  = d["buttonB"]
                thumbstick = d["thumbstick"]

                # --- Trigger edge: OFF -> ON ---
                if sending and not prev_trigger:
                    # Capture anchors (current EE pose + current VR pose)
                    pos_start_ee   = current_ee_pos.copy()
                    pos_start_ctrl = goal_pos.copy()
                    quat_start_ctrl = R.from_quat(goal_rot)
                    quat_start_ee   = R.from_matrix(current_ee_rot)
                    print(f"[VR] Trigger pressed — anchors recorded.")

                # --- If never calibrated yet, hold default pose ---
                if quat_start_ee is None:
                    target_pos = pos_save.copy()
                    target_rot = quat_save.copy()
                    prev_trigger = sending
                else:
                    if sending:
                        # Position: additive delta in world frame
                        ctrl_pos_delta = goal_pos - pos_start_ctrl
                        target_pos = (pos_start_ee + ctrl_pos_delta).copy()

                        # Rotation: world-frame delta
                        q_now_ctrl = R.from_quat(goal_rot)
                        rot_delta_world = q_now_ctrl * quat_start_ctrl.inv()
                        # Apply same direction flip as the Isaac script
                        dq = rot_delta_world.as_quat().copy()
                        dq[0] = -dq[0]  # negate x-component
                        rot_delta_world = R.from_quat(dq)
                        target_rot = (quat_start_ee * rot_delta_world).as_quat()
                    else:
                        target_pos = pos_save.copy()
                        target_rot = quat_save.copy()

                    # --- Trigger edge: ON -> OFF -> save FK pose ---
                    if prev_trigger and not sending:
                        pos_save  = target_pos.copy()
                        quat_save = target_rot.copy()
                        print(f"[VR] Trigger released — pose held.")

                    # --- Thumbstick: alpha blending with saved pose ---
                    if thumbstick == 1:
                        alpha_pos = min(alpha_pos + 0.1, 1.0)
                    elif thumbstick == 3:
                        alpha_pos = max(alpha_pos - 0.1, 0.0)
                    elif thumbstick == 0:
                        alpha_rot = min(alpha_rot + 0.1, 1.0)
                    elif thumbstick == 2:
                        alpha_rot = max(alpha_rot - 0.1, 0.0)
                    alpha_pos = round(alpha_pos, 1)
                    alpha_rot = round(alpha_rot, 1)

                    # Lerp position, slerp rotation toward saved pose
                    target_pos_live = target_pos.copy()
                    target_pos = target_pos * (1 - alpha_pos) + pos_save * alpha_pos

                    target_rot_live = target_rot.copy()
                    target_rot = slerp_quat(target_rot, quat_save, alpha_rot)

                    prev_trigger = sending

                # --- Convert target pose -> LIBERO action ---
                target_R = R.from_quat(target_rot).as_matrix()
                action = world_pose_to_action(
                    current_ee_pos=current_ee_pos,
                    current_ee_rot=current_ee_rot,
                    target_ee_pos=target_pos,
                    target_ee_rot=target_R,
                    grip_closed=grip,
                    pos_scale=args.pos_scale,
                    rot_scale=args.rot_scale,
                )

                # --- Step the env ---
                obs, reward, done, info = env.step(action)

                # --- Read new EEF pose ---
                current_ee_pos = env.env._eef_xpos.copy()
                current_ee_rot = env.env._eef_xmat.copy()

                # --- Capture images (if saving video) ---
                if args.save_video:
                    agentview_rgb = obs.get("agentview_image")
                    wrist_rgb = obs.get("robot0_eye_in_hand_image")
                    if agentview_rgb is not None:
                        episode_frames.append((agentview_rgb, wrist_rgb))

                # --- Status ---
                if step % 20 == 0:
                    print(f"  step {step:4d}  EE_pos={current_ee_pos.round(3).tolist()}  "
                          f"reward={reward:.2f}  done={done}")

                # --- Check success ---
                if env.check_success():
                    print(f"\n[SUCCESS] Task completed at step {step}!")
                    done = True

                # --- Episode end conditions ---
                if done or buttonA:
                    reason = "success" if env.check_success() else \
                             ("buttonA reset" if buttonA else "done flag")
                    print(f"[EPISODE] End at step {step} ({reason}).")

                    # Save video if requested
                    if args.save_video and episode_frames:
                        out_dir = Path("videos") / args.suite
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{task_name}.mp4"
                        save_video(episode_frames, str(out_path), args.video_fps)
                        saved_videos.append(str(out_path))
                        print(f"[VIDEO] Saved: {out_path}")

                    # Save dataset if requested
                    if args.save_dataset and done:
                        out_path = Path(args.save_dataset) / f"{task_name}.h5"
                        save_episode(obs, action, str(out_path))
                        saved_episodes.append(str(out_path))
                        print(f"[DATASET] Saved: {out_path}")

                    if buttonA:
                        # Reset same task (don't advance)
                        obs = env.reset()
                        current_ee_pos = env.env._eef_xpos.copy()
                        current_ee_rot = env.env._eef_xmat.copy()
                        pos_start_ee = None
                        pos_start_ctrl = None
                        quat_start_ctrl = None
                        quat_start_ee = None
                        pos_save = current_ee_pos.copy()
                        quat_save = R.from_matrix(current_ee_rot).as_quat()
                        alpha_pos = 0.0
                        alpha_rot = 0.0
                        prev_trigger = False
                        episode_frames = []
                        print(f"[RESET] Re-running task: {task_name}")
                        continue
                    break

            # Cycle to next task on normal completion
            task_idx += 1
            if task_idx < len(task_list):
                print(f"\n[BTN B hint] Press button B to advance to: "
                      f"{task_list[task_idx]}")
                print("[BTN B hint] Or Ctrl-C to exit.")

            # Wait for button B or timeout
            print("[WAIT] Press button B to advance to next task "
                  "(or Ctrl-C to exit).")
            while True:
                d = vr_state.snapshot()
                if d["buttonB"]:
                    # Drain the press so it doesn't double-fire
                    time.sleep(0.3)
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        mqtt_inst.loop_stop()
        mqtt_inst.disconnect()
        if saved_videos:
            print(f"\n[INFO] Saved {len(saved_videos)} videos to ./videos/{args.suite}/")
        if saved_episodes:
            print(f"[INFO] Saved {len(saved_episodes)} dataset files to "
                  f"{args.save_dataset}/")


def save_video(frames, output_path, fps):
    """Save a list of (agentview_rgb, wrist_rgb) tuples as a side-by-side MP4."""
    import cv2
    if not frames:
        return
    h, w, _ = frames[0][0].shape
    # Side-by-side layout
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h)
    )
    for agent, wrist in frames:
        if wrist is None:
            wrist = np.zeros_like(agent)
        # obs returns (H, W, 3) uint8
        combined = np.concatenate([agent, wrist], axis=1)
        # cv2 expects BGR; flip RGB -> BGR
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    out.release()


def save_episode(obs, action, output_path):
    """Save a successful episode as an HDF5 file (lightweight wrapper).

    For full alignment with ALIGN's data loader you'd want to save
    the full per-step trajectory, but this hook captures only the
    final observation + action to validate the format."""
    import h5py
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                try:
                    f.create_dataset(k, data=v)
                except Exception:
                    pass
        f.create_dataset("action", data=action)


if __name__ == "__main__":
    main()